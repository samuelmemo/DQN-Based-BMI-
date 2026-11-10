import os 
from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from tqdm import tqdm
import pickle


def extract_raw_features(trial_EEG):

    """
    Extract raw EEG features from trial_EEG data.  

    """    
    n_trials, nch, ntpoints = trial_EEG.shape
    RAWfeature = trial_EEG.reshape(n_trials, nch * ntpoints)
    print('RAW features are extracted.')



    return RAWfeature


def FTA_features(trials, fs, max_freq):

    """
    Extract fourier Transform amplitudes from EEG trials.

    """


    n_trials, n_channels, n_points = trials.shape
    
    # Frequency axis  and indices for positive frequencies up to max_freq
    freq_res = np.fft.fftfreq(n_points, d=1/fs) 
    pos_freqs = freq_res >= 0
    freques = freq_res[pos_freqs]
    freq_axis_indices= np.where(freques <= max_freq)[0]

    freq_axis_ = freques[freq_axis_indices]
    n_freq_components = len(freq_axis_indices)

    all_features = []

    real_features = []
    imag_features = []

    real_features_arr = np.zeros((n_trials, n_channels, n_freq_components))
    imag_features_arr = np.zeros((n_trials, n_channels, n_freq_components))

    for trial in range(n_trials):
        eeg_fragment = trials[trial, :, :]  # Current trial
        features = []
        real_feat = []
        imag_feat = []


        for ch in range(n_channels):
            fft_values = np.fft.fft(eeg_fragment[ch])  # Normalize the FFT output
            fft_values = fft_values[pos_freqs]  # Keep only positive frequencies
            amps = fft_values[freq_axis_indices]  # Keep only frequencies up to max_freq


            # dc component (real part only) + real and imaginary parts of other frequencies
            feat = [amps[0].real]
            r_feat = [amps[0].real]
            im_feat = [0.0]  # add DC imaginary explicitly
            real_features_arr[trial, ch, 0] = amps[0].real
            imag_features_arr[trial, ch, 0] = 0.0

            for a in amps[1:]:
                # Multiply by 2 to account for negative frequencies
                feat.extend([a.real*2, a.imag*2])
                r_feat.extend([a.real*2])
                im_feat.extend([a.imag*2])
                real_features_arr[trial, ch, len(r_feat)-1] = a.real*2
                imag_features_arr[trial, ch, len(im_feat)-1] = a.imag*2
            features.extend(feat)
            real_feat.extend(r_feat)
            imag_feat.extend(im_feat)

        all_features.append(features)
        real_features.append(real_feat)
        imag_features.append(imag_feat)
    
    real_features = np.array(real_features)  # Shape (n_trials, n_channels * n_freq_components)

    imag_features = np.array(imag_features)  # Shape (n_trials, n_channels  * n_freq_components)

    return np.array(all_features), real_features, imag_features, freq_axis_, n_freq_components, real_features_arr, imag_features_arr  



def downsample_antialias_spline(data: np.ndarray,
                                fs_in: float,
                                target_fs: float = 250.0,
                                cutoff_hz: float = 112.5,
                                iir_order: int = 4) -> np.ndarray:
        
        """
        Downsample a signal with anti-aliasing using a low-pass IIR filter,
        then resample via cubic spline interpolation to target_fs.

        Contract
        - Inputs:
            - data: array-like with time on the last axis (..., n_time)
            - fs_in: original sampling rate (Hz)
            - target_fs: desired sampling rate (Hz), default 250 Hz
            - cutoff_hz: low-pass cutoff prior to resampling (Hz), default 112.5 Hz
            - iir_order: Butterworth filter order, default 4
        - Output:
            - resampled array with shape (..., n_out) where n_out ≈ round(n_time * target_fs / fs_in)

        Notes
        - If cutoff_hz is above Nyquist of fs_in, it is clamped to 0.99 of Nyquist.
        """

        # DATA SHAPE channels x time
        data = np.asarray(data)

        if fs_in <= 0 or target_fs <= 0:
                raise ValueError("fs_in and target_fs must be positive.")

        # Design Butterworth low-pass filter 
        nyq = fs_in / 2.0
        # Normalize cutoff (clamp below 1.0 to satisfy design constraints)
        Wn = cutoff_hz / nyq
        Wn = min(max(Wn, 1e-6), 0.99)
        b, a = butter(iir_order, Wn, btype='low', analog=False)

        # Zero-phase filter along the last axis (time)
        filtered = filtfilt(b, a, data, axis=-1)

        # Build time vectors for cubic spline interpolation
        # number of sample points in the input signal
        n = filtered.shape[-1]
        n_out = int(round(n * float(target_fs) / float(fs_in)))
        n_out = max(n_out, 3)  # ensure at least 3 samples
        if n_out < 3:
            raise ValueError("Input data must have at least 3 time samples for interpolation.")

        t_in = np.arange(n, dtype=float) / float(fs_in)
        t_out = np.arange(n_out, dtype=float) / float(target_fs)

        # Cubic spline interpolation along the last axis
        interp_fn = interp1d(t_in, filtered, kind='cubic', axis=-1,
                                                 bounds_error=False, fill_value='extrapolate', assume_sorted=True)
        resampled = interp_fn(t_out)

        return resampled


def band_pass_filter(data: np.ndarray, fs: float, lowcut: float, highcut: float, order: int = 4) -> np.ndarray:
    """
    Apply a Butterworth band-pass filter to the data.

    Parameters:
    - data: array-like with time on the last axis (..., n_time)
    - fs: sampling rate (Hz)
    - lowcut: low cutoff frequency (Hz)
    - highcut: high cutoff frequency (Hz)
    - order: filter order (default 4)

    Returns:
    - filtered data with the same shape as input
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data, axis=-1)
    return filtered_data

class Large_EEG_Dataset:
    def __init__(self, dir_path, reload_dict_path = None):
        
        self.root_dir = dir_path
        self.reload_dict_path = reload_dict_path   

    def _load_reaching_grasping_data(self):
        """
        Load the Freewill Reaching and Grasping dataset.

        Parameters
        ----------
        root_dir : str
            Path to the 'derivatives/matfiles' folder 
            (e.g., '/path/to/Freewill_Reaching_Grasping/derivatives/matfiles').

        Returns
        -------
        data_dict : dict
            Nested dictionary of the form:
            {
                'sub-01': {
                    'ses-01': <mat data>,
                    'ses-02': <mat data>
                },
                'sub-02': {
                    'ses-01': <mat data>,
                    ...
                },
                ...
            }
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dict_path = os.path.join(current_dir, 'data_dict.pkl')
        print(data_dict_path)

        if os.path.exists(data_dict_path):

            print(f"[Loading from pickle] {data_dict_path}")
            data_dict = pd.read_pickle(data_dict_path)
            return data_dict
        
        elif self.reload_dict_path is not None and os.path.exists(self.reload_dict_path):
            print(f"[Loading from pickle] {self.reload_dict_path}")
            data_dict = pd.read_pickle(self.reload_dict_path)
            return data_dict
        

        else:
            print(f"[Scanning directory] {self.root_dir}")
      

            data_dict = {}
            # Group1: 'sub-01' to 'sub-23'
            # Group2: 'ses-01' to 'ses-0n' (n varies per subject)
            pattern = re.compile(r'(sub-\d+)_ses-(\d+)_task-reachingandgrasping_eeg\.mat')

            mat_paths = []
            for root, _, files in os.walk(self.root_dir):
                for file in files:
                    if file.endswith('.mat'):
                        match = pattern.search(file)
                        if match:
                            subj = match.group(1)       # e.g. 'sub-01'
                            ses = f"ses-{match.group(2)}" # e.g. 'ses-01'
                            mat_path = os.path.join(root, file)
                            mat_paths.append(mat_path)

                            if subj not in data_dict:
                                data_dict[subj] = {}
                            data_dict[subj][ses] = loadmat(mat_path, simplify_cells=True)['data']
        
            
            print(f"[Saving data to pickle] ")
            with open("data_dict.pkl", "wb") as f:
                pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

            return data_dict

    @staticmethod
    def _data_df(data):

        data_table = data["dataTable"]
        df = pd.DataFrame(data_table[1:], columns=data_table[0])
        # Coerce key columns to numeric to avoid object-dtype math issues
        for col in ["Run", "AccStartIndex", "TgtID", "TrialDiscardIndex"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # discard rows where TrialDiscardIndex is 1
        if 'TrialDiscardIndex' in df.columns:
            df = df[df['TrialDiscardIndex'] == 0]
        # Drop rows with NaNs in critical columns
        drop_cols = [c for c in ["Run", "AccStartIndex"] if c in df.columns]
        if drop_cols:
            df = df.dropna(subset=drop_cols)
        # Sort for stable ordering
        sort_cols = [c for c in ["Run", "AccStartIndex"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols)
        df = df.reset_index(drop=True)
        return df


    def get_trials(self, t_start: float, t_end: float, lowcut: float = 0.3, highcut: float = 3.0,
                    subject: str = None, verbose: bool = False) -> dict:
        """
        Get the EEG trials for a specific time window in a given session.

        Parameters:
        t_start (float): Start time in seconds relative to the trial start.
        t_end (float): End time in seconds relative to the trial start.
        lowcut (float): Low cutoff frequency for band-pass filter (default 0.3 Hz).
        highcut (float): High cutoff frequency for band-pass filter (default 3.0 Hz).
        subject (str): Optional. If provided, only extract data for this subject (e.g., 'sub-01').

        Returns:
        run_trials (dict):
        """

    

        default_fs = 250  # default sampling frequency

        trials_data = {}
        data_dict = self._load_reaching_grasping_data()

        if data_dict is None:
            raise RuntimeError("Failed to load dataset.")

        # If subject is specified, filter to just that subject
        if subject is not None:
            if subject not in data_dict:
                raise ValueError(f"Subject {subject} not found in dataset.")
            subjects = [subject]
        else:
            subjects = list(data_dict.keys())

        print("\n========== EEG Trial Extraction ==========\n")
        for subj in tqdm(subjects, desc="Subjects", unit="subject"):
            sessions = data_dict[subj]
            trials_data[subj] = {}

            for ses in tqdm(sessions.keys(), desc=f"  {subj} Sessions", leave=False, unit="session"):
                data = sessions[ses]
                trials_data[subj][ses] = {}
               
                print(f"\n--- Subject: {subj} | Session: {ses} ---")

                raw_data = data['rawEEG']
                fs = data['fs']
                channel_names = data['channelName']
                n_channels = len(channel_names)

                df = self._data_df(data)

                # split df according to the number of runs 
                run_dfs = {key: group for key, group in df.groupby('Run')}

                for run in tqdm(run_dfs.keys(), desc=f"{subj} {ses} Runs", leave=False, unit="run"):
                    run_df = run_dfs[run]
                    if type(run) is not int:
                        run = int(run)

                    run_key = f"run-{run}"
                    trials_data[subj][ses][run_key] = {}

                    data_per_run = raw_data[run - 1]  # run is 1-based

                    # Work on a copy to avoid SettingWithCopy warnings
                    run_df = run_df.copy()

                    # Decide effective sampling rate and adjust indices if resampled
                    if fs != int(default_fs):
                        if verbose:
                            print(f"      Downsampling: {fs} Hz → 250 Hz | {subj} {ses} {run_key}")
                        data_per_run = self.downsample_to_250(data_per_run, fs_in=fs)
                        run_df['AccStartIndex'] = pd.to_numeric(run_df['AccStartIndex'], errors='coerce')
                        run_df['AccStartIndex'] = np.rint(run_df['AccStartIndex'] * (default_fs / fs)).astype(int)
                        fs_effective = default_fs
                    else:
                        run_df['AccStartIndex'] = pd.to_numeric(run_df['AccStartIndex'], errors='coerce').astype(int)
                        fs_effective = fs

                    # apply band-pass filter 0.3-3 Hz after optional downsampling
                    if verbose:
                        print(f"      Band-pass filtering: {lowcut}-{highcut} Hz | {subj} {ses} {run_key}")

                    data_per_run = band_pass_filter(data_per_run, fs=fs_effective, lowcut=lowcut, 
                                                    highcut=highcut, order=4)

                    n_trials = run_df.shape[0]
                    labels = np.zeros((n_trials,), dtype=int)

                    # Number of samples based on effective sampling rate
                    if fs_effective != 250:
                        raise ValueError("fs_effective should be 250 Hz after downsampling.")
                    
                    n_samples = int(np.ceil((t_end - t_start) * fs_effective))
                    
                    if verbose: 
                        print(f"    Extracting {n_trials} trials, {n_samples} samples/trial ({t_start}s to {t_end}s)")

                    eeg_data = np.zeros((n_trials, n_channels, n_samples), dtype=float)

                    # AccStartIndex is 1-based from MATLAB. Converted to 0-based in Python
                    trial_starts = run_df['AccStartIndex'] - 1
                
                    for i, trial_idx in enumerate(trial_starts):
                        start_sample = int(trial_idx + np.ceil(t_start * fs_effective))
                        end_samples = start_sample + n_samples
                        # Clip to bounds and pad if needed
                        start_clip = max(start_sample, 0)
                        end_clip = min(end_samples, data_per_run.shape[1])
                        segment = data_per_run[:, start_clip:end_clip]
                        seg_len = segment.shape[1]
                        if seg_len < n_samples:
                            pad = np.zeros((n_channels, n_samples), dtype=float)
                            dst_start = max(0, -start_sample)
                            pad[:, dst_start:dst_start+seg_len] = segment
                            segment = pad
                        eeg_data[i, :, :] = segment
                        labels[i] = int(pd.to_numeric(run_df["TgtID"].iloc[i], errors='coerce'))

                    if eeg_data.shape[0] != labels.shape[0]:
                        raise ValueError("Mismatch between number of trials and labels.")

                    trials_data[subj][ses][run_key] = {
                        'eeg': eeg_data,
                        'labels': labels,
                        'fs': fs_effective,
                        'orig_fs': fs,
                        'channel_names': channel_names}
                    
        print("\n========== Extraction Complete ==========\n")
        return trials_data
    
    def concatenate_runs_per_section(self, trials_data) -> dict:
        """
        Concatenate runs per section for each subject.

        Parameters:
        - trials_data: Nested dictionary of trials data.

        Returns:
        - concatenated_data: Nested dictionary with concatenated runs per section.
        """

        
        concatenated_data = {}

        for subj, sessions in trials_data.items():
            concatenated_data[subj] = {}
            for ses, runs in sessions.items():
                concatenated_data[subj][ses] = {}
                all_eeg = []
                all_labels = []
                for run_key, run_data in runs.items():
                    # choose only EEG channels [first 31 channels]
                    all_eeg.append(run_data['eeg'][:, :31, :])
                    all_labels.append(run_data['labels'])
                if all_eeg:
                    concatenated_eeg = np.vstack(all_eeg)
                    concatenated_labels = np.concatenate(all_labels)
                    if concatenated_eeg.shape[0] != concatenated_labels.shape[0]:
                        raise ValueError("Mismatch between number of concatenated trials and labels.")
                    concatenated_data[subj][ses] = {
                        'eeg': concatenated_eeg,
                        'labels': concatenated_labels,
                        'fs': runs[next(iter(runs))]['fs'],  # fs from the first run
                        'channel_names': runs[next(iter(runs))]['channel_names'][:31]
                    }
        return concatenated_data
    

    def downsample_to_250(self, data: np.ndarray, fs_in: float,
                           cutoff_hz: float = 112.5, iir_order: int = 4) -> np.ndarray:
        """
        Convenience wrapper to anti-alias and resample data to 250 Hz.

        Parameters
        - data: array-like with time on the last axis (..., n_time)
        - fs_in: original sampling rate (Hz)
        - cutoff_hz: low-pass cutoff prior to resampling (Hz), default 112.5
        - iir_order: Butterworth filter order, default 4

        Returns
        - data resampled to 250 Hz using cubic spline interpolation after anti-alias IIR filtering
        """
        return downsample_antialias_spline(data, fs_in, target_fs=250.0,
                                           cutoff_hz=cutoff_hz, iir_order=iir_order)



def plot_ERPs(trials, channel_index=7, subjects=None, t_start=-0.85, t_end=0.0, ylim=(-40, 40)):
    if subjects is None:
        subjects = ['sub-01', 'sub-03']

    for subject in subjects:
        sections = sorted(trials[subject].keys())
        n_sections = len(sections)

        fig, axes = plt.subplots(1, n_sections, figsize=(10 * n_sections, 6), sharex=True, sharey=True, squeeze=False)
        axes = axes.ravel()

        channel_name = None

        for idx, section in enumerate(sections):
            ax = axes[idx]
            stacked_data_per_section = []
            stacked_labels_per_section = []

            runs = trials[subject][section].keys()
            for run in runs:
                data = trials[subject][section][run]['eeg']
                labels = trials[subject][section][run]['labels']
                ch_names = trials[subject][section][run]['channel_names']

                if channel_name is None and channel_index < len(ch_names):
                    channel_name = ch_names[channel_index]

                # Extract trials for the specified channel
                channel_data = data[:, channel_index, :]
                stacked_data_per_section.append(channel_data)
                stacked_labels_per_section.append(labels)

            # After collecting all runs for this section, compute the average ERP
            if stacked_data_per_section:
                all_data_section = np.vstack(stacked_data_per_section)          # (n_trials, n_samples)
                all_labels_section = np.concatenate(stacked_labels_per_section) # (n_trials,)

                t = np.linspace(t_start, t_end, all_data_section.shape[1])

                # Compute average ERP for each condition
                conditions = np.unique(all_labels_section)
                for cond in conditions:
                    cond_data = all_data_section[all_labels_section == cond]
                    if cond_data.size > 0:
                        avg = np.mean(cond_data, axis=0)
                        std = np.std(cond_data, axis=0)
                        ax.plot(t, avg, label=f'Target {int(cond)}')
                        ax.fill_between(t, avg - std, avg + std, alpha=0.2)

                ax.set_title(f'{section}')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude (µV)')
                if ylim is not None:
                    ax.set_ylim(*ylim)
                ax.grid(True, alpha=0.3)
            else:
                ax.set_visible(False)

        fig.suptitle(f'Average ERP - Subject: {subject}, Channel: {channel_name if channel_name else f"ch-{channel_index}"}')
        # Put a single legend for the whole figure if available
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper right')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


                
    

    

