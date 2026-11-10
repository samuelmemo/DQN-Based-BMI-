import numpy as np
import torch
from data_function import Large_EEG_Dataset
from utils import (Reaching_target_xy, channel_standardizer, train_agent, plot_success_rate)
from models import DQNAgent
import argparse 
import matplotlib.pyplot as plt
import os 
from scipy.io import savemat

parser = argparse.ArgumentParser(description='DQN Training')
parser.add_argument('--dataset_path', type=str, default='.../Freewill_EEG_Reaching_Grasping/derivatives/matfiles', help='Path to the dataset')
parser.add_argument('--t_start', type=float, default=-0.5, help='Start time of the trial')
parser.add_argument('--t_end', type=float, default=0.0, help='End time of the trial')
parser.add_argument('--low_cut', type=float, default=0.1, help='Low cut frequency')
parser.add_argument('--high_cut', type=float, default=4.0, help='High cut frequency')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--model', type=str, default='CNNLSTM_EEG', help='Model name')
parser.add_argument('--montecarlo_runs', type=int, default=10, help='Number of Monte Carlo runs')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--reaching_radius', type=float, default=1.0, help='Reaching radius')
parser.add_argument('--reaching_distance_threshold', type=float, default=0.1, help='Reaching distance threshold')
parser.add_argument('--reaching_center', type=tuple, default=(0.0, 0.0), help='Reaching center')
parser.add_argument('--reward_success', type=float, default=1.0, help='Reward for success')
parser.add_argument('--reward_failure', type=float, default=-1.0, help='Reward for failure')


args = parser.parse_args()


def main(args):
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ################### Data Loading #####################################
    dataset_path = args.dataset_path
    dataset = Large_EEG_Dataset(dataset_path)
    t_start = args.t_start
    t_end = args.t_end
    low_cut = args.low_cut
    high_cut = args.high_cut

    print("Loading trials...")

    trials = dataset.get_trials(t_start=t_start, t_end=t_end, lowcut=low_cut, highcut=high_cut, verbose=False)
    final_trials = dataset.concatenate_runs_per_section(trials)
    print("Data loaded.")

    ################### Organize Data across sessions if across_sessions is True #####################################
    all_curves= []
    for subject in final_trials.keys():
        print(f"Subject: {subject}")
        subject_trials = final_trials[subject]
        for section in subject_trials.keys():
            print(f"  Section: {section}")

            # Check if all_curves file exists for this subject and section
            # If it exists, skip to the next subject/section
            folder = f'{args.model}_success_rate_per_subject_section_band_{low_cut}_{high_cut}'
            if not os.path.exists(folder):
                os.makedirs(folder)
            curve_path = os.path.join(folder, f'success_rates_{subject}_{section}.npz')
            if os.path.exists(curve_path):
                print(f"    Success rate data already exists for {subject} {section}, skipping...")
                continue
            

            sub_data = final_trials[subject][section]['eeg']
            sub_labels = final_trials[subject][section]['labels']
            fs = final_trials[subject][section]['fs']


            n_targets = len(set(sub_labels))
            n_trials, n_channels, n_timepoints = sub_data.shape
            print(f"Number of targets: {n_targets}")
            print(f"Number of trials: {n_trials}")
            print(f"EEG Data shape: {sub_data.shape}")

            # Normalize raw EEG data
            normalized_data = channel_standardizer(sub_data)

            # Reshape for EEGNet: (Trials, 1, Channels, Time)
            if args.model == 'EEGNet':
                normalized_data = normalized_data[:, np.newaxis, :, :]
                
            print(f"Normalized data shape for EEGNet: {normalized_data.shape}")


            # ####################### Environment setup #########################

            reaching_radius = args.reaching_radius
            reaching_distance_threshold = args.reaching_distance_threshold
            reaching_center = args.reaching_center
            reaching_target_xy = Reaching_target_xy(n_targets=n_targets, radius=reaching_radius, center=reaching_center)
            reward_success = args.reward_success
            reward_failure = args.reward_failure


            ####################### Training Parameters #############################

            montecarlo_runs = args.montecarlo_runs
            batch_size = args.batch_size

            
            ####################### Training ########################################

            agent, success_rate = train_agent(
                    normalized_data=normalized_data,
                    labels=sub_labels,
                    epochs=args.epochs,
                    montecarlo_runs=montecarlo_runs,
                    chans=n_channels,
                    batch_size=args.batch_size,
                    samples=n_timepoints,
                    kernLength=fs//2,
                    model_name=args.model,
                    n_targets=n_targets,
                    reaching_target_xy=reaching_target_xy,
                    reaching_center=reaching_center,
                    reaching_radius=reaching_radius,
                    reaching_distance_threshold=reaching_distance_threshold,
                    reward_success=reward_success,
                    reward_failure=reward_failure
                )

            ############# Plotting ############################
            plot_success_rate(success_rate, 'Bohj', subject = subject, session=section,
                              model=args.model,
                              low_band=low_cut,
                               high_band=high_cut,
                               save=True)

            all_curves.append((subject, section, success_rate))

            # save all curves (updated to save per subject and section in case the
            # process is interrupted)
            
            np.savez(os.path.join(folder, f'success_rates_{subject}_{section}.npz'), success_rate=success_rate)
        

            

            # plot all sucess rates together

    plt.figure(figsize=(10, 6))
    # Load data from saved files

    # list npz files in the folder
    folder = f'{args.model}_success_rate_per_subject_section_band_{low_cut}_{high_cut}'
    npz_files = [f for f in os.listdir(folder) if f.endswith('.npz')]

    all_success_rates = []

    for npz_file in npz_files:
        data = np.load(os.path.join(folder, npz_file))
        success_rate = data['success_rate']  # shape: (montecarlo_runs, epochs)
        
        # Keep only epochs where at least one run has data (not all NaN)
        valid_epochs = ~np.all(np.isnan(success_rate), axis=0)
        mean_success_rate = np.nanmean(success_rate[:, valid_epochs], axis=0)
        all_success_rates.append(mean_success_rate)

    # Pad all arrays to the same length for averaging
    max_length = max(len(arr) for arr in all_success_rates)
    padded_rates = [np.pad(arr, (0, max_length - len(arr)), constant_values=np.nan) for arr in all_success_rates]
    padded_rates = np.array(padded_rates)

    # Calculate mean and std across subjects/sections
    mean_rate = np.nanmean(padded_rates, axis=0)
    std_rate = np.nanstd(padded_rates, axis=0)
    actual_epochs = np.sum(~np.isnan(mean_rate))

    # Plot with shaded std
    epochs = range(1, actual_epochs + 1)
    plt.plot(epochs, mean_rate[:actual_epochs], label='Average', linewidth=2, color='blue')
    plt.fill_between(epochs, 
                      mean_rate[:actual_epochs] - std_rate[:actual_epochs],
                      mean_rate[:actual_epochs] + std_rate[:actual_epochs],
                      alpha=0.3, color='blue')

    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Success Rate', fontsize=16)
    #plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(f'{args.model}_all_success_rates_{low_cut}_{high_cut}.png')
    
    # Save figure data as MATLAB file
    savemat(f'{args.model}_all_success_rates_{low_cut}_{high_cut}.mat', 
            {'mean_rate': mean_rate[:actual_epochs], 
             'std_rate': std_rate[:actual_epochs],
             'epochs': np.array(epochs)})
    
    plt.show()

if __name__ == '__main__':
    main(args)
