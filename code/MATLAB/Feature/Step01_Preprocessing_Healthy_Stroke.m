%% Step01_Preprocessing_Healthy_Stroke.m
% Combined preprocessing for both healthy and stroke subjects
% Processes healthy subjects from EDF files and stroke subjects from .mat files
clear; clc;

% Choose which dataset sections and BMI windows to process.
processHealthy = true;
processStroke = false;
processBMI05s = true;
processBMI1s = false;

%% Load Configuration
disp('Loading preprocessing configuration...');
config = Utility_Functions.get_preprocessing_config();
disp('Configuration loaded successfully.');

% Extract variables for convenience
resultsPath = config.resultsPath;
processedHealthyPath = config.healthyProcessedPath;
processedStrokePath = fullfile(resultsPath, 'stroke_preprocessing');
strokeRawBasePath = config.strokeRawBasePath;
healthyBasePath = config.healthyBasePath;
targetFs = config.targetFs;
butterOrder = config.butterOrder;
miBand = config.miBand;
deltaBand = config.deltaBand;
thetaBand = config.thetaBand;
muBand = config.muBand;
betaBand = config.betaBand;
gammaBand = config.gammaBand;
healthyRestSec = config.healthyRestSec;
healthyCueSec = config.healthyCueSec;
strokeTrialWindow = config.strokeTrialWindow;
analysisWindowSize = config.analysisWindowSize;
healthyTask2Runs = config.healthyTask2Runs;
removeChannels = config.removeChannels;
harmonizedLabels = config.harmonizedLabels;

% Subject BMI paths
% Healthy BMI inputs are stored on the G: drive; stroke inputs remain in the project.
bmiHealthyPath = config.healthyBmiInputPath;
bmiStrokePath = config.strokeBmiInputPath;
bmiWindows = {};
if processBMI05s
    bmiWindows{end + 1} = [0 0.5];
end
if processBMI1s
    bmiWindows{end + 1} = [0 1.0];
end

if ~exist(bmiHealthyPath, 'dir')
    mkdir(bmiHealthyPath);
end

if ~exist(bmiStrokePath, 'dir')
    mkdir(bmiStrokePath);
end

% Paths for feature analysis results
analysisResultsPath = fullfile(resultsPath, 'analysis');

%% Display Configuration Summary
Utility_Functions.display_preprocessing_config(config, processedHealthyPath, processedStrokePath);

%% Create output directories if needed
disp('Checking and creating output directories...');
if ~exist(processedHealthyPath, 'dir')
    mkdir(processedHealthyPath);
    disp(['Created directory: ', processedHealthyPath]);
end
if ~exist(processedStrokePath, 'dir')
    mkdir(processedStrokePath);
    disp(['Created directory: ', processedStrokePath]);
end

%% Design Filters Once
disp('Designing bandpass filters...');
filters = Utility_Functions.design_bandpass_filters(targetFs, butterOrder, ...
    deltaBand, thetaBand, miBand, muBand, betaBand, gammaBand);
disp('Filters designed successfully.');

%% Plot Filter Magnitude Spectra
disp('Plotting and saving filter magnitude spectra...');

% Create general filter plots directory in results folder
[resultsDir, ~, ~] = fileparts(processedHealthyPath);
plotsDir = fullfile(resultsDir, 'filter_plots');
if ~exist(plotsDir, 'dir')
    mkdir(plotsDir);
end

% Plot each filter band
Utility_Functions.plot_filter_magnitude_spectra(filters.b_delta, filters.a_delta, plotsDir, 'delta_filter_spectrum', deltaBand);
Utility_Functions.plot_filter_magnitude_spectra(filters.b_theta, filters.a_theta, plotsDir, 'theta_filter_spectrum', thetaBand);
Utility_Functions.plot_filter_magnitude_spectra(filters.b_mi, filters.a_mi, plotsDir, 'mi_filter_spectrum', miBand);
Utility_Functions.plot_filter_magnitude_spectra(filters.b_mu, filters.a_mu, plotsDir, 'mu_filter_spectrum', muBand);
Utility_Functions.plot_filter_magnitude_spectra(filters.b_beta, filters.a_beta, plotsDir, 'beta_filter_spectrum', betaBand);
Utility_Functions.plot_filter_magnitude_spectra(filters.b_gamma, filters.a_gamma, plotsDir, 'gamma_filter_spectrum', gammaBand);

disp('Filter plots saved successfully.');

%% ========================================================================
% PROCESS HEALTHY SUBJECTS
% ========================================================================

if processHealthy
disp('Starting Healthy Subjects Preprocessing...');
disp('Processed healthy path:');
disp(processedHealthyPath);
disp('Healthy EDF base path:');
disp(healthyBasePath);

%% Find Healthy Subject Folders
disp('Finding healthy subject folders...');
healthyFolders = dir(fullfile(healthyBasePath, "S*"));
healthyFolders = healthyFolders([healthyFolders.isdir]);
nHealthySubjects = numel(healthyFolders);

disp(['Healthy subjects found: ', num2str(nHealthySubjects)]);
%% Check if any healthy subjects found
if nHealthySubjects == 0
    warning('No healthy subject folders found. Check healthyBasePath.');
else
    %% Initialize Healthy Summary Struct
    allHealthyResults = struct();

    %% Loop Through Healthy Subjects
    disp(['Processing ', num2str(nHealthySubjects), ' healthy subjects...']);
    for s = 1:nHealthySubjects

        disp(' ');
        disp('==============================');
        disp(['Processing healthy subject ', num2str(s), ' of ', num2str(nHealthySubjects)]);
        disp('==============================');

        %% Subject name
        subjectName = healthyFolders(s).name;
        subjectPath = fullfile(healthyBasePath, subjectName);

        disp(['Subject folder: ', subjectPath]);

        %% Initialize run accumulators
        runEpochsMI = []; % channels x samples x trials (concatenated across runs)
        runEpochsDelta = []; % channels x samples x trials (concatenated across runs)
        runEpochsTheta = []; % channels x samples x trials (concatenated across runs)
        runEpochsMu = []; % channels x samples x trials (concatenated across runs)
        runEpochsBeta = []; % channels x samples x trials (concatenated across runs)
        runEpochsGamma = []; % channels x samples x trials (concatenated across runs)

        runLabels = []; % labels for each trial (concatenated across runs)
        runOnsetSamples = []; % onset sample indices for each trial (concatenated across runs)
        runEventsMI = table();     % valid T1/T2 event timing metadata
        runEventsAll = table();    % all T0/T1/T2 annotations from all runs
        edfFilesUsed = {}; % keep track of which EDF files were successfully processed for this subject

        matchedLabelsSaved = {}; % to store matched labels from the first successful run for this subject
        tEpoch = []; % to store epoch time vector (should be the same across runs)

        %% Loop Through Healthy MI Runs
        for r = healthyTask2Runs

            edfName = sprintf('%sR%02d.edf', subjectName, r);
            edfFile = fullfile(subjectPath, edfName);

            if ~isfile(edfFile)
                error(['Missing EDF file: ' edfFile]);
            end

            disp(['Loading ', edfName]);

            %% Load EDF
            T = edfread(edfFile);
            info = edfinfo(edfFile);

            Utility_Functions.inspect_edf_file(edfFile);

            X_H = Utility_Functions.convert_edf_into_matrix(T);   % [channels x samples]

            %% Get Original Sampling Frequency
            fsOld = Utility_Functions.get_sampling_frequency(info);
            
            %% CPz Re-reference Before Harmonization
            X_H = Utility_Functions.rereference_to_channel( ...
                X_H, info.SignalLabels, 'CPZ');

            %% Harmonize Channels
            [X_harm_H, matchedLabels] = Utility_Functions.apply_channel_harmonization( ...
                X_H, info.SignalLabels, harmonizedLabels, 'healthy', removeChannels);

            if isempty(matchedLabelsSaved)
                matchedLabelsSaved = matchedLabels;
            end

            %% Healthy sampling-rate handling
            % Resample before filtering so filters and epoch sample indexing use targetFs.
            if abs(fsOld - targetFs) > 1e-6
                warning('%s has fsOld = %.1f Hz, not targetFs = %.1f Hz. Resampling before filtering.', ...
                    edfName, fsOld, targetFs);
            
                X_harm_H = Utility_Functions.resample_2d_matrix(X_harm_H, fsOld, targetFs);
            end

            %% Apply Bandpass Filters
            [X_delta_H, X_theta_H, X_mi_H, X_mu_H, X_beta_H, X_gamma_H] = Utility_Functions.apply_bandpass_filters(X_harm_H, filters);

            %% Load Healthy MI Events
            [onsetSamples_H, labels_H, eventsMI_H, allEvents_H] = ...
            Utility_Functions.load_healthy_mi_events( ...
            info, targetFs, subjectName, r, edfName);

            % Validate healthy annotation timing
        if istable(eventsMI_H) && ~isempty(eventsMI_H)
        
            shortT0 = eventsMI_H.prevT0DurationSec < config.analysisWindowSize;
        
            if any(shortT0)
                warning(['%s has %d MI trials where previous T0 duration is shorter than %.2f s. ' ...
                         'These trials may not support the requested BMI/QKTD baseline window.'], ...
                         edfName, sum(shortT0), config.analysisWindowSize);
            end
        
            unusualT0 = eventsMI_H.prevT0DurationSec < 4.0 | eventsMI_H.prevT0DurationSec > 4.5;
        
            if any(unusualT0)
                warning('%s has %d MI trials with previous T0 duration outside [4.0 4.5] s.', ...
                    edfName, sum(unusualT0));
            end
        
            unusualMI = eventsMI_H.miDurationSec < 4.0 | eventsMI_H.miDurationSec > 4.5;
        
            if any(unusualMI)
                warning('%s has %d MI trials with T1/T2 duration outside [4.0 4.5] s.', ...
                    edfName, sum(unusualMI));
            end
        end       

            %% Epoch Each Band Using EDF Annotation Timing
            
            [epochsMI_H, tEpoch, validMask_H, skippedInfo_H] = ...
                Utility_Functions.extract_healthy_full_trials( ...
                X_mi_H, eventsMI_H, targetFs, ...
                config.healthyEpochPreMI_sec, config.healthyEpochPostMI_sec);
            
            [epochsDelta_H, ~, validMask_delta_H, skippedInfo_delta_H] = ...
                Utility_Functions.extract_healthy_full_trials( ...
                X_delta_H, eventsMI_H, targetFs, ...
                config.healthyEpochPreMI_sec, config.healthyEpochPostMI_sec);
            
            [epochsTheta_H, ~, validMask_theta_H, skippedInfo_theta_H] = ...
                Utility_Functions.extract_healthy_full_trials( ...
                X_theta_H, eventsMI_H, targetFs, ...
                config.healthyEpochPreMI_sec, config.healthyEpochPostMI_sec);
            
            [epochsMu_H, ~, validMask_mu_H, skippedInfo_mu_H] = ...
                Utility_Functions.extract_healthy_full_trials( ...
                X_mu_H, eventsMI_H, targetFs, ...
                config.healthyEpochPreMI_sec, config.healthyEpochPostMI_sec);
            
            [epochsBeta_H, ~, validMask_beta_H, skippedInfo_beta_H] = ...
                Utility_Functions.extract_healthy_full_trials( ...
                X_beta_H, eventsMI_H, targetFs, ...
                config.healthyEpochPreMI_sec, config.healthyEpochPostMI_sec);
            
            [epochsGamma_H, ~, validMask_gamma_H, skippedInfo_gamma_H] = ...
                Utility_Functions.extract_healthy_full_trials( ...
                X_gamma_H, eventsMI_H, targetFs, ...
                config.healthyEpochPreMI_sec, config.healthyEpochPostMI_sec);

            if ~isequal(validMask_H, validMask_delta_H, validMask_theta_H, validMask_mu_H, validMask_beta_H, validMask_gamma_H)
                error('Valid trial masks do not match across bands for %s (%s). Data integrity compromised.', subjectName, edfName);
            end

            if isempty(epochsMI_H)
                error(['No valid epochs extracted from ' edfName]);
            end

            labels_valid_H = labels_H(validMask_H);

            %% Concatenate Across Runs
            % After fixed MI-onset epoching, every healthy run should have the same
            % channel x sample dimensions. If not, stop because MI-onset alignment failed.
            
            Utility_Functions.assert_same_epoch_size(runEpochsMI, epochsMI_H, ...
                'runEpochsMI', 'epochsMI_H');
            runEpochsMI = cat(3, runEpochsMI, epochsMI_H);
            
            Utility_Functions.assert_same_epoch_size(runEpochsDelta, epochsDelta_H, ...
                'runEpochsDelta', 'epochsDelta_H');
            runEpochsDelta = cat(3, runEpochsDelta, epochsDelta_H);
            
            Utility_Functions.assert_same_epoch_size(runEpochsTheta, epochsTheta_H, ...
                'runEpochsTheta', 'epochsTheta_H');
            runEpochsTheta = cat(3, runEpochsTheta, epochsTheta_H);
            
            Utility_Functions.assert_same_epoch_size(runEpochsMu, epochsMu_H, ...
                'runEpochsMu', 'epochsMu_H');
            runEpochsMu = cat(3, runEpochsMu, epochsMu_H);
            
            Utility_Functions.assert_same_epoch_size(runEpochsBeta, epochsBeta_H, ...
                'runEpochsBeta', 'epochsBeta_H');
            runEpochsBeta = cat(3, runEpochsBeta, epochsBeta_H);
            
            Utility_Functions.assert_same_epoch_size(runEpochsGamma, epochsGamma_H, ...
                'runEpochsGamma', 'epochsGamma_H');
            runEpochsGamma = cat(3, runEpochsGamma, epochsGamma_H);

            runLabels = [runLabels; labels_valid_H];
            runOnsetSamples = [runOnsetSamples; onsetSamples_H(validMask_H)];
            edfFilesUsed{end+1} = edfFile;

            %% Store full annotation timing metadata
        if ~isempty(allEvents_H) && istable(allEvents_H)
            runEventsAll = [runEventsAll; allEvents_H];
        end
        
        % Store valid MI event timing metadata
        if ~isempty(eventsMI_H) && istable(eventsMI_H)
        
            eventsValid_H = eventsMI_H(validMask_H, :);
        
            if isempty(runEventsMI)
                runEventsMI = eventsValid_H;
            else
                runEventsMI = [runEventsMI; eventsValid_H];
            end
        end
        end

        %% Skip Subject if No Valid Runs
        if isempty(runEpochsMI)
            error(['No valid MI epochs across runs for ' subjectName]);
        end

        %% Build Output Struct
        subjectOut = struct();
        subjectOut.subjectName = subjectName;
        subjectOut.fs = targetFs;
        subjectOut.tEpoch = tEpoch;
        subjectOut.chanLabels = matchedLabelsSaved;
        subjectOut.labels = runLabels;

        subjectOut.referenceMethod = ...
            'CPz rereference before harmonization; CPz removed after rereference because CPz - CPz = 0';
        subjectOut.referenceChannel = 'CPZ';
        
        subjectOut.trialTimeMode = 'annotation_defined_full_trial';
        subjectOut.referenceTime = 'T1/T2 onset';
        subjectOut.cueTimeSec = 0;
        
        subjectOut.baselineWindow = [-config.analysisWindowSize 0];
        subjectOut.miWindow = [0 config.analysisWindowSize];
        subjectOut.trialWindow = [tEpoch(1) tEpoch(end)];
        
        subjectOut.analysisWindowSize = config.analysisWindowSize;
        subjectOut.baselineOnset = -config.analysisWindowSize;
        subjectOut.miOnset = 0;

        % Epoched data: channels x samples x trials
        subjectOut.epochsMI = runEpochsMI;
        subjectOut.epochsDelta = runEpochsDelta;
        subjectOut.epochsTheta = runEpochsTheta;
        subjectOut.epochsMu = runEpochsMu;
        subjectOut.epochsBeta = runEpochsBeta;
        subjectOut.epochsGamma = runEpochsGamma;
        subjectOut.onsetSamples = runOnsetSamples;
        subjectOut.edfFilesUsed = edfFilesUsed;
        subjectOut.eventsMI = runEventsMI;
        subjectOut.eventsAll = runEventsAll;

        %% Save Subject File
        saveFile = fullfile(processedHealthyPath, [subjectName '_processed.mat']);
        save(saveFile, 'subjectOut', '-v7.3');

        %% Build and Save BMI/QKTD Inputs for Multiple Windows
        for bw = 1:length(bmiWindows)
        
            bmiWindowRel = bmiWindows{bw};
        
            % Healthy epochs are MI-onset aligned if using annotation-defined timing.
            % Therefore [0 0.5] and [0 1.0] are already absolute in tEpoch.
            if strcmp(subjectOut.trialTimeMode, 'annotation_defined_full_trial')
                bmiWindowAbs = bmiWindowRel;
            else
                bmiWindowAbs = subjectOut.cueTimeSec + bmiWindowRel;
            end
        
            bmiOut = Utility_Functions.build_bmi_input_from_epochs( ...
                subjectOut.epochsMI, ...
                subjectOut.labels, ...
                subjectOut.tEpoch, ...
                subjectOut.chanLabels, ...
                subjectOut.fs, ...
                subjectOut.subjectName, ...
                'healthy', ...
                bmiWindowAbs);
        
            Utility_Functions.save_bmi_input_file(bmiOut, bmiHealthyPath);
        end

        clc; % Clear command window to reduce clutter for next subject processing
        fprintf('Saved: %s\n', saveFile);
        disp(['  Healthy subject ' subjectName ': ' num2str(size(runEpochsMI,3)) ' epochs, ' num2str(size(runEpochsMI,1)) ' channels, ' num2str(size(runEpochsMI,2)) ' samples/epoch']);
        if skippedInfo_H.totalSkipped > 0
            disp(['  ⚠ Skipped ' num2str(skippedInfo_H.totalSkipped) ' trials during epoching']);
        else
            disp('  ✓ All trials successfully epoched (no skipped trials)');
        end

        %% Summary Info
        allHealthyResults(s).subjectName = subjectName;
        allHealthyResults(s).nEpochs = size(runEpochsMI, 3);
        allHealthyResults(s).nChannels = size(runEpochsMI, 1);
        allHealthyResults(s).nSamplesPerEpoch = size(runEpochsMI, 2);
        allHealthyResults(s).fs = targetFs;
        allHealthyResults(s).nLeft = sum(runLabels == 1);
        allHealthyResults(s).nRight = sum(runLabels == 2);
        allHealthyResults(s).nSkipped = skippedInfo_H.totalSkipped;

        if istable(runEventsMI) && ~isempty(runEventsMI)
        
            allHealthyResults(s).prevT0DurationMin = min(runEventsMI.prevT0DurationSec, [], 'omitnan');
            allHealthyResults(s).prevT0DurationMax = max(runEventsMI.prevT0DurationSec, [], 'omitnan');
            allHealthyResults(s).prevT0DurationMean = mean(runEventsMI.prevT0DurationSec, 'omitnan');
        
            allHealthyResults(s).miDurationMin = min(runEventsMI.miDurationSec, [], 'omitnan');
            allHealthyResults(s).miDurationMax = max(runEventsMI.miDurationSec, [], 'omitnan');
            allHealthyResults(s).miDurationMean = mean(runEventsMI.miDurationSec, 'omitnan');
        
        else
        
            allHealthyResults(s).prevT0DurationMin = NaN;
            allHealthyResults(s).prevT0DurationMax = NaN;
            allHealthyResults(s).prevT0DurationMean = NaN;
        
            allHealthyResults(s).miDurationMin = NaN;
            allHealthyResults(s).miDurationMax = NaN;
            allHealthyResults(s).miDurationMean = NaN;
            
        end
    end

    %% Save Healthy Summary File
    disp('Saving healthy subjects summary...');
    save(fullfile(processedHealthyPath, 'allHealthyResults_summary.mat'), 'allHealthyResults');
    disp('Healthy subjects summary saved.');

    disp('Healthy preprocessing complete.');
end

end % processHealthy

%% ========================================================================
% PROCESS STROKE SUBJECTS
% ========================================================================

if processStroke
disp('Starting Stroke Subjects Preprocessing...');
disp('Processed stroke path:');
disp(processedStrokePath);
disp('Stroke raw base path:');
disp(strokeRawBasePath);

%% Find Stroke Subject Folders
disp('Finding stroke subject folders...');
allEntries = dir(strokeRawBasePath);
isSubj = [allEntries.isdir] & startsWith({allEntries.name}, 'sub-');
strokeFolders = allEntries(isSubj);
nStrokeSubjects = numel(strokeFolders);

disp(['Stroke subjects found: ', num2str(nStrokeSubjects)]);
%% Check if any stroke subjects found
if nStrokeSubjects == 0
    warning('No stroke subject folders found. Check strokeRawBasePath.');
else
    %% Initialize Stroke Summary Struct
    allStrokeResults = struct();

    %% Raw stroke channel labels after excluding channel 18, 31, 32, 33
    % Dataset description:
    % 1-30 EEG, ch18 reference, 31-32 EOG, 33 marker
    strokeRawLabels = { ...
        'FP1','FP2','FZ','F3','F4','F7','F8','FCZ','FC3', ...
        'FC4','FT7','FT8','CZ','C3','C4','T3','T4','CP3', ...
        'CP4','TP7','TP8','PZ','P3','P4','T5','T6','OZ','O1','O2'};

    %% Original raw sampling frequency
    fsOld = 500;

    %% Loop Through Stroke Subjects
    disp(['Processing ', num2str(nStrokeSubjects), ' stroke subjects...']);
    validCount = 0;

    for s = 1:nStrokeSubjects

        disp(' ');
        disp('==============================');
        disp(['Processing stroke subject ', num2str(s), ' of ', num2str(nStrokeSubjects)]);
        disp('==============================');

        %% Subject name
        subjectName = strokeFolders(s).name;

        %% Path to this subject folder
        subjPath = fullfile(strokeRawBasePath, subjectName);

        %% Find raw .mat file
        matStruct = dir(fullfile(subjPath, '*.mat'));

        if isempty(matStruct)
            error(['Missing .mat file for ' subjectName]);
        end

        matFile = fullfile(subjPath, matStruct(1).name);
        disp(["Loading raw file: ", matFile]);

        %% Load raw file
        D = load(matFile);

        if ~isfield(D, 'eeg') || ~isstruct(D.eeg)
            error(['Missing eeg struct in ' matFile '. Variables: ' strjoin(fieldnames(D), ', ')]);
        end

        if ~isfield(D.eeg, 'rawdata') || ~isfield(D.eeg, 'label')
            error(['eeg struct missing rawdata or label in ' subjectName '. Fields: ' strjoin(fieldnames(D.eeg), ', ')]);
        end

        rawdata = D.eeg.rawdata;   % nTrials x channels x samples
        labels  = D.eeg.label(:);  % [nTrials x 1]
        
        %% Basic checks
        if ndims(rawdata) ~= 3
            error(['rawdata is not 3D for ' subjectName ' (dims: ' num2str(ndims(rawdata)) ')']);
        end

        nTrials = size(rawdata,1);
        nChRaw  = size(rawdata,2);
        nSamp   = size(rawdata,3);

        disp(['Loaded ', subjectName, ' | trials=', num2str(nTrials), ' | channels=', num2str(nChRaw), ' | samples=', num2str(nSamp)]);

        if nTrials ~= numel(labels)
            error(['Labels do not match trial count for ' subjectName ' (' num2str(nTrials) ' trials vs ' num2str(numel(labels)) ' labels)']);
        end

        %% Keep usable EEG channels only
        % Exclude:
        %   channel 18 = reference
        %   channels 31-32 = EOG
        %   channel 33 = marker
        eegKeepIdx = [1:17, 19:30];   % 29 EEG channels
        rawEEG = rawdata(:, eegKeepIdx, :);   % trials x 29 x samples

        %% Convert to channels x samples x trials
        X_S = permute(rawEEG, [2 3 1]);         % 29 x samples x trials

        %% Harmonize channels to template
        [X_harm_S, matchedLabels] = Utility_Functions.apply_channel_harmonization_trialwise( ...
            X_S, strokeRawLabels, harmonizedLabels);

        %% Resample each trial
        X_resampled_S = Utility_Functions.resample_3d_trials(X_harm_S, fsOld, targetFs);

        %% Apply filters to each trial
        [X_delta_S, X_theta_S, X_mi_S, X_mu_S, X_beta_S, X_gamma_S] = Utility_Functions.apply_bandpass_filters_3d(X_resampled_S, filters);

        %% Epoch
        [epochsMI_S, tEpoch, validMask_S, skippedInfo_S] = ...
            Utility_Functions.extract_stroke_full_trials( ...
            X_mi_S, targetFs, config.strokeTrialWindow);
        
        [epochsDelta_S, ~, validMask_delta_S, ~] = ...
            Utility_Functions.extract_stroke_full_trials( ...
            X_delta_S, targetFs, config.strokeTrialWindow);
        
        [epochsTheta_S, ~, validMask_theta_S, ~] = ...
            Utility_Functions.extract_stroke_full_trials( ...
            X_theta_S, targetFs, config.strokeTrialWindow);
        
        [epochsMu_S, ~, validMask_mu_S, ~] = ...
            Utility_Functions.extract_stroke_full_trials( ...
            X_mu_S, targetFs, config.strokeTrialWindow);
        
        [epochsBeta_S, ~, validMask_beta_S, ~] = ...
            Utility_Functions.extract_stroke_full_trials( ...
            X_beta_S, targetFs, config.strokeTrialWindow);
        
        [epochsGamma_S, ~, validMask_gamma_S, ~] = ...
            Utility_Functions.extract_stroke_full_trials( ...
            X_gamma_S, targetFs, config.strokeTrialWindow);

        if ~isequal(validMask_S, validMask_delta_S, validMask_theta_S, validMask_mu_S, validMask_beta_S, validMask_gamma_S)
            error('Valid trial masks do not match across bands for %s. Data integrity compromised.', subjectName);
        end

        labels_valid_S = labels(validMask_S);

        %% Build output struct
        subjectOut = struct();
        subjectOut.subjectName = subjectName;
        subjectOut.fs = targetFs;
        subjectOut.tEpoch = tEpoch;
        subjectOut.chanLabels = matchedLabels;
        subjectOut.labels = labels_valid_S;

        subjectOut.referenceMethod = 'Original stroke reference channel omitted; no CPz rereference applied';
        subjectOut.referenceChannel = 'Dataset reference channel removed';

        subjectOut.trialTimeMode = 'full_trial';
        subjectOut.cueTimeSec = config.strokeCueTimeSec;
        subjectOut.baselineWindow = config.strokeBaselineWindow;
        subjectOut.miWindow = config.strokeMIWindow;
        subjectOut.cueWindow = config.strokeCueWindow;
        subjectOut.trialWindow = config.strokeTrialWindow;
        subjectOut.analysisWindowSize = config.analysisWindowSize;
        subjectOut.baselineOnset = config.strokeBaselineOnset;
        subjectOut.miOnset = config.strokeMIOnset;

        subjectOut.epochsMI = epochsMI_S;
        subjectOut.epochsDelta = epochsDelta_S;
        subjectOut.epochsTheta = epochsTheta_S;
        subjectOut.epochsMu = epochsMu_S;
        subjectOut.epochsBeta = epochsBeta_S;
        subjectOut.epochsGamma = epochsGamma_S;
        subjectOut.originalFs = fsOld;
        subjectOut.nTrials = nTrials;
        subjectOut.nChannels = size(X_mi_S,1);
        subjectOut.nSamplesPerEpoch = size(epochsMI_S,2);

        %% Save subject file
        saveFile = fullfile(processedStrokePath, [subjectName '_processed.mat']);
        save(saveFile, 'subjectOut', '-v7.3');

        %% Build and Save BMI/QKTD Inputs for Multiple Windows
        
        for bw = 1:length(bmiWindows)
        
            bmiWindowRel = bmiWindows{bw};
        
            % Stroke epochs are full-trial aligned, so add the cue/MI onset time.
            bmiWindowAbs = subjectOut.cueTimeSec + bmiWindowRel;
        
            bmiOut = Utility_Functions.build_bmi_input_from_epochs( ...
                subjectOut.epochsMI, ...
                subjectOut.labels, ...
                subjectOut.tEpoch, ...
                subjectOut.chanLabels, ...
                subjectOut.fs, ...
                subjectOut.subjectName, ...
                'stroke', ...
                bmiWindowAbs);
        
            Utility_Functions.save_bmi_input_file(bmiOut, bmiStrokePath);
        end

        clc; % Clear command window to reduce clutter for next subject processing

        disp(['Saved: ', saveFile]);
        disp(['  Stroke subject ' subjectName ': ' num2str(size(epochsMI_S,3)) ' epochs, ' num2str(size(epochsMI_S,1)) ' channels, ' num2str(size(epochsMI_S,2)) ' samples/epoch']);
        if skippedInfo_S.totalSkipped > 0
            disp(['  ⚠ Skipped ' num2str(skippedInfo_S.totalSkipped) ' trials during epoching']);
        else
            disp('  ✓ All trials successfully epoched (no skipped trials)');
        end

        %% Summary info
        validCount = validCount + 1;
        allStrokeResults(validCount).subjectName = subjectName;
        allStrokeResults(validCount).nTrials = nTrials;
        allStrokeResults(validCount).nChannels = size(X_mi_S,1);
        allStrokeResults(validCount).nSamplesPerEpoch = size(epochsMI_S,2);
        allStrokeResults(validCount).fs = targetFs;
        allStrokeResults(validCount).nLeft = sum(labels_valid_S == 1);
        allStrokeResults(validCount).nRight = sum(labels_valid_S == 2);
        allStrokeResults(validCount).nSkipped = skippedInfo_S.totalSkipped;
    end

    %% Save Stroke Summary File
    disp('Saving stroke subjects summary...');
    save(fullfile(processedStrokePath, 'allStrokeResults_summary.mat'), 'allStrokeResults');
    disp('Stroke subjects summary saved.');

    disp('Stroke preprocessing complete.');
end

end % processStroke

disp('Combined preprocessing for healthy and stroke subjects is complete.');
