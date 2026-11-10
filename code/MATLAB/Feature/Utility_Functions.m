classdef Utility_Functions
    % Utility_Functions contains shared functions and configurations
    % for EEG preprocessing and analysis

    methods (Static)
        function config = get_preprocessing_config()
            %% Configure Preprocessing
            % This will contain all necessary aspects needed to run preprocessing
            % portions as well as feature analysis and extraction

            % ZOV Datasets
            % Paths
            config.strokeRawBasePath = "C:\Users\sme319\Documents\RLBMI\data\stroke";

            config.healthyBasePath = "C:\Users\sme319\Documents\RLBMI\data\healthy";

            % Single results folder path
            config.resultsPath = "C:\Users\sme319\Documents\RLBMI\data\Results";

            % BMI/QKTD output folder
            config.bmiInputPath = fullfile(config.resultsPath, "BMI_inputs");
            
            % BMI/QKTD extraction windows relative to MI onset.
            % These control only the saved BMI input files, not Step02 feature extraction.
            config.bmiWindows = {
                [0 0.5]
                [0 1.0]
            };

            % Electrode file for topographic plotting
            config.electrodeFile = fullfile(fileparts(config.healthyBasePath), "task-motor-imagery_electrodes.tsv");

            %%
            % Sampling
            config.targetFs = 160;

            % Filter settings
            config.butterOrder = 4;   % used with filtfilt
            config.miBand = [0.1 30];
            config.deltaBand = [0.1 4]; % Literature suggests that 0.1-8 Hz contains useful stroke information
            config.thetaBand = [4 8];
            config.muBand = [8 13];
            config.betaBand = [13 30];
            config.gammaBand = 30;  % 30 Hz highpass cutoff

            % Full-trial extraction settings
            config.saveFullTrials = true;
            
            % ============================================================
            % Configurable feature-analysis window settings
            % ============================================================
            % Window length used later for baseline-vs-MI feature extraction.
            % Change this to 0.5 or 1.0 depending upon analysis.
            config.analysisWindowSize = 1.0;   % seconds

            % Fixed healthy MI-onset aligned epoch window.
            % Healthy epochs will be extracted from -4.1 to +4.1 s relative to MI onset.
            % At 160 Hz, this gives 1312 samples and MI onset at column 657.
            config.healthyEpochPreMI_sec  = 4.1;
            config.healthyEpochPostMI_sec = 4.1;
            
            % Healthy PhysioNet timing
            config.healthyRestSec = 4.2;
            config.healthyCueSec = 4.1;
            config.healthyTrialWindow = [0 8.3];
            config.healthyCueTimeSec = 4.2;
            
            % Healthy analysis-window anchors
            % Baseline window ends at cue/MI onset.
            % MI window starts at cue/MI onset.
            config.healthyBaselineOnset = config.healthyCueTimeSec - config.analysisWindowSize;
            config.healthyMIOnset = config.healthyCueTimeSec;
            
            config.healthyBaselineWindow = [ ...
                config.healthyBaselineOnset, ...
                config.healthyBaselineOnset + config.analysisWindowSize];
            
            config.healthyMIWindow = [ ...
                config.healthyMIOnset, ...
                config.healthyMIOnset + config.analysisWindowSize];
            
            % Stroke timing
            config.strokeTrialWindow = [0 8];
            config.strokeCueTimeSec = 2;
            config.strokeCueWindow = [2 6];
            
            % Stroke analysis-window anchors
            % MI window starts at cue/MI onset.
            % Baseline/rest window ends at the end of the trial.
            config.strokeMIOnset = config.strokeCueTimeSec;
            config.strokeBaselineOnset = config.strokeTrialWindow(2) - config.analysisWindowSize;
            
            config.strokeMIWindow = [ ...
                config.strokeMIOnset, ...
                config.strokeMIOnset + config.analysisWindowSize];
            
            config.strokeBaselineWindow = [ ...
                config.strokeBaselineOnset, ...
                config.strokeBaselineOnset + config.analysisWindowSize];

            fprintf('%-30s: %.2f s\n', 'Analysis Window Size', config.analysisWindowSize);

            % Healthy runs for left/right fist motor imagery according to the Physionet
            % dataset
            config.healthyTask2Runs = [4 8 12];

            % Channels to remove (EOG & Grounding Channels)
            config.removeChannels = {'HEOL','HEOR','CPZ'};

            % Harmonized channel template
            config.harmonizedLabels = { ...
                'FP1','FP2','FZ','F3','F4','F7','F8','FCZ','FC3', ...
                'FC4','FT7','FT8','CZ','C3','C4','T3','T4','CP3', ...
                'CP4','TP7','TP8','PZ','P3','P4','T5','T6','OZ','O1','O2'};
        end

        function X_ref = rereference_to_channel(X, rawLabels, refLabel)
            % Re-reference EEG data to a specified reference channel.
            %
            % X_ref(ch,t) = X(ch,t) - X(refChannel,t)
            %
            % After re-referencing, the reference channel should be zero/flat.
        
            if isstring(rawLabels)
                rawLabels = cellstr(rawLabels);
            end
        
            rawLabels = rawLabels(:);
        
            cleanLabels = upper(strtrim(rawLabels));
            cleanLabels = regexprep(cleanLabels, '\.', '');
        
            refLabel = upper(strtrim(refLabel));
            refLabel = regexprep(refLabel, '\.', '');
        
            refIdx = find(strcmpi(cleanLabels, refLabel), 1);
        
            if isempty(refIdx)
                error('Reference channel %s was not found in raw channel labels.', refLabel);
            end
        
            refSignal = double(X(refIdx, :));
        
            % Subtract CPz from every channel sample-by-sample
            X_ref = double(X) - refSignal;
        
            % Verify CPz is now flatlined at zero
            maxRefResidual = max(abs(X_ref(refIdx, :)), [], 'omitnan');
        
            fprintf('Applied %s re-reference to all channels.\n', refLabel);
            fprintf('Post-reference %s max absolute residual: %.3e\n', refLabel, maxRefResidual);
        
            if maxRefResidual > 1e-10
                warning('%s did not flatline exactly to zero after re-referencing. Check numerical precision or channel indexing.', refLabel);
            end
        end

        function [trials, tTrial, validMask, skippedInfo] = extract_healthy_full_trials(X, eventsMI, fs, preMISec, postMISec)
            % Extract fixed MI-onset aligned healthy epochs.
            %
            % Epoch window:
            %   -preMISec to +postMISec relative to T1/T2 MI onset.
            %
            % Missing pre- or post-MI samples are padded with NaN.
            % Extra samples are cropped.
            %
            % At fs = 160 Hz and pre/post = 4.1 s:
            %   preSamp = 656
            %   postSamp = 656
            %   total = 1312 samples
            %   MI onset is column 657.
        
            if nargin < 4 || isempty(preMISec)
                preMISec = 4.1;
            end
        
            if nargin < 5 || isempty(postMISec)
                postMISec = 4.1;
            end
        
            if isempty(eventsMI) || ~istable(eventsMI)
                error('eventsMI must be a non-empty table.');
            end
        
            requiredVars = {'miOnsetSec','miDurationSec','prevT0DurationSec'};
        
            for v = 1:length(requiredVars)
                if ~ismember(requiredVars{v}, eventsMI.Properties.VariableNames)
                    error('eventsMI is missing required variable: %s', requiredVars{v});
                end
            end
        
            nCh = size(X, 1);
            nSamp = size(X, 2);
            nTrials = height(eventsMI);
        
            preSamp = round(preMISec * fs);
            postSamp = round(postMISec * fs);
            epochLen = preSamp + postSamp;
        
            tTrial = ((0:epochLen-1) - preSamp) / fs;
        
            trials = nan(nCh, epochLen, nTrials);
            validMask = true(nTrials, 1);
        
            paddedPre = false(nTrials, 1);
            paddedPost = false(nTrials, 1);
        
            for tr = 1:nTrials
        
                if isnan(eventsMI.miOnsetSec(tr))
                    error('Trial %d is missing MI onset timing.', tr);
                end
        
                miOnsetSample = round(eventsMI.miOnsetSec(tr) * fs) + 1;
        
                srcStart = miOnsetSample - preSamp;
                srcEnd   = miOnsetSample + postSamp - 1;
        
                dstStart = 1;
                dstEnd   = epochLen;
        
                if srcStart < 1
                    paddedPre(tr) = true;
                    dstStart = 2 - srcStart;
                    srcStart = 1;
                end
        
                if srcEnd > nSamp
                    paddedPost(tr) = true;
                    dstEnd = epochLen - (srcEnd - nSamp);
                    srcEnd = nSamp;
                end
        
                if srcStart > srcEnd || dstStart > dstEnd
                    error('Trial %d has no valid overlap with EEG data after MI-aligned epoching.', tr);
                end
        
                trials(:, dstStart:dstEnd, tr) = X(:, srcStart:srcEnd);
            end
        
            skippedInfo = struct();
            skippedInfo.totalSkipped = 0;
            skippedInfo.skippedTrialIndices = [];
            skippedInfo.skippedReasons = {};
            skippedInfo.preMISec = preMISec;
            skippedInfo.postMISec = postMISec;
            skippedInfo.epochLenSamples = epochLen;
            skippedInfo.miOnsetColumn = preSamp + 1;
            skippedInfo.paddedPre = paddedPre;
            skippedInfo.paddedPost = paddedPost;
            skippedInfo.nPaddedPre = sum(paddedPre);
            skippedInfo.nPaddedPost = sum(paddedPost);
        
            fprintf('Extracted %d fixed MI-onset aligned healthy trials.\n', nTrials);
            fprintf('Epoch window: %.3f to %.3f s | %d samples\n', ...
                -preMISec, postMISec, epochLen);
            fprintf('MI onset column: %d\n', preSamp + 1);
            fprintf('Trials padded before MI: %d | after MI: %d\n', ...
                sum(paddedPre), sum(paddedPost));
        end

        function assert_same_epoch_size(A, B, labelA, labelB)
            % Strictly verify epoch arrays have matching channel and sample dimensions.
        
            if isempty(A) || isempty(B)
                return
            end
        
            if size(A,1) ~= size(B,1) || size(A,2) ~= size(B,2)
                error(['Epoch size mismatch between %s and %s.\n' ...
                       '%s size = %s\n' ...
                       '%s size = %s\n' ...
                       'After fixed MI-onset epoching, this should not happen.'], ...
                       labelA, labelB, ...
                       labelA, mat2str(size(A)), ...
                       labelB, mat2str(size(B)));
            end
        end

        function [trials, tTrial, validMask, skippedInfo] = extract_stroke_full_trials(X, fs, trialWindow)

            startIdx = round(trialWindow(1) * fs) + 1;
            endIdx = round(trialWindow(2) * fs);
        
            nCh = size(X,1);
            nSamp = size(X,2);
            nTrials = size(X,3);
        
            if startIdx < 1 || endIdx > nSamp
                error('Requested stroke trial window [%d %d] is out of bounds for trial length %d.', ...
                    startIdx, endIdx, nSamp);
            end
        
            trials = X(:, startIdx:endIdx, :);
            tTrial = (0:(size(trials,2)-1)) / fs;
        
            validMask = true(nTrials,1);
        
            skippedInfo.totalSkipped = 0;
            skippedInfo.skippedTrialIndices = [];
            skippedInfo.skippedReasons = {};
        
            fprintf('Extracted %d full stroke trials.\n', nTrials);
            fprintf('Trial length: %d samples, %.2f s\n', size(trials,2), size(trials,2)/fs);
        end

        function filters = design_bandpass_filters(targetFs, butterOrder, deltaBand, thetaBand, miBand, muBand, betaBand, gammaCutoff)
            % Design delta, theta, MI, mu, beta bandpass filters and gamma highpass filter

            [filters.b_delta, filters.a_delta] = butter(butterOrder, deltaBand / (targetFs/2), 'bandpass');
            [filters.b_theta, filters.a_theta] = butter(butterOrder, thetaBand / (targetFs/2), 'bandpass');
            [filters.b_mi,    filters.a_mi]    = butter(butterOrder, miBand    / (targetFs/2), 'bandpass');
            [filters.b_mu,    filters.a_mu]    = butter(butterOrder, muBand    / (targetFs/2), 'bandpass');
            [filters.b_beta,  filters.a_beta]  = butter(butterOrder, betaBand  / (targetFs/2), 'bandpass');
            [filters.b_gamma, filters.a_gamma] = butter(butterOrder, gammaCutoff / (targetFs/2), 'high');
        end

        function display_preprocessing_config(config, processedHealthyPath, processedStrokePath)
            % Display preprocessing configuration in a formatted table
            disp(' ');
            disp('================================================================================');
            disp('PREPROCESSING CONFIGURATION SUMMARY');
            disp('================================================================================');

            fprintf('\n%-30s: %d Hz\n', 'Target Sampling Frequency', config.targetFs);
            fprintf('%-30s: %d\n', 'Butterworth Filter Order', config.butterOrder);
            fprintf('%-30s: [%.1f %.1f] Hz\n', 'MI Band', config.miBand(1), config.miBand(2));
            fprintf('%-30s: [%.1f %.1f] Hz\n', 'Delta Band', config.deltaBand(1), config.deltaBand(2));
            fprintf('%-30s: [%.1f %.1f] Hz\n', 'Theta Band', config.thetaBand(1), config.thetaBand(2));
            fprintf('%-30s: [%.1f %.1f] Hz\n', 'Mu Band', config.muBand(1), config.muBand(2));
            fprintf('%-30s: [%.1f %.1f] Hz\n', 'Beta Band', config.betaBand(1), config.betaBand(2));
            fprintf('%-30s: %.1f Hz (highpass)\n', 'Gamma Band', config.gammaBand);
            fprintf('%-30s: %s\n', 'Healthy Timing Mode', ...
                'Annotation-defined from EDF T0/T1/T2 events');
            fprintf('%-30s: %s\n', 'Healthy Feature Windows', ...
                'Saved per subjectOut relative to T1/T2 onset');
            
            fprintf('%-30s: [%.1f %.1f] s\n', 'Stroke Trial Window', config.strokeTrialWindow(1), config.strokeTrialWindow(2));
            fprintf('%-30s: %.1f s\n', 'Stroke Cue Time', config.strokeCueTimeSec);
            fprintf('%-30s: [%.1f %.1f] s\n', 'Stroke Baseline Window', config.strokeBaselineWindow(1), config.strokeBaselineWindow(2));
            fprintf('%-30s: [%.1f %.1f] s\n', 'Stroke MI Window', config.strokeMIWindow(1), config.strokeMIWindow(2));

            disp(' ');
            disp('FILE PATHS:');
            fprintf('%-30s: %s\n', 'Healthy Base Path', config.healthyBasePath);
            fprintf('%-30s: %s\n', 'Processed Healthy Path', processedHealthyPath);
            fprintf('%-30s: %s\n', 'Stroke Raw Base Path', config.strokeRawBasePath);
            fprintf('%-30s: %s\n', 'Processed Stroke Path', processedStrokePath);

            disp(' ');
            disp('================================================================================');
        end

        function plot_filter_magnitude_spectra(b, a, savePath, filename, cutoffFreqs)
            % Plot magnitude spectrum and save as .tif/.fig.
        
            if ~exist(savePath, 'dir')
                mkdir(savePath);
            end
        
            targetFs = 160;
            [H, f] = freqz(b, a, 1024, targetFs);
        
            fig = figure('Visible','off', ...
                'Color','w', ...
                'Position',[100 100 900 650]);
        
            ax = axes(fig);
            hold(ax, 'on');
        
            plot(f, 20*log10(abs(H)), ...
                'LineWidth',2, ...
                'Color','b', ...
                'DisplayName','Single pass');
        
            plot(f, 20*log10(abs(H).^2), ...
                'LineWidth',2, ...
                'Color','r', ...
                'LineStyle','--', ...
                'DisplayName','filtfilt');
        
            if nargin >= 5 && ~isempty(cutoffFreqs)
                for i = 1:length(cutoffFreqs)
                    if cutoffFreqs(i) > 0 && cutoffFreqs(i) < targetFs/2
                        xline(cutoffFreqs(i), '--m', ...
                            'LineWidth',1.2, ...
                            'DisplayName',sprintf('Cutoff %.1f Hz', cutoffFreqs(i)));
                    end
                end
            end
        
            xlabel('Frequency [Hz]');
            ylabel('Magnitude [dB]');
            title('Filter Magnitude Spectrum');
            legend('Location','northeast');
        
            xlim([0 targetFs/2]);
            ylim([-120 10]);
        
            Utility_Functions.style_publication_axes(ax);
            hold(ax, 'off');
        
            Utility_Functions.save_figure_tif_fig(fig, fullfile(savePath, filename), 300);
            close(fig);
        end

        function X = convert_edf_into_matrix(T)
            % Convert EDF read output into numeric EEG matrix [channels x samples]

            disp('========================================');
            disp('CONVERTING EDF DATA INTO EEG MATRIX');
            disp('========================================');

            % Analyze input data structure
            disp(['Input data type: ' class(T)]);
            disp(['Input data size: ' num2str(size(T))]);

            % We know it's a timetable
            disp('✓ Input is TIMETABLE with time information');

            % Check the time index (row times)
            rowTimes = T.Properties.RowTimes;
            disp(['Time index type: ' class(rowTimes)]);
            disp(['Time index length: ' num2str(length(rowTimes))]);
            disp(['Time range: ' char(rowTimes(1)) ' to ' char(rowTimes(end))]);

            % Calculate actual duration
            timeRange = rowTimes(end) - rowTimes(1);
            disp(['Actual duration: ' char(timeRange)]);

            % Check time step
            if length(rowTimes) > 1
                timeStep = rowTimes(2) - rowTimes(1);
                disp(['Time step between points: ' char(timeStep)]);
            end

            disp(['Number of time points: ' num2str(height(T))]);

            disp(['Number of variables (channels): ' num2str(width(T))]);
            varNames = T.Properties.VariableNames;
            disp(['Channel names (first 10): ' strjoin(varNames(1:min(10, length(varNames))), ', ')]);
            if length(varNames) > 10
                disp(['... and ' num2str(length(varNames) - 10) ' more channels']);
            end
            disp(' ');

            % Display first few rows from T
            disp('=== FIRST FEW ROWS OF TIMETABLE ===');
            nRowsToShow = min(3, height(T));
            nColsToShow = min(5, width(T));
            disp(['Showing first ' num2str(nRowsToShow) ' rows and ' num2str(nColsToShow) ' columns:']);

            for rowIdx = 1:nRowsToShow
                disp(['Row ' num2str(rowIdx) ' (Time: ' char(T.Properties.RowTimes(rowIdx)) '):']);
                for colIdx = 1:nColsToShow
                    cellData = T{rowIdx, colIdx};
                    disp(['  ' varNames{colIdx} ': size=' num2str(size(cellData)) ', class=' class(cellData)]);
                    % If it's numeric and not too large, show a bit of the data
                    if isnumeric(cellData)
                        if numel(cellData) <= 10
                            disp(['    Values: ' num2str(cellData)]);
                        else
                            disp(['    First 3 values: ' num2str(cellData(1:min(3, numel(cellData))))]);
                            disp(['    Last 3 values: ' num2str(cellData(max(1, numel(cellData)-2):end))]);
                        end
                    end
                end
            end
            disp(' ');

            % Pull raw contents from the table/timetable
            disp('Extracting raw data from timetable...');
            raw = T{:,:};
            disp(['Raw data type: ' class(raw)]);
            disp(['Raw data size: ' num2str(size(raw))]);

            % If the table contents are not cells, just transpose to channels x samples
            if ~iscell(raw)
                disp('✓ SIMPLE NUMERIC DATA: No cell structure detected');
                disp(['Data type: ' class(raw)]);
                disp(['Amplitude (µV): ' num2str(min(raw(:))) ' to ' num2str(max(raw(:)))]);
                X = double(raw');
                disp(['✓ Conversion complete: ' num2str(size(X,1)) ' channels x ' num2str(size(X,2)) ' samples']);
                disp(['Final matrix type: ' class(X)]);
                disp(['Amplitude range (µV): ' num2str(min(X(:))) ' to ' num2str(max(X(:)))]);
                disp('========================================');
                return
            end

            % Handle cell-based data (multiple data records)
            nRows = size(raw,1);   % usually time blocks / records
            nCols = size(raw,2);   % usually channels

            disp('✓ CELL-BASED DATA: Multiple data records detected');
            disp(['Number of data records (rows): ' num2str(nRows)]);
            disp(['Number of channels (columns): ' num2str(nCols)]);
            disp(['Total cells to process: ' num2str(nRows * nCols)]);
            disp(' ');

            % Analyze cell structure
            disp('Analyzing cell structure...');
            sampleCell = raw{1,1};
            disp(['Sample cell type: ' class(sampleCell)]);
            if isnumeric(sampleCell)
                disp(['Sample cell size: ' num2str(size(sampleCell))]);
                disp(['Sample cell amplitude (µV): ' num2str(min(sampleCell(:))) ' to ' num2str(max(sampleCell(:)))]);
            end
            disp(' ');

            Xrows = cell(nCols,1);

            disp('Processing channels and concatenating data records...');
            for ch = 1:nCols
                pieces = cell(1,nRows);

                for r = 1:nRows
                    block = raw{r,ch};

                    if isempty(block)
                        error('Empty block found at row %d, channel %d.', r, ch);
                    end

                    if ~isnumeric(block)
                        error('Non-numeric block found at row %d, channel %d.', r, ch);
                    end

                    pieces{r} = double(block(:))';   % force row vector
                end

                Xrows{ch} = [pieces{:}];

                % Progress update with more details
                if mod(ch, 5) == 0 || ch == nCols
                    totalSamples = length(Xrows{ch});
                    samplesPerRecord = totalSamples / nRows;
                    disp(['  Channel ' num2str(ch) '/' num2str(nCols) ': ' num2str(totalSamples) ' total samples']);
                    disp(['    → ' num2str(nRows) ' records × ' num2str(samplesPerRecord) ' samples/record']);
                    disp(['    → Amplitude (µV): ' num2str(min(Xrows{ch})) ' to ' num2str(max(Xrows{ch}))]);
                end
            end

            disp(' ');
            disp('Verifying channel consistency...');
            % Verify that all channels have the same number of samples
            lengths = cellfun(@length, Xrows);
            uniqueLengths = unique(lengths);

            if numel(uniqueLengths) ~= 1
                disp('⚠ INCONSISTENT CHANNEL LENGTHS DETECTED:');
                for ch = 1:nCols
                    disp(['  Channel ' num2str(ch) ' (' varNames{ch} '): ' num2str(lengths(ch)) ' samples']);
                end
                error('Channels do not all have the same number of samples after concatenation.');
            else
                disp(['✓ All channels have consistent length: ' num2str(uniqueLengths) ' samples']);
            end

            disp(' ');
            disp('Creating final matrix...');
            X = vertcat(Xrows{:});

            disp('✓ EDF CONVERSION COMPLETE');
            disp(['  Final matrix size: ' num2str(size(X,1)) ' channels × ' num2str(size(X,2)) ' samples']);
            disp(['  Matrix data type: ' class(X)]);
            disp(['  Amplitude range (µV): ' num2str(min(X(:))) ' to ' num2str(max(X(:)))]);
            disp(['  Memory usage: ~' num2str(numel(X) * 8 / 1024 / 1024, '%.1f') ' MB (double precision)']);
            disp('========================================');
        end

        function bmiOut = build_bmi_input_from_epochs(epochs, labels, tEpoch, chanLabels, fs, subjectName, groupName, bmiWindow)
            % Build BMI/QKTD input from epoched EEG data.
            %
            % epochs: channels x samples x trials
            %
            % Output:
            %   InputTensor: nTrials x nSamples x nChannels
            %   InputMatrix: nTrials x (nSamples*nChannels)
        
            if nargin < 8 || isempty(bmiWindow)
                error('bmiWindow must be provided explicitly.');
            end
        
            labels = labels(:);
            tEpoch = tEpoch(:)';
        
            bmiWindow = double(bmiWindow);
            bmiWindowSize = bmiWindow(2) - bmiWindow(1);
        
            if bmiWindowSize <= 0
                error('Invalid BMI window [%.3f %.3f]. End time must be greater than start time.', ...
                    bmiWindow(1), bmiWindow(2));
            end
        
            bmiIdx = tEpoch >= bmiWindow(1) & tEpoch < bmiWindow(2);
        
            if ~any(bmiIdx)
                error('No samples found in BMI window [%.2f %.2f]. tEpoch range is [%.2f %.2f].', ...
                    bmiWindow(1), bmiWindow(2), tEpoch(1), tEpoch(end));
            end
        
            nExpected = round(bmiWindowSize * fs);
            nFound = sum(bmiIdx);
        
            if nFound ~= nExpected
                warning('BMI window found %d samples, expected %d samples for %.2f s at %.1f Hz.', ...
                    nFound, nExpected, bmiWindowSize, fs);
            end
        
            Xwin = epochs(:, bmiIdx, :);              % channels x samples x trials
            InputTensor = permute(Xwin, [3 2 1]);     % trials x samples x channels
        
            nTrials = size(InputTensor, 1);
            nSamples = size(InputTensor, 2);
            nChannels = size(InputTensor, 3);
        
            InputMatrix = reshape(InputTensor, nTrials, nSamples*nChannels);
        
            if length(labels) ~= nTrials
                error('Label count does not match BMI trial count.');
            end
        
            bmiOut = struct();
            bmiOut.subjectName = subjectName;
            bmiOut.group = groupName;
            bmiOut.fs = fs;
        
            bmiOut.bmiWindow = bmiWindow;
            bmiOut.bmiWindowSize = bmiWindowSize;
            bmiOut.bmiWindowStart = bmiWindow(1);
            bmiOut.bmiWindowEnd = bmiWindow(2);
        
            bmiOut.tBMI = tEpoch(bmiIdx);
            bmiOut.chanLabels = chanLabels;
            bmiOut.labels = labels;
        
            bmiOut.InputTensor = InputTensor;
            bmiOut.InputMatrix = InputMatrix;
        
            bmiOut.nTrials = nTrials;
            bmiOut.nSamples = nSamples;
            bmiOut.nChannels = nChannels;
            bmiOut.nFeatures = size(InputMatrix, 2);
        end

        function save_bmi_input_file(bmiOut, saveDir)
        
            if ~exist(saveDir, 'dir')
                mkdir(saveDir);
            end
        
            if isfield(bmiOut, 'bmiWindowSize')
                windowSize = bmiOut.bmiWindowSize;
            else
                windowSize = bmiOut.bmiWindow(2) - bmiOut.bmiWindow(1);
            end
        
            % Creates 0p5s or 1p0s
            windowTag = strrep(sprintf('%.1fs', windowSize), '.', 'p');
        
            saveFile = fullfile(saveDir, ...
                [char(bmiOut.subjectName) '_BMI_input_' windowTag '.mat']);
        
            InputTensor = bmiOut.InputTensor;
            InputMatrix = bmiOut.InputMatrix;
            labels = bmiOut.labels;
            tBMI = bmiOut.tBMI;
            chanLabels = bmiOut.chanLabels;
            fs = bmiOut.fs;
        
            bmiWindow = bmiOut.bmiWindow;
            bmiWindowSize = windowSize;
            bmiWindowStart = bmiOut.bmiWindow(1);
            bmiWindowEnd = bmiOut.bmiWindow(2);
        
            subjectName = bmiOut.subjectName;
            group = bmiOut.group;
        
            save(saveFile, ...
                'InputTensor', ...
                'InputMatrix', ...
                'labels', ...
                'tBMI', ...
                'chanLabels', ...
                'fs', ...
                'bmiWindow', ...
                'bmiWindowSize', ...
                'bmiWindowStart', ...
                'bmiWindowEnd', ...
                'subjectName', ...
                'group', ...
                '-v7.3');
        
            fprintf('Saved BMI input file: %s\n', saveFile);
            fprintf('  BMI window: [%.2f %.2f] s | size = %.2f s\n', ...
                bmiWindowStart, bmiWindowEnd, bmiWindowSize);
            fprintf('  InputTensor: %d trials x %d samples x %d channels\n', ...
                size(InputTensor,1), size(InputTensor,2), size(InputTensor,3));
            fprintf('  InputMatrix: %d trials x %d features\n', ...
                size(InputMatrix,1), size(InputMatrix,2));
        end

        function erdersOut = compute_subject_erders_all_channels(epochs, labels, baselineIdx, leftCode, rightCode)
            % Compute ERD/ERS per trial first, then average across trials.
            %
            % No smoothing is applied here. Smoothing should be visualization-only.
        
            labels = labels(:);
        
            leftIdx = labels == leftCode;
            rightIdx = labels == rightCode;
        
            nCh = size(epochs,1);
            nSamp = size(epochs,2);
            nTrials = size(epochs,3);
        
            erdTrial = nan(nCh, nSamp, nTrials);
        
            for tr = 1:nTrials
                for ch = 1:nCh
        
                    sig = double(squeeze(epochs(ch,:,tr)));
                    pow = sig.^2;
        
                    basePow = mean(pow(baselineIdx), 'omitnan');
        
                    if isnan(basePow) || basePow <= 0
                        continue
                    end
        
                    erdTrial(ch,:,tr) = 100 .* ((pow - basePow) ./ (basePow + eps));
                end
            end
        
            erdersOut = struct();
            erdersOut.left = mean(erdTrial(:,:,leftIdx), 3, 'omitnan');
            erdersOut.right = mean(erdTrial(:,:,rightIdx), 3, 'omitnan');
        
            erdersOut.left_trials = erdTrial(:,:,leftIdx);
            erdersOut.right_trials = erdTrial(:,:,rightIdx);
        
            erdersOut.smoothingAppliedToSavedValues = false;
        end

        function plot_subject_erders_all_channels(erdersSubject, saveBasePath, plotTitle)
            % Plot all-channel ERD/ERS heatmaps for mu and beta bands.
        
            t = erdersSubject.t(:)';
            chanLabels = erdersSubject.chanLabels;
        
            if isstring(chanLabels)
                chanLabels = cellstr(chanLabels);
            end
        
            bands = {'mu','beta'};
            conds = {'left','right'};
        
            % Visualization-only smoothing settings
            if isfield(erdersSubject, 'plotSmoothWin_sec') && ~isempty(erdersSubject.plotSmoothWin_sec)
                smoothWin_sec = erdersSubject.plotSmoothWin_sec;
            elseif isfield(erdersSubject, 'smoothWin_sec') && ~isempty(erdersSubject.smoothWin_sec)
                % Backward-compatible fallback, but this should now be visualization-only.
                smoothWin_sec = erdersSubject.smoothWin_sec;
            else
                smoothWin_sec = 0;
            end
        
            if isfield(erdersSubject, 'fs') && ~isempty(erdersSubject.fs)
                fs = erdersSubject.fs;
            else
                if length(t) > 1
                    fs = 1 / median(diff(t), 'omitnan');
                else
                    fs = 1;
                end
            end
        
            smoothWin_samp = round(smoothWin_sec * fs);
        
            % Fixed ERD/ERS color limits for fair comparison
            if isfield(erdersSubject, 'plotColorLimits') && ~isempty(erdersSubject.plotColorLimits)
                colorLimits = erdersSubject.plotColorLimits;
            else
                colorLimits = [-75 75];
            end
        
            for b = 1:length(bands)
        
                band = bands{b};
        
                if ~isfield(erdersSubject, band)
                    error('erdersSubject is missing band field: %s', band);
                end
        
                fig = figure('Visible','off', ...
                    'Color','w', ...
                    'Position',[100 100 1450 850]);
        
                tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
        
                for c = 1:length(conds)
        
                    cond = conds{c};
        
                    if ~isfield(erdersSubject.(band), cond)
                        error('erdersSubject.%s is missing condition field: %s', band, cond);
                    end
        
                    % Pull UNSMOOTHED saved data
                    dataRaw = erdersSubject.(band).(cond);
        
                    % Make visualization-only copy
                    dataPlot = dataRaw;
        
                    % Smooth only across time dimension for plotting.
                    % Do not overwrite saved feature values.
                    if smoothWin_samp > 1
                        dataPlot = movmean(dataPlot, smoothWin_samp, 2, 'omitnan');
                    end
        
                    ax = nexttile;
                    imagesc(ax, t, 1:length(chanLabels), dataPlot);
                    axis(ax, 'xy');
        
                    colormap(ax, turbo);
                    cb = colorbar(ax);
                    cb.Label.String = 'ERD/ERS [%]';
        
                    clim(ax, colorLimits);
        
                    Utility_Functions.add_time_window_markers( ...
                        ax, ...
                        erdersSubject.cueTimeSec, ...
                        erdersSubject.baselineWindow, ...
                        erdersSubject.miWindow);
        
                    yticks(ax, 1:length(chanLabels));
                    yticklabels(ax, chanLabels);
        
                    xlabel(ax, 'Time [s]');
                    ylabel(ax, 'Channel');
        
                    if strcmpi(cond, 'left')
                        condLabel = 'Left MI';
                    elseif strcmpi(cond, 'right')
                        condLabel = 'Right MI';
                    else
                        condLabel = cond;
                    end
        
                    title(ax, sprintf('%s | %s | Smoothed for display only', upper(band), condLabel), ...
                        'Interpreter','none');
        
                    Utility_Functions.style_publication_axes(ax);
                end
        
                sgtitle(sprintf('%s | %s ERD/ERS | Plot smoothing %.2f s', ...
                    plotTitle, upper(band), smoothWin_sec), ...
                    'Interpreter','none', ...
                    'FontWeight','bold');
        
                Utility_Functions.save_figure_tif_fig( ...
                    fig, ...
                    [char(saveBasePath) '_' band], ...
                    300);
        
                close(fig);
            end
        end

        function plot_subject_tfr_all_channels(tfrSubject, colorLimits, saveBasePath, plotTitle)
            % Plot channel-time TFR summaries for left and right MI.
            %
            % This function expects:
            %   tfrSubject.bands.<bandName>.left.<channel>
            %   tfrSubject.bands.<bandName>.right.<channel>
            %
            % Each channel entry is frequency x time. For visualization, this function
            % averages across the frequency dimension to create a channel x time summary.
            %
            % Smoothing is applied only to the plotting copy, not to saved TFR values.
        
            t = tfrSubject.t(:)';
            chanLabels = tfrSubject.chanLabels;
        
            if isstring(chanLabels)
                chanLabels = cellstr(chanLabels);
            end
        
            nCh = length(chanLabels);
        
            if nargin < 2 || isempty(colorLimits)
                if isfield(tfrSubject, 'plotColorLimits') && ~isempty(tfrSubject.plotColorLimits)
                    colorLimits = tfrSubject.plotColorLimits;
                else
                    colorLimits = [-3 3];
                end
            end
        
            if nargin < 4 || isempty(plotTitle)
                plotTitle = 'TFR Channel-Time Summary';
            end
        
            if isfield(tfrSubject, 'plotSmoothWin_sec') && ~isempty(tfrSubject.plotSmoothWin_sec)
                smoothWin_sec = tfrSubject.plotSmoothWin_sec;
            else
                smoothWin_sec = 0.25;
            end
        
            smoothSamp = round(smoothWin_sec * tfrSubject.fs);
        
            if ~isfield(tfrSubject, 'bands') || isempty(fieldnames(tfrSubject.bands))
                error('tfrSubject.bands is missing or empty. Rerun Step02 TFR extraction with multi-band TFR enabled.');
            end
        
            bandNames = fieldnames(tfrSubject.bands);
        
            for b = 1:length(bandNames)
        
                bandName = bandNames{b};
                bandStruct = tfrSubject.bands.(bandName);
        
                if ~isfield(bandStruct, 'range')
                    error('TFR band %s is missing range field.', bandName);
                end
        
                bandRange = bandStruct.range;
        
                leftSummary = nan(nCh, length(t));
                rightSummary = nan(nCh, length(t));
        
                for ch = 1:nCh
        
                    chName = matlab.lang.makeValidName(chanLabels{ch});
        
                    if isfield(bandStruct.left, chName)
                        % left TFR is frequency x time
                        leftSummary(ch,:) = mean(bandStruct.left.(chName), 1, 'omitnan');
                    end
        
                    if isfield(bandStruct.right, chName)
                        % right TFR is frequency x time
                        rightSummary(ch,:) = mean(bandStruct.right.(chName), 1, 'omitnan');
                    end
                end
        
                % Smooth only the plotting copy across time.
                leftPlot = leftSummary;
                rightPlot = rightSummary;
        
                if smoothSamp > 1
                    leftPlot = movmean(leftPlot, smoothSamp, 2, 'omitnan');
                    rightPlot = movmean(rightPlot, smoothSamp, 2, 'omitnan');
                end
        
                fig = figure('Visible','off', ...
                    'Color','w', ...
                    'Position',[100 100 1450 850]);
        
                tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
        
                ax1 = nexttile;
                imagesc(ax1, t, 1:nCh, leftPlot);
                axis(ax1, 'xy');
                colormap(ax1, turbo);
                cb1 = colorbar(ax1);
                ylabel(cb1, 'Baseline-normalized power [dB]');
                clim(ax1, colorLimits);
        
                Utility_Functions.add_time_window_markers( ...
                    ax1, ...
                    tfrSubject.cueTimeSec, ...
                    tfrSubject.baselineWindow, ...
                    tfrSubject.miWindow);
        
                yticks(ax1, 1:nCh);
                yticklabels(ax1, chanLabels);
                xlabel(ax1, 'Time [s]');
                ylabel(ax1, 'Channel');
                title(ax1, sprintf('Left MI | Mean %.0f-%.0f Hz TFR channel-time summary', ...
                    bandRange(1), bandRange(2)), ...
                    'Interpreter','none');
        
                Utility_Functions.style_publication_axes(ax1);
        
                ax2 = nexttile;
                imagesc(ax2, t, 1:nCh, rightPlot);
                axis(ax2, 'xy');
                colormap(ax2, turbo);
                cb2 = colorbar(ax2);
                ylabel(cb2, 'Baseline-normalized power [dB]');
                clim(ax2, colorLimits);
        
                Utility_Functions.add_time_window_markers( ...
                    ax2, ...
                    tfrSubject.cueTimeSec, ...
                    tfrSubject.baselineWindow, ...
                    tfrSubject.miWindow);
        
                yticks(ax2, 1:nCh);
                yticklabels(ax2, chanLabels);
                xlabel(ax2, 'Time [s]');
                ylabel(ax2, 'Channel');
                title(ax2, sprintf('Right MI | Mean %.0f-%.0f Hz TFR channel-time summary', ...
                    bandRange(1), bandRange(2)), ...
                    'Interpreter','none');
        
                Utility_Functions.style_publication_axes(ax2);
        
                sgtitle(sprintf('%s | %s | plot smoothing = %.2f s', ...
                    plotTitle, bandName, smoothWin_sec), ...
                    'Interpreter','none', ...
                    'FontWeight','bold');
        
                savePathBand = [char(saveBasePath) '_' bandName];
        
                Utility_Functions.save_figure_tif_fig(fig, savePathBand, 300);
                close(fig);
            end
        end

        function summary = compute_channel_band_power_with_baseline(Xband, labels, leftCode, rightCode, baselineIdx, miIdx)
            % Compute channel-wise band power with baseline normalization
            %
            % Inputs:
            %   Xband: [channels x samples x trials] filtered EEG data
            %   labels: [trials x 1] condition labels
            %   leftCode, rightCode: numeric codes for conditions
            %   baselineIdx, miIdx: logical indices for time windows
            %
            % Outputs:
            %   summary: struct with mean power and ERD% for each channel and condition

            leftIdx  = labels == leftCode;
            rightIdx = labels == rightCode;

            nCh = size(Xband,1);
            epsVal = 1e-10;

            summary.left_baseline_mean_ch  = zeros(nCh,1);
            summary.right_baseline_mean_ch = zeros(nCh,1);
            summary.left_mi_mean_ch        = zeros(nCh,1);
            summary.right_mi_mean_ch       = zeros(nCh,1);

            for ch = 1:nCh
                % Left trials
                leftTrials = squeeze(Xband(ch,:,leftIdx));
                if isvector(leftTrials)
                    leftTrials = reshape(leftTrials, [], 1);
                end

                leftPow = leftTrials.^2;
                summary.left_baseline_mean_ch(ch) = mean(leftPow(baselineIdx,:), 'all');
                summary.left_mi_mean_ch(ch)       = mean(leftPow(miIdx,:), 'all');

                % Right trials
                rightTrials = squeeze(Xband(ch,:,rightIdx));
                if isvector(rightTrials)
                    rightTrials = reshape(rightTrials, [], 1);
                end

                rightPow = rightTrials.^2;
                summary.right_baseline_mean_ch(ch) = mean(rightPow(baselineIdx,:), 'all');
                summary.right_mi_mean_ch(ch)       = mean(rightPow(miIdx,:), 'all');
            end

            % Summary ERD%
            summary.left_erders_mean_ch = ...
                ((summary.left_mi_mean_ch - summary.left_baseline_mean_ch) ./ ...
                (summary.left_baseline_mean_ch + epsVal)) * 100;

            summary.right_erders_mean_ch = ...
                ((summary.right_mi_mean_ch - summary.right_baseline_mean_ch) ./ ...
                (summary.right_baseline_mean_ch + epsVal)) * 100;
        end

        function inspect_edf_file(edfFile)
            % Inspect EDF file for raw signals and data structure
            % Displays comprehensive information about the EDF file using edfinfo
            %
            % Input:
            %   edfFile: full path to the EDF file to inspect

            disp('========================================');
            disp('EDF FILE INSPECTION STARTED');
            disp('========================================');

            if ~exist(edfFile, 'file')
                disp(['ERROR: EDF file not found: ' edfFile]);
                disp('Please update the file path.');
                return;
            end

            disp(['File: ' edfFile]);
            disp(' ');

            %% 1. Basic File Information from edfinfo
            disp('=== BASIC FILE INFORMATION (edfinfo) ===');
            info = edfinfo(edfFile);

            disp(['Number of Signals: ' num2str(info.NumSignals)]);
            disp(['Number of Data Records: ' num2str(info.NumDataRecords)]);

            % Handle DataRecordDuration - it might be numeric, duration, or string
            if isnumeric(info.DataRecordDuration)
                dataRecordDuration = info.DataRecordDuration;
            elseif isduration(info.DataRecordDuration)
                dataRecordDuration = seconds(info.DataRecordDuration);
            elseif ischar(info.DataRecordDuration) || isstring(info.DataRecordDuration)
                % Try to parse as number, or extract numeric part
                dataRecordDuration = str2double(info.DataRecordDuration);
                if isnan(dataRecordDuration)
                    % If parsing fails, try to extract number from string like "1.000 seconds"
                    tokens = regexp(info.DataRecordDuration, '(\d+\.?\d*)', 'tokens');
                    if ~isempty(tokens)
                        dataRecordDuration = str2double(tokens{1}{1});
                    else
                        dataRecordDuration = NaN;
                    end
                end
            else
                dataRecordDuration = NaN;
            end

            if isnan(dataRecordDuration)
                disp(['Data Record Duration: ' char(info.DataRecordDuration)]);
                disp('Total Duration: Unknown (cannot parse duration)');
                disp('Sampling Frequency: Unknown (cannot parse duration)');
            else
                disp(['Data Record Duration: ' num2str(dataRecordDuration) ' seconds']);
                disp(['Total Duration: ' num2str(info.NumDataRecords * dataRecordDuration) ' seconds']);
                disp(['Sampling Frequency: ' num2str(1/dataRecordDuration) ' Hz']);
            end
            disp(' ');

            %% 2. Signal Labels (Channel Names)
            disp('=== SIGNAL LABELS (CHANNELS) ===');
            disp(['Channels: ' strjoin(info.SignalLabels, ', ')]);
            disp(' ');

            %% 3. Signal Properties from edfinfo
            disp('=== SIGNAL PROPERTIES (edfinfo) ===');

            % Check what properties are available
            availableProps = fieldnames(info);
            disp(['Available edfinfo properties: ' strjoin(availableProps', ', ')]);
            disp(' ');

            for i = 1:min(5, info.NumSignals)  % Check first 5 signals
                disp(['Signal ' num2str(i) ' (' info.SignalLabels{i} '):']);

                % Check and display PhysicalMin/Max if available
                if ismember('PhysicalMin', availableProps) && ismember('PhysicalMax', availableProps)
                    disp(['  Physical Range: ' num2str(info.PhysicalMin(i)) ' to ' num2str(info.PhysicalMax(i))]);
                end

                % Check and display PhysicalUnits if available
                if ismember('PhysicalUnits', availableProps)
                    disp(['  Physical Units: ' info.PhysicalUnits{i}]);
                end

                % Check and display DigitalMin/Max if available
                if ismember('DigitalMin', availableProps) && ismember('DigitalMax', availableProps)
                    disp(['  Digital Range: ' num2str(info.DigitalMin(i)) ' to ' num2str(info.DigitalMax(i))]);
                end

                % Check and display Prefiltering if available
                if ismember('Prefiltering', availableProps)
                    disp(['  Prefiltering: ' info.Prefiltering{i}]);

                    % Check if prefiltering indicates raw data
                    if isempty(info.Prefiltering{i}) || contains(lower(info.Prefiltering{i}), 'none')
                        disp('  ✓ LIKELY RAW: No prefiltering reported');
                    else
                        disp(['  ⚠ POSSIBLY FILTERED: ' info.Prefiltering{i}]);
                    end
                end

                disp(' ');
            end

            %% 4. Data Structure Summary
            disp('=== DATA STRUCTURE SUMMARY ===');
            disp(['File contains ' num2str(info.NumSignals) ' signals (channels)']);
            disp(['Data is organized in ' num2str(info.NumDataRecords) ' data records']);

            if info.NumDataRecords > 1
                disp('⚠ MULTIPLE DATA RECORDS: File contains blocked data');
                if ~isnan(dataRecordDuration)
                    disp(['  → Each record: ' num2str(dataRecordDuration) ' seconds']);
                    disp(['  → Total duration: ' num2str(info.NumDataRecords * dataRecordDuration) ' seconds']);
                end
                disp('  → Use edfread() to load data as timetable with cell structure');
            else
                disp('✓ SINGLE DATA RECORD: Continuous data stream');
                disp('  → Use edfread() to load data as simple numeric matrix');
            end

            % Display units and ranges if available
            if ismember('PhysicalUnits', availableProps) && ~isempty(info.PhysicalUnits)
                disp(['Physical units (first channel): ' info.PhysicalUnits{1}]);
            end

            if ismember('DigitalMin', availableProps) && ismember('DigitalMax', availableProps)
                disp(['Digital range (first channel): ' num2str(info.DigitalMin(1)) ' to ' num2str(info.DigitalMax(1))]);
            end

            disp(' ');
            disp('========================================');
            disp('EDF FILE INSPECTION COMPLETE');
            disp('========================================');
            disp('Note: Use edfread() in convert_edf_into_matrix() to see actual data structure');
        end

        function [labels_clean, keepMask] = clean_channel_labels(labels, datasetType, removeChannels)
            % Clean and standardize channel labels
            %
            % Inputs:
            %   labels: cell array of channel labels
            %   datasetType: 'healthy' or 'stroke' (affects channel name standardization)
            %   removeChannels: cell array of channel names to remove
            %
            % Outputs:
            %   labels_clean: cleaned and standardized channel labels
            %   keepMask: logical mask indicating which original channels to keep

            disp('Cleaning and standardizing channel labels...');

            % Convert string array to cell array if needed
            if isstring(labels)
                labels = cellstr(labels);
                disp('✓ Converted labels from string to cell array');
            end

            labels = labels(:);
            originalCount = length(labels);
            disp(['Original labels count: ' num2str(originalCount)]);

            % Remove empty labels
            nonEmptyMask = ~cellfun(@isempty, labels);
            labels = labels(nonEmptyMask);
            disp(['After removing empty labels: ' num2str(length(labels))]);

            % Clean and standardize each label
            for i = 1:length(labels)
                lbl = labels{i};
                originalLbl = lbl;

                lbl = strtrim(lbl);        % Remove whitespace
                lbl = regexprep(lbl, '\.', '');  % Remove dots
                lbl = upper(lbl);          % Convert to uppercase

                % Standardize channels for combined dataset analysis (international 10-20 system)
                if strcmpi(datasetType, 'healthy')
                    switch lbl
                        case 'T7'
                            lbl = 'T3';
                        case 'T8'
                            lbl = 'T4';
                        case 'P7'
                            lbl = 'T5';
                        case 'P8'
                            lbl = 'T6';
                    end
                end

                labels{i} = lbl;
                if ~strcmp(originalLbl, lbl)
                    disp(['  Standardized: ' originalLbl ' → ' lbl]);
                end
            end

            % Remove unwanted channels
            keepMaskLocal = ~ismember(labels, removeChannels);
            removedChannels = labels(~keepMaskLocal);

            if ~isempty(removedChannels)
                disp(['Removing channels: ' strjoin(removedChannels, ', ')]);
            end

            labels_clean = labels(keepMaskLocal);
            disp(['Final clean labels: ' num2str(length(labels_clean))]);
            disp(['Clean labels: ' strjoin(labels_clean, ', ')]);

            % Rebuild keep mask relative to original label list
            keepMask = false(originalCount, 1);
            idxNonEmpty = find(nonEmptyMask);
            keepMask(idxNonEmpty(keepMaskLocal)) = true;

            disp(['Keep mask: ' num2str(sum(keepMask)) '/' num2str(originalCount) ' channels kept']);
        end

        function fs = get_sampling_frequency(info)
            % Compute sampling frequency from EDF metadata
            % Uses the exact method from Zander's original code

            disp('Computing sampling frequency from EDF metadata...');
            fs = info.NumSamples(1) / seconds(info.DataRecordDuration);
            disp(['✓ Sampling frequency: ' num2str(fs) ' Hz']);
        end

        function [X_harm, matchedLabels, foundMask, idxMap] = apply_channel_harmonization(X, rawLabels, harmonizedLabels, datasetType, removeChannels)
            % Reorder EEG channels to match harmonized template
            %
            % Inputs:
            %   X: EEG data matrix [channels x samples]
            %   rawLabels: cell array of raw channel labels from EDF file
            %   harmonizedLabels: cell array of target harmonized channel labels
            %   datasetType: 'healthy' or 'stroke' (for channel cleaning)
            %   removeChannels: cell array of channels to remove (EOG, ground, etc.)
            %
            % Outputs:
            %   X_harm: Harmonized EEG data [matched_channels x samples]
            %   matchedLabels: Cell array of successfully matched channel labels
            %   foundMask: Logical mask indicating which harmonized channels were found
            %   idxMap: Index mapping from harmonized to raw channel positions

            disp('========================================');
            disp('APPLYING CHANNEL HARMONIZATION');
            disp('========================================');

            % Convert string array to cell array if needed
            if isstring(rawLabels)
                rawLabels = cellstr(rawLabels);
                disp('✓ Converted rawLabels from string to cell array');
            end

            rawLabels = rawLabels(:);
            disp(['Input channels: ' num2str(length(rawLabels))]);
            disp(['Input channel names (first 10): ' strjoin(rawLabels(1:min(10, length(rawLabels))), ', ')]);

            % Clean channel labels (remove unwanted channels)
            disp(['Removing channels: ' strjoin(removeChannels, ', ')]);
            [labels_clean, keepMask] = Utility_Functions.clean_channel_labels(rawLabels, datasetType, removeChannels);

            disp(['Channels after cleaning: ' num2str(length(labels_clean))]);
            disp(['Cleaned channel names (first 10): ' strjoin(labels_clean(1:min(10, length(labels_clean))), ', ')]);

            % Apply cleaning to data
            X_clean = X(keepMask, :);
            disp(['Data shape after cleaning: ' num2str(size(X_clean, 1)) ' channels × ' num2str(size(X_clean, 2)) ' samples']);

            % Find matching channels
            harmonizedLabels = harmonizedLabels(:);
            disp(['Target harmonized channels: ' num2str(length(harmonizedLabels))]);
            disp(['Target channel names: ' strjoin(harmonizedLabels, ', ')]);

            [foundMask, idxMap] = ismember(harmonizedLabels, labels_clean);

            nFound = sum(foundMask);
            nMissing = sum(~foundMask);

            disp(['✓ Found ' num2str(nFound) ' matching channels']);
            if nMissing > 0
                disp(['⚠ Missing ' num2str(nMissing) ' channels:']);
                missingChannels = harmonizedLabels(~foundMask);
                disp(['  ' strjoin(missingChannels', ', ')]);
            end

            % Extract harmonized data
            X_harm = X_clean(idxMap(foundMask), :);
            matchedLabels = harmonizedLabels(foundMask);

            disp(['Final harmonized data shape: ' num2str(size(X_harm, 1)) ' channels × ' num2str(size(X_harm, 2)) ' samples']);
            disp(['Matched channel names: ' strjoin(matchedLabels, ', ')]);
            disp('========================================');
        end

        function [X_delta, X_theta, X_mi, X_mu, X_beta, X_gamma] = apply_bandpass_filters(X, filters)
            % Apply delta, theta, MI, mu, beta, and gamma filters to EEG matrix
            %
            % Inputs:
            %   X: EEG data matrix [channels x samples]
            %   filters: struct containing filter coefficients (b_*, a_* fields)
            %
            % Outputs:
            %   X_delta, X_theta, X_mi, X_mu, X_beta, X_gamma: Filtered data matrices

            disp('Applying bandpass filters to all channels...');

            X_delta = zeros(size(X));
            X_theta = zeros(size(X));
            X_mi    = zeros(size(X));
            X_mu    = zeros(size(X));
            X_beta  = zeros(size(X));
            X_gamma = zeros(size(X));

            nCh = size(X, 1);

            for ch = 1:nCh
                sig = double(X(ch,:));

                X_delta(ch,:) = filtfilt(filters.b_delta, filters.a_delta, sig);
                X_theta(ch,:) = filtfilt(filters.b_theta, filters.a_theta, sig);
                X_mi(ch,:)    = filtfilt(filters.b_mi, filters.a_mi, sig);
                X_mu(ch,:)    = filtfilt(filters.b_mu, filters.a_mu, sig);
                X_beta(ch,:)  = filtfilt(filters.b_beta, filters.a_beta, sig);
                X_gamma(ch,:) = filtfilt(filters.b_gamma, filters.a_gamma, sig);

                if mod(ch, max(1, round(nCh/10))) == 0
                    disp(['  Filtered ' num2str(ch) '/' num2str(nCh) ' channels']);
                end
            end

            disp('✓ Filtering complete (delta, theta, MI, mu, beta, gamma)');
        end

        function [X_delta, X_theta, X_mi, X_mu, X_beta, X_gamma] = apply_bandpass_filters_3d(X, filters)
            % Apply delta, theta, MI, mu, beta, and gamma filters to 3D EEG matrix
            %
            % Inputs:
            %   X: EEG data matrix [channels x samples x trials]
            %   filters: struct containing filter coefficients (b_*, a_* fields)
            %
            % Outputs:
            %   X_delta, X_theta, X_mi, X_mu, X_beta, X_gamma: Filtered data matrices [channels x samples x trials]

            disp('Applying bandpass filters to 3D EEG data (all trials)...');

            nCh = size(X,1);
            nSamp = size(X,2);
            nTrials = size(X,3);

            X_delta = zeros(nCh, nSamp, nTrials);
            X_theta = zeros(nCh, nSamp, nTrials);
            X_mi    = zeros(nCh, nSamp, nTrials);
            X_mu    = zeros(nCh, nSamp, nTrials);
            X_beta  = zeros(nCh, nSamp, nTrials);
            X_gamma = zeros(nCh, nSamp, nTrials);

            for tr = 1:nTrials
                for ch = 1:nCh
                    sig = double(squeeze(X(ch,:,tr)));

                    X_delta(ch,:,tr) = filtfilt(filters.b_delta, filters.a_delta, sig);
                    X_theta(ch,:,tr) = filtfilt(filters.b_theta, filters.a_theta, sig);
                    X_mi(ch,:,tr)    = filtfilt(filters.b_mi,    filters.a_mi,    sig);
                    X_mu(ch,:,tr)    = filtfilt(filters.b_mu,    filters.a_mu,    sig);
                    X_beta(ch,:,tr)  = filtfilt(filters.b_beta,  filters.a_beta,  sig);
                    X_gamma(ch,:,tr) = filtfilt(filters.b_gamma, filters.a_gamma, sig);
                end

                if mod(tr, max(1, round(nTrials/10))) == 0
                    disp(['  Filtered ' num2str(tr) '/' num2str(nTrials) ' trials']);
                end
            end

            disp('✓ 3D filtering complete (delta, theta, MI, mu, beta, gamma)');
        end

        function [onsetSamples, labels, eventsMI, allEvents] = load_healthy_mi_events(info, fs, subjectName, runNumber, edfName)
            % Load PhysioNet EEGMMI annotations from EDF info.
            %
            % Outputs:
            %   onsetSamples - T1/T2 onset samples at sampling rate fs
            %   labels       - 1 = T1, 2 = T2
            %   eventsMI     - table of T1/T2 events with previous T0 timing
            %   allEvents    - table of all T0/T1/T2 annotations for this EDF run
        
            if nargin < 3 || isempty(subjectName)
                subjectName = "";
            end
        
            if nargin < 4 || isempty(runNumber)
                runNumber = NaN;
            end
        
            if nargin < 5 || isempty(edfName)
                edfName = "";
            end
        
            if ~isprop(info, 'Annotations') && ~isfield(info, 'Annotations')
                error('EDF info does not contain Annotations.');
            end
        
            ann = info.Annotations;
        
            if isempty(ann)
                error('EDF annotations table is empty.');
            end
        
            if ~istimetable(ann) && ~istable(ann)
                error('EDF Annotations must be a table or timetable.');
            end
        
            if ~ismember('Annotations', ann.Properties.VariableNames)
                error('EDF annotations table does not contain an Annotations column.');
            end
        
            if ~ismember('Duration', ann.Properties.VariableNames)
                error('EDF annotations table does not contain a Duration column.');
            end
        
            % Onset is usually the row time in the timetable.
            if istimetable(ann)
                onsetSec = seconds(ann.Properties.RowTimes);
            elseif ismember('Onset', ann.Properties.VariableNames)
                onsetSec = Utility_Functions.annotation_time_to_seconds(ann.Onset);
            else
                error('Could not find annotation onset times.');
            end
        
            durationSec = Utility_Functions.annotation_time_to_seconds(ann.Duration);
            annotationText = upper(strtrim(string(ann.Annotations)));
        
            nEvents = height(ann);
        
            allEvents = table();
            allEvents.subjectName = repmat(string(subjectName), nEvents, 1);
            allEvents.runNumber = repmat(runNumber, nEvents, 1);
            allEvents.edfFile = repmat(string(edfName), nEvents, 1);
            allEvents.annotationIndex = (1:nEvents)';
            allEvents.annotation = annotationText;
            allEvents.onsetSec = onsetSec(:);
            allEvents.durationSec = durationSec(:);
            allEvents.endSec = onsetSec(:) + durationSec(:);
        
            isT0 = annotationText == "T0";
            isT1 = annotationText == "T1";
            isT2 = annotationText == "T2";
            isMI = isT1 | isT2;
        
            miRows = find(isMI);
        
            if isempty(miRows)
                onsetSamples = [];
                labels = [];
                eventsMI = table();
                return
            end
        
            nMI = numel(miRows);
        
            onsetSamples = zeros(nMI, 1);
            labels = zeros(nMI, 1);
        
            prevT0Index = nan(nMI, 1);
            prevT0OnsetSec = nan(nMI, 1);
            prevT0DurationSec = nan(nMI, 1);
            prevT0EndSec = nan(nMI, 1);
            prevT0GapToMISec = nan(nMI, 1);
        
            for k = 1:nMI
        
                idx = miRows(k);
        
                if isT1(idx)
                    labels(k) = 1;
                elseif isT2(idx)
                    labels(k) = 2;
                end
        
                onsetSamples(k) = round(onsetSec(idx) * fs) + 1;
        
                % Closest T0 immediately before this T1/T2 event.
                candidateT0 = find(isT0 & onsetSec < onsetSec(idx), 1, 'last');
        
                if ~isempty(candidateT0)
                    prevT0Index(k) = candidateT0;
                    prevT0OnsetSec(k) = onsetSec(candidateT0);
                    prevT0DurationSec(k) = durationSec(candidateT0);
                    prevT0EndSec(k) = onsetSec(candidateT0) + durationSec(candidateT0);
                    prevT0GapToMISec(k) = onsetSec(idx) - prevT0EndSec(k);
                end
            end
        
            eventsMI = table();
            eventsMI.subjectName = repmat(string(subjectName), nMI, 1);
            eventsMI.runNumber = repmat(runNumber, nMI, 1);
            eventsMI.edfFile = repmat(string(edfName), nMI, 1);
        
            eventsMI.miEventIndex = miRows(:);
            eventsMI.miTrialIndex = (1:nMI)';
            eventsMI.annotation = annotationText(miRows);
            eventsMI.label = labels(:);
        
            eventsMI.miOnsetSec = onsetSec(miRows);
            eventsMI.miDurationSec = durationSec(miRows);
            eventsMI.miEndSec = onsetSec(miRows) + durationSec(miRows);
            eventsMI.onsetSample = onsetSamples(:);
        
            eventsMI.prevT0Index = prevT0Index;
            eventsMI.prevT0OnsetSec = prevT0OnsetSec;
            eventsMI.prevT0DurationSec = prevT0DurationSec;
            eventsMI.prevT0EndSec = prevT0EndSec;
            eventsMI.prevT0GapToMISec = prevT0GapToMISec;
        
            eventsMI.hasPreviousT0 = ~isnan(prevT0DurationSec);
            eventsMI.hasAtLeast0p5sBaseline = prevT0DurationSec >= 0.5;
            eventsMI.hasAtLeast1sBaseline = prevT0DurationSec >= 1.0;
            eventsMI.hasAtLeast2sBaseline = prevT0DurationSec >= 2.0;
            eventsMI.hasAtLeast4p2sBaseline = prevT0DurationSec >= 4.2;
        
            fprintf('Loaded %d MI events from %s.\n', nMI, string(edfName));
            fprintf('  T0 duration range before MI: %.3f to %.3f s\n', ...
                min(prevT0DurationSec, [], 'omitnan'), ...
                max(prevT0DurationSec, [], 'omitnan'));
            fprintf('  T1/T2 duration range: %.3f to %.3f s\n', ...
                min(eventsMI.miDurationSec, [], 'omitnan'), ...
                max(eventsMI.miDurationSec, [], 'omitnan'));
        end

        function sec = annotation_time_to_seconds(x)
            % Convert EDF annotation onset/duration fields into seconds.
        
            if isduration(x)
                sec = seconds(x);
                return
            end
        
            if isnumeric(x)
                sec = double(x);
                return
            end
        
            if iscell(x)
                sec = nan(numel(x), 1);
                for i = 1:numel(x)
                    sec(i) = Utility_Functions.annotation_time_to_seconds(x{i});
                end
                return
            end
        
            if isstring(x) || ischar(x)
                xs = string(x);
                sec = nan(numel(xs), 1);
        
                for i = 1:numel(xs)
                    token = regexp(xs(i), '[-+]?\d*\.?\d+', 'match', 'once');
        
                    if ~isempty(token)
                        sec(i) = str2double(token);
                    end
                end
        
                return
            end
        
            error('Unsupported annotation time type: %s', class(x));
        end

        function [X_harm, matchedLabels] = apply_channel_harmonization_trialwise(X, inputLabels, harmonizedLabels)
            % Apply channel harmonization to 3D trial-based EEG data
            %
            % Inputs:
            %   X: 3D EEG data [channels x samples x trials]
            %   inputLabels: Cell array of input channel labels
            %   harmonizedLabels: Cell array of target channel labels (10-20 template)
            %
            % Outputs:
            %   X_harm: Harmonized data [nKeep x samples x trials]
            %   matchedLabels: Labels corresponding to harmonized channels

            if isstring(inputLabels)
                inputLabels = cellstr(inputLabels);
            end

            inputLabels = inputLabels(:);
            harmonizedLabels = harmonizedLabels(:);

            nTrials = size(X,3);
            nSamples = size(X,2);
            nKeep = numel(harmonizedLabels);

            X_harm = zeros(nKeep, nSamples, nTrials);
            matchedLabels = harmonizedLabels;

            for i = 1:nKeep
                idx = find(strcmpi(strtrim(inputLabels), strtrim(harmonizedLabels{i})), 1);

                if isempty(idx)
                    error('Missing channel during harmonization: %s', harmonizedLabels{i});
                end

                X_harm(i,:,:) = X(idx,:,:);
            end
        end

        function X_resampled = resample_2d_matrix(X, fsOld, fsNew)
            % Resample continuous EEG matrix channel-by-channel.
            % X is channels x samples.
        
            if fsOld == fsNew
                X_resampled = X;
                return
            end
        
            nCh = size(X, 1);
        
            testSig = resample(double(X(1,:)), fsNew, fsOld);
            nSampNew = length(testSig);
        
            X_resampled = zeros(nCh, nSampNew);
        
            for ch = 1:nCh
                X_resampled(ch,:) = resample(double(X(ch,:)), fsNew, fsOld);
            end
        
            fprintf('Resampled continuous EEG from %.1f Hz to %.1f Hz.\n', fsOld, fsNew);
            fprintf('Samples: %d -> %d\n', size(X,2), size(X_resampled,2));
        end

        function X_resampled = resample_3d_trials(X, fsOld, fsNew)
            % Resample 3D trial-based EEG data channel-by-channel and trial-by-trial
            % First applies 0.1 Hz highpass filter (4th order Butterworth), then resamples
            %
            % Inputs:
            %   X: 3D EEG data [channels x samples x trials]
            %   fsOld: Original sampling frequency (Hz)
            %   fsNew: Target sampling frequency (Hz)
            %
            % Outputs:
            %   X_resampled: Filtered and resampled data [channels x newSamples x trials]

            % Design and apply 0.1 Hz highpass filter (4th order Butterworth) BEFORE resampling
            filterOrder_hp = 4;
            cutoff_hp = 0.1; % Hz
            nyquist_old = fsOld / 2;
            normalized_cutoff = cutoff_hp / nyquist_old;

            if normalized_cutoff >= 1
                warning('Highpass cutoff frequency >= Nyquist frequency. Skipping highpass filter.');
                X_filtered = X;
            else
                [b_hp, a_hp] = butter(filterOrder_hp, normalized_cutoff, 'high');

                disp(['Applying 0.1 Hz highpass filter (4th order Butterworth) to raw 3D trial data...']);

                nCh = size(X, 1);
                nTrials = size(X, 3);
                X_filtered = zeros(size(X));

                for tr = 1:nTrials
                    for ch = 1:nCh
                        sig = double(squeeze(X(ch,:,tr)));
                        X_filtered(ch,:,tr) = filtfilt(b_hp, a_hp, sig);
                    end
                end

                disp(['✓ Highpass filtering complete']);
            end

            % Now resample the filtered data
            if fsOld == fsNew
                error('fsOld and fsNew are equal. Resampling is required - cannot proceed with identical frequencies.');
            end

            filterOrder = 512; % High order for better anti-aliasing, especially when downsampling
            nCh = size(X_filtered,1);
            nTrials = size(X_filtered,3);

            testSig = resample(double(squeeze(X_filtered(1,:,1))), fsNew, fsOld, filterOrder);
            nSampNew = length(testSig);

            X_resampled = zeros(nCh, nSampNew, nTrials);

            disp(['Resampling filtered 3D trial data: ' num2str(nCh) ' channels × ' num2str(nTrials) ' trials...']);

            for tr = 1:nTrials
                for ch = 1:nCh
                    X_resampled(ch,:,tr) = resample(double(squeeze(X_filtered(ch,:,tr))), fsNew, fsOld, filterOrder);
                end
            end

            disp(['✓ Resampling complete: ' num2str(nCh) ' channels, ' num2str(nTrials) ' trials']);
            disp(['  ' num2str(size(X,2)) ' → ' num2str(nSampNew) ' samples per trial']);
        end

        
        function [tfr_avg_db, f] = compute_time_frequency(trials_2d, fs, freqLimits, t, baselineWindow)
            % Compute average time-frequency representation across trials.
            %
            % Handles NaN-padded trials by computing CWT only over finite samples.
            % This is important for healthy PhysioNet trials because annotation-defined
            % trial lengths can differ across runs and shorter trials are padded with NaN.
        
            if isvector(trials_2d)
                trials_2d = trials_2d(:);
            end
        
            % Expected format: samples x trials
            if size(trials_2d, 1) == 1
                trials_2d = trials_2d';
            end
        
            nSamp = size(trials_2d, 1);
            nTrials = size(trials_2d, 2);
        
            if nargin < 4 || isempty(t)
                t = (0:nSamp-1) / fs;
            else
                t = t(:)';
            end
        
            if nargin < 5 || isempty(baselineWindow)
                baselineWindow = [t(1), t(1) + 2];
            end
        
            TF_all = [];
            f = [];
        
            for tr = 1:nTrials
        
                sigFull = double(trials_2d(:, tr));
                finiteIdx = isfinite(sigFull);
        
                if ~any(finiteIdx)
                    warning('Skipping trial %d in TFR because all samples are non-finite.', tr);
                    continue
                end
        
                % Use the continuous finite span. This handles NaN padding at the end
                % while preserving the true time positions in the full trial.
                firstFinite = find(finiteIdx, 1, 'first');
                lastFinite  = find(finiteIdx, 1, 'last');
        
                sig = sigFull(firstFinite:lastFinite);
        
                % If there are interior NaNs, interpolate them. End NaNs are already excluded.
                if any(~isfinite(sig))
                    sig = fillmissing(sig, 'linear', 'EndValues', 'nearest');
                end
        
                if any(~isfinite(sig)) || isempty(sig)
                    warning('Skipping trial %d in TFR because finite-sample repair failed.', tr);
                    continue
                end
        
                [wt, f_temp] = cwt(sig, fs, 'amor', 'FrequencyLimits', freqLimits);
                powerTemp = abs(wt).^2;
        
                % Initialize storage after the first valid CWT
                if isempty(TF_all)
                    f = f_temp;
                    TF_all = nan(length(f), nSamp, nTrials);
                end
        
                % Store power back into the correct time locations
                TF_all(:, firstFinite:lastFinite, tr) = powerTemp;
            end
        
            if isempty(TF_all)
                error('No valid finite trials were available for TFR computation.');
            end
        
            avgTF = mean(TF_all, 3, 'omitnan');
        
            baselineIdx = t >= baselineWindow(1) & t < baselineWindow(2);
        
            if any(baselineIdx)
                baselinePower = mean(avgTF(:, baselineIdx), 2, 'omitnan');
                tfr_avg_db = 10 * log10((avgTF + eps) ./ (baselinePower + eps));
            else
                warning('No baseline samples found for TFR baseline correction. Returning uncorrected log power.');
                tfr_avg_db = 10 * log10(avgTF + eps);
            end
        end

        function power_trials = compute_channel_band_power_fft(Xband, fs, idx)

            [nCh, ~, nTrials] = size(Xband);
            power_trials = zeros(nCh, nTrials);

            for tr = 1:nTrials
                for ch = 1:nCh
                    sig = double(squeeze(Xband(ch,idx,tr)));
                    power_trials(ch,tr) = Utility_Functions.fft_bandpower(sig, fs);
                end
            end
        end

        function bp = fft_bandpower(sig, fs)
        
            sig = sig(:);
            sig = sig(isfinite(sig));
        
            if numel(sig) < 4
                bp = NaN;
                return
            end
        
            sig = sig - mean(sig, 'omitnan');
        
            n = length(sig);
            win = hamming(n);
            sig = sig .* win;
        
            Y = fft(sig);
            P2 = abs(Y / n).^2;
            P1 = P2(1:floor(n/2)+1);
            P1(2:end-1) = 2 * P1(2:end-1);
        
            f = fs * (0:floor(n/2)) / n;
        
            bp = trapz(f, P1);
        end


        function erpOut = compute_subject_channel_erp(X, labels, t, chanLabels, leftCode, rightCode)
            % Compute subject-level ERP by direct trial averaging.
            %
            % No ERP baseline correction is applied here. Data are assumed to already
            % be filtered in Step01.
        
            labels = labels(:);
            t = t(:)';
        
            Xuse = double(X);
        
            leftIdx = labels == leftCode;
            rightIdx = labels == rightCode;
        
            erpOut = struct();
            erpOut.t = t;
            erpOut.chanLabels = chanLabels;
            erpOut.doBaselineCorrection = false;
        
            erpOut.nAll = size(Xuse, 3);
            erpOut.nLeft = sum(leftIdx);
            erpOut.nRight = sum(rightIdx);
        
            erpOut.all = mean(Xuse, 3, 'omitnan');
            erpOut.all_sem = std(Xuse, 0, 3, 'omitnan') ./ sqrt(size(Xuse, 3));
        
            if any(leftIdx)
                erpOut.left = mean(Xuse(:,:,leftIdx), 3, 'omitnan');
                erpOut.left_sem = std(Xuse(:,:,leftIdx), 0, 3, 'omitnan') ./ sqrt(sum(leftIdx));
            else
                erpOut.left = nan(size(Xuse,1), size(Xuse,2));
                erpOut.left_sem = nan(size(Xuse,1), size(Xuse,2));
            end
        
            if any(rightIdx)
                erpOut.right = mean(Xuse(:,:,rightIdx), 3, 'omitnan');
                erpOut.right_sem = std(Xuse(:,:,rightIdx), 0, 3, 'omitnan') ./ sqrt(sum(rightIdx));
            else
                erpOut.right = nan(size(Xuse,1), size(Xuse,2));
                erpOut.right_sem = nan(size(Xuse,1), size(Xuse,2));
            end
        
            erpOut.left_minus_right = erpOut.left - erpOut.right;
        end

        function plot_erp_all_channels_with_windows(t, erpOut, chanLabels, baselineWindow, miWindow, cueTimeSec, saveBasePath, plotTitle)
            % Plot and save ERP waveforms for every channel.
            %
            % Saves one .fig and one .tif per channel.
            %
            % Fixed y-axis limits:
            %   MI-band ERP    : [-20 20] uV
            %   Delta-band ERP : [-11 10] uV
            %
            % Expected erpOut fields:
            %   erpOut.all
            %   erpOut.left
            %   erpOut.right
            %   erpOut.band
        
            t = t(:)';
        
            if nargin < 8 || isempty(plotTitle)
                plotTitle = 'ERP';
            end
        
            if nargin < 7 || isempty(saveBasePath)
                error('saveBasePath must be provided.');
            end
        
            if isstring(chanLabels)
                chanLabels = cellstr(chanLabels);
            end
        
            nCh = length(chanLabels);
        
            requiredFields = {'all','left','right'};
            for f = 1:length(requiredFields)
                if ~isfield(erpOut, requiredFields{f})
                    error('erpOut is missing required field: %s', requiredFields{f});
                end
            end
        
            if size(erpOut.all,1) ~= nCh
                error('Number of ERP channels does not match chanLabels.');
            end
        
            if size(erpOut.all,2) ~= length(t)
                error('ERP time dimension does not match t.');
            end
        
            % Determine band name
            if isfield(erpOut, 'band') && ~isempty(erpOut.band)
                bandName = char(erpOut.band);
            else
                bandName = '';
            end
        
            % Fixed y-limits for comparable ERP plots
            switch upper(string(bandName))
                case "MI"
                    yLimits = [-20 20];
                    yLabelText = 'Amplitude [\muV]';
                case "DELTA"
                    yLimits = [-11 10];
                    yLabelText = 'Amplitude [\muV]';
                otherwise
                    % Fallback if band is not set
                    allVals = [
                        erpOut.all(:)
                        erpOut.left(:)
                        erpOut.right(:)
                    ];
        
                    allVals = allVals(isfinite(allVals));
        
                    if isempty(allVals)
                        yLimits = [-1 1];
                    else
                        yMin = min(allVals);
                        yMax = max(allVals);
        
                        if yMin == yMax
                            pad = max(abs(yMax)*0.1, 1);
                        else
                            pad = 0.12 * (yMax - yMin);
                        end
        
                        yLimits = [yMin - pad, yMax + pad];
                    end
        
                    yLabelText = 'Amplitude [\muV]';
            end
        
            [saveDir, baseName, ~] = fileparts(char(saveBasePath));
        
            if isempty(saveDir)
                saveDir = pwd;
            end
        
            if ~exist(saveDir, 'dir')
                mkdir(saveDir);
            end
        
            % Loop through channels and save one file per channel
            for ch = 1:nCh
        
                chLabel = chanLabels{ch};
                chNameSafe = matlab.lang.makeValidName(chLabel);
        
                figBase = fullfile(saveDir, [baseName '_' chNameSafe]);
        
                fig = figure('Visible','off', ...
                    'Color','w', ...
                    'Position',[100 100 950 650]);
        
                ax = axes(fig);
                hold(ax, 'on');
        
                % Set fixed y-limits before patches so shaded windows span full axis
                ylim(ax, yLimits);
                yl = yLimits;
        
                % Baseline/rest window - darker red shading, hidden from legend
                if ~isempty(baselineWindow) && numel(baselineWindow) == 2
                    patch(ax, ...
                        [baselineWindow(1) baselineWindow(2) baselineWindow(2) baselineWindow(1)], ...
                        [yl(1) yl(1) yl(2) yl(2)], ...
                        [1.00 0.65 0.65], ...
                        'FaceAlpha',0.32, ...
                        'EdgeColor','none', ...
                        'HandleVisibility','off');
                end
        
                % MI window - darker green shading, hidden from legend
                if ~isempty(miWindow) && numel(miWindow) == 2
                    patch(ax, ...
                        [miWindow(1) miWindow(2) miWindow(2) miWindow(1)], ...
                        [yl(1) yl(1) yl(2) yl(2)], ...
                        [0.65 1.00 0.65], ...
                        'FaceAlpha',0.32, ...
                        'EdgeColor','none', ...
                        'HandleVisibility','off');
                end
        
                % Zero-amplitude reference line, hidden from legend
                yline(ax, 0, 'k:', ...
                    'LineWidth',1.0, ...
                    'HandleVisibility','off');
        
                % Cue / MI onset marker, hidden from legend
                xline(ax, cueTimeSec, 'k--', ...
                    'MI onset', ...
                    'LineWidth',1.3, ...
                    'LabelOrientation','aligned', ...
                    'LabelVerticalAlignment','top', ...
                    'FontSize',10, ...
                    'HandleVisibility','off');
        
                % Optional baseline and MI boundaries, hidden from legend
                if ~isempty(baselineWindow) && numel(baselineWindow) == 2
                    xline(ax, baselineWindow(1), 'r--', ...
                        'LineWidth',0.9, ...
                        'HandleVisibility','off');
        
                    xline(ax, baselineWindow(2), 'r--', ...
                        'LineWidth',0.9, ...
                        'HandleVisibility','off');
                end
        
                if ~isempty(miWindow) && numel(miWindow) == 2
                    xline(ax, miWindow(1), 'g--', ...
                        'LineWidth',0.9, ...
                        'HandleVisibility','off');
        
                    xline(ax, miWindow(2), 'g--', ...
                        'LineWidth',0.9, ...
                        'HandleVisibility','off');
                end
        
                % Plot ERP waveforms. Only these handles go into the legend.
                hAll = plot(ax, t, erpOut.all(ch,:), ...
                    'Color',[0 0 0], ...
                    'LineWidth',1.6, ...
                    'DisplayName','All trials');
        
                hLeft = plot(ax, t, erpOut.left(ch,:), ...
                    'Color',[0 0.20 1.00], ...
                    'LineWidth',1.4, ...
                    'DisplayName','Left MI');
        
                hRight = plot(ax, t, erpOut.right(ch,:), ...
                    'Color',[1.00 0 0], ...
                    'LineWidth',1.4, ...
                    'DisplayName','Right MI');
        
                % Axis formatting
                xlim(ax, [t(1) t(end)]);
                ylim(ax, yLimits);
        
                xlabel(ax, 'Time [s]');
                ylabel(ax, yLabelText);
        
                title(ax, sprintf('%s | %s', plotTitle, chLabel), ...
                    'Interpreter','none', ...
                    'FontWeight','bold');
        
                legend(ax, [hAll hLeft hRight], ...
                    {'All trials','Left MI','Right MI'}, ...
                    'Location','best', ...
                    'Interpreter','none');
        
                Utility_Functions.style_publication_axes(ax);
        
                hold(ax, 'off');
        
                % Save as .fig and .tif using shared helper
                Utility_Functions.save_figure_tif_fig(fig, figBase, 500);
                close(fig);
            end
        end


        function val = mean_psd_fft(sig, fs, bandRange)
        
            sig = double(sig(:));
            sig = sig - mean(sig, 'omitnan');
        
            n = length(sig);
        
            if n < 4
                val = NaN;
                return
            end
        
            win = hamming(n);
            sigWin = sig .* win;
        
            nfft = max(256, 2^nextpow2(n));
        
            Y = fft(sigWin, nfft);
        
            P2 = abs(Y).^2 / (fs * sum(win.^2));
            P1 = P2(1:nfft/2+1);
            P1(2:end-1) = 2 * P1(2:end-1);
        
            f = fs * (0:nfft/2) / nfft;
        
            if isscalar(bandRange)
                idx = f >= bandRange;
            else
                idx = f >= bandRange(1) & f <= bandRange(2);
            end
        
            if ~any(idx)
                val = NaN;
            else
                val = mean(P1(idx), 'omitnan');
            end
        end

        function psdOut = compute_subject_channel_psd_by_condition(epochs, labels, fs, baselineIdx, miIdx, bandRange, leftCode, rightCode)
            % Compute windowed band power from already band-filtered EEG.
            %
            % The input epochs are already filtered to the target band, so this function
            % computes mean squared amplitude in the baseline and MI windows.
            % dB fields are stored and should be used for plots/statistics.
        
            labels = labels(:);
        
            leftIdx = labels == leftCode;
            rightIdx = labels == rightCode;
        
            nCh = size(epochs,1);
        
            psdOut = struct();
        
            fields = {'left_baseline','left_mi','right_baseline','right_mi'};
            for f = 1:length(fields)
                psdOut.(fields{f}) = nan(nCh,1);
                psdOut.([fields{f} '_db']) = nan(nCh,1);
                psdOut.([fields{f} '_sem_db']) = nan(nCh,1);
            end
        
            psdOut.nLeft = sum(leftIdx);
            psdOut.nRight = sum(rightIdx);
            psdOut.fs = fs;
            psdOut.bandRange = bandRange;
            psdOut.powerMethod = 'mean_squared_amplitude_from_bandpass_filtered_signal';
        
            for ch = 1:nCh
        
                leftBaseVals = Utility_Functions.window_power_per_trial(epochs(ch,baselineIdx,leftIdx));
                leftMiVals   = Utility_Functions.window_power_per_trial(epochs(ch,miIdx,leftIdx));
        
                rightBaseVals = Utility_Functions.window_power_per_trial(epochs(ch,baselineIdx,rightIdx));
                rightMiVals   = Utility_Functions.window_power_per_trial(epochs(ch,miIdx,rightIdx));
        
                psdOut.left_baseline(ch)  = mean(leftBaseVals, 'omitnan');
                psdOut.left_mi(ch)        = mean(leftMiVals, 'omitnan');
                psdOut.right_baseline(ch) = mean(rightBaseVals, 'omitnan');
                psdOut.right_mi(ch)       = mean(rightMiVals, 'omitnan');
        
                leftBaseDB  = Utility_Functions.pow_to_db(leftBaseVals);
                leftMiDB    = Utility_Functions.pow_to_db(leftMiVals);
                rightBaseDB = Utility_Functions.pow_to_db(rightBaseVals);
                rightMiDB   = Utility_Functions.pow_to_db(rightMiVals);
        
                psdOut.left_baseline_db(ch)  = mean(leftBaseDB, 'omitnan');
                psdOut.left_mi_db(ch)        = mean(leftMiDB, 'omitnan');
                psdOut.right_baseline_db(ch) = mean(rightBaseDB, 'omitnan');
                psdOut.right_mi_db(ch)       = mean(rightMiDB, 'omitnan');
        
                psdOut.left_baseline_sem_db(ch)  = Utility_Functions.sem_omitnan(leftBaseDB);
                psdOut.left_mi_sem_db(ch)        = Utility_Functions.sem_omitnan(leftMiDB);
                psdOut.right_baseline_sem_db(ch) = Utility_Functions.sem_omitnan(rightBaseDB);
                psdOut.right_mi_sem_db(ch)       = Utility_Functions.sem_omitnan(rightMiDB);
            end
        end

        function vals = window_power_per_trial(X)
            % X can be 1 x samples x trials or samples x trials.
            % Returns one power value per trial.
        
            X = squeeze(X);
        
            if isempty(X)
                vals = NaN;
                return
            end
        
            if isvector(X)
                X = X(:);
            end
        
            % samples x trials expected
            vals = mean(double(X).^2, 1, 'omitnan')';
        end

        function s = sem_omitnan(x)
        
            x = x(:);
            x = x(~isnan(x));
        
            if isempty(x)
                s = NaN;
            else
                s = std(x) / sqrt(length(x));
            end
        end

        function plot_subject_psd_bars_by_condition(psdOut, chanLabels, saveBasePath, plotTitle, scaleMode)
            % Plot subject-level windowed band power by condition.
            %
            % scaleMode:
            %   'db'     -> uses *_db fields, primary plot/statistics scale
            %   'linear' -> uses linear power fields, inspection/debugging only
        
            if nargin < 5 || isempty(scaleMode)
                scaleMode = 'db';
            end
        
            if isstring(chanLabels)
                chanLabels = cellstr(chanLabels);
            end
        
            scaleMode = lower(string(scaleMode));
        
            switch scaleMode
        
                case "db"
                    requiredFields = { ...
                        'left_baseline_db', ...
                        'left_mi_db', ...
                        'right_baseline_db', ...
                        'right_mi_db'};
        
                    for rf = 1:length(requiredFields)
                        if ~isfield(psdOut, requiredFields{rf})
                            error('psdOut is missing dB field: %s', requiredFields{rf});
                        end
                    end
        
                    vals = [
                        psdOut.left_baseline_db, ...
                        psdOut.left_mi_db, ...
                        psdOut.right_baseline_db, ...
                        psdOut.right_mi_db
                    ];
        
                    yLabelText = 'Band power [dB]';
                    includeZero = true;
        
                case "linear"
                    requiredFields = { ...
                        'left_baseline', ...
                        'left_mi', ...
                        'right_baseline', ...
                        'right_mi'};
        
                    for rf = 1:length(requiredFields)
                        if ~isfield(psdOut, requiredFields{rf})
                            error('psdOut is missing linear field: %s', requiredFields{rf});
                        end
                    end
        
                    vals = [
                        psdOut.left_baseline, ...
                        psdOut.left_mi, ...
                        psdOut.right_baseline, ...
                        psdOut.right_mi
                    ];
        
                    yLabelText = 'Band power [linear]';
                    includeZero = true;
        
                otherwise
                    error('Unknown scaleMode: %s. Use ''db'' or ''linear''.', scaleMode);
            end
        
            fig = figure('Visible','off', ...
                'Color','w', ...
                'Position',[100 100 1550 750]);
        
            ax = axes(fig);
            hold(ax, 'on');
        
            bar(ax, vals, 'grouped');
        
            xlabel(ax, 'Channel');
            ylabel(ax, yLabelText);
        
            title(ax, plotTitle, ...
                'Interpreter','none', ...
                'FontWeight','bold');
        
            legend(ax, ...
                {'Left baseline','Left MI','Right baseline','Right MI'}, ...
                'Location','best', ...
                'Interpreter','none');
        
            xticks(ax, 1:length(chanLabels));
            xticklabels(ax, chanLabels);
            ax.TickLabelInterpreter = 'none';
            xtickangle(ax, 45);
        
            yLimits = Utility_Functions.get_bar_ylim(vals, includeZero);
            ylim(ax, yLimits);
        
            Utility_Functions.style_publication_axes(ax);
        
            hold(ax, 'off');
        
            Utility_Functions.save_figure_tif_fig(fig, saveBasePath, 300);
            close(fig);
        end

        function plot_subject_channel_power_summary(topoSubject, saveBasePath, plotTitle)
            % Plot TOPO/channel-power summaries using hand-specific dB values.
            %
            % Plots:
            %   left baseline
            %   left MI
            %   right baseline
            %   right MI
            %
            % Mean TOPO fields can still be stored, but they are not the only plotted values.
        
            chanLabels = topoSubject.chanLabels;
        
            if isstring(chanLabels)
                chanLabels = cellstr(chanLabels);
            end
        
            bandNames = topoSubject.bandNames;
        
            for b = 1:length(bandNames)
        
                band = bandNames{b};
        
                if ~isfield(topoSubject, band)
                    error('topoSubject is missing band field: %s', band);
                end
        
                T = topoSubject.(band);
        
                requiredFields = { ...
                    'left_baseline_power_db', ...
                    'left_mi_power_db', ...
                    'right_baseline_power_db', ...
                    'right_mi_power_db'};
        
                for rf = 1:length(requiredFields)
                    if ~isfield(T, requiredFields{rf})
                        error('topoSubject.%s is missing required field: %s', ...
                            band, requiredFields{rf});
                    end
                end
        
                vals = [
                    T.left_baseline_power_db(:), ...
                    T.left_mi_power_db(:), ...
                    T.right_baseline_power_db(:), ...
                    T.right_mi_power_db(:)
                ];
        
                fig = figure('Visible','off', ...
                    'Color','w', ...
                    'Position',[100 100 1550 750]);
        
                ax = axes(fig);
                hold(ax, 'on');
        
                bar(ax, vals, 'grouped');
        
                xlabel(ax, 'Channel');
                ylabel(ax, 'Channel power [dB]');
        
                title(ax, sprintf('%s | %s | Hand-specific baseline vs MI', ...
                    plotTitle, upper(band)), ...
                    'Interpreter','none', ...
                    'FontWeight','bold');
        
                legend(ax, ...
                    {'Left baseline','Left MI','Right baseline','Right MI'}, ...
                    'Location','best', ...
                    'Interpreter','none');
        
                xticks(ax, 1:length(chanLabels));
                xticklabels(ax, chanLabels);
                ax.TickLabelInterpreter = 'none';
                xtickangle(ax, 45);
        
                yLimits = Utility_Functions.get_bar_ylim(vals, true);
                ylim(ax, yLimits);
        
                Utility_Functions.style_publication_axes(ax);
        
                hold(ax, 'off');
        
                Utility_Functions.save_figure_tif_fig( ...
                    fig, ...
                    [char(saveBasePath) '_' band], ...
                    300);
        
                close(fig);
            end
        end

        function plot_subject_allpair_assi(assiSubject, saveBasePath, plotTitle)
            % Plot all-pair ASSI values with left-hand and right-hand MI separated.
            %
            % This avoids cancellation of lateralization direction across hands.
            % Pair labels use hyphens instead of underscores to prevent MATLAB
            % subscript formatting.
        
            bandNames = assiSubject.bandNames;
            pairInfo = assiSubject.pairInfo;
        
            pairLabels = strings(length(pairInfo), 1);
        
            for p = 1:length(pairInfo)
                pairLabels(p) = string(pairInfo(p).pairName);
            end
        
            pairLabels = strrep(pairLabels, "_", "-");
        
            for b = 1:length(bandNames)
        
                band = bandNames{b};
        
                leftBaseline = nan(length(pairInfo), 1);
                leftMI = nan(length(pairInfo), 1);
                rightBaseline = nan(length(pairInfo), 1);
                rightMI = nan(length(pairInfo), 1);
        
                for p = 1:length(pairInfo)
        
                    pairName = pairInfo(p).pairName;
        
                    if ~isfield(assiSubject.(band), pairName)
                        error('ASSI band %s is missing pair field: %s', band, pairName);
                    end
        
                    A = assiSubject.(band).(pairName);
        
                    leftBaseline(p) = A.left_baseline_ai;
                    leftMI(p) = A.left_ai_mi;
                    rightBaseline(p) = A.right_baseline_ai;
                    rightMI(p) = A.right_ai_mi;
                end
        
                vals = [leftBaseline, leftMI, rightBaseline, rightMI];
        
                fig = figure('Visible','off', ...
                    'Color','w', ...
                    'Position',[100 100 1500 750]);
        
                ax = axes(fig);
                hold(ax, 'on');
        
                bar(ax, vals, 'grouped');
        
                yline(ax, 0, 'k:', ...
                    'LineWidth',1.0, ...
                    'HandleVisibility','off');
        
                xlabel(ax, 'Electrode pair');
                ylabel(ax, 'ASSI');
        
                title(ax, sprintf('%s | %s | Hand-specific lateralization', ...
                    plotTitle, upper(band)), ...
                    'Interpreter','none', ...
                    'FontWeight','bold');
        
                legend(ax, ...
                    {'Left baseline','Left MI','Right baseline','Right MI'}, ...
                    'Location','best', ...
                    'Interpreter','none');
        
                xticks(ax, 1:length(pairLabels));
                xticklabels(ax, pairLabels);
                ax.TickLabelInterpreter = 'none';
                xtickangle(ax, 45);
        
                yLimits = Utility_Functions.get_bar_ylim(vals, true);
                ylim(ax, yLimits);
        
                Utility_Functions.style_publication_axes(ax);
        
                hold(ax, 'off');
        
                Utility_Functions.save_figure_tif_fig( ...
                    fig, ...
                    [char(saveBasePath) '_' band], ...
                    300);
        
                close(fig);
            end
        end

        function pairInfo = get_left_right_channel_pairs(chanLabels)
        
            pairNames = {
                'FP1','FP2'
                'F3', 'F4'
                'F7', 'F8'
                'FC3','FC4'
                'FT7','FT8'
                'C3', 'C4'
                'T3', 'T4'
                'CP3','CP4'
                'TP7','TP8'
                'P3', 'P4'
                'T5', 'T6'
                'O1', 'O2'
            };
        
            pairInfo = struct([]);
            count = 0;
        
            for i = 1:size(pairNames,1)
        
                leftName = pairNames{i,1};
                rightName = pairNames{i,2};
        
                leftIdx = find(strcmpi(chanLabels, leftName), 1);
                rightIdx = find(strcmpi(chanLabels, rightName), 1);
        
                if isempty(leftIdx) || isempty(rightIdx)
                    continue
                end
        
                count = count + 1;
                pairInfo(count).leftName = leftName;
                pairInfo(count).rightName = rightName;
                pairInfo(count).pairName = [leftName '_' rightName];
                pairInfo(count).leftIdx = leftIdx;
                pairInfo(count).rightIdx = rightIdx;
            end
        end

        function assiOut = compute_summary_allpair_asymmetry( ...
            deltaSummary, thetaSummary, muSummary, betaSummary, gammaSummary, chanLabels)
        
            pairInfo = Utility_Functions.get_left_right_channel_pairs(chanLabels);
        
            summaries = {deltaSummary, thetaSummary, muSummary, betaSummary, gammaSummary};
            bandNames = {'delta','theta','mu','beta','gamma'};
        
            epsVal = 1e-10;
        
            assiOut = struct();
            assiOut.pairInfo = pairInfo;
            assiOut.chanLabels = chanLabels;
            assiOut.bandNames = bandNames;
        
            for b = 1:length(bandNames)
        
                band = bandNames{b};
                S = summaries{b};
        
                for p = 1:length(pairInfo)
        
                    L = pairInfo(p).leftIdx;
                    R = pairInfo(p).rightIdx;
                    pairName = pairInfo(p).pairName;
        
                    leftBaseL = S.left_baseline_mean_ch(L);
                    leftBaseR = S.left_baseline_mean_ch(R);
                    rightBaseL = S.right_baseline_mean_ch(L);
                    rightBaseR = S.right_baseline_mean_ch(R);
        
                    leftMiL = S.left_mi_mean_ch(L);
                    leftMiR = S.left_mi_mean_ch(R);
                    rightMiL = S.right_mi_mean_ch(L);
                    rightMiR = S.right_mi_mean_ch(R);
        
                    leftErdL = S.left_erders_mean_ch(L);
                    leftErdR = S.left_erders_mean_ch(R);
                    rightErdL = S.right_erders_mean_ch(L);
                    rightErdR = S.right_erders_mean_ch(R);
        
                    % Positive = greater left-hemisphere value than right-hemisphere value.
                    assiOut.(band).(pairName).left_baseline_ai = ...
                        (leftBaseL - leftBaseR) ./ (leftBaseL + leftBaseR + epsVal);
        
                    assiOut.(band).(pairName).right_baseline_ai = ...
                        (rightBaseL - rightBaseR) ./ (rightBaseL + rightBaseR + epsVal);
        
                    assiOut.(band).(pairName).left_ai_mi = ...
                        (leftMiL - leftMiR) ./ (leftMiL + leftMiR + epsVal);
        
                    assiOut.(band).(pairName).right_ai_mi = ...
                        (rightMiL - rightMiR) ./ (rightMiL + rightMiR + epsVal);
        
                    % Task-aware ERD/ERS lateralization:
                    % Left-hand MI: contralateral = right hemisphere.
                    % Right-hand MI: contralateral = left hemisphere.
                    assiOut.(band).(pairName).left_ai_erd_task = leftErdR - leftErdL;
                    assiOut.(band).(pairName).right_ai_erd_task = rightErdL - rightErdR;
        
                    % Simple left-minus-right ERD/ERS, not task-aware.
                    assiOut.(band).(pairName).left_erd_lr = leftErdL - leftErdR;
                    assiOut.(band).(pairName).right_erd_lr = rightErdL - rightErdR;
                end
            end
        end

        function groupERP = average_group_erp_mi_delta(healthySubjects)
        
            nSubj = length(healthySubjects);
        
            first = healthySubjects(1).ERP;
        
            groupERP = struct();
            groupERP.nSubjects = nSubj;
            groupERP.subjectNames = cell(nSubj,1);
            groupERP.t = first.t;
            groupERP.chanLabels = first.chanLabels;
            groupERP.baselineWindow = first.baselineWindow;
            groupERP.miWindow = first.miWindow;
            groupERP.cueTimeSec = first.cueTimeSec;
        
            bands = {'MI','Delta'};
        
            for b = 1:length(bands)
        
                band = bands{b};
        
                allStack = nan([size(first.(band).all), nSubj]);
                leftStack = nan([size(first.(band).left), nSubj]);
                rightStack = nan([size(first.(band).right), nSubj]);
        
                for s = 1:nSubj
                    groupERP.subjectNames{s} = char(healthySubjects(s).subjectName);
                    allStack(:,:,s) = healthySubjects(s).ERP.(band).all;
                    leftStack(:,:,s) = healthySubjects(s).ERP.(band).left;
                    rightStack(:,:,s) = healthySubjects(s).ERP.(band).right;
                end
        
                groupERP.(band).all = mean(allStack, 3, 'omitnan');
                groupERP.(band).left = mean(leftStack, 3, 'omitnan');
                groupERP.(band).right = mean(rightStack, 3, 'omitnan');
        
                groupERP.(band).all_sem = std(allStack, 0, 3, 'omitnan') ./ sqrt(nSubj);
                groupERP.(band).left_sem = std(leftStack, 0, 3, 'omitnan') ./ sqrt(nSubj);
                groupERP.(band).right_sem = std(rightStack, 0, 3, 'omitnan') ./ sqrt(nSubj);
        
                groupERP.(band).t = groupERP.t;
                groupERP.(band).chanLabels = groupERP.chanLabels;
                groupERP.(band).baselineWindow = groupERP.baselineWindow;
                groupERP.(band).miWindow = groupERP.miWindow;
                groupERP.(band).cueTimeSec = groupERP.cueTimeSec;
                groupERP.(band).band = band;
                groupERP.(band).nSubjects = nSubj;
            end
        end


        function groupERDS = average_group_erders_all_channels(healthySubjects)
        
            nSubj = length(healthySubjects);
        
            first = healthySubjects(1).ERDS;
        
            groupERDS = struct();
            groupERDS.nSubjects = nSubj;
            groupERDS.t = first.t;
            groupERDS.chanLabels = first.chanLabels;
            groupERDS.baselineWindow = first.baselineWindow;
            groupERDS.miWindow = first.miWindow;
            groupERDS.cueTimeSec = first.cueTimeSec;
        
            bands = {'mu','beta'};
            conds = {'left','right'};
        
            for b = 1:length(bands)
                band = bands{b};
        
                for c = 1:length(conds)
                    cond = conds{c};
        
                    firstMat = first.(band).(cond);
                    stack = nan([size(firstMat), nSubj]);
        
                    for s = 1:nSubj
                        stack(:,:,s) = healthySubjects(s).ERDS.(band).(cond);
                    end
        
                    groupERDS.(band).(cond) = mean(stack, 3, 'omitnan');
                    groupERDS.(band).([cond '_sem']) = std(stack, 0, 3, 'omitnan') ./ sqrt(nSubj);
                end
            end
        end

        function plot_group_erp_heatmap_all_channels(groupERP, saveBasePath, plotTitle)
            % Plot group-average ERP heatmaps for all channels.
            %
            % Expected groupERP fields:
            %   t, chanLabels, left, right, left_minus_right
            % Optional fields:
            %   cueTimeSec, baselineWindow, miWindow
        
            t = groupERP.t(:)';
            chanLabels = groupERP.chanLabels;
            nCh = length(chanLabels);
        
            % Pull available matrices
            if isfield(groupERP, 'left')
                leftData = groupERP.left;
            else
                error('groupERP is missing field: left');
            end
        
            if isfield(groupERP, 'right')
                rightData = groupERP.right;
            else
                error('groupERP is missing field: right');
            end
        
            if isfield(groupERP, 'left_minus_right')
                diffData = groupERP.left_minus_right;
            else
                diffData = leftData - rightData;
            end
        
            % Robust symmetric color limits across all ERP panels
            allVals = [leftData(:); rightData(:); diffData(:)];
            climAbs = max(abs(allVals), [], 'omitnan');
        
            if isempty(climAbs) || isnan(climAbs) || climAbs == 0
                climAbs = 1;
            end
        
            cLim = [-climAbs climAbs];
        
            fig = figure('Visible','off', ...
                'Color','w', ...
                'Position',[100 100 1500 900]);
        
            tiledlayout(1,3, ...
                'TileSpacing','compact', ...
                'Padding','compact');
        
            % ------------------------------------------------------------
            % Left MI ERP heatmap
            % ------------------------------------------------------------
            ax1 = nexttile;
            imagesc(ax1, t, 1:nCh, leftData);
            axis(ax1, 'xy');
            clim(ax1, cLim);
            colormap(ax1, turbo);
            colorbar(ax1);
        
            yticks(ax1, 1:nCh);
            yticklabels(ax1, chanLabels);
        
            xlabel(ax1, 'Time [s]');
            ylabel(ax1, 'Channel');
            title(ax1, 'Left MI', 'Interpreter','none');
        
            Utility_Functions.style_publication_axes(ax1);
        
            % ------------------------------------------------------------
            % Right MI ERP heatmap
            % ------------------------------------------------------------
            ax2 = nexttile;
            imagesc(ax2, t, 1:nCh, rightData);
            axis(ax2, 'xy');
            clim(ax2, cLim);
            colormap(ax2, turbo);
            colorbar(ax2);
        
            yticks(ax2, 1:nCh);
            yticklabels(ax2, chanLabels);
        
            xlabel(ax2, 'Time [s]');
            ylabel(ax2, 'Channel');
            title(ax2, 'Right MI', 'Interpreter','none');
        
            Utility_Functions.style_publication_axes(ax2);
        
            % ------------------------------------------------------------
            % Left minus Right heatmap
            % ------------------------------------------------------------
            ax3 = nexttile;
            imagesc(ax3, t, 1:nCh, diffData);
            axis(ax3, 'xy');
            clim(ax3, cLim);
            colormap(ax3, turbo);
            colorbar(ax3);
        
            yticks(ax3, 1:nCh);
            yticklabels(ax3, chanLabels);
        
            xlabel(ax3, 'Time [s]');
            ylabel(ax3, 'Channel');
            title(ax3, 'Left - Right', 'Interpreter','none');
        
            Utility_Functions.style_publication_axes(ax3);
        
            % ------------------------------------------------------------
            % Add timing markers if available
            % ------------------------------------------------------------
            if isfield(groupERP, 'cueTimeSec') && ...
                    isfield(groupERP, 'baselineWindow') && ...
                    isfield(groupERP, 'miWindow')
        
                Utility_Functions.add_time_window_markers(ax1, ...
                    groupERP.cueTimeSec, groupERP.baselineWindow, groupERP.miWindow);
        
                Utility_Functions.add_time_window_markers(ax2, ...
                    groupERP.cueTimeSec, groupERP.baselineWindow, groupERP.miWindow);
        
                Utility_Functions.add_time_window_markers(ax3, ...
                    groupERP.cueTimeSec, groupERP.baselineWindow, groupERP.miWindow);
            else
                % Fallback: mark zero if group ERP is relative to cue.
                xline(ax1, 0, 'k--', 'Cue', 'HandleVisibility','off');
                xline(ax2, 0, 'k--', 'Cue', 'HandleVisibility','off');
                xline(ax3, 0, 'k--', 'Cue', 'HandleVisibility','off');
            end
        
            sgtitle(plotTitle, ...
                'Interpreter','none', ...
                'FontWeight','bold');
        
            Utility_Functions.save_figure_tif_fig(fig, saveBasePath, 300);
            close(fig);
        end

        function plot_group_erders_all_channels(groupERDS, saveBasePath, plotTitle)
        
            tmp = groupERDS;
            tmp.fs = [];
            tmp.labels = [];
        
            Utility_Functions.plot_subject_erders_all_channels( ...
                tmp, saveBasePath, plotTitle);
        end
        
        
        function plot_group_tfr_all_channels(groupTFR, colorLimits, saveBasePath, plotTitle)
        
            Utility_Functions.plot_subject_tfr_all_channels( ...
                groupTFR, colorLimits, saveBasePath, plotTitle);
        end
        
        
        function plot_group_assi_all_pairs(groupASSI, saveBasePath, plotTitle)
        
            Utility_Functions.plot_subject_allpair_assi( ...
                groupASSI, saveBasePath, plotTitle);
        end
        
        
        function plot_group_channel_power_summary(groupTOPO, saveBasePath, plotTitle)
        
            Utility_Functions.plot_subject_channel_power_summary( ...
                groupTOPO, saveBasePath, plotTitle);
        end

        function groupTFR = average_group_tfr_all_channels(subjects)
            % Average subject-level TFR channel-time summaries across a group.
            %
            % Updated for multi-band TFR structure:
            %   subjects(s).TFR.bands.<bandName>.left.<channel>
            %   subjects(s).TFR.bands.<bandName>.right.<channel>
            %
            % Each channel TFR is frequency x time.
        
            if isempty(subjects)
                error('No subjects provided to average_group_tfr_all_channels.');
            end
        
            nSubj = length(subjects);
            first = subjects(1).TFR;
        
            if ~isfield(first, 'bands') || isempty(fieldnames(first.bands))
                error(['TFR structure does not contain TFR.bands. ' ...
                       'Rerun Step02 subject-level TFR extraction.']);
            end
        
            groupTFR = struct();
        
            groupTFR.nSubjects = nSubj;
            groupTFR.subjectNames = strings(nSubj, 1);
        
            for s = 1:nSubj
                if isfield(subjects(s), 'subjectName')
                    groupTFR.subjectNames(s) = string(subjects(s).subjectName);
                elseif isfield(subjects(s).TFR, 'subjectName')
                    groupTFR.subjectNames(s) = string(subjects(s).TFR.subjectName);
                else
                    groupTFR.subjectNames(s) = "subject_" + string(s);
                end
            end
        
            groupTFR.t = first.t;
            groupTFR.chanLabels = first.chanLabels;
            groupTFR.baselineWindow = first.baselineWindow;
            groupTFR.miWindow = first.miWindow;
            groupTFR.cueTimeSec = first.cueTimeSec;
        
            if isfield(first, 'plotColorLimits')
                groupTFR.plotColorLimits = first.plotColorLimits;
            else
                groupTFR.plotColorLimits = [-3 3];
            end
        
            if isfield(first, 'plotSmoothWin_sec')
                groupTFR.plotSmoothWin_sec = first.plotSmoothWin_sec;
            else
                groupTFR.plotSmoothWin_sec = 0.25;
            end
        
            if isfield(first, 'fs')
                groupTFR.fs = first.fs;
            elseif length(first.t) > 1
                groupTFR.fs = 1 / median(diff(first.t), 'omitnan');
            else
                groupTFR.fs = NaN;
            end
        
            bandNames = fieldnames(first.bands);
            groupTFR.bands = struct();
        
            for b = 1:length(bandNames)
        
                bandName = bandNames{b};
                firstBand = first.bands.(bandName);
        
                if ~isfield(firstBand, 'left') || ~isfield(firstBand, 'right')
                    error('TFR band %s is missing left or right fields.', bandName);
                end
        
                groupTFR.bands.(bandName).name = bandName;
        
                if isfield(firstBand, 'range')
                    groupTFR.bands.(bandName).range = firstBand.range;
                else
                    groupTFR.bands.(bandName).range = [];
                end
        
                if isfield(firstBand, 'f')
                    groupTFR.bands.(bandName).f = firstBand.f;
                else
                    groupTFR.bands.(bandName).f = [];
                end
        
                groupTFR.bands.(bandName).left = struct();
                groupTFR.bands.(bandName).right = struct();
                groupTFR.bands.(bandName).left_sem = struct();
                groupTFR.bands.(bandName).right_sem = struct();
        
                nCh = length(first.chanLabels);
        
                for ch = 1:nCh
        
                    chName = matlab.lang.makeValidName(first.chanLabels{ch});
        
                    if ~isfield(firstBand.left, chName) || ~isfield(firstBand.right, chName)
                        error('First subject TFR.%s is missing channel %s.', bandName, chName);
                    end
        
                    firstLeft = firstBand.left.(chName);
                    firstRight = firstBand.right.(chName);
        
                    leftStack = nan([size(firstLeft), nSubj]);
                    rightStack = nan([size(firstRight), nSubj]);
        
                    for s = 1:nSubj
        
                        if ~isfield(subjects(s), 'TFR')
                            error('Subject %d is missing TFR field.', s);
                        end
        
                        Ts = subjects(s).TFR;
        
                        if ~isfield(Ts, 'bands') || ~isfield(Ts.bands, bandName)
                            error('Subject %d is missing TFR.bands.%s.', s, bandName);
                        end
        
                        Bs = Ts.bands.(bandName);
        
                        if ~isfield(Bs.left, chName) || ~isfield(Bs.right, chName)
                            error('Subject %d TFR.bands.%s is missing channel %s.', ...
                                s, bandName, chName);
                        end
        
                        leftMat = Bs.left.(chName);
                        rightMat = Bs.right.(chName);
        
                        if ~isequal(size(leftMat), size(firstLeft))
                            error(['Left TFR size mismatch for subject %d, band %s, channel %s.\n' ...
                                   'Expected %s, got %s.'], ...
                                   s, bandName, chName, ...
                                   mat2str(size(firstLeft)), mat2str(size(leftMat)));
                        end
        
                        if ~isequal(size(rightMat), size(firstRight))
                            error(['Right TFR size mismatch for subject %d, band %s, channel %s.\n' ...
                                   'Expected %s, got %s.'], ...
                                   s, bandName, chName, ...
                                   mat2str(size(firstRight)), mat2str(size(rightMat)));
                        end
        
                        leftStack(:,:,s) = leftMat;
                        rightStack(:,:,s) = rightMat;
                    end
        
                    groupTFR.bands.(bandName).left.(chName) = ...
                        mean(leftStack, 3, 'omitnan');
        
                    groupTFR.bands.(bandName).right.(chName) = ...
                        mean(rightStack, 3, 'omitnan');
        
                    nValidLeft = sum(isfinite(leftStack), 3);
                    nValidRight = sum(isfinite(rightStack), 3);
        
                    leftSEM = std(leftStack, 0, 3, 'omitnan') ./ sqrt(nValidLeft);
                    rightSEM = std(rightStack, 0, 3, 'omitnan') ./ sqrt(nValidRight);
        
                    leftSEM(nValidLeft < 2) = NaN;
                    rightSEM(nValidRight < 2) = NaN;
        
                    groupTFR.bands.(bandName).left_sem.(chName) = leftSEM;
                    groupTFR.bands.(bandName).right_sem.(chName) = rightSEM;
                end
            end
        end


        function groupASSI = average_group_assi_all_pairs(healthySubjects)
        
            nSubj = length(healthySubjects);
        
            first = healthySubjects(1).ASSI;
        
            groupASSI = struct();
            groupASSI.nSubjects = nSubj;
            groupASSI.pairInfo = first.pairInfo;
            groupASSI.chanLabels = first.chanLabels;
            groupASSI.bandNames = first.bandNames;
        
            bandNames = first.bandNames;
            pairInfo = first.pairInfo;
        
            fieldsToAverage = { ...
                'left_baseline_ai', ...
                'right_baseline_ai', ...
                'left_ai_mi', ...
                'right_ai_mi', ...
                'left_ai_erd_task', ...
                'right_ai_erd_task', ...
                'left_erd_lr', ...
                'right_erd_lr'};
        
            for b = 1:length(bandNames)
        
                band = bandNames{b};
        
                for p = 1:length(pairInfo)
        
                    pairName = pairInfo(p).pairName;
        
                    for f = 1:length(fieldsToAverage)
        
                        fieldName = fieldsToAverage{f};
                        vals = nan(nSubj,1);
        
                        for s = 1:nSubj
                            vals(s) = healthySubjects(s).ASSI.(band).(pairName).(fieldName);
                        end
        
                        groupASSI.(band).(pairName).(fieldName) = mean(vals, 'omitnan');
                        groupASSI.(band).(pairName).([fieldName '_sem']) = ...
                            std(vals, 'omitnan') ./ sqrt(sum(~isnan(vals)));
                    end
                end
            end
        end


        function groupTOPO = average_group_topo_channel_power(subjects)
        
            if isempty(subjects)
                error('No subjects provided for TOPO group averaging.');
            end
        
            nSubj = length(subjects);
            first = subjects(1).TOPO;
        
            groupTOPO = struct();
            groupTOPO.nSubjects = nSubj;
            groupTOPO.chanLabels = first.chanLabels;
            groupTOPO.bandNames = first.bandNames;
            groupTOPO.baselineWindow = first.baselineWindow;
            groupTOPO.miWindow = first.miWindow;
            groupTOPO.cueTimeSec = first.cueTimeSec;
        
            for b = 1:length(first.bandNames)
        
                band = first.bandNames{b};
        
                if ~isfield(first, band)
                    continue
                end
        
                nCh = length(first.chanLabels);
        
                baselineDBStack = nan(nCh, nSubj);
                miDBStack = nan(nCh, nSubj);
        
                for s = 1:nSubj
        
                    T = subjects(s).TOPO.(band);
        
                    if ~isfield(T, 'mean_baseline_power_db') || ~isfield(T, 'mean_mi_power_db')
                        error(['Subject %d TOPO band %s is missing baseline/MI dB fields. ' ...
                               'Rerun Step02 subject-level extraction.'], s, band);
                    end
        
                    baselineDBStack(:,s) = T.mean_baseline_power_db;
                    miDBStack(:,s) = T.mean_mi_power_db;
                end
        
                [baselineMeanDB, baselineSEMDB] = Utility_Functions.mean_sem_from_db_stack(baselineDBStack);
                [miMeanDB, miSEMDB] = Utility_Functions.mean_sem_from_db_stack(miDBStack);
        
                groupTOPO.(band).mean_baseline_power_db = baselineMeanDB;
                groupTOPO.(band).mean_mi_power_db = miMeanDB;
        
                groupTOPO.(band).mean_baseline_power_sem_db = baselineSEMDB;
                groupTOPO.(band).mean_mi_power_sem_db = miSEMDB;
        
                groupTOPO.(band).delta_mi_minus_baseline_db = miMeanDB - baselineMeanDB;
            end
        end


        function groupPSD = average_group_psd_healthy_stroke(healthySubjects, strokeSubjects)
        
            if isempty(healthySubjects)
                error('No healthy subjects provided for PSD group averaging.');
            end
        
            if isempty(strokeSubjects)
                error('No stroke subjects provided for PSD group averaging.');
            end
        
            firstPSD = healthySubjects(1).PSD;
        
            bandNames = firstPSD.bandNames;
            chanLabels = firstPSD.chanLabels;
        
            groupPSD = struct();
            groupPSD.bandNames = bandNames;
            groupPSD.chanLabels = chanLabels;
            groupPSD.nHealthy = length(healthySubjects);
            groupPSD.nStroke = length(strokeSubjects);
        
            for b = 1:length(bandNames)
        
                band = bandNames{b};
        
                nCh = length(chanLabels);
        
                healthyBaselineStack = nan(nCh, length(healthySubjects));
                healthyMIStack       = nan(nCh, length(healthySubjects));
        
                strokeBaselineStack = nan(nCh, length(strokeSubjects));
                strokeMIStack       = nan(nCh, length(strokeSubjects));
        
                % Healthy subjects
                for s = 1:length(healthySubjects)
        
                    P = healthySubjects(s).PSD.(band);
        
                    % Collapse left/right conditions within subject
                    healthyBaselineStack(:,s) = mean([ ...
                        P.left_baseline, ...
                        P.right_baseline], 2, 'omitnan');
        
                    healthyMIStack(:,s) = mean([ ...
                        P.left_mi, ...
                        P.right_mi], 2, 'omitnan');
                end
        
                % Stroke subjects
                for s = 1:length(strokeSubjects)
        
                    P = strokeSubjects(s).PSD.(band);
        
                    % Collapse left/right conditions within subject
                    strokeBaselineStack(:,s) = mean([ ...
                        P.left_baseline, ...
                        P.right_baseline], 2, 'omitnan');
        
                    strokeMIStack(:,s) = mean([ ...
                        P.left_mi, ...
                        P.right_mi], 2, 'omitnan');
                end
        
                groupPSD.(band).stroke_baseline = mean(strokeBaselineStack, 2, 'omitnan');
                groupPSD.(band).stroke_mi       = mean(strokeMIStack, 2, 'omitnan');
        
                groupPSD.(band).healthy_baseline = mean(healthyBaselineStack, 2, 'omitnan');
                groupPSD.(band).healthy_mi       = mean(healthyMIStack, 2, 'omitnan');
        
                groupPSD.(band).stroke_baseline_sem = std(strokeBaselineStack, 0, 2, 'omitnan') ./ sqrt(length(strokeSubjects));
                groupPSD.(band).stroke_mi_sem       = std(strokeMIStack, 0, 2, 'omitnan') ./ sqrt(length(strokeSubjects));
        
                groupPSD.(band).healthy_baseline_sem = std(healthyBaselineStack, 0, 2, 'omitnan') ./ sqrt(length(healthySubjects));
                groupPSD.(band).healthy_mi_sem       = std(healthyMIStack, 0, 2, 'omitnan') ./ sqrt(length(healthySubjects));
        
                % Keep subject-level stacks for later statistics/debugging
                groupPSD.(band).stroke_baseline_subjects = strokeBaselineStack;
                groupPSD.(band).stroke_mi_subjects = strokeMIStack;
                groupPSD.(band).healthy_baseline_subjects = healthyBaselineStack;
                groupPSD.(band).healthy_mi_subjects = healthyMIStack;
        
                groupPSD.(band).bandName = band;
                groupPSD.(band).nHealthy = length(healthySubjects);
                groupPSD.(band).nStroke = length(strokeSubjects);
            end
        end

        function plot_group_psd_healthy_stroke_bars(psdGroupBand, chanLabels, saveBasePath, plotTitle)
            % Group PSD plot: Stroke baseline, Stroke MI, Healthy baseline, Healthy MI.
        
            useDbFields = isfield(psdGroupBand, 'stroke_baseline_db') && ...
                isfield(psdGroupBand, 'stroke_mi_db') && ...
                isfield(psdGroupBand, 'healthy_baseline_db') && ...
                isfield(psdGroupBand, 'healthy_mi_db');
        
            if useDbFields
                vals = [
                    psdGroupBand.stroke_baseline_db, ...
                    psdGroupBand.stroke_mi_db, ...
                    psdGroupBand.healthy_baseline_db, ...
                    psdGroupBand.healthy_mi_db
                ];
        
                if isfield(psdGroupBand, 'stroke_baseline_sem_db')
                    semVals = [
                        psdGroupBand.stroke_baseline_sem_db, ...
                        psdGroupBand.stroke_mi_sem_db, ...
                        psdGroupBand.healthy_baseline_sem_db, ...
                        psdGroupBand.healthy_mi_sem_db
                    ];
                else
                    semVals = nan(size(vals));
                end
        
            else
                vals = Utility_Functions.pow_to_db([
                    psdGroupBand.stroke_baseline, ...
                    psdGroupBand.stroke_mi, ...
                    psdGroupBand.healthy_baseline, ...
                    psdGroupBand.healthy_mi
                ]);
        
                semVals = nan(size(vals));
            end
        
            nCh = length(chanLabels);
        
            fig = figure('Visible','off', ...
                'Color','w', ...
                'Position',[100 100 1750 800]);
        
            ax = axes(fig);
            b = bar(ax, vals, 'grouped');
            hold(ax, 'on');
        
            if ~all(isnan(semVals(:)))
                for k = 1:length(b)
                    x = b(k).XEndPoints;
                    errorbar(ax, x, vals(:,k), semVals(:,k), ...
                        'k', ...
                        'LineStyle','none', ...
                        'LineWidth',0.8);
                end
            end
        
            xticks(ax, 1:nCh);
            xticklabels(ax, chanLabels);
            xtickangle(ax, 90);
        
            ylabel(ax, 'PSD [dB]');
            xlabel(ax, 'Channel');
            title(ax, plotTitle, 'Interpreter','none');
        
            legend(ax, ...
                {'Stroke baseline','Stroke MI','Healthy baseline','Healthy MI'}, ...
                'Location','bestoutside');
        
            ylim(ax, Utility_Functions.get_bar_ylim(vals, true));
        
            Utility_Functions.style_publication_axes(ax);
            hold(ax, 'off');
        
            Utility_Functions.save_figure_tif_fig(fig, saveBasePath, 300);
            close(fig);
        end

        function save_figure_tif_fig(fig, saveBasePath, resolution)
            % Save figure as editable .fig and high-resolution .tif.
            %
            % Uses saveas + print so figure export behavior is consistent
            % across ERP, PSD, ERDS, TFR, ASSI, and TOPO plots.
        
            if nargin < 3 || isempty(resolution)
                resolution = 500;
            end
        
            [saveDir, baseName, ~] = fileparts(char(saveBasePath));
        
            if isempty(saveDir)
                saveDir = pwd;
            end
        
            if ~exist(saveDir, 'dir')
                mkdir(saveDir);
            end
        
            baseName = regexprep(baseName, '[<>:"/\\|?*]', '_');
        
            figPath = fullfile(saveDir, [baseName '.fig']);
            tifPath = fullfile(saveDir, [baseName '.tif']);
        
            saveas(fig, figPath);
            print(fig, tifPath, '-dtiff', sprintf('-r%d', resolution));
        end
        
        function valsDB = pow_to_db(vals)
            % Convert linear power or PSD values to dB safely.
            %
            % This helper is used for PSD/TOPO/channel-power features.
            % It protects against log10(0) by replacing zero or invalid values
            % with eps before conversion.
        
            vals = double(vals);
        
            % Replace zeros, negative values, Inf, and NaN with eps before log10.
            % Power values should be nonnegative, so negative values are invalid.
            badIdx = vals <= 0 | isnan(vals) | isinf(vals);
            vals(badIdx) = eps;
        
            valsDB = 10 .* log10(vals);
        end
        
        function [meanDB, semDB] = mean_sem_from_db_stack(dbStack)
            % Compute mean and SEM from already dB-transformed subject-level values.
            %
            % dbStack should be features/channels x subjects.
        
            meanDB = mean(dbStack, 2, 'omitnan');
        
            nValid = sum(~isnan(dbStack), 2);
            semDB = std(dbStack, 0, 2, 'omitnan') ./ sqrt(nValid);
        
            semDB(nValid < 2) = NaN;
        end
        
        
        function yLimits = get_bar_ylim(vals, includeZero)
            % Build fair y-axis limits for comparable bar plots.
        
            if nargin < 2
                includeZero = true;
            end
        
            vals = vals(:);
            vals = vals(~isnan(vals));
        
            if isempty(vals)
                yLimits = [-1 1];
                return
            end
        
            yMin = min(vals);
            yMax = max(vals);
        
            if includeZero
                yMin = min(yMin, 0);
                yMax = max(yMax, 0);
            end
        
            if yMin == yMax
                pad = max(abs(yMax) * 0.1, 1);
            else
                pad = 0.12 * (yMax - yMin);
            end
        
            yLimits = [yMin - pad, yMax + pad];
        end
        
        
        function add_time_window_markers(ax, cueTimeSec, baselineWindow, miWindow)
            % Add cue/MI onset, baseline window, and MI window markers.
        
            if nargin < 1 || isempty(ax)
                ax = gca;
            end
        
            xline(ax, cueTimeSec, 'k--', ...
                'Cue / MI onset', ...
                'LineWidth',1.2, ...
                'LabelOrientation','aligned', ...
                'LabelVerticalAlignment','top', ...
                'FontSize',10, ...
                'HandleVisibility','off');
        
            xline(ax, baselineWindow(1), 'r--', ...
                'LineWidth',1.1, ...
                'HandleVisibility','off');
        
            xline(ax, baselineWindow(2), 'r--', ...
                'LineWidth',1.1, ...
                'HandleVisibility','off');
        
            xline(ax, miWindow(1), 'g--', ...
                'LineWidth',1.1, ...
                'HandleVisibility','off');
        
            xline(ax, miWindow(2), 'g--', ...
                'LineWidth',1.1, ...
                'HandleVisibility','off');
        end
        
        
        function style_publication_axes(ax)
            % Basic white-background publication styling.
        
            if nargin < 1 || isempty(ax)
                ax = gca;
            end
        
            ax.Color = 'w';
            ax.XColor = 'k';
            ax.YColor = 'k';
            ax.GridColor = [0.80 0.80 0.80];
            ax.GridAlpha = 0.35;
            ax.LineWidth = 1.0;
            ax.FontSize = 11;
        
            grid(ax, 'on');
            box(ax, 'on');
        end

        function statsOut = run_windowed_feature_stats(featureType, featureNames, healthyBaselineMat, healthyMIMat, strokeBaselineMat, strokeMIMat, alpha)
            % General statistics engine for subject-level windowed EEG features.
            %
            % Inputs:
            %   featureType         - string, e.g., 'PSD', 'ERDS', 'ASSI'
            %   featureNames        - 1 x nFeatures or nFeatures x 1 cell/string array
            %   healthyBaselineMat  - healthy subjects x features
            %   healthyMIMat        - healthy subjects x features
            %   strokeBaselineMat   - stroke subjects x features
            %   strokeMIMat         - stroke subjects x features
            %   alpha               - significance threshold
            %
            % Outputs:
            %   statsOut.healthyBaselineVsMI
            %   statsOut.strokeBaselineVsMI
            %   statsOut.healthyMIvsStrokeMI
            %   statsOut.all
        
            if nargin < 7 || isempty(alpha)
                alpha = 0.05;
            end
        
            featureNames = string(featureNames(:));
        
            healthyBaselineVsMI = Utility_Functions.run_paired_window_stats( ...
                featureType, ...
                "Healthy_Baseline_vs_MI", ...
                featureNames, ...
                healthyBaselineMat, ...
                healthyMIMat, ...
                alpha);
        
            strokeBaselineVsMI = Utility_Functions.run_paired_window_stats( ...
                featureType, ...
                "Stroke_Baseline_vs_MI", ...
                featureNames, ...
                strokeBaselineMat, ...
                strokeMIMat, ...
                alpha);
        
            healthyMIvsStrokeMI = Utility_Functions.run_independent_window_stats( ...
                featureType, ...
                "Healthy_MI_vs_Stroke_MI", ...
                featureNames, ...
                healthyMIMat, ...
                strokeMIMat, ...
                alpha);
        
            allStats = [
                healthyBaselineVsMI
                strokeBaselineVsMI
                healthyMIvsStrokeMI
            ];
        
            statsOut = struct();
            statsOut.featureType = featureType;
            statsOut.healthyBaselineVsMI = healthyBaselineVsMI;
            statsOut.strokeBaselineVsMI = strokeBaselineVsMI;
            statsOut.healthyMIvsStrokeMI = healthyMIvsStrokeMI;
            statsOut.all = allStats;
        end

        function T = run_paired_window_stats(featureType, comparisonName, featureNames, baselineMat, miMat, alpha)
    % Paired baseline-vs-MI statistics within the same subject group.
    %
    % Primary test:
    %   paired t-test
    %
    % Optional secondary nonparametric check:
    %   signrank(baseline, MI)

    featureNames = string(featureNames(:));
    nFeatures = length(featureNames);

    rows = cell(0, 19);

    for f = 1:nFeatures

        baselineVals = baselineMat(:,f);
        miVals = miMat(:,f);

        valid = ~isnan(baselineVals) & ~isnan(miVals);

        baselineVals = baselineVals(valid);
        miVals = miVals(valid);

        nPaired = length(baselineVals);

        meanBaseline = mean(baselineVals, 'omitnan');
        sdBaseline = std(baselineVals, 'omitnan');

        meanMI = mean(miVals, 'omitnan');
        sdMI = std(miVals, 'omitnan');

        diffVals = miVals - baselineVals;

        meanDiff = mean(diffVals, 'omitnan');
        sdDiff = std(diffVals, 'omitnan');

        p_ttest = NaN;
        tstat = NaN;
        df = NaN;
        p_nonparametric = NaN;
        effectSize = NaN;

        if nPaired >= 2

            [~, p_ttest, ~, testStats] = ttest(miVals, baselineVals);

            tstat = testStats.tstat;
            df = testStats.df;

            try
                p_nonparametric = signrank(baselineVals, miVals);
            catch
                p_nonparametric = NaN;
            end

            if sdDiff > 0
                effectSize = meanDiff / sdDiff;  % Cohen's dz
            end
        end

        % Primary p-value is the paired t-test.
        % The nonparametric p-value is retained only as a secondary sensitivity check.
        p_main = p_ttest;

        rows(end+1,:) = { ...
            string(featureType), ...
            string(comparisonName), ...
            featureNames(f), ...
            nPaired, ...
            NaN, ...
            NaN, ...
            meanBaseline, ...
            sdBaseline, ...
            meanMI, ...
            sdMI, ...
            meanDiff, ...
            sdDiff, ...
            p_ttest, ...
            tstat, ...
            df, ...
            p_nonparametric, ...
            effectSize, ...
            p_main, ...
            p_main < alpha};
    end

    T = cell2table(rows, ...
        'VariableNames', { ...
        'FeatureType', ...
        'Comparison', ...
        'Feature', ...
        'N_Paired', ...
        'N_Healthy', ...
        'N_Stroke', ...
        'Mean_A', ...
        'SD_A', ...
        'Mean_B', ...
        'SD_B', ...
        'Mean_B_minus_A', ...
        'SD_B_minus_A', ...
        'p_ttest', ...
        'tstat', ...
        'df', ...
        'p_nonparametric', ...
        'EffectSize', ...
        'p_main', ...
        'Significant_uncorrected'});

    T = Utility_Functions.add_fdr_by_comparison(T, alpha);
        end

        function T = run_independent_window_stats(featureType, comparisonName, featureNames, healthyMat, strokeMat, alpha)
    % Independent healthy-vs-stroke statistics.
    %
    % Primary test:
    %   Welch's two-sample t-test
    %
    % Secondary sensitivity check:
    %   Mann-Whitney U / ranksum test

    featureNames = string(featureNames(:));
    nFeatures = length(featureNames);

    rows = cell(0, 19);

    for f = 1:nFeatures

        healthyVals = healthyMat(:,f);
        strokeVals = strokeMat(:,f);

        healthyVals = healthyVals(~isnan(healthyVals));
        strokeVals = strokeVals(~isnan(strokeVals));

        nHealthy = length(healthyVals);
        nStroke = length(strokeVals);

        meanHealthy = mean(healthyVals, 'omitnan');
        sdHealthy = std(healthyVals, 'omitnan');

        meanStroke = mean(strokeVals, 'omitnan');
        sdStroke = std(strokeVals, 'omitnan');

        p_ttest = NaN;
        tstat = NaN;
        df = NaN;
        p_nonparametric = NaN;
        effectSize = NaN;

        if nHealthy >= 2 && nStroke >= 2

            [~, p_ttest, ~, testStats] = ttest2(healthyVals, strokeVals, 'Vartype', 'unequal');

            tstat = testStats.tstat;
            df = testStats.df;

            try
                p_nonparametric = ranksum(healthyVals, strokeVals);
            catch
                p_nonparametric = NaN;
            end

            pooledSD = sqrt(((nHealthy - 1) * sdHealthy^2 + ...
                             (nStroke - 1) * sdStroke^2) / ...
                             max(nHealthy + nStroke - 2, 1));

            if pooledSD > 0
                effectSize = (meanStroke - meanHealthy) / pooledSD;
            end
        end

        % Primary p-value is Welch's independent-samples t-test.
        % The nonparametric p-value is retained only as a secondary sensitivity check.
        p_main = p_ttest;

        rows(end+1,:) = { ...
            string(featureType), ...
            string(comparisonName), ...
            featureNames(f), ...
            NaN, ...
            nHealthy, ...
            nStroke, ...
            meanHealthy, ...
            sdHealthy, ...
            meanStroke, ...
            sdStroke, ...
            meanStroke - meanHealthy, ...
            NaN, ...
            p_ttest, ...
            tstat, ...
            df, ...
            p_nonparametric, ...
            effectSize, ...
            p_main, ...
            p_main < alpha};
    end

    T = cell2table(rows, ...
        'VariableNames', { ...
        'FeatureType', ...
        'Comparison', ...
        'Feature', ...
        'N_Paired', ...
        'N_Healthy', ...
        'N_Stroke', ...
        'Mean_A', ...
        'SD_A', ...
        'Mean_B', ...
        'SD_B', ...
        'Mean_B_minus_A', ...
        'SD_B_minus_A', ...
        'p_ttest', ...
        'tstat', ...
        'df', ...
        'p_nonparametric', ...
        'EffectSize', ...
        'p_main', ...
        'Significant_uncorrected'});

    T = Utility_Functions.add_fdr_by_comparison(T, alpha);
        end

        function T = add_fdr_by_comparison(T, alpha)
    % Apply Benjamini-Hochberg FDR correction separately within each comparison.

    if nargin < 2 || isempty(alpha)
        alpha = 0.05;
    end

    T.p_FDR = nan(height(T),1);
    T.Significant_FDR = false(height(T),1);

    comparisons = unique(T.Comparison);

    for c = 1:length(comparisons)

        thisComparison = comparisons(c);
        idx = T.Comparison == thisComparison;

        p = T.p_main(idx);

        if all(isnan(p))
            continue
        end

        q = Utility_Functions.bh_fdr_local(p);

        T.p_FDR(idx) = q;
        T.Significant_FDR(idx) = q < alpha;
    end
        end

        function q = bh_fdr_local(p)

    p = p(:);
    q = nan(size(p));

    validIdx = find(~isnan(p));
    pValid = p(validIdx);

    if isempty(pValid)
        return
    end

    [pSorted, sortIdx] = sort(pValid);
    m = length(pSorted);

    qSorted = pSorted .* m ./ (1:m)';

    for i = m-1:-1:1
        qSorted(i) = min(qSorted(i), qSorted(i+1));
    end

    qValid = nan(m,1);
    qValid(sortIdx) = qSorted;

    q(validIdx) = qValid;
        end

        function [featureNames, healthyBaselineMat, healthyMIMat, strokeBaselineMat, strokeMIMat] = ...
    collect_windowed_feature_matrices(featureType, healthySubjects, strokeSubjects)

    featureType = upper(string(featureType));

    switch featureType

        case "PSD"
            [featureNames, healthyBaselineMat, healthyMIMat] = ...
                Utility_Functions.collect_psd_window_matrix(healthySubjects);

            [featureNamesStroke, strokeBaselineMat, strokeMIMat] = ...
                Utility_Functions.collect_psd_window_matrix(strokeSubjects);

        case "ERDS"
            [featureNames, healthyBaselineMat, healthyMIMat] = ...
                Utility_Functions.collect_erders_window_matrix(healthySubjects);

            [featureNamesStroke, strokeBaselineMat, strokeMIMat] = ...
                Utility_Functions.collect_erders_window_matrix(strokeSubjects);

        case "ASSI"
            [featureNames, healthyBaselineMat, healthyMIMat] = ...
                Utility_Functions.collect_assi_window_matrix(healthySubjects);

            [featureNamesStroke, strokeBaselineMat, strokeMIMat] = ...
                Utility_Functions.collect_assi_window_matrix(strokeSubjects);

        case "TOPO"
            [featureNames, healthyBaselineMat, healthyMIMat] = ...
                Utility_Functions.collect_topo_window_matrix(healthySubjects);

            [featureNamesStroke, strokeBaselineMat, strokeMIMat] = ...
                Utility_Functions.collect_topo_window_matrix(strokeSubjects);

        otherwise
            error('Unsupported feature type for windowed stats: %s', featureType);
    end

    if length(featureNames) ~= length(featureNamesStroke) || ...
            any(featureNames(:) ~= featureNamesStroke(:))
        error('Feature names do not match between healthy and stroke for %s.', featureType);
    end
        end

        function [featureNames, baselineMat, miMat] = collect_psd_window_matrix(subjects)
        
            if isempty(subjects)
                error('No subjects provided to collect_psd_window_matrix.');
            end
        
            bandNames = subjects(1).PSD.bandNames;
            chanLabels = subjects(1).PSD.chanLabels;
        
            nSubj = length(subjects);
            nFeatures = length(bandNames) * length(chanLabels);
        
            baselineMat = nan(nSubj, nFeatures);
            miMat = nan(nSubj, nFeatures);
            featureNames = strings(nFeatures,1);
        
            col = 0;
        
            for b = 1:length(bandNames)
        
                band = bandNames{b};
        
                for ch = 1:length(chanLabels)
        
                    col = col + 1;
        
                    featureNames(col) = "PSD_" + string(band) + "_" + string(chanLabels{ch});
        
                    for s = 1:nSubj
        
                        P = subjects(s).PSD.(band);
        
                        baselineMat(s,col) = mean([ ...
                            P.left_baseline_db(ch), ...
                            P.right_baseline_db(ch)], 'omitnan');
                        
                        miMat(s,col) = mean([ ...
                            P.left_mi_db(ch), ...
                            P.right_mi_db(ch)], 'omitnan');
                    end
                end
            end
        end

        function [featureNames, baselineMat, miMat] = collect_erders_window_matrix(subjects)
        
            if isempty(subjects)
                error('No subjects provided to collect_erders_window_matrix.');
            end
        
            bandNames = {'mu','beta'};
            condNames = {'left','right'};
        
            chanLabels = subjects(1).ERDS.chanLabels;
        
            nSubj = length(subjects);
            nFeatures = length(bandNames) * length(condNames) * length(chanLabels);
        
            baselineMat = nan(nSubj, nFeatures);
            miMat = nan(nSubj, nFeatures);
            featureNames = strings(nFeatures,1);
        
            col = 0;
        
            for b = 1:length(bandNames)
        
                band = bandNames{b};
        
                for c = 1:length(condNames)
        
                    cond = condNames{c};
        
                    for ch = 1:length(chanLabels)
        
                        col = col + 1;
        
                        featureNames(col) = "ERDS_" + string(band) + "_" + string(cond) + "_" + string(chanLabels{ch});
        
                        for s = 1:nSubj
        
                            t = subjects(s).ERDS.t;
                            baselineWindow = subjects(s).ERDS.baselineWindow;
                            miWindow = subjects(s).ERDS.miWindow;
        
                            baselineIdx = t >= baselineWindow(1) & t < baselineWindow(2);
                            miIdx = t >= miWindow(1) & t < miWindow(2);
        
                            X = subjects(s).ERDS.(band).(cond);
        
                            baselineMat(s,col) = mean(X(ch,baselineIdx), 'omitnan');
                            miMat(s,col) = mean(X(ch,miIdx), 'omitnan');
                        end
                    end
                end
            end
        end

        function [featureNames, baselineMat, miMat] = collect_assi_window_matrix(subjects)
            % Collect ASSI baseline and MI values for statistics.
            %
            % Left-hand MI and right-hand MI are kept as separate features.
            % This prevents left/right lateralization effects from canceling out.
            %
            % Output:
            %   baselineMat: subjects x features
            %   miMat:       subjects x features
            %
            % For each band and electrode pair, this creates:
            %   ASSI_<band>_leftMI_<pair>
            %   ASSI_<band>_rightMI_<pair>
        
            if isempty(subjects)
                error('No subjects provided to collect_assi_window_matrix.');
            end
        
            if ~isfield(subjects(1), 'ASSI')
                error('Subjects do not contain ASSI field.');
            end
        
            bandNames = subjects(1).ASSI.bandNames;
            pairInfo = subjects(1).ASSI.pairInfo;
        
            nSubj = length(subjects);
        
            % Two features per band/pair:
            %   1) left-hand MI asymmetry feature
            %   2) right-hand MI asymmetry feature
            nFeatures = length(bandNames) * length(pairInfo) * 2;
        
            baselineMat = nan(nSubj, nFeatures);
            miMat = nan(nSubj, nFeatures);
            featureNames = strings(nFeatures, 1);
        
            col = 0;
        
            for b = 1:length(bandNames)
        
                band = bandNames{b};
        
                for p = 1:length(pairInfo)
        
                    pairName = pairInfo(p).pairName;
        
                    % Make a clean display/statistics name.
                    % Example: C3_C4 becomes C3-C4.
                    pairNameClean = strrep(string(pairName), "_", "-");
        
                    %% ------------------------------------------------------------
                    % Left-hand MI-specific ASSI feature
                    % -------------------------------------------------------------
                    col = col + 1;
        
                    featureNames(col) = ...
                        "ASSI_" + string(band) + "_leftMI_" + pairNameClean;
        
                    for s = 1:nSubj
        
                        if ~isfield(subjects(s).ASSI, band)
                            error('Subject %d is missing ASSI band field: %s', s, band);
                        end
        
                        if ~isfield(subjects(s).ASSI.(band), pairName)
                            error('Subject %d is missing ASSI pair field: %s.%s', s, band, pairName);
                        end
        
                        A = subjects(s).ASSI.(band).(pairName);
        
                        requiredFields = {'left_baseline_ai','left_ai_mi'};
        
                        for rf = 1:length(requiredFields)
                            if ~isfield(A, requiredFields{rf})
                                error('Subject %d ASSI.%s.%s is missing field: %s', ...
                                    s, band, pairName, requiredFields{rf});
                            end
                        end
        
                        baselineMat(s,col) = A.left_baseline_ai;
                        miMat(s,col) = A.left_ai_mi;
                    end
        
                    %% ------------------------------------------------------------
                    % Right-hand MI-specific ASSI feature
                    % -------------------------------------------------------------
                    col = col + 1;
        
                    featureNames(col) = ...
                        "ASSI_" + string(band) + "_rightMI_" + pairNameClean;
        
                    for s = 1:nSubj
        
                        if ~isfield(subjects(s).ASSI, band)
                            error('Subject %d is missing ASSI band field: %s', s, band);
                        end
        
                        if ~isfield(subjects(s).ASSI.(band), pairName)
                            error('Subject %d is missing ASSI pair field: %s.%s', s, band, pairName);
                        end
        
                        A = subjects(s).ASSI.(band).(pairName);
        
                        requiredFields = {'right_baseline_ai','right_ai_mi'};
        
                        for rf = 1:length(requiredFields)
                            if ~isfield(A, requiredFields{rf})
                                error('Subject %d ASSI.%s.%s is missing field: %s', ...
                                    s, band, pairName, requiredFields{rf});
                            end
                        end
        
                        baselineMat(s,col) = A.right_baseline_ai;
                        miMat(s,col) = A.right_ai_mi;
                    end
                end
            end
        
            % Safety check
            if col ~= nFeatures
                error('ASSI feature collection mismatch: expected %d features, created %d.', ...
                    nFeatures, col);
            end
        end

        function [featureNames, baselineMat, miMat] = collect_topo_window_matrix(subjects)
            % Collect TOPO/channel-power baseline and MI features for statistics.
            %
            % Left-hand MI and right-hand MI are kept as separate features.
            % This prevents hand-specific topographic effects from being hidden by
            % averaging left and right MI together.
        
            if isempty(subjects)
                error('No subjects provided to collect_topo_window_matrix.');
            end
        
            if ~isfield(subjects(1), 'TOPO')
                error('Subjects do not contain TOPO field.');
            end
        
            bandNames = subjects(1).TOPO.bandNames;
            chanLabels = subjects(1).TOPO.chanLabels;
        
            if isstring(chanLabels)
                chanLabels = cellstr(chanLabels);
            end
        
            nSubj = length(subjects);
        
            % Two features per band/channel:
            %   1) left-hand MI TOPO feature
            %   2) right-hand MI TOPO feature
            nFeatures = length(bandNames) * length(chanLabels) * 2;
        
            baselineMat = nan(nSubj, nFeatures);
            miMat = nan(nSubj, nFeatures);
            featureNames = strings(nFeatures, 1);
        
            col = 0;
        
            for b = 1:length(bandNames)
        
                band = bandNames{b};
        
                for ch = 1:length(chanLabels)
        
                    chLabel = string(chanLabels{ch});
        
                    %% ------------------------------------------------------------
                    % Left-hand MI-specific TOPO feature
                    % -------------------------------------------------------------
                    col = col + 1;
        
                    featureNames(col) = ...
                        "TOPO_" + string(band) + "_leftMI_" + chLabel;
        
                    for s = 1:nSubj
        
                        if ~isfield(subjects(s).TOPO, band)
                            error('Subject %d is missing TOPO band field: %s', s, band);
                        end
        
                        T = subjects(s).TOPO.(band);
        
                        requiredFields = { ...
                            'left_baseline_power_db', ...
                            'left_mi_power_db'};
        
                        for rf = 1:length(requiredFields)
                            if ~isfield(T, requiredFields{rf})
                                error(['Subject %d TOPO.%s is missing field: %s. ' ...
                                       'Rerun Step02 subject-level TOPO extraction.'], ...
                                       s, band, requiredFields{rf});
                            end
                        end
        
                        baselineMat(s,col) = T.left_baseline_power_db(ch);
                        miMat(s,col) = T.left_mi_power_db(ch);
                    end
        
                    %% ------------------------------------------------------------
                    % Right-hand MI-specific TOPO feature
                    % -------------------------------------------------------------
                    col = col + 1;
        
                    featureNames(col) = ...
                        "TOPO_" + string(band) + "_rightMI_" + chLabel;
        
                    for s = 1:nSubj
        
                        if ~isfield(subjects(s).TOPO, band)
                            error('Subject %d is missing TOPO band field: %s', s, band);
                        end
        
                        T = subjects(s).TOPO.(band);
        
                        requiredFields = { ...
                            'right_baseline_power_db', ...
                            'right_mi_power_db'};
        
                        for rf = 1:length(requiredFields)
                            if ~isfield(T, requiredFields{rf})
                                error(['Subject %d TOPO.%s is missing field: %s. ' ...
                                       'Rerun Step02 subject-level TOPO extraction.'], ...
                                       s, band, requiredFields{rf});
                            end
                        end
        
                        baselineMat(s,col) = T.right_baseline_power_db(ch);
                        miMat(s,col) = T.right_mi_power_db(ch);
                    end
                end
            end
        
            % Safety check
            if col ~= nFeatures
                error('TOPO feature collection mismatch: expected %d features, created %d.', ...
                    nFeatures, col);
            end
        end

        function plot_significant_stats_by_comparison(T, saveDir, featureType, topN)
            % Plot top statistically significant features for each comparison.
            %
            % Uses only rows where Significant_FDR == true.
            %
            % For each comparison, this creates a grouped bar plot:
            %   Mean_A vs Mean_B
            %
            % The plotted features are ranked by absolute effect size if available,
            % otherwise by absolute Mean_B_minus_A.
            %
            % Expected table fields:
            %   FeatureType
            %   Comparison
            %   Feature
            %   Mean_A
            %   Mean_B
            %   Mean_B_minus_A
            %   p_FDR
            %   Significant_FDR
        
            if nargin < 4 || isempty(topN)
                topN = 20;
            end
        
            if ~exist(saveDir, 'dir')
                mkdir(saveDir);
            end
        
            requiredVars = { ...
                'Comparison', ...
                'Feature', ...
                'Mean_A', ...
                'Mean_B', ...
                'Mean_B_minus_A', ...
                'p_FDR', ...
                'Significant_FDR'};
        
            for v = 1:length(requiredVars)
                if ~ismember(requiredVars{v}, T.Properties.VariableNames)
                    error('Statistics table is missing required variable: %s', requiredVars{v});
                end
            end
        
            if isempty(T)
                warning('Statistics table is empty for %s. No significant plots created.', featureType);
                return
            end
        
            sigT = T(T.Significant_FDR == true, :);
        
            if isempty(sigT)
                warning('No FDR-significant %s features found. No significant feature plots created.', featureType);
                return
            end
        
            comparisons = unique(sigT.Comparison, 'stable');
        
            for c = 1:length(comparisons)
        
                compName = comparisons(c);
                compRows = sigT(sigT.Comparison == compName, :);
        
                if isempty(compRows)
                    continue
                end
        
                % Rank features by effect magnitude.
                if ismember('EffectSize', compRows.Properties.VariableNames)
                    rankMetric = abs(compRows.EffectSize);
                else
                    rankMetric = abs(compRows.Mean_B_minus_A);
                end
        
                rankMetric(~isfinite(rankMetric)) = -Inf;
        
                [~, sortIdx] = sort(rankMetric, 'descend');
                compRows = compRows(sortIdx, :);
        
                nPlot = min(topN, height(compRows));
                compRows = compRows(1:nPlot, :);
        
                vals = [compRows.Mean_A, compRows.Mean_B];
        
                featureLabels = string(compRows.Feature);
                featureLabels = Utility_Functions.shorten_feature_labels(featureLabels, 45);
        
                fig = figure('Visible','off', ...
                    'Color','w', ...
                    'Position',[100 100 1600 850]);
        
                ax = axes(fig);
                hold(ax, 'on');
        
                bar(ax, vals, 'grouped');
        
                yline(ax, 0, 'k:', ...
                    'LineWidth',1.0, ...
                    'HandleVisibility','off');
        
                xticks(ax, 1:nPlot);
                xticklabels(ax, featureLabels);
                ax.TickLabelInterpreter = 'none';
                xtickangle(ax, 45);
        
                ylabel(ax, 'Feature value');
                xlabel(ax, 'FDR-significant feature');
        
                legend(ax, {'Condition A','Condition B'}, ...
                    'Location','best', ...
                    'Interpreter','none');
        
                title(ax, sprintf('%s | %s | Top %d FDR-significant features', ...
                    featureType, string(compName), nPlot), ...
                    'Interpreter','none', ...
                    'FontWeight','bold');
        
                % Add FDR p-values above bars as text labels
                yLimits = Utility_Functions.get_bar_ylim(vals, true);
                ylim(ax, yLimits);
        
                yRange = yLimits(2) - yLimits(1);
                textY = yLimits(2) - 0.05*yRange;
        
                for i = 1:nPlot
                    txt = sprintf('q=%.3g', compRows.p_FDR(i));
                    text(ax, i, textY, txt, ...
                        'Rotation',90, ...
                        'HorizontalAlignment','right', ...
                        'VerticalAlignment','middle', ...
                        'FontSize',8, ...
                        'Interpreter','none');
                end
        
                Utility_Functions.style_publication_axes(ax);
        
                hold(ax, 'off');
        
                compSafe = matlab.lang.makeValidName(char(compName));
                saveBase = fullfile(saveDir, ...
                    [char(featureType) '_' compSafe '_top_significant_features']);
        
                Utility_Functions.save_figure_tif_fig(fig, saveBase, 500);
                close(fig);
            end
        end

        function plot_significant_feature_count_summary(T, saveBasePath, plotTitle)
            % Plot the number of FDR-significant features by feature type/comparison.
            %
            % This gives a compact overview of where significant effects are occurring.
        
            requiredVars = {'FeatureType','Comparison','Significant_FDR'};
        
            for v = 1:length(requiredVars)
                if ~ismember(requiredVars{v}, T.Properties.VariableNames)
                    error('Statistics table is missing required variable: %s', requiredVars{v});
                end
            end
        
            if isempty(T)
                warning('Empty statistics table. No significant count plot created.');
                return
            end
        
            featureTypes = unique(T.FeatureType, 'stable');
            comparisons = unique(T.Comparison, 'stable');
        
            countMat = zeros(length(featureTypes), length(comparisons));
        
            for f = 1:length(featureTypes)
                for c = 1:length(comparisons)
        
                    idx = T.FeatureType == featureTypes(f) & ...
                          T.Comparison == comparisons(c);
        
                    countMat(f,c) = sum(T.Significant_FDR(idx) == true);
                end
            end
        
            fig = figure('Visible','off', ...
                'Color','w', ...
                'Position',[100 100 1200 700]);
        
            ax = axes(fig);
            hold(ax, 'on');
        
            bar(ax, countMat, 'grouped');
        
            xticks(ax, 1:length(featureTypes));
            xticklabels(ax, string(featureTypes));
            ax.TickLabelInterpreter = 'none';
        
            ylabel(ax, 'Number of FDR-significant features');
            xlabel(ax, 'Feature type');
        
            legend(ax, string(comparisons), ...
                'Location','bestoutside', ...
                'Interpreter','none');
        
            title(ax, plotTitle, ...
                'Interpreter','none', ...
                'FontWeight','bold');
        
            Utility_Functions.style_publication_axes(ax);
        
            hold(ax, 'off');
        
            Utility_Functions.save_figure_tif_fig(fig, saveBasePath, 500);
            close(fig);
        end

        function shortLabels = shorten_feature_labels(labels, maxLen)
            % Shorten long feature labels for readable x-axis tick labels.
        
            if nargin < 2 || isempty(maxLen)
                maxLen = 45;
            end
        
            labels = string(labels);
            shortLabels = labels;
        
            for i = 1:length(labels)
        
                thisLabel = labels(i);
        
                if strlength(thisLabel) > maxLen
                    shortLabels(i) = extractBefore(thisLabel, maxLen) + "...";
                end
            end
        end


    end % methods
end % classdef