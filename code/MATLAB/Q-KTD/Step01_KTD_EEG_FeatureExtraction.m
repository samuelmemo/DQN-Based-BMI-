clear; clc; close all;

% This code prepares segmented EEG BMI input files for Q-KTD model training.
% The received EEG data have already been segmented from 0 to 0.5 seconds
% after motor imagery onset and concatenated into trial-by-feature matrices.
%
% Three Q-KTD input cases are generated:
%   Case 0: All healthy subjects concatenated together
%   Case 1: Individual stroke subject files
%   Case 2: All stroke subjects except one stroke subject
%   Case 3: All healthy subjects + all stroke subjects except one stroke subject


%% ===================== USER SETTINGS =====================
inputRoot = 'D:\Bhoj Raj Thapa\EEGReaching\EEGKTD\results\BMI_inputs';      % Main folder containing healthy and stroke BMI input folders
baseResultsRoot = "D:\Bhoj Raj Thapa\EEGReaching\EEGKTD\results";   % Results folder

outputRoot = fullfile(baseResultsRoot, 'QKTD');                                      % Q-KTD-specific output folder

case0Folder = fullfile(outputRoot, 'case0');                                         % Case 0 output: all healthy subjects
case1Folder = fullfile(outputRoot, 'case1');                                         % Case 1 output: individual stroke subjects
case2Folder = fullfile(outputRoot, 'case2');                                         % Case 2 output: all stroke minus one subject
case3Folder = fullfile(outputRoot, 'case3');                                         % Case 3 output: all healthy + all stroke minus one subject

windowStart = 0;                                                                    % Start time of EEG window in seconds
windowEnd = 1.0;                                                                      % End time of EEG window in seconds

windowSize = windowEnd - windowStart;                                               % Duration of selected EEG window in seconds
windowTag = makeWindowTag(windowSize);                                              % Window tag used in BMI input filename, e.g. 0p5s or 1p0s

windowStartText = formatWindowTime(windowStart);                                    % Start time text for output folder name
windowEndText = formatWindowTime(windowEnd);                                        % End time text for output folder name

dataSubfolderName = ['Data_from', windowStartText, '_to', windowEndText];           % Output subfolder, e.g. Data_from0_to1
outputFileName = 'Data_EEGfeatureRAW.mat';                                          % Q-KTD expected data filename

healthyFolder = fullfile(inputRoot, 'healthy');                                     % Folder containing SXXX_BMI_Input.mat files
strokeFolder = fullfile(inputRoot, 'stroke');                                       % Folder containing sub-XX_BMI_Input.mat files

healthyFilePattern = ['S*_BMI_input_', windowTag, '.mat'];                          % Window-specific healthy BMI input pattern
strokeFilePattern = ['sub-*_BMI_input_', windowTag, '.mat'];                        % Window-specific stroke BMI input pattern                                          % File pattern for stroke subject BMI inputs

%% ===================== CHECK INPUT / OUTPUT FOLDERS =====================
if ~exist(healthyFolder, 'dir')
    error('Healthy input folder does not exist: %s', healthyFolder);
end

if ~exist(strokeFolder, 'dir')
    error('Stroke input folder does not exist: %s', strokeFolder);
end

if ~exist(outputRoot, 'dir')
    mkdir(outputRoot);
end

%% ===================== FIND BMI INPUT FILES =====================
healthyFiles = dir(fullfile(healthyFolder, healthyFilePattern));                    % List of healthy BMI input files
strokeFiles = dir(fullfile(strokeFolder, strokeFilePattern));                       % List of stroke BMI input files

if isempty(healthyFiles)
    error('No healthy BMI input files found in: %s', healthyFolder);
end

if isempty(strokeFiles)
    error('No stroke BMI input files found in: %s', strokeFolder);
end

healthyFiles = sortFileStructByName(healthyFiles);                                 % Sort healthy files by filename
strokeFiles = sortFileStructByName(strokeFiles);                                   % Sort stroke files by filename

numHealthySubjects = length(healthyFiles);                                         % Number of healthy subject files found
numStrokeSubjects = length(strokeFiles);                                           % Number of stroke subject files found

fprintf('Found %d healthy subject files.\n', numHealthySubjects);
fprintf('Found %d stroke subject files.\n', numStrokeSubjects);

%% ===================== LOAD HEALTHY DATA INTO MEMORY =====================
fprintf('\n===================== LOAD HEALTHY DATA =====================\n');

healthySubjectNames = strings(numHealthySubjects,1);                               % Stores healthy subject names
healthyRAW = cell(numHealthySubjects,1);                                           % Stores healthy RAWfeature matrices
healthyClassID = cell(numHealthySubjects,1);                                       % Stores healthy classID vectors

for i = 1:numHealthySubjects

    inputFile = fullfile(healthyFiles(i).folder, healthyFiles(i).name);             % Current healthy subject file
    subjectName = erase(healthyFiles(i).name, ['_BMI_input_', windowTag, '.mat']);       % Subject name from window-specific filename
    [RAWfeature, classID] = loadBMIInputAsQKTD(inputFile, windowSize);                  % Convert InputMatrix/labels to RAWfeature/classID
    healthySubjectNames(i) = string(subjectName);                                  % Store healthy subject name
    healthyRAW{i} = RAWfeature;                                                    % Store healthy subject features
    healthyClassID{i} = classID;                                                   % Store healthy subject class labels

    fprintf('Loaded %s: RAWfeature = %d x %d, classID = %d x %d\n', ...
        subjectName, size(RAWfeature,1), size(RAWfeature,2), ...
        size(classID,1), size(classID,2));
end

%% ===================== LOAD STROKE DATA INTO MEMORY =====================
fprintf('\n===================== LOAD STROKE DATA =====================\n');

strokeSubjectNames = strings(numStrokeSubjects,1);                                 % Stores stroke subject names
strokeRAW = cell(numStrokeSubjects,1);                                             % Stores stroke RAWfeature matrices
strokeClassID = cell(numStrokeSubjects,1);                                         % Stores stroke classID vectors

for i = 1:numStrokeSubjects

    inputFile = fullfile(strokeFiles(i).folder, strokeFiles(i).name);               % Current stroke subject file
    subjectName = erase(strokeFiles(i).name, ['_BMI_input_', windowTag, '.mat']);        % Subject name from window-specific filename
    [RAWfeature, classID] = loadBMIInputAsQKTD(inputFile, windowSize);                  % Convert InputMatrix/labels to RAWfeature/classID
    strokeSubjectNames(i) = string(subjectName);                                   % Store stroke subject name
    strokeRAW{i} = RAWfeature;                                                     % Store stroke subject features
    strokeClassID{i} = classID;                                                    % Store stroke subject class labels

    fprintf('Loaded %s: RAWfeature = %d x %d, classID = %d x %d\n', ...
        subjectName, size(RAWfeature,1), size(RAWfeature,2), ...
        size(classID,1), size(classID,2));
end

%% ==========================================================
%  CASE 0: ALL HEALTHY SUBJECTS CONCATENATED
%
%  Output example:
%     results\QKTD\case0\healthy_all\Data_from0_to0.5\Data_EEGfeatureRAW.mat
%
%% ==========================================================
fprintf('\n===================== CASE 0: HEALTHY ALL =====================\n');

RAWfeature_healthy_all = [];                                                       % Concatenated healthy EEG features
classID_healthy_all = [];                                                          % Concatenated healthy class labels

for i = 1:numHealthySubjects

    RAWfeature_healthy_all = [RAWfeature_healthy_all; healthyRAW{i}];              % Append current healthy subject features
    classID_healthy_all = [classID_healthy_all; healthyClassID{i}];                % Append current healthy subject labels

end

RAWfeature = RAWfeature_healthy_all;                                               % Q-KTD variable name
classID = classID_healthy_all;                                                     % Q-KTD variable name

healthyAllOutputFolder = fullfile(case0Folder, 'healthy_all', dataSubfolderName);   % Output folder for healthy_all

if ~exist(healthyAllOutputFolder, 'dir')
    mkdir(healthyAllOutputFolder);
end

save(fullfile(healthyAllOutputFolder, outputFileName), ...
    'RAWfeature', 'classID', '-v7.3');

fprintf('Saved case0 healthy_all: RAWfeature = %d x %d, classID = %d x %d\n', ...
    size(RAWfeature,1), size(RAWfeature,2), size(classID,1), size(classID,2));

%% ==========================================================
%  CASE 1: INDIVIDUAL STROKE SUBJECT FILES
%
%  Output example:
%     results\QKTD\case1\sub-01\Data_from0_to0.5\Data_EEGfeatureRAW.mat
%     results\QKTD\case1\sub-02\Data_from0_to0.5\Data_EEGfeatureRAW.mat
%
%% ==========================================================
fprintf('\n===================== CASE 1: INDIVIDUAL STROKE SUBJECTS =====================\n');

for i = 1:numStrokeSubjects

    subjectName = strokeSubjectNames(i);                                           % Current stroke subject name

    RAWfeature = strokeRAW{i};                                                     % Current stroke subject EEG features
    classID = strokeClassID{i};                                                    % Current stroke subject class labels

    subjectOutputFolder = fullfile(case1Folder, subjectName, dataSubfolderName);    % Output folder for current subject

    if ~exist(subjectOutputFolder, 'dir')
        mkdir(subjectOutputFolder);
    end

    save(fullfile(subjectOutputFolder, outputFileName), ...
        'RAWfeature', 'classID', '-v7.3');

    fprintf('Saved case1 %s: RAWfeature = %d x %d, classID = %d x %d\n', ...
        subjectName, size(RAWfeature,1), size(RAWfeature,2), ...
        size(classID,1), size(classID,2));
end

%% ==========================================================
%  CASE 2: ALL STROKE MINUS ONE STROKE SUBJECT
%
%  Output example:
%     results\QKTD\case2\minus-01\Data_from0_to0.5\Data_EEGfeatureRAW.mat
%     results\QKTD\case2\minus-02\Data_from0_to0.5\Data_EEGfeatureRAW.mat
%
%% ==========================================================
fprintf('\n===================== CASE 2: STROKE MINUS ONE =====================\n');

for leaveOutIndex = 1:numStrokeSubjects

    leaveOutSubject = strokeSubjectNames(leaveOutIndex);                           % Stroke subject excluded from this dataset

    RAWfeature_case2 = [];                                                         % Case 2 EEG features for this leave-one-out dataset
    classID_case2 = [];                                                            % Case 2 class labels for this leave-one-out dataset

    for strokeIndex = 1:numStrokeSubjects

        if strokeIndex ~= leaveOutIndex

            RAWfeature_case2 = [RAWfeature_case2; strokeRAW{strokeIndex}];         % Append stroke subject if not excluded
            classID_case2 = [classID_case2; strokeClassID{strokeIndex}];           % Append stroke labels if not excluded

        end
    end

    RAWfeature = RAWfeature_case2;                                                 % Q-KTD variable name
    classID = classID_case2;                                                       % Q-KTD variable name

    leaveOutFolderID = strrep(char(leaveOutSubject), 'sub-', '');                   
    minusFolderName = ['minus-', leaveOutFolderID];                                 % Output folder name for case2
    minusOutputFolder = fullfile(case2Folder, minusFolderName, dataSubfolderName);  % Output folder for current case2 dataset

    if ~exist(minusOutputFolder, 'dir')
        mkdir(minusOutputFolder);
    end

    save(fullfile(minusOutputFolder, outputFileName), ...
        'RAWfeature', 'classID', '-v7.3');

    fprintf('Saved case2 %s: RAWfeature = %d x %d, classID = %d x %d\n', ...
        minusFolderName, size(RAWfeature,1), size(RAWfeature,2), ...
        size(classID,1), size(classID,2));

end

%% ==========================================================
%  CASE 3: ALL HEALTHY + ALL STROKE MINUS ONE STROKE SUBJECT
%
%  Output example:
%     results\QKTD\case3\allminus-01\Data_from0_to0.5\Data_EEGfeatureRAW.mat
%     results\QKTD\case3\allminus-02\Data_from0_to0.5\Data_EEGfeatureRAW.mat
%% ==========================================================
fprintf('\n===================== CASE 3: HEALTHY + STROKE MINUS ONE =====================\n');

for leaveOutIndex = 1:numStrokeSubjects

    leaveOutSubject = strokeSubjectNames(leaveOutIndex);                               % Stroke subject excluded from this dataset

    RAWfeature_case3 = RAWfeature_healthy_all;                                         % Start with all healthy features
    classID_case3 = classID_healthy_all;                                               % Start with all healthy labels

    for strokeIndex = 1:numStrokeSubjects

        if strokeIndex ~= leaveOutIndex

            RAWfeature_case3 = [RAWfeature_case3; strokeRAW{strokeIndex}];            % Append stroke subject if not excluded
            classID_case3 = [classID_case3; strokeClassID{strokeIndex}];              % Append stroke labels if not excluded

        end
    end

    RAWfeature = RAWfeature_case3;                                                     % Q-KTD variable name
    classID = classID_case3;                                                           % Q-KTD variable name

    leaveOutFolderID = strrep(char(leaveOutSubject), 'sub-', '');                     
    minusFolderName = ['allminus-', leaveOutFolderID];                                 % Output folder name for case3
    minusOutputFolder = fullfile(case3Folder, minusFolderName, dataSubfolderName);     % Output folder for current case3 dataset

    if ~exist(minusOutputFolder, 'dir')
        mkdir(minusOutputFolder);
    end

    save(fullfile(minusOutputFolder, outputFileName), ...
        'RAWfeature', 'classID', '-v7.3');

    fprintf('Saved case3 %s: RAWfeature = %d x %d, classID = %d x %d\n', ...
        minusFolderName, size(RAWfeature,1), size(RAWfeature,2), ...
        size(classID,1), size(classID,2));

end

disp('Finished creating Q-KTD input files.');

%% ===================== LOCAL FUNCTIONS =====================

function sortedFiles = sortFileStructByName(fileStruct)
% Sorts a MATLAB dir() file structure alphabetically by filename.

    fileNames = string({fileStruct.name});                                         % Extract filenames from dir structure
    [~, sortIndex] = sort(fileNames);                                              % Sort filenames alphabetically
    sortedFiles = fileStruct(sortIndex);                                           % Reorder dir structure

end

function [RAWfeature, classID] = loadBMIInputAsQKTD(inputFile, expectedWindowSize)
% Loads a BMI input file and converts it into Q-KTD variable names.
%
% Required input variables:
%   InputMatrix
%   labels
%
% Output variables:
%   RAWfeature
%   classID

    if ~exist(inputFile, 'file')
        error('BMI input file does not exist: %s', inputFile);
    end

    data = load(inputFile);                                                        % Load BMI input file
        if isfield(data, 'bmiWindowSize')
        loadedWindowSize = double(data.bmiWindowSize);                             % Window size stored in loaded BMI input file

            if abs(loadedWindowSize - expectedWindowSize) > 1e-9
            error('Window size mismatch in %s: expected %.3f s but file contains %.3f s.', ...
                inputFile, expectedWindowSize, loadedWindowSize);
            end
        else
        warning('Variable bmiWindowSize not found in %s. Filename pattern was used, but metadata could not be checked.', inputFile);
        end


    if ~isfield(data, 'InputMatrix')
        error('Missing variable InputMatrix in file: %s', inputFile);
    end

    if ~isfield(data, 'labels')
        error('Missing variable labels in file: %s', inputFile);
    end

    RAWfeature = double(data.InputMatrix);                                         % Q-KTD feature matrix
    classID = double(data.labels);                                                 % Q-KTD class labels

    if size(RAWfeature,1) ~= length(classID)
        error('Trial mismatch in %s: RAWfeature has %d rows but classID has %d labels.', ...
            inputFile, size(RAWfeature,1), length(classID));
    end

    classID = classID(:);                                                          % Force classID to be column vector

end

function timeText = formatWindowTime(timeValue)
% Converts a numeric window time into clean folder-name text.

    timeText = sprintf('%g', timeValue);                                            % Convert number to compact text

end

function windowTag = makeWindowTag(windowSize)
% Converts numeric window size into BMI input filename tag.

    windowTag = sprintf('%.1fs', windowSize);                                       % Create one-decimal text, e.g. 0.5s or 1.0s
    windowTag = strrep(windowTag, '.', 'p');                                        % Replace decimal point with p

end
