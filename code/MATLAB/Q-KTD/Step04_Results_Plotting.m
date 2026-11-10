clear; clc; close all;

%% USER SETTINGS
scriptDir = fileparts(mfilename('fullpath'));      % .../BRT/AMK
projectRoot = fileparts(fileparts(scriptDir));      % .../code
resultsRoot = fullfile(projectRoot, 'results');
rootPath = fullfile(resultsRoot, 'stroke_preprocessing');
savePath = fullfile(resultsRoot, 'amk', 'TL_Comparison_FirstEpoch');

withoutTL_prefix = 'ResultsWparms_TLsource_';
withTL_prefix = 'ResultsWparms_TLtrainingRAW_sourcehealthyfrom0to0.5_';

numSubjects = 50;
numPlotSubjects = 10;

if ~exist(savePath, 'dir')
    mkdir(savePath);
end

%% PREALLOCATE
subjectNames = strings(numSubjects,1);

mean_withoutTL = nan(numSubjects,1);
std_withoutTL = nan(numSubjects,1);

mean_withTL = nan(numSubjects,1);
std_withTL = nan(numSubjects,1);

relativePercentIncrease = nan(numSubjects,1);

%% LOAD FIRST-EPOCH SUCCESS RATES
for s = 1:numSubjects

    subjectName = sprintf('sub-%02d', s);
    subjectNames(s) = subjectName;

    withoutTL_file = fullfile(rootPath, [withoutTL_prefix subjectName '.mat']);
    withTL_file = fullfile(rootPath, [withTL_prefix subjectName '.mat']);

    if ~isfile(withoutTL_file) || ~isfile(withTL_file)
        warning('Missing files for %s', subjectName);
        continue;
    end

    data_without = load(withoutTL_file, 'successRate');
    data_with = load(withTL_file, 'successRate');

    epoch1_without = data_without.successRate(:,1);
    epoch1_with = data_with.successRate(:,1);

    mean_withoutTL(s) = mean(epoch1_without, 'omitnan');
    std_withoutTL(s) = std(epoch1_without, 'omitnan');

    mean_withTL(s) = mean(epoch1_with, 'omitnan');
    std_withTL(s) = std(epoch1_with, 'omitnan');

    relativePercentIncrease(s) = (mean_withTL(s) - mean_withoutTL(s)) * 100;
end

%% TABLE OUTPUT
ResultsTable = table( ...
    subjectNames, ...
    mean_withoutTL, ...
    mean_withTL, ...
    relativePercentIncrease, ...
    'VariableNames', { ...
    'Subject', ...
    'Mean_FirstEpoch_WithoutTL', ...
    'Mean_FirstEpoch_WithTL', ...
    'RelativePercent_Increase'} );

disp(ResultsTable);

writetable(ResultsTable, fullfile(savePath, 'FirstEpoch_TL_RelativeIncrease_Table.csv'));

%% SUBJECT-WISE SIGNIFICANCE
p_values = nan(numSubjects,1);
h_values = nan(numSubjects,1);

for s = 1:numSubjects

    subjectName = sprintf('sub-%02d', s);

    withoutTL_file = fullfile(rootPath, [withoutTL_prefix subjectName '.mat']);
    withTL_file = fullfile(rootPath, [withTL_prefix subjectName '.mat']);

    if ~isfile(withoutTL_file) || ~isfile(withTL_file)
        continue;
    end

    data_without = load(withoutTL_file, 'successRate');
    data_with = load(withTL_file, 'successRate');

    epoch1_without = data_without.successRate(:,1);
    epoch1_with = data_with.successRate(:,1);

    [h, p] = ttest(epoch1_with, epoch1_without);

    p_values(s) = p;
    h_values(s) = h;
end

%% BAR PLOT: SUB-01 THROUGH SUB-10
plot_idx = 1:numPlotSubjects;

plot_without = mean_withoutTL(plot_idx) * 100;
plot_with = mean_withTL(plot_idx) * 100;

err_without = std_withoutTL(plot_idx) * 100;
err_with = std_withTL(plot_idx) * 100;

hFig = figure;
barData = [plot_with, plot_without];

b = bar(barData);
hold on;

ngroups = size(barData,1);
nbars = size(barData,2);

x = nan(nbars, ngroups);
for i = 1:nbars
    x(i,:) = b(i).XEndPoints;
end

errorbar(x(1,:), plot_with, err_with, 'k', 'linestyle', 'none');
errorbar(x(2,:), plot_without, err_without, 'k', 'linestyle', 'none');

grid on;
ylabel('Success Rate (%)');
xlabel('Subject');
title('First-Epoch Success Rate: With TL vs Without TL');

xticks(1:numPlotSubjects);
xticklabels(subjectNames(plot_idx));
xtickangle(45);

legend({'with TL', 'without TL'}, 'Location', 'best');

ylim([0 100]);
set(gca, 'fontsize', 14);

saveas(hFig, fullfile(savePath, 'FirstEpoch_WithTL_vs_WithoutTL.fig'));
saveas(hFig, fullfile(savePath, 'FirstEpoch_WithTL_vs_WithoutTL.tif'));

disp('Finished TL comparison.');
