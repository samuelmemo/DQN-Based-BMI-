%% This code implements offline center out reaching task using KTD algorithm based on EEG data
% The code is framed to conduct a reaching task.
% The reward is assigned when the cursor is closed to the taget: i.e.
% distance to the target is less then a determined threshold.
% This code is updated by Jihye Bae on 12/21/2021
% This code is last updated by Bhoj Raj Thapa on 08/16/2024

clear; close all; clc;
%% Global Variables
featurePath = 'C:\DATA\2026_School\UKY - SP2026\EE 595\Code\Ava Code\results';
codePath = pwd;

% dataName = 'FREEFORMSubjectB1511112StLRHand';
% dataName = 'FREEFORMSubjectC1512082StLRHand';
% dataName = 'FREEFORMSubjectC1512102StLRHand';

dataNameSource = 'minus sub-02';%'FREEFORMSubjectC1512102StLRHand'; %Source
dataNameTraining = 'sub-02'; %Target
featureName = 'RAW';
tstartSource = 0; % Define start time to extract EEG. Relative to the start of a trial.
tendSource = 0.5; % Define end time to extract EEG. Relative to the start of trial.
tstartTraining = tstartSource;
tendTraining = tendSource; 
classifierName = 'KTD_TL';

testRunName = '';

dataPathSource = [featurePath '\' dataNameSource ...
    '\Data_from' num2str(tstartSource) '_to' num2str(tendSource) '\' classifierName 'source' featureName]; 
%loading TL parameters
TLsource = load(fullfile(dataPathSource,['ResultsWparms_TLsource' testRunName])); 

dataPathTraining = [featurePath '\' dataNameTraining ...
    '\Data_from' num2str(tstartTraining) '_to' num2str(tendTraining)];

resultPath = dataPathTraining; 
folderName = [classifierName 'training' featureName '_source' ...
    dataNameSource 'from' num2str(tstartSource) 'to' num2str(tendSource)];
if exist(fullfile(resultPath,folderName),'dir')~= 7
    mkdir(fullfile(resultPath,folderName))
end
resultPath = fullfile(resultPath,folderName); 

dataFileName = ['Data_EEGfeature' featureName '.mat'];
load(fullfile(dataPathTraining,dataFileName));
NSinput = eval([featureName 'feature']);
TargetIndex = classID;

%% setting up parameters
parms = TLsource.parms;
[parms.ntrial, parms.nch] = size(NSinput);
parms.ntarget = max(classID);
parms.npossibleAction = parms.ntarget;

% KTD implementation
ktdReachingIndexAll = cell(1,parms.nMCrun);
sucessAll = cell(1,parms.nMCrun);
QAll = cell(1,parms.nMCrun);
kernelBWtraceAll = cell(1,parms.nMCrun);
EpochLength = cell(1,parms.nMCrun);
successRate = nan(parms.nMCrun,parms.nepoch);
trialIndex = nan(parms.nMCrun,parms.ntrial);
for imcrun = 1:parms.nMCrun

    Unit = TLsource.TLunit{1,imcrun};
    Weight = TLsource.TLweight{1,imcrun};
    ikernel = size(Unit,1);
    parms.RLepsilon = TLsource.parms.RLepsilon;

    ktdReachingIndex = nan(parms.nepoch,parms.ntrial);
    Q = nan(parms.ntrial*parms.nepoch,parms.npossibleAction);
    kernelBWtrace = nan(parms.ntrial*parms.nepoch,1);
    success = zeros(parms.nepoch,parms.ntarget); % need to assign 0 values in sucess 
    % so that it starts adding +1 whenever a successful trial is observed.

    trialIndex(imcrun,:) = randperm(parms.ntrial);
    
    for iepoch = 1: parms.nepoch
        tic

        fprintf('Current Implementation: "%d" Monte Carlo run and "%d" epoch\n', imcrun, iepoch)
        
        if mod(iepoch,parms.RLepsilonDecayEpoch) == 0
            parms.RLepsilon = parms.RLepsilon*parms.RLepsilonDecayRate;
        end
        
        for itrial = 1: parms.ntrial
            cursorPreXY = parms.ReachingCenterXY;
            % normalization NSinput
            ns_max = max(NSinput(trialIndex(imcrun,itrial),:));
            ns_min = min(NSinput(trialIndex(imcrun,itrial),:));
            if ns_max == ns_min
                fprintf('=====================ERROR========================\n')
                fprintf('NSinput has max and min the same value, at %d th epoch and %d th trial', iepoch, itrial)
                pause
            end
            ns_amp = (parms.NormalizationUpperBound - parms.NormalizationLowerBound)/(ns_max-ns_min);
            ns_off = parms.NormalizationUpperBound - ns_amp*ns_max;
            NormalizedNS = ns_amp*NSinput(trialIndex(imcrun,itrial),:) + ns_off;
            
            % assigning the first unit values in KTD
%             if iepoch == 1 && itrial == 1
%                 Unit(ikernel,:) = NormalizedNS;
%                 Weight(ikernel,1:parms.npossibleAction) = zeros(ikernel,parms.npossibleAction);
%                 % assgining dummy kernel size
%                 kernelBW = parms.KTDkernelBWini;
%             else
                                
                % computing the current Q(t)
                indif = bsxfun(@minus,Unit,NormalizedNS);
                inputDifference = sum(indif.*indif,2);
                kernelBW = sqrt(mean(inputDifference))*parms.KTDkernelBWfactor; % computing kernel size
                kernelf = exp(inputDifference/(-2*kernelBW.^2));
                out1 = parms.KTDstepsize*kernelf'*Weight;
                
                maxout1 = find(out1 == max(out1));
                action1 = func_ActionSelection(maxout1,parms.npossibleAction,parms.RLepsilon);
                Q1 = out1(action1);
                Q((iepoch-1)*parms.ntrial+itrial,:) = out1;
                % tracking selected action
                ktdReachingIndex(iepoch,itrial) = action1;
                
                % quantization application
                quantizationIndex = find(inputDifference < parms.KTDquantizationThr);
                if isempty(quantizationIndex)
                    ikernel = ikernel+1;
                    Unit(ikernel,:) = NormalizedNS;
                    Weight(ikernel,:) = Weight(ikernel-1,:);
                    centerIndex = ikernel;
                else
                    if length(quantizationIndex)>1
                        fprintf('=====================ERROR========================\n')
                        fprintf('Quantizatio Weight! Please modify your quantization threshold.\n')
                        pause
                    else
                        centerIndex = quantizationIndex;
                    end
                end
                
                % computing the future Q(t+1)
                if itrial < parms.ntrial
                    indif = bsxfun(@minus, Unit, NormalizedNS);
                    inputDifference = sum(indif.*indif,2);
                    out2 = parms.KTDstepsize*exp(inputDifference/(-2*kernelBW.^2))'*Weight;
                    maxout2 = find(out2 == max(out2));
                    action2 = func_ActionSelection(maxout2,parms.npossibleAction,parms.RLepsilon);
                    Q2 = out2(action2);
                end

                % updating cursor xy location
                cursorNextXY = func_CursorUpdateXY(cursorPreXY, action1, parms.npossibleAction, parms.ReachingCenterXY, parms.ReachingRadius);
                cursorDistance = sqrt(sum((cursorNextXY-parms.ReachingTargetXY(classID(trialIndex(imcrun,itrial)),:)).^2));
                
                % evaluating cursor to target distance
                if cursorDistance < (parms.ReachingRadius-parms.ReachingDisanceThr)
                    reward = parms.RLpreward;
                else
                    reward = parms.RLnreward;
                end
                
                % computing TD value
                %if itrial == parms.ntrial
                    TD = reward - Q1;
               %else
                %    TD = reward + parms.RLgamma*Q2 - Q1;
                %end
                
                % counting success trials
                if reward == parms.RLpreward
                    success(iepoch,classID(trialIndex(imcrun,itrial))) = ...
                        success(iepoch,classID(trialIndex(imcrun,itrial))) + 1;
                end
                
                % assigning weights
                if isempty(quantizationIndex)
                        Weight(centerIndex,:) = zeros(1,parms.ntarget);
                        Weight(centerIndex,action1) = TD;
                else
                    Weight(centerIndex,action1) = Weight(centerIndex,action1)+TD;
                end
            %end
                        
            kernelBWtrace((iepoch-1)*parms.ntrial+itrial,1) = kernelBW;
        end % end of trial

        % success rate calculation
        successRate(imcrun,iepoch) = (sum(success(iepoch,:)))/parms.ntrial;
        fprintf('Current success rates: %d.\n',successRate(imcrun,iepoch))


        if iepoch >= 3
            if ((abs(successRate(imcrun, iepoch)-successRate(imcrun, iepoch-1)) +...
                    abs(successRate(imcrun, iepoch-1) - successRate(imcrun, iepoch-2)))/2 < parms.stopThr)
                fprintf("Stopping criteria reached\n");
                break;
            end
        end
        fprintf("Epoch completed at " + iepoch + " epoch in " + toc + " seconds.\n");

    end % end of epoch

    ktdReachingIndexAll{1,imcrun} = ktdReachingIndex;
    sucessAll{1,imcrun} = success;
    QAll{1,imcrun} = Q;
    kernelBWtraceAll{1,imcrun} = kernelBWtrace;
    EpochLength{1,imcrun} = iepoch;
    
    clear Unit
    clear Weight
end % end of MC run

% plotting success rate
h = figure;
plot(successRate','LineWidth',2)
grid on
xlabel('Epoch')
ylabel('Success Rate')
set(gca,'fontsize', 18);

saveas(h,fullfile(resultPath,['Fig_SucessRatesAll' testRunName '.tif']))
saveas(h,fullfile(resultPath,['Fig_SucessRatesAll' testRunName '.fig']))

%% Prepare success rates with TL for plotting

%Extrapolite TL success rate rows where stopping criteria is met
nepoch = max(cell2mat(EpochLength));
successRateExtrapolated = successRate(:,1:nepoch);
for imcrun = 1:parms.nMCrun
    extrapolationIfo = isnan(successRateExtrapolated(imcrun,:));
    if sum(extrapolationIfo) > 0
        extrapolationIndex = find(extrapolationIfo == 1);
        successRateExtrapolated(imcrun, extrapolationIndex) = successRateExtrapolated(imcrun, min(extrapolationIndex)-1);
    end
end

successRateStd = std(successRateExtrapolated);
successRateMean = mean(successRateExtrapolated, 1);

%Prepare success rates without TL for plotting
% Get success rates from original implementation without Transfer learning
cd([featurePath '\' dataNameTraining ...
   '\Data_from' num2str(tstartTraining) '_to' num2str(tendTraining) '\' classifierName 'source' featureName])
OriginalPerformanceWoTL = load(['ResultsWparms_TLsource' testRunName]); 
cd(codePath) %Getting back to the script's directory
%Extrapolate no TL success rate rows where stopping criteria is met
nepochWoTL = max(cell2mat(OriginalPerformanceWoTL.TLepochLength));
successRateExtrapolatedWoTL = OriginalPerformanceWoTL.successRate(:,1:nepochWoTL);
for imcrun = 1:OriginalPerformanceWoTL.parms.nMCrun
    extrapolationIfoWoTL = isnan(successRateExtrapolatedWoTL(imcrun,:));
    if sum(extrapolationIfoWoTL) > 0
        extrapolationIndexWoTL = find(extrapolationIfoWoTL == 1);
        successRateExtrapolatedWoTL(imcrun, extrapolationIndexWoTL) = successRateExtrapolatedWoTL(imcrun, min(extrapolationIndexWoTL)-1);
    end
end

successRateStdWoTL = std(successRateExtrapolatedWoTL);
successRateMeanWoTL = mean(successRateExtrapolatedWoTL, 1);

%Plot TL and no TL success rates for comparison
h = figure;
hold on;
patch([1:nepoch flip(1:nepoch)],[successRateMean-successRateStd flip(successRateMean+successRateStd)] * 100, 'black', 'faceAlpha', 0.4,'LineStyle','none');
patch([1:nepochWoTL flip(1:nepochWoTL)], [successRateMeanWoTL-successRateStdWoTL flip(successRateMeanWoTL+successRateStdWoTL)] * 100, 'black', 'faceAlpha', 0.1,'LineStyle','none')
plot(successRateMean  * 100, 'LineWidth',2, 'LineStyle', '-', 'Color','black');
plot(successRateMeanWoTL * 100, 'LineWidth', 2, 'LineStyle', '--', 'Color', 'black');
grid on
xlabel('Epoch');
ylabel('Success Rate (%)')
ylim([50 100]);
xlim([1,inf]);
set(gca, 'fontsize', 18);
legend({'','','with TL','without TL'}, 'location','southeast');
%title({[dataNameSource  '-->'  dataNameTraining], [featureName ',[' num2str(tstartSource) ',' num2str(tendSource) ']']});

saveas(h,fullfile(resultPath,['Fig_TLPerformance' testRunName '.tif']))
saveas(h,fullfile(resultPath,['Fig_TLPerformance' testRunName '.fig']))

% saving parameters and performances
save(fullfile(resultPath,['ResultsWparms' testRunName '.mat']), ...
    'parms', ...
    'ktdReachingIndexAll','sucessAll','trialIndex', ...
    'QAll','kernelBWtraceAll', ...
    'successRate', 'EpochLength')

%% End of the script
disp('The script executed succefully.')
%% Function Definitions
function ReachingTargetXY = func_GenerateTargetXY(ntarget, ReachingRadius, ReachingCenterXY)
% Generate a x-y target location in a center-out reaching task
% based on the the number of targets.
% The center is located at the origin, (0,0)
%
% ReachingTargetXY = func_GenerateTarget(ntarget)
%
% N(ntarget) regularly spaced center out format
% ntarget: number of targets
% ReachingRadius: ReachingRadius from the center (origin) to the target

theta = 0:2*pi/ntarget:2*pi-2*pi/ntarget;
ReachingTargetXY = round([cos(theta')+ReachingCenterXY(1,1) sin(theta')+ReachingCenterXY(1,2)].*ReachingRadius,3);

end

function action = func_ActionSelection(maxout,npossibleAction,RLepsilon)
% Select one action based on RLepsilon-greedy method
%
% action = func_ActionSelection(maxout,ntarget,RLepsilon)
% 
% maxout: maximum Q value
% ntarget: number of targets
% RLepsilon: RLepsilon in the RLepsilon-greedy method

if rand > 1-RLepsilon
    rnum = rand(1,npossibleAction);
    action = find(rnum == max(rnum));
else
    if isempty(maxout) == 1
        rnum = rand(1,npossibleAction);
        action = find(rnum == max(rnum));
    elseif length(maxout) > 1
        rnum = rand(1,npossibleAction);
        action = find(rnum == max(rnum));
    else
        action = maxout;
    end
end

end

function CursorNextXY = func_CursorUpdateXY(CursorPreXY, selectedAction, npossibleAction, ReachingCenterXY, ReachingRadius)
% Update a x-y cursor location in a center-out reaching task 
% based on a selected action from the current cursor position.
%
% CursorNextXY = func_CursorUpdateXY(CursorPreXY,npossibleAction, ReachingRadius)
%
% CursorPreXY = current X,Y position
% ntarget = number of targets
% ReachingRadius = ReachingRadius from the center (origin) to the target

theta = 0:2*pi/npossibleAction:2*pi-2*pi/npossibleAction;
possibleXY = round([cos(theta')+ReachingCenterXY(1,1) sin(theta')+ReachingCenterXY(1,2)].*ReachingRadius,3);
CursorNextXY = CursorPreXY + possibleXY(selectedAction,:);

end
