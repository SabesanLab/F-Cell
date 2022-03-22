% 2021-07-26
% Robert F Cooper
% This code uses the F(Cell) toolkit to obtain coordinate-based iORGs from
% any MEAO dataset.
%
% It outputs an avi that is the same duration and size as the original, but
% Where each pixel value is the iORG for that pixel location.
% Assumes the data has been registered previously.

close all force;
clear;

rootDir = uigetdir(pwd, 'Select the folder containing all videos of interest.');

fNames = read_folder_contents(rootDir, 'avi', '760nm');

% Remove this from consideration.
fNames = fNames(cellfun(@(name) ~contains(name, 'ALL_ACQ_STK'), fNames)); 

% Load our reference image.
all_acq_avg_im = read_folder_contents(rootDir, 'tif', 'ALL_ACQ_AVG');

if length(all_acq_avg_im) > 1
    warning('More than one pipeline average file found. Results may be inaccurate.');
end
ref_coords=[];
if ~isempty(all_acq_avg_im) && exist(fullfile(rootDir, all_acq_avg_im{1}), 'file')
    ref_im = imread(fullfile(rootDir, all_acq_avg_im{1}));
    
    % Load our reference coordinates.
    if exist(fullfile(rootDir, [all_acq_avg_im{1}(1:end-4) '_coords.csv'] ), 'file')
        ref_coords = dlmread(fullfile(rootDir, [all_acq_avg_im{1}(1:end-4) '_coords.csv'] ));
    end
else
    error('No reference image found. Please re-run pipeline to fix issue.');
end

[stimFile, stimPath] = uigetfile(fullfile(rootDir, '*.csv'), 'Select the stimulus train file that was used for this dataset.');

stimTrain = dlmread(fullfile(stimPath, stimFile), ',');

wbh = waitbar(0,['Loading dataset: ' strrep(fNames{1},'_','\_')]);

startind = 1;
endind = length(fNames);

finalized_temporal_profiles = cell(endind, 1);
framestamps = cell(endind, 1);
pop_iORGs = cell(endind, 1);
framerate = zeros(endind, 1);
num_extant_profiles = zeros(endind, 1);

legendfNames = fNames(startind:endind);
%% Load the temporal profiles from each pipelined dataset.

for f=startind:endind
    
    
    % Load the pipelined MEAO data.
    waitbar(0, wbh, ['Loading dataset: ' strrep(fNames{f},'_','\_')]);    
    [temporal_data, framestamps{f}, framerate(f)] = load_pipelined_MEAO_data(fullfile(rootDir, fNames{f}));

    % Extract temporal profiles at each pixel
    [temporal_profiles, ~]=extract_temporal_profiles(temporal_data,'SegmentationRadius',2, 'Coordinates', ref_coords, 'ProgBarHandle', wbh );

    num_extant_profiles(f) = sum(any(~isnan(temporal_profiles),2));
    % Normalize the temporal profiles to each frame's mean
    [norm_temporal_profiles]=framewise_normalize_temporal_profiles(temporal_profiles, 'ProgBarHandle', wbh);

    % Standardize the temporal profiles to their *pre stimulus* behavior    
    [finalized_temporal_profiles{f}]=standardize_temporal_profiles(norm_temporal_profiles, framestamps{f}', [1 stimTrain(1)], framerate(f),...
                                                                  'Method', 'relative_change', 'ProgBarHandle', wbh);


    writematrix([framestamps{f}'; finalized_temporal_profiles{f}], [fNames{f}(1:end-4) '_profiles.csv'], 'Delimiter',',');
end


%% Some PCA fitting.
max_framestamp = max(cellfun(@max,framestamps));
min_framestamp = min(cellfun(@min,framestamps));

all_profiles = nan(length(finalized_temporal_profiles)*size(ref_coords,1), max_framestamp);
for f=startind:endind
    
    all_profiles( ((f-1)*size(ref_coords,1)+1):f*size(ref_coords,1), framestamps{f}) = finalized_temporal_profiles{f};
end
notallnan = ~all(isnan(all_profiles),2);

[COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU] = pca(all_profiles(notallnan, :),'Algorithm','als');

all_profiles(notallnan, :) = SCORE(:,1:5)*COEFF(:, 1:5)'; % Reconstructed signals, using the first 5 components.
% all_profiles(notallnan, :) = SCORE(:,:)*COEFF(:, :)'; 

% Put them back in the finalized temporal profiles, and make population
% iORGs out of them.
figure(708); clf;
for f=startind:endind
    
    finalized_temporal_profiles{f} = all_profiles( ((f-1)*size(ref_coords,1)+1):f*size(ref_coords,1), :);
    
    figure(707); hold on;
    pop_iORGs{f} = Population_iORG(finalized_temporal_profiles{f}, 0:(size(finalized_temporal_profiles{f},2)-1) ); 
    
end
% legend(legendfNames, 'Interpreter', 'none')
hold off;

%% Make an average population iORG from our controls using pooled variance (weighted avg)

avg_pop_iORG = nan(endind, max_framestamp);

for f=startind:endind
    
    avg_pop_iORG(f, framestamps{f}) = pop_iORGs{f}*(num_extant_profiles(f)-1);
    
end

pointwise_numextprofiles = repmat(num_extant_profiles,[1 size(avg_pop_iORG,2)]);
pointwise_numextprofiles = pointwise_numextprofiles.*~isnan(avg_pop_iORG);

validframes = any(~isnan(avg_pop_iORG), 1);
allframes = 1:max_framestamp;
allframes = allframes(validframes);

avg_pop_iORG = sum(avg_pop_iORG,'omitnan') ./ sum(pointwise_numextprofiles);
avg_pop_iORG = avg_pop_iORG(validframes);

figure(707); hold on;
plot(allframes, avg_pop_iORG,'k-*')
% iORG_Map(ref_coords, finalized_temporal_profiles, framestamps);

close(wbh) 

return;

%% Code 


%%
legendfNames = fNames(startind:endind);

acqtimes =23;%12; %:32;

peakval = nan(size(finalized_temporal_profiles{acqtimes(1)},1), 1);
for coi=1:size(finalized_temporal_profiles{acqtimes(1)},1)
%     goodforlegend = true(length(legendfNames),1);
    
%     std(finalized_temporal_profiles{f}(coi, [1 stimTrain(1)]))
    
    
    for f=1:length(acqtimes)
        t = acqtimes(f);

        

        fullsig = finalized_temporal_profiles{t}(coi,:);
        origsig = fullsig;
        if ~all(isnan(fullsig))
%             minsig = min(fullsig);
%             maxsig = max(fullsig);
%             fullsig = fullsig-minsig;
%             fullsig = 2*fullsig./max(fullsig);
%             fullsig = fullsig -1;

%             figure(708); clf;
%             hold on;
%             plot((0:179)/framerate(1), fullsig);
%             plot([58/framerate(1) 58/framerate(1)],[-2 2], 'k')
%             axis([0 180/framerate(1) -2 2])
%             hold off;

            filtbank =cwtfilterbank('Wavelet','morse','SignalLength',numel(fullsig),'SamplingFrequency',framerate(t),...
                                    'WaveletParameters',[3 9], 'FrequencyLimits',[0.25 10]);
%             [psi, t]=wavelets(filtbank);
%             wavesc = scales(filtbank);
%              figure(21); clf;
%             cwt(fullsig,'Filterbank',filtbank);
            
            [wt, f, cone_of_inter,fb,scalingfs]=cwt(fullsig,'Filterbank',filtbank);
            wtpwrspect =(abs(wt));
            
            [maxvalrows, maxrow]=max(wtpwrspect(:,stimTrain(1):stimTrain(1)+30),[], 1);
            [maxvalcols, maxcol]=max(maxvalrows);
            maxvalcols;
            peakval(coi) = maxvalcols;

%             maxvalcols = sum(sum(wtpwrspect(:,stimTrain(1):stimTrain(1)+10).^2 )); %./numel(wtpwrspect(:,stimTrain(1):stimTrain(1)+10));
            

                figure(708); clf;
                hold on;
                plot((0:numel(origsig)-1)/framerate(t), origsig);
                plot([58/framerate(t) 58/framerate(t)],[-2 2], 'k')
                title(num2str(peakval(coi)));
                hold off;
                figure(21); clf;
            cwt(fullsig,'Filterbank',filtbank);
            caxis([0 0.5])
                pause;



            
        end
   
        
    end
    
%     legend(legendfNames(goodforlegend), 'Interpreter', 'none')
%     ref_coords(coi,:)
    
end

histogram((peakval))