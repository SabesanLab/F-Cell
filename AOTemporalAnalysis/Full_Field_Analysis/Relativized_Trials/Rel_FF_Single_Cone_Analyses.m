% function [fitCharacteristics]=Rel_FF_Single_Cone_Analyses(stimRootDir, controlRootDir)
% [fitCharacteristics]=Rel_FF_Single_Cone_Analyses(stimRootDir, controlRootDir)
%
%   Calculates pooled variance across a set of pre-analyzed 
%   signals from a single cone's stimulus and control trials, performs 
%   the subtraction between its standard deviations, and performs a
%   piecewise fit of the subtraction.
%
%   This script is designed to work with FULL FIELD datasets- that is, each
%   dataset (mat file) contains *only* control or stimulus data.
%
%   Normally, the user doesn't need to select a stimulus or control root
%   directory (that will be found automatically by
%   "FF_Aggregate_Multi_Trial_Run.m"), but if the software is run by
%   itself it will prompt the user for the folders containing the
%   pre-analyzed mat files generated by Rel_FF_Temporal_Reflectivity_Analysis.m.
%
% Inputs:
%       stimRootDir: The folder path of the pre-analyzed (.mat) stimulus
%       trials. Each mat file must contain valid stimulus signals.
%
%       controlRootDir: The folder path of the pre-analyzed (.mat) control
%       trials. Each mat file must contain valid control signals.
%
%
% Outputs:
%       fitCharacteristics: Information extracted from the mat files and
%       the fitted subtracted signal.
%
% Created by Robert F Cooper 2017-10-31
%

clear;


% load('lowest_responders.mat');

CUTOFF = 26;
NUMTRIALS = 20;
CRITICAL_REGION = 72:108; % 72:90;

CELL_OF_INTEREST = [1:2000];

if isempty(CELL_OF_INTEREST)
    close all force;
end

if ~exist('stimRootDir','var')
    stimRootDir = uigetdir(pwd, 'Select the directory containing the stimulus profiles');
    controlRootDir = uigetdir(pwd, 'Select the directory containing the control profiles');
end

profileSDataNames = read_folder_contents(stimRootDir,'mat');
profileCDataNames = read_folder_contents(controlRootDir,'mat');


% For structure:
% /stuff/id/date/wavelength/time/intensity/location/data/Profile_Data

[remain kid] = getparent(stimRootDir); % data
[remain stim_loc] = getparent(remain); % location 
[remain stim_intensity] = getparent(remain); % intensity 
[remain stim_time] = getparent(remain); % time
[remain stimwave] = getparent(remain); % wavelength
% [remain sessiondate] = getparent(remain); % date
[~, id] = getparent(remain); % id
outFname = [id '_' stimwave '_' stim_intensity '_' stim_time  '_' stim_loc '_' num2str(length(profileSDataNames)) '_single_cone_signals'];


%% Code for determining variance across all signals at given timepoint

THEwaitbar = waitbar(0,'Loading stimulus profiles...');

max_index=0;

load(fullfile(stimRootDir, profileSDataNames{1}));
stim_coords = ref_coords;

if ~isempty(CELL_OF_INTEREST)
    stim_cell_reflectance = cell(length(profileSDataNames),1);
end
stim_cell_reflectance_nonorm = cell(length(profileSDataNames),1);
stim_time_indexes = cell(length(profileSDataNames),1);
stim_cell_prestim_mean = cell(length(profileSDataNames),1);

for j=1:length(profileSDataNames)

    waitbar(j/length(profileSDataNames),THEwaitbar,'Loading stimulus profiles...');
    
    ref_coords=[];
    profileSDataNames{j}
    load(fullfile(stimRootDir,profileSDataNames{j}));
    
    if ~isempty(CELL_OF_INTEREST)
        stim_cell_reflectance_nonorm{j} = cell_reflectance;
    end
    stim_cell_reflectance{j} = norm_cell_reflectance;
    stim_time_indexes{j} = cell_times;
    stim_cell_prestim_mean{j} = cell_prestim_mean;
    
    thesecoords = union(stim_coords, ref_coords,'rows');
    
    % These all must be the same length! (Same coordinate set)
    if size(ref_coords,1) ~= size(thesecoords,1)
        error('Coordinate lists different between mat files in this directory. Unable to perform analysis.')
    end
    
    for k=1:length(cell_times)
        max_index = max([max_index max(cell_times{k})]);
    end
    
end

%%
if ~isempty(CELL_OF_INTEREST)
    control_cell_reflectance = cell(length(profileCDataNames),1);
end
control_cell_reflectance_nonorm = cell(length(profileCDataNames),1);
control_time_indexes = cell(length(profileCDataNames),1);
control_cell_prestim_mean = cell(length(profileCDataNames),1);

load(fullfile(controlRootDir, profileCDataNames{1}));
control_coords = ref_coords;

for j=1:length(profileCDataNames)

    waitbar(j/length(profileCDataNames),THEwaitbar,'Loading control profiles...');
    
    ref_coords=[];
    profileCDataNames{j}
    load(fullfile(controlRootDir,profileCDataNames{j}));
    
    if ~isempty(CELL_OF_INTEREST)
        control_cell_reflectance_nonorm{j} = cell_reflectance;
    end
    control_cell_reflectance{j} = norm_cell_reflectance;
    control_time_indexes{j} = cell_times;
    control_cell_prestim_mean{j} = cell_prestim_mean;

    thesecoords = union(control_coords, ref_coords,'rows');
    
    % The length of the cell reflectance lists *must* be the same, because the
    % coordinate lists *must* be the same in each mat file.
    if size(ref_coords,1) ~= size(thesecoords,1)
        error('Coordinate lists different between mat files in this directory. Unable to perform analysis.')
    end
    
    for k=1:length(cell_times)
        max_index = max([max_index max(cell_times{k})]);
    end
    
end

%% The coordinate lists must the same length,
% otherwise it's not likely they're from the same set.

if size(stim_coords,1) ~= size(control_coords,1)
    error('Coordinate lists different between control and stimulus directories. Unable to perform analysis.')
end

allcoords = stim_coords;


%% Aggregation of all trials

percentparula = parula(101);

numstimcoords = size(stim_coords,1);

stim_cell_var = nan(numstimcoords, max_index);
stim_cell_median = nan(numstimcoords, max_index);
stim_cell_pca_std = nan(numstimcoords, 1);
stim_trial_count = zeros(numstimcoords,1);
stim_posnegratio = nan(numstimcoords,max_index);
stim_prestim_means = nan(numstimcoords,length(profileSDataNames));

i=1;

        
for i=1:numstimcoords
    waitbar(i/size(stim_coords,1),THEwaitbar,'Processing stimulus signals...');

    
    numtrials = 0;
    all_times_ref = nan(length(profileSDataNames), max_index);
    if ~isempty(CELL_OF_INTEREST)
        nonorm_ref = nan(length(profileSDataNames), max_index);
    end
    for j=1:length(profileSDataNames)
        
        if ~isempty(stim_cell_reflectance{j}{i}) && ...
           sum(stim_time_indexes{j}{i} >= CRITICAL_REGION(1) & stim_time_indexes{j}{i} <= CRITICAL_REGION(end)) >= CUTOFF && ...
           stim_cell_prestim_mean{j}(i) <= 240

            
            stim_prestim_means(i,j) = stim_cell_prestim_mean{j}(i);
            numtrials = numtrials+1;
            if ~isempty(CELL_OF_INTEREST)
                nonorm_ref(j, stim_time_indexes{j}{i} ) = stim_cell_reflectance_nonorm{j}{i}(~isnan(stim_cell_reflectance_nonorm{j}{i}))-stim_prestim_means(i,j);
                
%                 figure(10); plot(stim_cell_reflectance{j}{i}); title(num2str(stim_prestim_means(i,j)));
            end
            all_times_ref(j, stim_time_indexes{j}{i} ) = stim_cell_reflectance{j}{i};
        end
    end 
    stim_trial_count(i) = numtrials;
    
    quants = quantile(all_times_ref, [0.1 0.90]);
    
    for j=1:max_index
        nonan_ref = all_times_ref(~isnan(all_times_ref(:,j)), j);
        refcount = sum(~isnan(all_times_ref(:,j)));
        refmedian = median(nonan_ref);
        if ~isnan(refmedian)
            stim_cell_median(i,j) = double(median(nonan_ref));
            
            stim_cell_var(i,j) = ( sum((nonan_ref-mean(nonan_ref)).^2)./ (refcount-1) );
        end
    end
    
    if any(i==CELL_OF_INTEREST) && stim_trial_count(i)>CUTOFF 
        figure(1); clf;
        subplot(4,1,1); plot( nonorm_ref' );axis([1 max_index -150 150]); xlabel('Time index'); ylabel('Raw Response');
        subplot(4,1,2); plot(all_times_ref');  axis([1 max_index -10 10]); xlabel('Time index'); ylabel('Standardized Response');
        subplot(4,1,3); plot(stim_cell_median(i,:)); hold on;
                        plot(sqrt(stim_cell_var(i,:))); hold off; axis([1 max_index -2 10]); xlabel('Time index'); ylabel('Response');
        subplot(4,1,4); plot(stim_prestim_means(i,:),'*'); xlabel('Trial #'); ylabel('Prestimulus mean (A.U.)'); axis([0 50 0 255]);
        title(['Cell #:' num2str(i)]);
        drawnow;
%         saveas(gcf, ['Cell_' num2str(i) '_stimulus.png']);
        
        THEstimref = all_times_ref;
        
%         figure(5); imagesc(ref_image); colormap gray; axis image;hold on; 
%         plot(ref_coords(i,1),ref_coords(i,2),'r*'); hold off;
%         saveas(gcf, ['785_Cell_' num2str(i) '_location.png']);
        drawnow;
        
%         stim_prestim_means
        pause;
    end

end


%%
numcontrolcoords = size(control_coords,1);

control_cell_var = nan(size(control_coords,1), max_index);
control_cell_median = nan(size(control_coords,1), max_index);
control_cell_pca_std = nan(size(stim_coords,1), 1);
control_trial_count = zeros(size(control_coords,1),1);
control_posnegratio = nan(size(control_coords,1),max_index);
cont_prestim_means=[];


for i=1:numcontrolcoords
    waitbar(i/size(control_coords,1),THEwaitbar,'Processing control signals...');

    numtrials = 0;
    all_times_ref = nan(length(profileCDataNames), max_index);
    if ~isempty(CELL_OF_INTEREST)
        nonorm_ref = nan(length(profileCDataNames), max_index);
    end
    for j=1:length(profileCDataNames)
                        
        if ~isempty(control_cell_reflectance{j}{i}) && ...
           sum(control_time_indexes{j}{i} >= CRITICAL_REGION(1) & control_time_indexes{j}{i} <= CRITICAL_REGION(end)) >=  CUTOFF && ...
           control_cell_prestim_mean{j}(i) <= 240
       
            cont_prestim_means = [cont_prestim_means; control_cell_prestim_mean{j}(i)];
            numtrials = numtrials+1;
            if ~isempty(CELL_OF_INTEREST)
                nonorm_ref(j, control_time_indexes{j}{i} ) = control_cell_reflectance_nonorm{j}{i}(~isnan(control_cell_reflectance_nonorm{j}{i}));
            end
            all_times_ref(j, control_time_indexes{j}{i} ) = control_cell_reflectance{j}{i};
        end
    end
    control_trial_count(i) = numtrials;
    

    for j=1:max_index
        nonan_ref = all_times_ref(~isnan(all_times_ref(:,j)), j);
        refcount = sum(~isnan(all_times_ref(:,j)));
        refmedian = median(nonan_ref);
        if ~isnan(refmedian)
            control_cell_median(i,j) = refmedian;
            control_cell_var(i,j) = ( sum((nonan_ref-mean(nonan_ref)).^2)./ (refcount-1) );
        end
    end
    
    if any(i==CELL_OF_INTEREST) %&& control_trial_count(i)>CUTOFF
        figure(2); clf;
        subplot(3,1,1); plot(bsxfun(@minus,nonorm_ref, nonorm_ref(:,2))');axis([CRITICAL_REGION(1) CRITICAL_REGION(end) -150 150]);  xlabel('Time index'); ylabel('Raw Response');
        subplot(3,1,2); plot(all_times_ref'); axis([CRITICAL_REGION(1) CRITICAL_REGION(end) -10 10]); xlabel('Time index'); ylabel('Standardized Response');
        subplot(3,1,3); plot(control_cell_median(i,:)); hold on;
                        plot(sqrt(control_cell_var(i,:))); hold off; axis([CRITICAL_REGION(1) CRITICAL_REGION(end) -2 10]); xlabel('Time index'); ylabel('Response');
        title(['Cell #:' num2str(i)]);   
        THEcontref = all_times_ref;
        drawnow;
        critreg=all_times_ref(:,CRITICAL_REGION);
        quantile(critreg(:),.95)
        quantile(critreg(:),.5)
        quantile(critreg(:),.05)
        
%         saveas(gcf, ['Cell_' num2str(i) '_control.png']);
        pause;
    end
end


% for k=1:size(all_times_ref,1)
% figure(1); clf; plot(THEstimref(k,:)); axis([2 166 -10 10]);
% pause;
% end

% figure;
% histogram(stim_prestim_means, 255); hold on; histogram(cont_prestim_means, 255);
% numover = sum(stim_prestim_means>200) + sum(cont_prestim_means>200);
% title(['Pre-stimulus means of all available trials (max 50) from ' num2str(size(control_coords,1)) ' cones. ' num2str(numover) ' trials >200 ']);


valid = (stim_trial_count >= NUMTRIALS) & (control_trial_count >= NUMTRIALS);

% Calculate the pooled std deviation
std_dev_sub = sqrt(stim_cell_var)-mean(sqrt(control_cell_var),'omitnan');
median_sub = stim_cell_median-mean(control_cell_median,'omitnan');
%%

if ~isempty(CELL_OF_INTEREST )
%     figure(3);clf;
%     
% %     load('control_avgs.mat')
%     
%     plot( stim_cell_median(CELL_OF_INTEREST,:)-allcontrolmed );  hold on;
%     plot(sqrt(stim_cell_var(CELL_OF_INTEREST,:))-allcontrolstd);
%     axis([2 166 -3 3]);
%     title(['Cell #:' num2str(CELL_OF_INTEREST)]);  
%     drawnow;
%     saveas(gcf, ['Cell_' num2str(CELL_OF_INTEREST) '_subs.svg']);
end
%% Calculate PCA on the crtiical area of the signals
critical_nonnan_ref = sqrt(stim_cell_var(:,CRITICAL_REGION));
[std_dev_coeff, std_dev_score, std_dev_latent, tquare, std_dev_explained, std_dev_mu] = pca(critical_nonnan_ref);


critical_nonnan_ref = stim_cell_median(:,CRITICAL_REGION);
[median_coeff, median_score, latent, tquare, median_explained]=pca(critical_nonnan_ref);

timeBase = ((1:max_index-1)/16.6)';

%% Analyze the signals

AmpResp = nan(size(std_dev_sub,1),1);
MedianResp = nan(size(std_dev_sub,1),1);
TTPResp = nan(size(std_dev_sub,1),1);
PrestimVal = nan(size(std_dev_sub,1),1);
ControlPrestimVal = nan(size(std_dev_sub,1),1);

allinds = 1:length(std_dev_sub);
for i=1:size(std_dev_sub,1)
waitbar(i/size(std_dev_sub,1),THEwaitbar,'Analyzing subtracted signals...');

    if ~all( isnan(std_dev_sub(i,2:end)) ) && (stim_trial_count(i) >= NUMTRIALS) && (control_trial_count(i) >= NUMTRIALS)
        std_dev_sig = std_dev_sub(i,2:end);
        
        nanners = ~isnan(std_dev_sig);
        
        firstind=find(cumsum(nanners)>0);
        [~, lastind] = max(cumsum(nanners)-sum(nanners));
        interpinds = firstind(1):lastind;
        goodinds = allinds(nanners);
        
        std_dev_sig = interp1(goodinds, std_dev_sig(nanners), interpinds, 'linear');        

        filt_stddev_sig = wavelet_denoise( std_dev_sig );

        median_sig = median_sub(i,2:end);
        
        nanners = ~isnan(median_sig);
        
        firstind=find(cumsum(nanners)>0);
        [~, lastind] = max(cumsum(nanners)-sum(nanners));
        interpinds = firstind(1):lastind;
        goodinds = allinds(nanners);
        
        median_sig = interp1(goodinds, median_sig(nanners), interpinds, 'linear');
        
        filt_median_sig = wavelet_denoise( median_sig );

        critical_filt = filt_stddev_sig( CRITICAL_REGION );
        
%         [height,loc]= findpeaks(filt_stddev_sig(CRITICAL_REGION))
        
        [~, TTPResp(i)] = max( abs(critical_filt) );    
        AmpResp(i) = critical_filt(TTPResp(i))-mean(filt_stddev_sig(1:CRITICAL_REGION(1)));
        MedianResp(i) = mean(abs(filt_median_sig(CRITICAL_REGION)));%abs( max(abs(filt_median_sig(CRITICAL_REGION)))-mean(filt_median_sig(1:CRITICAL_REGION(1))) );
        PrestimVal(i) = mean( stim_prestim_means(i,:),2, 'omitnan');
        ControlPrestimVal(i) = mean( cont_prestim_means(i,:),2, 'omitnan');
%         histogram(filt_stddev_sig(CRITICAL_REGION),20)
%         if AmpResp(i) == 0
           plot( filt_stddev_sig ); hold on; plot(std_dev_sig);
           plot(sqrt(control_cell_var(i,2:end))-1);
           hold off; drawnow; pause;
%         end
    end
end
close(THEwaitbar);
%%
save([ outFname '.mat'],'AmpResp','MedianResp','TTPResp', 'std_dev_coeff',...
     'valid','std_dev_explained','median_explained','median_coeff',...
     'std_dev_score','median_score','allcoords','ref_image','control_cell_median',...
     'control_cell_var','stim_cell_median','stim_cell_var','stim_prestim_means');

 

%% Plot the pos/neg ratio of the mean vs the amplitude
posnegratio=nan(size(control_coords,1),1);


figure(101); clf; hold on;
for i=1:size(control_coords,1)
    if ~isnan(AmpResp(i))
        % Find out what percentage of time the signal spends negative
        % or positive after stimulus delivery (66th frame)
%         numposneg = sign(mean_sub(i,:));
%         pos = sum(numposneg == 1);
% 
%         posnegratio(i) = 100*pos/length(numposneg);

%         if mean(stim_prestim_means(i,:),'omitnan') > 150 && mean(stim_prestim_means(i,:),'omitnan') < 152
%             disp(num2str(i));
%         end
        plot( mean(stim_prestim_means(i,:),'omitnan'),AmpResp(i)+abs(MedianResp(i)),'k.');        
    end
end
% ylabel('Median response amplitude');
% xlabel('Reflectance response amplitude');
% title('Median reflectance vs reflectance response amplitude')
hold off;
% saveas(gcf,['posneg_vs_amp_' num2str(stim_intensity) '.png']);
%% Plot histograms of the amplitudes
figure(7);
histogram( AmpResp(~isnan(AmpResp)) ,'Binwidth',0.1);
title('Stim-Control per cone subtraction amplitudes');
xlabel('Amplitude difference from control');
ylabel('Number of cones');

%% TEMP to prove control equivalence!
% stim_resp = sum(sqrt(stim_cell_var(:,critical_region)),2) + abs(sum(stim_cell_median(:,critical_region),2));
% control_resp = sum(sqrt(control_cell_var(:,critical_region)),2) + abs(sum(control_cell_median(:,critical_region),2));
% 
% figure(8); plot(stim_resp,control_resp,'k.'); hold on;
% plot([25 90],[25 90],'k'); hold off; axis square;
% ylabel('450nW control response');
% xlabel('0nW control response');
