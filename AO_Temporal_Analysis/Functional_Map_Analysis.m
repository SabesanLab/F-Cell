
% Created by Robert F Cooper 2021-03-24
%

clear;


CUTOFF = 10;
NUM_NEIGHBORS = 10;
NUMTRIALS=20 * (NUM_NEIGHBORS + 1);
CRITICAL_REGION = 68:102;
window_size = 5;
half_window = floor(window_size/2);

CELL_OF_INTEREST = [];

if isempty(CELL_OF_INTEREST)
    close all force;
end

if ~exist('stimRootDir','var')
    stimRootDir = uigetdir(pwd, 'Select the directory containing the stimulus profiles');
end

profileSDataNames = read_folder_contents(stimRootDir,'mat');


% For structure:
% /stuff/id/date/wavelength/time/intensity/location/data/Profile_Data


% [remain kid] = getparent(stimRootDir); % data
% [remain stim_loc] = getparent(remain); % location 
% [remain stim_intensity] = getparent(remain); % intensity 
% [remain stim_time] = getparent(remain); % time
% [remain stimwave] = getparent(remain); % wavelength
% % [remain sessiondate] = getparent(remain); % date
% [~, id] = getparent(remain); % id
% outFname = [id '_' stimwave '_' stim_intensity '_' stim_time  '_' stim_loc '_' num2str(length(profileSDataNames)) '_' num2str(NUM_NEIGHBORS) '_neighbor_cone_signals'];



%% Code for determining variance across all signals at given timepoint

THEwaitbar = waitbar(0,'Loading profiles...');

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
    

    for k=1:length(cell_times)
        max_index = max([max_index max(cell_times{k})]);
    end
    
end

allcoords = stim_coords;

%% Aggregation of all trials

percentparula = parula(101);

numstimcoords = size(stim_coords,1);

stim_cell_rms = nan(numstimcoords, max_index);
stim_cell_median = double(nan(numstimcoords, max_index));
stim_trial_count = zeros(numstimcoords,1);

stim_prestim_means=[];

i=1;

[~, neighinds]=pdist2(stim_coords,stim_coords,'euclidean','Smallest',NUM_NEIGHBORS+1);

neighinds = neighinds(2:end,:)';

for i=1:numstimcoords
    waitbar(i/size(stim_coords,1),THEwaitbar,'Processing signals...');


    numconesignals = 0;
    all_times_ref = nan(length(profileSDataNames)*(NUM_NEIGHBORS+1), max_index);
    
    % Load cones 
    j=1;
%     for j=1:length(profileSDataNames)
        
        if ~isempty(stim_cell_reflectance{j}{i}) && ...
           sum(stim_time_indexes{j}{i} >= CRITICAL_REGION(1) & stim_time_indexes{j}{i} <=CRITICAL_REGION(end)) >= CUTOFF

            numconesignals = numconesignals+1;
            all_times_ref(j, stim_time_indexes{j}{i} ) = stim_cell_reflectance{j}{i};
        end
%     end     
    
        %Load neighbors into same structure
        for n=1:NUM_NEIGHBORS
            theind=neighinds(i,n);
            offset = n*length(profileSDataNames);
    %         for j=1:length(profileSDataNames)

                if ~isempty(stim_cell_reflectance{j}{theind}) && ...
                   sum(stim_time_indexes{j}{theind} >= CRITICAL_REGION(1) & stim_time_indexes{j}{theind} <=CRITICAL_REGION(end)) >= CUTOFF

                    numconesignals = numconesignals+1;
                    all_times_ref(j+offset, stim_time_indexes{j}{theind} ) = stim_cell_reflectance{j}{theind};
                end
    %         end 
        end
        
    stim_trial_count(i) = numconesignals;
    
    if stim_trial_count(i) > 5
        % Calculate the power over the 6 neighboring cones
        padded_all_times_ref = padarray(all_times_ref, [0 half_window], 'symmetric','both');
        for j=1:size(padded_all_times_ref,2)-window_size
            window =  padded_all_times_ref(:, j:j+window_size-1);
            refcount = sum(~isnan(all_times_ref(:,j)));

            refmedian = median(window(:),'omitnan');


            if ~isnan(refmedian)
                stim_cell_median(i,j) = refmedian;
                stim_cell_rms(i,j) = rms(window(~isnan(window)));
            end
        end


    %     if i==CELL_OF_INTEREST 
%             figure(1); clf;
    %         subplot(3,1,1); plot( bsxfun(@minus,nonorm_ref, nonorm_ref(:,2))');axis([2 166 -75 75]);
%             subplot(3,1,2); plot(all_times_ref');  axis([2 166 -10 10]);       
    %         subplot(3,1,3); plot(stim_cell_median(i,:)); hold on;
    %                         plot(sqrt(stim_cell_var(i,:))); hold off; axis([2 166 -2 4]);
    %         title(['Cell #:' num2str(i)]);
    %         drawnow;
    %         saveas(gcf, ['Cell_' num2str(i) '_stimulus.svg']);
    %         
    %         THEstimref = all_times_ref;
    %         
    %         figure(5); imagesc(ref_image); colormap gray; axis image;hold on; 
    %         plot(ref_coords(i,1),ref_coords(i,2),'r*'); hold off;
    %         saveas(gcf, ['Cell_' num2str(i) '_location.svg']);
%             drawnow;
    %         
    end
%         pause;
%     end

end

% Generate our power maps.
[xq, yq] = meshgrid(min(ref_coords(:,1)):max(ref_coords(:,1)), min(ref_coords(:,2)):max(ref_coords(:,2)));


for t=1:max_index
   
    F = scatteredInterpolant(ref_coords(:,1),ref_coords(:,2), stim_cell_rms(:,t));
    
    imagesc( reshape(F(xq(:),yq(:)), size(xq)) ); caxis([0 6]); title(num2str(t)); drawnow; 
    
end

% figure;
% histogram(stim_prestim_means, 255); hold on; histogram(cont_prestim_means, 255);
% numover = sum(stim_prestim_means>200) + sum(cont_prestim_means>200);
% title(['Pre-stimulus means of all available trials (max 50) from ' num2str(size(control_coords,1)) ' cones. ' num2str(numover) ' trials >200 ']);

valid = (stim_trial_count >= NUMTRIALS) & (control_trial_count >= NUMTRIALS);

% Calculate the pooled std deviation
% std_dev_sub = sqrt(stim_cell_rms)-mean(sqrt(control_cell_var),'omitnan');
% control_std_dev_sub = sqrt(control_cell_var)-mean(sqrt(control_cell_var),'omitnan');
% median_sub = stim_cell_median-mean(control_cell_median,'omitnan');
% control_median_sub = control_cell_median-mean(control_cell_median,'omitnan');

%%

if ~isempty(CELL_OF_INTEREST )
    figure(3);clf;
    
    load('control_avgs.mat')
    
    plot( stim_cell_median(CELL_OF_INTEREST,:)-allcontrolmed );  hold on;
    plot(sqrt(stim_cell_rms(CELL_OF_INTEREST,:))-allcontrolstd);
    axis([2 166 -3 3]);
    title(['Cell #:' num2str(CELL_OF_INTEREST)]);  
    drawnow;
    saveas(gcf, ['Cell_' num2str(CELL_OF_INTEREST) '_subs.svg']);
end


AmpResp = nan(size(std_dev_sub,1),1);
MedianResp = nan(size(std_dev_sub,1),1);
TTPResp = nan(size(std_dev_sub,1),1);
PrestimVal = nan(size(std_dev_sub,1),1);

ControlAmpResp = nan(size(std_dev_sub,1),1);
ControlMedianResp = nan(size(std_dev_sub,1),1);
ControlPrestimVal = nan(size(std_dev_sub,1),1);
hz=17.85;
allinds = 1:length(std_dev_sub);
for i=1:size(std_dev_sub,1)
 waitbar(i/size(std_dev_sub,1),THEwaitbar,'Analyzing subtracted signals...');

    if ~all( isnan(std_dev_sub(i,2:end)) ) && valid(i)
        
        % Stimulus with control subtracted
        std_dev_sig = std_dev_sub(i,2:end);        
        nanners = ~isnan(std_dev_sig);
        firstind=find(cumsum(nanners)>0);
        [~, lastind] = max(cumsum(nanners)-sum(nanners));
        interpinds = firstind(1):lastind;
        goodinds = allinds(nanners);
        std_dev_sig = interp1(goodinds, std_dev_sig(nanners), interpinds, 'linear');
        filt_stddev_sig = std_dev_sig;

        median_sig = median_sub(i,2:end);
        nanners = ~isnan(median_sig);
        firstind=find(cumsum(nanners)>0);
        [~, lastind] = max(cumsum(nanners)-sum(nanners));
        interpinds = firstind(1):lastind;
        goodinds = allinds(nanners);
        median_sig = interp1(goodinds, median_sig(nanners), interpinds, 'linear');
        filt_median_sig = median_sig;

        critical_filt = filt_stddev_sig( CRITICAL_REGION );
        
        [~, TTPResp(i)] = max( abs(critical_filt) );    
        AmpResp(i) = quantile(critical_filt,0.95);
        MedianResp(i) = max(abs(filt_median_sig(CRITICAL_REGION))-mean(filt_median_sig(1:CRITICAL_REGION(1))) );        
        PrestimVal(i) = mean( stim_prestim_means(i,:),2, 'omitnan');
        
        if any(i==CELL_OF_INTEREST)
           figure(1);
           subplot(2,1,1);
           plot( interpinds/hz, filt_stddev_sig ); hold on; plot(interpinds/hz,std_dev_sig);
           plot( interpinds/hz,filt_median_sig ); plot(interpinds/hz, median_sig);
%            plot(sqrt(control_cell_var(i,2:end))-1);
%            plot(sqrt(stim_cell_var(i,2:end))-1);
%            plot(mean(sqrt(control_cell_var),'omitnan'));
           title(['#: ' num2str(i) ' A: ' num2str(AmpResp(i)) ', M: ' num2str(MedianResp(i)) ', LogResp: ' num2str(log10(AmpResp(i)+MedianResp(i)+1)) ]);
           axis([0 9 -5 10]); hold off;
           
           subplot(2,1,2); 
           plot((0:69)/hz, criticalfit(i,:));hold on;
           plot(densitometry_vect_times{i},densitometry_vect_ref{i},'.'); hold off;           
           title(['FitAmp: ' num2str(densitometry_fit_amplitude(i))]);axis([0 2 0 1.5]);
           drawnow; 
           
%            saveas(gcf, ['Cell_' num2str(i) '_stimulus.png']);
        end
        
        % Control only
        std_dev_sig = control_std_dev_sub(i,2:end);
        nanners = ~isnan(std_dev_sig);
        firstind=find(cumsum(nanners)>0);
        [~, lastind] = max(cumsum(nanners)-sum(nanners));
        interpinds = firstind(1):lastind;
        goodinds = allinds(nanners);
        std_dev_sig = interp1(goodinds, std_dev_sig(nanners), interpinds, 'linear');
        filt_stddev_sig = std_dev_sig;

        
        median_sig = control_median_sub(i,2:end);        
        nanners = ~isnan(median_sig);        
        firstind=find(cumsum(nanners)>0);
        [~, lastind] = max(cumsum(nanners)-sum(nanners));
        interpinds = firstind(1):lastind;
        goodinds = allinds(nanners);        
        median_sig = interp1(goodinds, median_sig(nanners), interpinds, 'linear');
        filt_median_sig = median_sig;


        critical_filt = filt_stddev_sig( CRITICAL_REGION );
        
        ControlAmpResp(i) = quantile(critical_filt,0.95);
        ControlMedianResp(i) = max(abs(filt_median_sig(CRITICAL_REGION))-mean(filt_median_sig(1:CRITICAL_REGION(1))) );        
        ControlPrestimVal(i) = mean( cont_prestim_means(i,:),2, 'omitnan');
%         histogram(filt_stddev_sig(CRITICAL_REGION),20)

    end
end
%%
close(THEwaitbar);
%%
save([ outFname '.mat'], 'AmpResp','MedianResp','TTPResp',...
     'ControlAmpResp','ControlMedianResp','ControlPrestimVal',...
     'valid','allcoords','ref_image','control_cell_median',...
     'control_cell_var','stim_cell_median','stim_cell_var','stim_prestim_means');

 
 

%% Plot the pos/neg ratio of the mean vs the amplitude
% posnegratio=nan(size(control_coords,1),1);
% 
% 
% figure(101); clf; hold on;
% for i=1:size(control_coords,1)
%     if ~isnan(AmpResp(i))
%         % Find out what percentage of time the signal spends negative
%         % or positive after stimulus delivery (66th frame)
% %         numposneg = sign(mean_sub(i,:));
% %         pos = sum(numposneg == 1);
% % 
% %         posnegratio(i) = 100*pos/length(numposneg);
% 
%         plot( AmpResp(i), MedianResp(i),'k.');        
%     end
% end
% ylabel('Median response amplitude');
% xlabel('Reflectance response amplitude');
% title('Median reflectance vs reflectance response amplitude')
% hold off;
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
