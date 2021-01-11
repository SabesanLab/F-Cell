% function []=Pixelwise_Temporal_Reflectivity_Analysis(mov_path, this_image_fname, stimulus_frames, vid_type)
% []=Pixelwise_Temporal_Reflectivity_Analysis(mov_path, ref_image_fname)
% Robert F Cooper 06-20-2017
%
% @param mov_path - The path to the video which stores the reflectance
% information. Must have an associated file that ends with
% '_acceptable_frames.csv'. That file contains a list of the frame numbers 
% (time points) which were extracted from the original, raw video. If, 
% for example, frame 4 from a 10 frame video did not register, 
% the file would contain the list: [0 1 2 4 5 6 7 8 9].
%
% @param ref_image_fname - The path to the reference image. Must have an
% associated coordinate file that ends in '_coords.csv'.
%
% @param stimulus_frames - A 2 element vector containing the temporal bound
% when the stimulus was delivered, inclusive. (ex: A [67 99] means that a
% stimulus was delivered from the 67th to the 99th frame.
%
% @param vid_type - The type of the video. This can either be the string
% 'control' or 'stimulus'.
%
% This software is responsible for the data processing of temporal
% datasets, by taking the data from separate stim folders and a control
% folders.
% This is designed to work with FULL FIELD datasets, and only performs a
% a normalization across the entire frame, with a standardization relative
% to the prestimulus behavior of EACH cell.
%
% 
%
clear
% if ~exist('contains','builtin')
%     contains = @(t,p)~isempty(strfind(t,p));
% end

% *** Constants ***
%
% The shape used to isolate the reflectance at each time point. Box is
% best choice, as it is both the fastest and most reliable.
profile_method = 'box';

% For release, this version only contains the normalizations used in the
% paper. However, the code is structured such that you can add more if desired.
norm_type = 'global_norm_linear_prestim_stdiz'; 

% mov_path=pwd;
if ~exist('mov_path','var') || ~exist('this_image_fname','var')
    close all force;
    [this_image_fname, mov_path]  = uigetfile(fullfile(pwd,'*.tif'));
    stimulus_frames=[72 90];
    
end
ref_image_fname = this_image_fname;
ref_coords_fname = [this_image_fname(1:end-4) '_coords.csv'];
stack_fnames = read_folder_contents( mov_path,'avi' );

for i=1:length(stack_fnames)
    if ~isempty( strfind( stack_fnames{i}, this_image_fname(1:end - length('_AVG.tif') ) ) )
        temporal_stack_fname = stack_fnames{i};

        acceptable_frames_fname = [stack_fnames{i}(1:end-4) '_acceptable_frames.csv'];
        break;
    end
end


%% Load the dataset(s)

ref_image  = double(imread(  fullfile(mov_path, ref_image_fname) ));

stimd_images = dlmread( fullfile(mov_path,acceptable_frames_fname) );
stimd_images = sort(stimd_images)' +1; % For some dumb reason it doesn't store them in the order they're put in the avi.

[X_grid, Y_grid]= meshgrid(1:size(ref_image,2), 1:size(ref_image,1));

ref_coords = [X_grid(:) Y_grid(:)];

temporal_stack_reader = VideoReader( fullfile(mov_path,temporal_stack_fname) );

i=1;
while(hasFrame(temporal_stack_reader))
    temporal_stack(:,:,i) = double(readFrame(temporal_stack_reader));
    i = i+1;
end


%% Find the frames where the stimulus was on, based on the acceptable frames.
stim_inds = find(stimd_images>=stimulus_frames(1) & stimd_images<=stimulus_frames(2));

% Make them relative to their actual temporal locations.
stim_times = stimd_images(stim_inds);


%% Isolate individual profiles
ref_coords = round(ref_coords);

cellseg = cell(size(ref_coords,1),1);
cellseg_inds = cell(size(ref_coords,1),1);

roiradius = 2;
    
im_size=size(ref_image);
wbh = waitbar(0,'Segmenting coordinates...');
parfor i=1:size(ref_coords,1)

%     waitbar(i/size(ref_coords,1),wbh, ['Segmenting coordinate ' num2str(i)]);
        


    if (ref_coords(i,1) - roiradius) > 1 && (ref_coords(i,1) + roiradius) < im_size(2) &&...
       (ref_coords(i,2) - roiradius) > 1 && (ref_coords(i,2) + roiradius) < im_size(1)

        [R, C ] = meshgrid((ref_coords(i,2) - roiradius) : (ref_coords(i,2) + roiradius), ...
                           (ref_coords(i,1) - roiradius) : (ref_coords(i,1) + roiradius));

        cellseg_inds{i} = sub2ind(im_size, R, C );

        cellseg_inds{i} = cellseg_inds{i}(:);

    end

end

clear X Y R C

%% Crop the analysis area to a certain size
% cropsize = 7.5;
% croprect = [0 0 cropsize/.4678 cropsize/.4678];
% 
% figure(cropfig); 
% title('Select the crop region for the STIMULUS cones');
% h=imrect(gca, croprect);
% stim_mask_rect = wait(h);
% close(cropfig)
% stim_mask = poly2mask([stim_mask_rect(1) stim_mask_rect(1)                   stim_mask_rect(1)+stim_mask_rect(3) stim_mask_rect(1)+stim_mask_rect(3) stim_mask_rect(1)],...
%                       [stim_mask_rect(2) stim_mask_rect(2)+stim_mask_rect(4) stim_mask_rect(2)+stim_mask_rect(4) stim_mask_rect(2)                   stim_mask_rect(2)],...
%                       size(colorcoded_im,1), size(colorcoded_im,2));
% 
% cropfig = figure(1); 
% imagesc( uint8(colorcoded_im) ); axis image; title('Select the crop region for the CONTROL cones');
% h=imrect(gca, croprect);
% control_mask_rect = wait(h);
% close(cropfig)
% control_mask = poly2mask([control_mask_rect(1) control_mask_rect(1)                      control_mask_rect(1)+control_mask_rect(3) control_mask_rect(1)+control_mask_rect(3) control_mask_rect(1)],...
%                          [control_mask_rect(2) control_mask_rect(2)+control_mask_rect(4) control_mask_rect(2)+control_mask_rect(4) control_mask_rect(2)                      control_mask_rect(2)],...
%                          size(colorcoded_im,1), size(colorcoded_im,2));

%% Extract the raw reflectance of each cell.

% coords_used = ref_coords(~cellfun(@isempty,cellseg_inds), :);
% cellseg_inds = cellseg_inds(~cellfun(@isempty,cellseg_inds));


cell_reflectance = cell( length(cellseg_inds),1  );
cell_times = cell( length(cellseg_inds),1  );

j=1;

if ~ishandle(wbh)
    wbh = waitbar(0.25, 'Creating reflectance profile for cells...');
end

vid_size = size(temporal_stack(:,:,1));
% if isempty(gcp)
%     myPool=parpool;
% else
%     myPool=gcp('nocreate');
% end

parfor i=1:length(cellseg_inds)
%     waitbar(i/length(cellseg_inds),wbh, ['Creating reflectance profile for cell: ' num2str(i)]);

    cell_times{i} = stimd_images;
    cell_reflectance{i} = zeros(1, size(temporal_stack,3));

        [m,n] = ind2sub(vid_size, cellseg_inds{i});
        
        thisstack = temporal_stack(min(m):max(m), min(n):max(n),:);        
        
        thisstack(thisstack == 0) = NaN;
        
        thisstack = sum(thisstack,1);
        thisstack = squeeze(sum(thisstack,2));
        thisstack = thisstack./ (length(cellseg_inds{i})*ones(size(thisstack)));
        
        cell_reflectance{i} = thisstack';
   
end
close(wbh);

%% Find the means / std devs
cell_ref = cell2mat(cell_reflectance);
cellinds = find( ~all(isnan(cell_ref),2) );
cell_ref = cell_ref( cellinds, :); % Remove any cells that have NaNs.

for t=1:size(cell_ref,2)
    ref_mean(t) = mean(cell_ref( ~isnan(cell_ref(:,t)) ,t));        
    ref_hist(t,:) = hist(cell_ref( ~isnan(cell_ref(:,t)) ,t),255);    
    ref_std(t) = std(cell_ref( ~isnan(cell_ref(:,t)) ,t));
end



%% Normalization to the mean
norm_cell_reflectance = nan( [length(cell_reflectance), length(cell_reflectance{1})] );
cell_prestim_mean = nan(size(cell_reflectance));

for i=1:length( cell_reflectance )
    
    if contains(  norm_type, 'global_norm' )
        norm_cell_reflectance(i,:) = cell_reflectance{i} ./ ref_mean;
    elseif contains(  norm_type, 'no_norm' )
        norm_cell_reflectance(i,:) = cell_reflectance{i};
        warn('No normalization selected!')
    end

%     no_ref = ~isnan(norm_cell_reflectance{i});
    
%     if all(~no_ref)
%         norm_cell_reflectance{i} = norm_cell_reflectance{i}(no_ref);
%         cell_times{i}       = cell_times{i}(no_ref);
%     end
       
end

notnans= ~all(isnan(norm_cell_reflectance),2);



% Why not consider summed normalized reflectance to be the total number of photons hitting
% detector? Seems like it'd be less sensitive to outliers AND moving
% profiles.
% coi = 4;
% temporalhist = zeros(length(0:.1:3.5)-1,size(temporal_stack,3));
% for i=1:size(temporal_stack,3)
%     [temporalhist(:,i), edges, bins]= histcounts(cellseg{coi}(:,:,i)./ref_mean(i), 0:.1:3.5);
%    figure(1); imagesc( cellseg{coi}(:,:,i) ); colormap gray; axis image; title(['Timepoint ' num2str(i) ' maxval: ' num2str(sum(sum(cellseg{1}(:,:,i)./ref_mean(i),'omitnan'),'omitnan'))]);
%    pause();
% 
% end


%% Standardization

% Then normalize to the average intensity of each cone BEFORE stimulus.
prestim_std=nan(1,length( norm_cell_reflectance ));
prestim_mean=nan(1,length( norm_cell_reflectance ));

maxtime =stim_times(1);



parfor i=1:length( norm_cell_reflectance )

    if notnans(i)
        
        
%         prestim_sig = norm_cell_reflectance(i, 1:stim_times(1)) & ~isnan( norm_cell_reflectance(i,1:stim_times(1)) );
%         prestim_time = cell_times{i}(cell_times{i}<stim_times(1))/17.85;
% 
%         linreg = [prestim_time; ones(size(prestim_time))]'\prestim_sig';
% 
%         prestim_sig = prestim_sig-(linreg(2)+prestim_time.*linreg(1));
% 
%         prestim_mean(i) = mean( norm_cell_reflectance(i, cell_times{i}<stim_times(1) & ~isnan( norm_cell_reflectance{i} ) ) );
% 
%         norm_cell_reflectance(i,:) = norm_cell_reflectance(i,:)-prestim_mean(i);
% 
%         prestim_std(i) = std( prestim_sig );
% 
%         norm_cell_reflectance(i,:) = norm_cell_reflectance(i,:)/( prestim_std(i) ); % /sqrt(length(norm_control_cell_reflectance{i})) );

        
%         elseif contains( norm_type, 'prestim_stdiz')
        % Then normalize to the average intensity of each cone BEFORE stimulus.
        

        sig = norm_cell_reflectance(i, : );
        prestim_mean(i) = mean(sig(1:maxtime) , 'omitnan' );

        norm_cell_reflectance(i,:) = norm_cell_reflectance(i,:)-prestim_mean(i);

        prestim_std(i) = std( sig( 1:maxtime ), 'omitnan' );

%         norm_cell_reflectance(i,:) = norm_cell_reflectance(i,:)/prestim_std(i); % /sqrt(length(norm_control_cell_reflectance{i})) );

    end
end

norm_cell_reflectance = norm_cell_reflectance ./ median(prestim_std,'omitnan');

% delete(myPool)

clear cell_reflectance;

norm_cell_reflectance(isinf(norm_cell_reflectance)) = NaN;

maxresp = max(abs(norm_cell_reflectance(notnans,:)),[],2);


%% output this stuff to a video...
vidObj = VideoWriter([this_image_fname(1:end-4) '_resp_video.avi']);
open(vidObj);


all_ref_low =abs(norm_cell_reflectance);


minall = min(all_ref_low(:));
maxall = max(all_ref_low(:)-minall);
for i=1:size(all_ref_low,2)

    scaled_norm_chg_im = reshape(all_ref_low(:,i)-min(all_ref_low(:)), size(ref_image,1),size(ref_image,2));
    scaled_norm_chg_im = scaled_norm_chg_im ./maxall;
    
    writeVideo(vidObj, scaled_norm_chg_im);
end

close(vidObj);
return;

%% Remove the all nan rows, drop the first never used index.
% Explains 95% of variance, selecting the first 5 components
tic;
[COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU] = pca(norm_cell_reflectance(notnans, :),'Algorithm','als', 'NumComponents',10);
toc;

all_ref(notnans,:) = SCORE*COEFF';
all_ref_low = zeros(size(norm_cell_reflectance,1),151);
all_ref_low(notnans,:) = SCORE(:,1)*COEFF(:,1)';
% save('Piecewise_PCA_run.mat', '-v7.3');

%% Conversion of cells to complete matrix.
norm_cell_reflectance_mat = nan(length(norm_cell_reflectance),161);



for c=1:length(norm_cell_reflectance)
    for t=1:length(cell_times{c})
    
       norm_cell_reflectance_mat(c, cell_times{c}-1) = norm_cell_reflectance(c,:);
    
    end
end
norm_cell_reflectance_mat(:,1) =[];
all_ref_low = zeros(size(norm_cell_reflectance_mat,1),161);
notnans = ~all(isnan(norm_cell_reflectance_mat),2);

tic;
[COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU] = pca(norm_cell_reflectance_mat(notnans, :),'Algorithm','als', 'NumComponents',2);
toc;

all_ref_low(notnans,:) = SCORE(:,1:2)*COEFF(:,1:2)';


%% Standard deviation of all cells before first stimulus

thatmax = max( cellfun(@max, cell_times( ~cellfun(@isempty,cell_times)) ) );   

[ ref_stddev,ref_times ] = reflectance_std_dev( cell_times( ~cellfun(@isempty,cell_times) ), ...
                                                norm_cell_reflectance( ~cellfun(@isempty,norm_cell_reflectance) ), thatmax );

clipped_ref_times = [];

if ~isempty(ref_times)
    i=1;
    while i<= length( ref_times )

        % Remove timepoints from cells that are NaN
        if isnan(ref_times(i))
            ref_times(i) = [];
            ref_stddev(i) = [];        
        else
            clipped_ref_times = [clipped_ref_times; ref_times(i)];
            i = i+1;
        end

    end    
end


for i=1:length(norm_cell_reflectance)
   
    
    
end

hz=16.6;

figure(10); 
if ~isempty(ref_stddev)
    plot( clipped_ref_times/hz,ref_stddev,'k'); hold on;
end
if strcmp(vid_type,'stimulus')
    plot(stim_times/hz, max(ref_stddev)*ones(size(stim_times)),'r*'); 
end
hold off;
ylabel('Standard deviation'); xlabel('Time (s)'); title( strrep( [this_image_fname(1:end - length('_AVG.tif') ) '_' profile_method '_stddev_ref_plot' ], '_',' ' ) );
axis([0 15 -1 4])

% ref_image_fname = strrep(ref_image_fname,'confocal','split_det');
if ~exist( fullfile(mov_path, 'Std_Dev_Plots'), 'dir' )
    mkdir(fullfile(mov_path, 'Std_Dev_Plots'))
end
saveas(gcf, fullfile(mov_path, 'Std_Dev_Plots' , [this_image_fname(1:end - length('_AVG.tif') ) '_' profile_method '_' norm_type '_' vid_type '_stddev.png' ] ) );

if ~exist( fullfile(mov_path, 'Profile_Data'), 'dir' )
    mkdir(fullfile(mov_path, 'Profile_Data'))
end
% Dump all the analyzed data to disk
% save(fullfile(mov_path, 'Profile_Data' ,[this_image_fname(1:end - length('_AVG.tif') ) '_' profile_method '_' norm_type '_' vid_type '_profiledata.mat']), ...
%      'this_image_fname', 'cell_times', 'norm_cell_reflectance','ref_coords','ref_image','ref_mean','ref_stddev','vid_type','cell_prestim_mean','cell_reflectance' );

  
