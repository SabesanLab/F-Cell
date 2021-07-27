function [temporal_data, framestamps, framerate, reference_coordinates, mask_data, reference_image]=load_reg_MEAO_data(temporal_data_path, varargin)
% function [...]=LOAD_REG_MEAO_DATA(temporal_data_path, ...)
%
% 	This function is designed to load in a temporal dataset obtained from
%   the MEAOSLO designed by Boston Micromachines Corporation. For videos to
%   load, you MUST install LAVFilters codecs, found here: 
%   https://github.com/Nevcairiel/LAVFilters, or any codec set that can load
%   Y800 encoded datasets.
%
% 	[temporal_data, framestamps, framerate]=LOAD_REG_MEAO_DATA(temporal_data_path) 
%   loads in an MEAOSLO temporal dataset into an NxMxT array,
%	alongside both the "frame stamps" which correspond to the indexes of the
%   frames from the original video that were included in the registration
%	step, and the detected framerate of the video.
%
% 	[temporal_data, framestamps, framerate, reference_coordinates]=
%                                   LOAD_REG_MEAO_DATA(temporal_data_path)
%	in addition to the above, the functon also loads in any coordinates 
%   that can be used in following steps (see EXTRACT_TEMPORAL_PROFILES).
%
%		** IMPORTANT: The coordinate file is expected to share a common 
%       path root with the registered tif from the 'Confocal' modality.
%
%		For example, if one obtains a registered temporal dataset named 
%       "MEAO_dataset_760nm1_extract_reg_cropped_small.avi", then the function form 
%		will assume you have included a coordinate file named 
%       "MEAO_dataset_Confocal1_extract_reg_avg_coords.csv" generated 
%       from a reference image "MEAO_dataset_Confocal1_extract_reg_avg.tif"
%
%		If this coordinate file cannot be found, an empty array will be
%       returned instead.
%
% 	[temporal_data, framestamps, framerate, reference_coordinates, mask_data]=
%                                    LOAD_REG_MEAO_DATA(temporal_data_path) 
%	in addition to the above, the function returns any associated masking 
%   data (in an NxMxT array) from the temporal dataset.
%
%		** Similar to above, the mask video file is expected to share a 
%       common path root with the registered video in 'temporal_data_path'.
%
%		For example, if one obtains a registered temporal dataset named 
%       "MEAO_dataset_760nm1_extract_reg_cropped_small.avi", then the function form 
%		will expect the existence of a 
%       "MEAO_dataset_760nm1_extract_reg_cropped_small_mask.avi"
%
% 	[temporal_data, framestamps, framerate, reference_coordinates, mask_data, reference_image]=
%                                    LOAD_REG_MEAO_DATA(temporal_data_path) 
%	in addition to the above, the function returns the reference image 
%   from the temporal dataset.
%
%		** Similar to above, the mask video file is expected to share a 
%       common path root with the registered video in 'temporal_data_path'.
%
%		For example, if one obtains a registered temporal dataset named 
%       "MEAO_dataset_760nm1_extract_reg_cropped_small.avi", then the function
%       will load the reference image "MEAO_dataset_Confocal1_extract_reg_avg.tif"
%
%
% 	[...]=LOAD_REG_MEAO_DATA(temporal_data_path, 'PARAM1', VALUE1, 'PARAM2', VALUE2,...) 
%   loads in an MEAOSLO temporal dataset into an NxMxT array, using named
%   parameters altered to specific values. Parameters and values may be:
%
%       LoadCoordinates - Specifies whether or not to load the coordinates.
%       [{'true'} | 'false']
%
%       LoadMasks - Specifies whether or not to load (and use) the associated video mask.
%       [{'true'} | 'false']
%
%       ReferenceModality - Defines the string embedded in the reference modality filename.
%       [{'Confocal'} | character vector]
%
%       ReferenceImage - Specifies the method for obtaining a reference
%       image. Generating a reference image (default) creates a reference
%       from the input dataset. Otherwise, it is loaded from disk using the
%       reference modality string.
%       [{'generated'} | 'loaded']

p = inputParser;

refimage = 'generated';
validrefimage = {'generated', 'loaded'};
checkrefimage = @(x) any(validatestring(x,validrefimage));

addRequired(p,'temporal_data_path', @ischar);
addParameter(p,'ReferenceModality', 'Confocal', @ischar);
addParameter(p,'LoadCoordinates', true, @islogical);
addParameter(p,'LoadMasks', true, @islogical);
addParameter(p,'ReferenceImage', refimage, checkrefimage);

% Parse our inputs.
parse(p,temporal_data_path,varargin{:})

ref_modality = p.Results.ReferenceModality;
load_coords = p.Results.LoadCoordinates;
ref_im = p.Results.ReferenceImage;
load_masks = p.Results.LoadMasks;

%Grab the base path provided; all other paths relevant to it can be derived
%from it.
[parentpath, filename] = getparent(temporal_data_path);

under_indices=regexp(filename,'_');

common_prefix = filename(1:under_indices(6));

reference_image=[];
if strcmp(ref_im, 'loaded')
    imfile = fullfile(parentpath,[common_prefix ref_modality '1_extract_reg_avg.tif']);
    if exist(imfile,'file')
        reference_image = imread(imfile);
    else
        warning(['Reference image file: ' imfile ' Not found.']);
    end
end

reference_coordinates=[];
if load_coords
    coordfile_base = fullfile(parentpath,[common_prefix ref_modality '1_extract_reg_avg_coords.csv']);
    coordfile_ref = fullfile(parentpath,[filename(1:under_indices(7)) 'extract_reg_avg_coords.csv']);
    if exist(coordfile_base,'file') % Could draw coordinates from EITHER the base modality, or some reference modality.
        reference_coordinates = dlmread(coordfile_base);
    elseif exist(coordfile_ref,'file')
        reference_coordinates = dlmread(coordfile_ref);
    else
        warning(['Coordinate file: ' coordfile_base ' (or ' coordfile_ref ' ) Not found.']);
    end
end


regdata = readtable(fullfile(parentpath, [filename(1:end-3) 'csv'] ));
framestamps = regdata.OriginalFrameNumber; % The framestamps column.
[~, minind] = min(1-regdata.NCC);
referenceidx = framestamps(minind); % The reference frame should be perfectly correlated to itself; use this as the reference.
% floor(regdata.Strip0_NCC)
temporal_data_reader = VideoReader( fullfile(parentpath, filename) );

framerate = temporal_data_reader.FrameRate;
num_frames = round(temporal_data_reader.Duration*temporal_data_reader.FrameRate);
% For videos to load, you MUST install LAVFilters codecs! https://github.com/Nevcairiel/LAVFilters
temporal_data = zeros(temporal_data_reader.Height, temporal_data_reader.Width, num_frames);

for f=1:num_frames
    temporal_data(:,:,f) = rgb2gray(readFrame(temporal_data_reader));
end

delete(temporal_data_reader)

mask_data=[];
if load_masks
    maskpath = fullfile(parentpath,[filename(1:end-4) '_mask.avi']);
    if exist(maskpath,'file')
        mask_data_reader = VideoReader( maskpath );

        num_mask_frames = round(mask_data_reader.Duration*mask_data_reader.FrameRate);

        if num_mask_frames ~= num_frames
           error('Number of frames in mask video file doesn''t match main temporal dataset!');
        end
        
        mask_data = zeros(mask_data_reader.Height, mask_data_reader.Width, num_mask_frames);

        for f=1:num_mask_frames
            mask_data(:,:,f) = rgb2gray(readFrame(mask_data_reader));
            
        end

        delete(mask_data_reader)
    else
        warning(['Mask video file: ' maskpat ' Not found.']);
    end    
end

% Remove residual distortions and torsion.
shiftheaders = regdata.Properties.VariableNames(4:end-2);
shiftvalues = regdata.Variables;

coarseX = shiftvalues(:,end-1);
coarseY = shiftvalues(:,end);

shiftvalues = shiftvalues(:,4:end-2);

% Find our headers
xshiftheaders = cellfun(@(head)contains(head,'XShift'), shiftheaders);
yshiftheaders = cellfun(@(head)contains(head,'YShift'), shiftheaders);

% Resort these to be in numerical instead of alphabetical order.
striporder = cellfun(@(x)str2double(x(6:end)), cellfun(@(head)strtok(head,'_'), shiftheaders(xshiftheaders), 'UniformOutput', false));
[~, sortind] = sort(striporder);


xshifts = shiftvalues(:,xshiftheaders);
xshifts = xshifts(:,sortind);
xshift_medians = median(xshifts,1);

roweval = linspace(1,length(xshift_medians), size(temporal_data,1));

indivxshift = zeros([num_frames, size(temporal_data,2)]);
%Use a poly8 as this is what BMC's imreg software uses.
for f=1:num_frames

    ind_xshiftfit = fit( (1:length(xshift_medians))',xshifts(f,:)','poly8','Normalize','on','Exclude', isnan(xshifts(f,:)')); 
    
    indivxshift(f,:) = feval(ind_xshiftfit, roweval);
end
% Do the INVERSE of what is represented in the data, to counteract it.
xgriddistortion = repmat(-median(indivxshift,1)', [1 size(temporal_data,2)]);
% figure(1); plot(indivxshift'); hold on; plot(median(indivxshift,1),'r*')

yshifts = shiftvalues(:,yshiftheaders);
yshifts = yshifts(:,sortind);
yshift_medians = median(yshifts,1);

indivyshift = zeros([num_frames, size(temporal_data,2)]);
%Use a poly8 as this is what BMC's imreg software uses. 
for f=1:num_frames
    ind_yshiftfit = fit( (1:length(yshift_medians))',yshifts(f,:)','poly8','Normalize','on','Exclude', isnan(yshifts(f,:)')); 
    indivyshift(f,:) = feval(ind_yshiftfit, roweval);
end
% Do the INVERSE of what is represented in the data, to counteract it.
ygriddistortion = repmat(-median(indivyshift,1)', [1 size(temporal_data,2)]);

% figure(2); plot(indivyshift'); hold on; plot(median(indivyshift,1),'r*')

disp_field = cat(3,xgriddistortion,ygriddistortion);

for f=1:num_frames
    temporal_data(:,:,f) = imwarp(temporal_data(:,:,f), disp_field,'FillValues',0);
end

[temporal_data, framestamps] = Residual_Torsion_Removal_Pipl(temporal_data, framestamps, mask_data, referenceidx);

if strcmp(ref_im, 'generated')
    reference_image = sum(temporal_data,3)./sum(mask_data/255,3);
    reference_image(isinf(reference_image)) = 0;
    reference_image(isnan(reference_image)) = 0;
    figure(3);  imagesc(reference_image); axis image; colormap gray; title(filename);
end

end