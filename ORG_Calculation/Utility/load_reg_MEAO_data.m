function [temporal_data, framestamps, reference_coordinates, mask_data, reference_image]=load_reg_MEAO_data(temporal_data_path, varargin)
% function [...]=LOAD_REG_MEAO_DATA(temporal_data_path, ...)
%
% 	This function is designed to load in a temporal dataset obtained from
%   the MEAOSLO designed by Boston Micromachines Corporation.
%
% 	[temporal_data, framestamps]=LOAD_REG_MEAO_DATA(temporal_data_path) 
%   loads in an MEAOSLO temporal dataset into an NxMxT array,
%	along with the "frame stamps" which correspond to the indexes of the
%   frames from the original video that were included in the registration
%	step.
%
% 	[temporal_data, framestamps, reference_coordinates]=
%                                   LOAD_REG_MEAO_DATA(temporal_data_path)
%	in addition to the above, the functon also loads in any coordinates 
%   that can be used in following steps (see EXTRACT_TEMPORAL_PROFILES).
%		** IMPORTANT: The coordinate file is expected to share a common 
%       path root with the registered tif from the 'Confocal' modality.
%
%		For example, if one obtains a registered temporal dataset named 
%       "MEAO_dataset_760nm1_extract_reg_small.avi", then the function form 
%		will assume you have included a coordinate file named 
%       "MEAO_dataset_Confocal1_extract_reg_avg_coords.csv" generated 
%       from the reference image "MEAO_dataset_Confocal1_extract_reg_avg.tif"
%
%		If this coordinate file cannot be found, an empty array will be
%       returned instead.
%
% 	[temporal_data, framestamps, reference_coordinates, mask_data]=
%                                    LOAD_REG_MEAO_DATA(temporal_data_path) 
%	in addition to the above, the function returns any associated masking 
%   data (in an NxMxT array) from the temporal dataset.
%		** Similar to above, the mask video file is expected to share a 
%       common path root with the registered video in 'temporal_data_path'.
%
%		For example, if one obtains a registered temporal dataset named 
%       "MEAO_dataset_760nm1_extract_reg_small.avi", then the function form 
%		will expect the existence of a 
%       "MEAO_dataset_760nm1_extract_reg_small_mask.avi"
%


p = inputParser;

addRequired(p,'filename', @ischar);
addParameter(p,'ReferenceModality', 'Confocal', @ischar);
addParameter(p,'LoadCoordinates', true, @islogical);
addParameter(p,'LoadReferenceImage', false, @islogical);
addParameter(p,'LoadMasks', true, @islogical);

% Parse our inputs.
parse(p,temporal_data_path,varargin{:})

ref_modality = p.Results.ReferenceModality;
load_coords = p.Results.LoadCoordinates;
load_ref_im = p.Results.LoadReferenceImage;
load_masks = p.Results.LoadMasks;

%Grab the base path provided; all other paths relevant to it can be derived
%from it.
[parentpath, filename] = getparent(temporal_data_path);

under_indices=regexp(filename,'_');

common_prefix = filename(1:under_indices(6));

reference_image=[];
if load_ref_im
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
        warning(['Coordinate file: ' coordfile_base '(or ' coordfile_ref ' ) Not found.']);
    end
end

framestamps = csvread(fullfile(parentpath, [filename(1:end-3) 'csv'] ), 1, 0);
framestamps = framestamps(:,3)';

temporal_data_reader = VideoReader( fullfile(parentpath, filename) );

num_frames = round(temporal_data_reader.Duration*temporal_data_reader.FrameRate);
% For this to work, you MUST install LAVFilters codecs! https://github.com/Nevcairiel/LAVFilters
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

[temporal_data] = Residual_Torsion_Removal_Pipl(temporal_data, mask_data);


end