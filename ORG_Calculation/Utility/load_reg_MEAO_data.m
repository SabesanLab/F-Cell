function [temporal_data, framestamps, reference_coordinates, mask_data, reference_image]=load_reg_MEAO_data(temporal_data_filename, varargin)

p = inputParser;

addRequired(p,'filename', @ischar);
addParameter(p,'ReferenceModality', 'Confocal', @ischar);
addParameter(p,'LoadCoordinates', true, @islogical);
addParameter(p,'LoadReferenceImage', true, @islogical);
addParameter(p,'LoadMasks', true, @islogical);

% Parse our inputs.
parse(p,temporal_data_filename,varargin{:})

ref_modality = p.Results.ReferenceModality;
load_coords = p.Results.LoadCoordinates;
load_ref_im = p.Results.LoadReferenceImage;
load_masks = p.Results.LoadMasks;

%Grab the base path provided; all other paths relevant to it can be derived
%from it.
[parentpath, filename] = getparent(temporal_data_filename);

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
    coordfile = fullfile(parentpath,[common_prefix ref_modality '1_extract_reg_avg_coords.csv']);
    if exist(coordfile,'file')
        reference_coordinates = dlmread(coordfile);
    else
        warning(['Coordinate file: ' coordfile ' Not found.']);
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
    maskfile = fullfile(parentpath,[temporal_data_filename(1:end-4) '_mask.avi']);
    if exist(maskfile,'file')
        mask_data_reader = VideoReader( fullfile(parentpath, maskfile) );

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
        warning(['Mask video file: ' maskfile ' Not found.']);
    end    
end

[temporal_data] = Residual_Torsion_Removal_Pipl(temporal_data, mask_data);


end