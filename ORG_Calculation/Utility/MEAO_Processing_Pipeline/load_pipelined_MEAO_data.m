function [temporal_data, framestamps, framerate, reference_coordinates]=load_pipelined_MEAO_data(temporal_data_path, varargin)
% function [...]=LOAD_PIPELINED_MEAO_DATA(temporal_data_path, ...)
%
% 	This function is designed to load in a temporal dataset that has been 
%   run though the MEAO_Functional_Imaging_Pipeline
%
% 	[temporal_data, framestamps, framerate]=LOAD_REG_MEAO_DATA(temporal_data_path) 
%   loads in an MEAOSLO temporal dataset into an NxMxT array,
%	alongside both the "frame stamps" which correspond to the indexes of the
%   frames from the original video that were included in the pipelined
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
%		For example, if one obtains a pipelined temporal dataset named 
%       "MEAO_dataset_760nm1_extract_reg_cropped_small.avi", then the function form 
%		will assume you have included a coordinate file named 
%       "MEAO_dataset_Confocal1_extract_reg_avg_coords.csv" generated 
%       from a reference image "MEAO_dataset_Confocal1_extract_reg_avg.tif"
%
%		If this coordinate file cannot be found, an empty array will be
%       returned instead.
%
%
% 	[temporal_data, framestamps, framerate, reference_coordinates, reference_image]=
%                                    LOAD_REG_MEAO_DATA(temporal_data_path) 
%	in addition to the above, the function returns the reference image 
%   from the temporal dataset, as created via the pipeline.
%
%		For example, if one obtains a pipelined temporal dataset named 
%       "MEAO_dataset_760nm1_extract_reg_cropped_small.avi", then the function
%       will load the reference image "MEAO_dataset_Confocal1_extract_reg_avg.tif"
%
%
% 	[...]=LOAD_REG_MEAO_DATA(temporal_data_path, 'PARAM1', VALUE1, 'PARAM2', VALUE2,...) 
%   loads in an MEAOSLO temporal dataset into an NxMxT array, using named
%   parameters altered to specific values. Parameters and values may be:
%


p = inputParser;


addRequired(p,'temporal_data_path', @ischar);
addParameter(p,'LoadCoordinates', false, @islogical);

% Parse our inputs.
parse(p,temporal_data_path,varargin{:})

load_coords = p.Results.LoadCoordinates;

%Grab the base path provided; all other paths relevant to it can be derived
%from it.
[parentpath, filename] = getparent(temporal_data_path);

under_indices=regexp(filename,'_');

reference_coordinates=[];
if load_coords
    coordfile_base = fullfile(parentpath,[filename(1:end-4) '_coords.csv']);
    if exist(coordfile_base,'file')
        reference_coordinates = dlmread(coordfile_base);
    else
        warning(['Coordinate file: ' coordfile_base ' (or ' coordfile_ref ' ) Not found.']);
    end
end

regdata = readtable(fullfile(parentpath, [filename(1:end-3) 'csv'] ));
framestamps = regdata.FrameStamps; % The framestamps column.

temporal_data_reader = VideoReader( fullfile(parentpath, filename) );
framerate = temporal_data_reader.FrameRate;

num_frames = round(temporal_data_reader.Duration*temporal_data_reader.FrameRate);
% For this to work, you MUST install LAVFilters codecs! https://github.com/Nevcairiel/LAVFilters
temporal_data = zeros(temporal_data_reader.Height, temporal_data_reader.Width, num_frames);

for f=1:num_frames
    temporal_data(:,:,f) = readFrame(temporal_data_reader);
end

delete(temporal_data_reader)


end