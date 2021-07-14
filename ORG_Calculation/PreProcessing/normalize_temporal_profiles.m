function [norm_cell_reflectance]=extract_temporal_profiles(temporal_data, varargin)


p = inputParser;

dataset_size = size(temporal_data);

if length(dataset_size) > 3
   fourDeeData = true; 
end

defaultextmethod = 'frame';
validextmethods = {'mean', 'median'};
checkExtMethods = @(x) any(validatestring(x,validextmethods));


addRequired(p,'temporal_data', @isnumeric)
addOptional(p,'Method', defaultextmethod, checkExtMethods);
addOptional(p,'ProgBarHandle', [], @ishandle);

% Parse our inputs.
parse(p,temporal_data,varargin{:})

segmentation_method = p.Results.SegmentationMethod;
extraction_method = p.Results.ExtractionMethod;
roiradius = p.Results.SegmentationRadius;
coordinates = p.Results.Coordinates;

if ~isempty(p.Results.ProgBarHandle)
    wbh = p.Results.ProgBarHandle;
else
    wbh = waitbar(0,'Segmenting coordinates...');
end

for i=1:length( cell_reflectance )
    
    if contains(  norm_type, 'global_norm' )
        norm_cell_reflectance{i} = cell_reflectance{i} ./ ref_mean;
    elseif contains(  norm_type, 'regional_norm' )
        norm_cell_reflectance{i} = cell_reflectance{i} ./ ref_mean;
    end
end

% If we didn't supply a progress bar, close it at the end to be a good
% neighbor.
if isempty(p.Results.ProgBarHandle)
   close(wbh) 
end

end