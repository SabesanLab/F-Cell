function [iORG]=Simple_Population_iORG(temporal_data, framestamps, reference_coordinates, prestimulus_range, fps, varargin)

p = inputParser;

checkFramestamps = @(x) isnumeric(x) && length(framestamps)==size(temporal_data,3);
checkcoords = @(x) size(x,2) == 2; % the coordinate list should be Nx2.
checkrange = @(x) size(x,1) == 1 && size(x,2) == 2; % the prestimulus range should be 1x2.

addRequired(p,'temporal_data', @isnumeric)
addRequired(p,'framestamps', checkFramestamps)
addOptional(p,'reference_coordinates', checkcoords);
addOptional(p,'prestimulus_range', checkrange);
addOptional(p,'fps', @isnumeric);

% Parse our inputs.
parse(p,temporal_data,framestamps,varargin{:})

wbh = waitbar(0,'Segmenting coordinates...');

[temporal_profiles]=extract_temporal_profiles(temporal_data,'Coordinates', reference_coordinates, ...
                                                            'SegmentationMethod','box',...
                                                            'SegmentationRadius',1,...
                                                            'ExtractionMethod', 'mean',...
                                                            'ProgBarHandle', wbh );

[norm_temporal_profiles]=framewise_normalize_temporal_profiles(temporal_profiles, 'ProgBarHandle', wbh);

if isempty(prestimulus_range)
   prestimulus_range=[1 framestamps(end)];
end

[stdz_temporal_profiles]=standardize_temporal_profiles(norm_temporal_profiles, framestamps, prestimulus_range, fps,'Method', 'relative_change', 'ProgBarHandle', wbh);

iORG=Population_iORG(stdz_temporal_profiles, framestamps/fps,'SummaryMethod','var','WindowSize',5);

close(wbh) 
end
