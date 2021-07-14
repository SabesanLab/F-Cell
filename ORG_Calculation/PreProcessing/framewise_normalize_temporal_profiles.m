function [norm_temporal_profiles, norm_values]=framewise_normalize_temporal_profiles(temporal_profiles, varargin)


p = inputParser;

dataset_size = size(temporal_profiles);

if length(dataset_size) > 3
   fourDeeData = true;
   warning('4D data is not yet supported.');
end

defaultregion = 'population';
validregions = {'population', 'regional'};
checkRegion = @(x) any(validatestring(x,validregions));

defaultextmethod = 'mean';
validextmethods = {'mean'};
checkExtMethods = @(x) any(validatestring(x,validextmethods));

addRequired(p,'temporal_profiles', @isnumeric)

addParameter(p,'Region', defaultregion, checkRegion);
addParameter(p,'Method', defaultextmethod, checkExtMethods);
addParameter(p,'ProgBarHandle', [], @ishandle);

% Parse our inputs.
parse(p,temporal_profiles,varargin{:})

region = p.Results.Region;
method = p.Results.Method;

if ~isempty(p.Results.ProgBarHandle)
    wbh = p.Results.ProgBarHandle;
else
    wbh = waitbar(0,'Framewise normalizing coordinates...');
end

switch method
    case 'mean'
        norm_values = mean(temporal_profiles,1,'omitnan');
    otherwise
        norm_values = mean(temporal_profiles,1,'omitnan');
end


switch region
    case 'population'
        norm_temporal_profiles = temporal_profiles ./ norm_values;
        
end

% If we didn't supply a progress bar, close it at the end to be a good
% neighbor.
if isempty(p.Results.ProgBarHandle)
   close(wbh) 
end

end