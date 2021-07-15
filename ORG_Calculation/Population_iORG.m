function []=Population_iORG(temporal_profiles, timestamps, varargin)

window_size = 5;
half_window = 2;

p = inputParser;

addRequired(p, 'temporal_profiles', @isnumeric)
addOptional(p,'timestamps',@isnumeric);

defaultmethod = 'moving_rms';
validmethods = {'moving_rms', 'stddev'};
checkMethods = @(x) any(validatestring(x,validmethods));

addParameter(p, 'SummaryMethod', defaultmethod, checkMethods);

parse(p,temporal_profiles, varargin{:})

method = p.Results.SummaryMethod;

iORG = nan(1, size(temporal_profiles,2));

switch method
    case 'moving_rms'
        padded_profiles = padarray(temporal_profiles, [0 half_window], 'symmetric','both');
        
        for j=1:size(padded_profiles,2)-window_size
            window =  padded_profiles(:, j:j+window_size-1);    

            iORG(j) = rms(window(~isnan(window)));
        end
    case 'stddev'
        iORG = std(temporal_profiles,0,2,'omitnan');

end

plot(timestamps, iORG);

end