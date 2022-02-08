function [population_iORG]=Population_iORG(temporal_profiles, timestamps, varargin)




p = inputParser;

addRequired(p, 'temporal_profiles', @isnumeric)
addOptional(p,'timestamps',@isnumeric);

defaultmethod = 'moving_rms';
validmethods = {'moving_rms', 'var'};
checkMethods = @(x) any(validatestring(x,validmethods));

addParameter(p, 'SummaryMethod', defaultmethod, checkMethods);
addParameter(p, 'WindowSize', 7, @isnumeric);


parse(p,temporal_profiles, varargin{:})

method = p.Results.SummaryMethod;
window_size = p.Results.WindowSize;
half_window = floor(window_size/2);
iORG = nan(1, size(temporal_profiles,2));

switch method
    case 'moving_rms'
        padded_profiles = padarray(temporal_profiles, [0 half_window], 'symmetric','both');
        
        for j=1:size(padded_profiles,2)-window_size
            window =  padded_profiles(:, j:j+window_size-1);    

            iORG(j) = rms(window(~isnan(window)));
        end
    case 'var'
        iORG = var(temporal_profiles,0,1,'omitnan');

end

population_iORG = iORG;

if ~exist('timestamps','var') || isempty(timestamps)
    timestamps=1:length(iORG);
end
figure(707); plot(timestamps, iORG); %axis([0 180 0 4]);

end