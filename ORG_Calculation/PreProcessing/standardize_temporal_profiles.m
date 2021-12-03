function [standardized_profile_data]=standardize_temporal_profiles(temporal_profiles, framestamps, range, fps, varargin)


p = inputParser;

dataset_size = size(temporal_profiles);

if length(dataset_size) > 3
   fourDeeData = true;
   warning('4D data is not yet supported.');
end


defaultmethod = 'linear_stddev';
validmethods = {'linear_stddev', 'stddev','linear_vast','vast','relative_change'};
checkMethods = @(x) any(validatestring(x,validmethods));

addRequired(p,'temporal_profiles', @isnumeric)
addRequired(p,'framestamps',@isnumeric)
addRequired(p,'range',@isnumeric)
addRequired(p,'fps',@isnumeric)

addParameter(p,'Method', defaultmethod, checkMethods);
addParameter(p,'ProgBarHandle', [], @ishandle);

% Parse our inputs.
parse(p,temporal_profiles,framestamps,range,fps,varargin{:})

method = p.Results.Method;

if ~isempty(p.Results.ProgBarHandle)
    wbh = p.Results.ProgBarHandle;
else
    wbh = waitbar(0,'Standardizing temporal profiles...');
end

standardized_profile_data = nan(size(temporal_profiles));
ranged_std=nan(size(temporal_profiles,1),1);
ranged_mean=nan(size(temporal_profiles,1),1);
        
switch method
    case 'linear_stddev'
        % Standardize using Autoscaling preceded by a linear fit to remove
        % any residual low-frequency changes
        for c=1:size(temporal_profiles,1)
            if mod(c, size(temporal_profiles,1)/10) == 0
                waitbar(c/size(temporal_profiles,1),wbh, ['Standardizing temporal profiles (' num2str(c) ' of ' num2str(size(temporal_profiles,1)) ')']);
            end
            
            % Determine the framestamps that fall within our selected
            % range, as well as any nan values in this profile.
            validrange = framestamps>=range(1) & framestamps<=range(2) & ~isnan( temporal_profiles(c,:) );
            
            if sum(validrange) >= (range(2)-range(1))/4
                ranged_sig = temporal_profiles(c, validrange); % Isolate the profile.
                ranged_time = framestamps(validrange) / fps; % Get the timestamps for this signal.

                ranged_mean(c) = mean( ranged_sig ); % Calculate the mean value of the signal before we do anything to it.

                linreg = [ranged_time; ones(size(ranged_time))]'\ranged_sig'; % Do a simple linear regression of the signal in our range.
                delinear_ranged_sig = ranged_sig-(linreg(2)+ranged_time.*linreg(1)); % Subtract that regression from our signal.

                ranged_std(c) = std( delinear_ranged_sig ); % Calculate the standard deviation of that range.

                standardized_profile_data(c,:) = (temporal_profiles(c,:)-ranged_mean(c))/ranged_std(c); % Standardize our signal.
            end
        end
    case 'stddev'
        % Standardize using Autoscaling.
        for c=1:size(temporal_profiles,1)
            if mod(c, size(temporal_profiles,1)/10) == 0
                waitbar(c/size(temporal_profiles,1),wbh, ['Standardizing temporal profiles (' num2str(c) ' of ' num2str(size(temporal_profiles,1)) ')']);
            end
            
            % Determine the framestamps that fall within our selected
            % range, as well as any nan values in this profile.
            validrange = framestamps>=range(1) & framestamps<=range(2) & ~isnan( temporal_profiles(c,:) );
            
            if sum(validrange) >= (range(2)-range(1))/4
                ranged_sig = temporal_profiles(c, validrange); % Isolate the profile.

                ranged_mean(c) = mean( ranged_sig ); % Calculate the mean value of the signal before we do anything to it.
                ranged_std(c) = std( ranged_sig ); % Calculate the standard deviation of that range.

                standardized_profile_data(c,:) = (temporal_profiles(c,:)-ranged_mean(c))/ranged_std(c); % Standardize our signal.
            end
        end
        
    case 'linear_vast' 
        % Standardize using variable stability, or VAST scaling, preceeded by a linear fit: 
        % https://www.sciencedirect.com/science/article/pii/S0003267003000941
        % this scaling is defined as autoscaling divided by the CoV.
        for c=1:size(temporal_profiles,1)  
            if mod(c, size(temporal_profiles,1)/10) == 0
                waitbar(c/size(temporal_profiles,1),wbh, ['Standardizing temporal profiles (' num2str(c) ' of ' num2str(size(temporal_profiles,1)) ')']);
            end
            
            % Determine the framestamps that fall within our selected
            % range, as well as any nan values in this profile.
            validrange = framestamps>=range(1) & framestamps<=range(2) & ~isnan( temporal_profiles(c,:) );
            
            if sum(validrange) >= (range(2)-range(1))/4
                ranged_sig = temporal_profiles(c, validrange); % Isolate the profile.
                ranged_time = framestamps(validrange) / fps; % Get the timestamps for this signal.

                ranged_mean(c) = mean( ranged_sig ); % Calculate the mean value of the signal before we do anything to it.

                linreg = [ranged_time; ones(size(ranged_time))]'\ranged_sig'; % Do a simple linear regression of the signal in our range.
                delinear_ranged_sig = ranged_sig-(linreg(2)+ranged_time.*linreg(1)); % Subtract that regression from our signal.

                ranged_std(c) = std( delinear_ranged_sig ); % Calculate the standard deviation of that range.

                standardized_profile_data(c,:) = ((temporal_profiles(c,:)-ranged_mean(c))/ranged_std(c)) / (ranged_std(c)/ranged_mean(c)); % Standardize our signal.
            end
        end
    case 'vast' 
        % Standardize using variable stability, or VAST scaling: 
        % https://www.sciencedirect.com/science/article/pii/S0003267003000941
        % this scaling is defined as autoscaling divided by the CoV.
        for c=1:size(temporal_profiles,1)
            if mod(c, size(temporal_profiles,1)/10) == 0
                waitbar(c/size(temporal_profiles,1),wbh, ['Standardizing temporal profiles (' num2str(c) ' of ' num2str(size(temporal_profiles,1)) ')']);
            end
            
            % Determine the framestamps that fall within our selected
            % range, as well as any nan values in this profile.
            validrange = framestamps>=range(1) & framestamps<=range(2) & ~isnan( temporal_profiles(c,:) );
            
            if sum(validrange) >= (range(2)-range(1))/4
                ranged_sig = temporal_profiles(c, validrange); % Isolate the profile.

                ranged_mean(c) = mean( ranged_sig ); % Calculate the mean value of the signal before we do anything to it.
                ranged_std(c) = std( ranged_sig ); % Calculate the standard deviation of that range.

                standardized_profile_data(c,:) = ((temporal_profiles(c,:)-ranged_mean(c))/ranged_std(c)) / (ranged_std(c)/ranged_mean(c)); % Standardize our signal.
            end
        end
    case 'relative_change'
        % Standardize using percentage preceded by a linear fit to remove
        % any residual low-frequency changes
        for c=1:size(temporal_profiles,1)
            if mod(c, size(temporal_profiles,1)/10) == 0
                waitbar(c/size(temporal_profiles,1),wbh, ['Standardizing temporal profiles (' num2str(c) ' of ' num2str(size(temporal_profiles,1)) ')']);
            end
            
            % Determine the framestamps that fall within our selected
            % range, as well as any nan values in this profile.
            validrange = framestamps>=range(1) & framestamps<=range(2) & ~isnan( temporal_profiles(c,:) );
            
            if sum(validrange) >= (range(2)-range(1))/4
                ranged_sig = temporal_profiles(c, validrange); % Isolate the profile.
                ranged_time = framestamps(validrange) / fps; % Get the timestamps for this signal.

                ranged_mean(c) = mean( ranged_sig ); % Calculate the mean value of the signal before we do anything to it.

                standardized_profile_data(c,:) = (temporal_profiles(c,:)-ranged_mean(c))/ranged_mean(c); % Standardize our signal.
            end
        end
    
end


% If we didn't supply a progress bar, close it at the end to be a good
% neighbor.
if isempty(p.Results.ProgBarHandle)
   close(wbh) 
end

end