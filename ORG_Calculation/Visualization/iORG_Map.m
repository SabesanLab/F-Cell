function [iORG_map] = iORG_Map(coords, temporal_profiles, framestamps, varargin)


p = inputParser;

addRequired(p,'coords', @isnumeric);
addRequired(p,'temporal_profiles', @iscell);
addRequired(p,'framestamps', @iscell);
addParameter(p,'FractionValid', 0.6, @isnumeric);
addParameter(p,'CoordsPerPixelTarget', false, @islogical);
addParameter(p,'ProfilesPerPixelTarget', false, @islogical);

% Parse our inputs.
parse(p,coords,temporal_profiles,framestamps,varargin{:})

fraction_valid=p.Results.FractionValid;



profilesize = cell2mat(cellfun(@size, temporal_profiles, 'UniformOutput', false));

if length(unique(profilesize(:,1)))~=1
    error('Number of cells in provided profiles do not match!');
end

nancells = cellfun(@isnan, temporal_profiles, 'UniformOutput', false);
% Determine the number of profiles that each cone has that are valid.
numvalid = zeros(profilesize(1),1);
for acqind=1:size(temporal_profiles,1)

    fract_good = sum( ~nancells{acqind},2)./size(nancells{acqind},2);
    
    valid_profiles{acqind} = (fract_good >= fraction_valid);
    numvalid = numvalid + valid_profiles{acqind};
end

% Determine the window size dynamically for each coordinate, given our
% constraints defined above.
neighborinds=cell(size(coords,1),1);
pixelwindowsize = zeros(size(coords,1),1);

maxrowval = max(coords(:,2));
maxcolval = max(coords(:,1));
max_framestamp = max(cellfun(@max,framestamps));

max_cells = [];

target_profiles = 35;

local_iORG={};
parfor c=1:size(coords,1)

    thiswindowsize=0;
    num_profiles=0;
    clipped_inds=0;
    while num_profiles <= target_profiles
        thiswindowsize = thiswindowsize+1;
        rowborders = ([coords(c,2)-(thiswindowsize/2) coords(c,2)+(thiswindowsize/2)]);
        colborders = ([coords(c,1)-(thiswindowsize/2) coords(c,1)+(thiswindowsize/2)]);

        rowborders(rowborders<=1)=1;
        colborders(colborders<=1)=1;
        rowborders(rowborders>=maxrowval)=maxrowval;
        colborders(colborders>=maxcolval)=maxcolval;

        [~, clipped_inds]=coordclip(coords, colborders, rowborders,'i');
        
        num_profiles=sum(numvalid(clipped_inds));
    end
    % Track all the neighboring coordinates that we'll be including.
    neighborinds{c} = clipped_inds;
    pixelwindowsize(c) = thiswindowsize;
    
    
    
    % Collect all of the profiles from each acquisition, and combine into a
    % single matrix.
    
    local_temporal_profiles = nan(num_profiles, max_framestamp);
    start_ind=1;
    for acqind=1:size(temporal_profiles,1)
        
        coi_inds = neighborinds{c}( valid_profiles{acqind}(neighborinds{c}) );
        coi_profiles = temporal_profiles{acqind}(coi_inds,:);
        coi_framestamps = framestamps{acqind};
        
        num_coi = size(coi_profiles, 1);
        
        local_temporal_profiles(start_ind:(start_ind-1+num_coi),coi_framestamps) = coi_profiles;
        start_ind = start_ind+num_coi;
    end
%    figure(1); plot(sum(~isnan(local_temporal_profiles)));
%     clf;plot(local_temporal_profiles'); hold on;
% figure(707); hold on;
    local_iORG{c} = Population_iORG(local_temporal_profiles,[],'SummaryMethod','moving_rms', 'WindowSize', 7);
end    

% Create a map from our results.
interped_map=zeros([maxrowval maxcolval max_framestamp]);
% sum_map=zeros([maxrowval maxcolval ]);

for f=1:max_framestamp
    sum_map=zeros([maxrowval maxcolval ]);
    
    for c=1:size(coords,1)

            thisval = local_iORG{c}(f);

            rowrange = round(coords(c,2)-(pixelwindowsize(c)/2):coords(c,2)+(pixelwindowsize(c)/2));
            colrange = round(coords(c,1)-(pixelwindowsize(c)/2):coords(c,1)+(pixelwindowsize(c)/2));

            rowrange(rowrange<1) =[];
            colrange(colrange<1) =[];
            rowrange(rowrange>maxrowval) =[];
            colrange(colrange>maxcolval) =[];

            interped_map(rowrange,colrange,f) = interped_map(rowrange,colrange,f) + thisval;
            sum_map(rowrange, colrange) = sum_map(rowrange, colrange) + 1;

    end


    
interped_map(:,:,f) = interped_map(:,:,f)./sum_map;

% interped_map(isnan(interped_map(:,:,f)),f) =0;

figure(1); imagesc(log(interped_map(:,:,f))); axis image; caxis([1.5 4]); title(num2str(f));
colorbar; pause(0.1);
end

end

