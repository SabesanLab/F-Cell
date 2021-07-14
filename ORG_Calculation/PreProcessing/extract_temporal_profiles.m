function [temporal_profiles, coordinates]=extract_temporal_profiles(temporal_data, varargin)


p = inputParser;

dataset_size = size(temporal_data);

if length(dataset_size) > 3
   fourDeeData = true; 
end

[X_grid, Y_grid]= meshgrid(1:dataset_size(1), 1:dataset_size(2));

defaultcoords = [X_grid(:) Y_grid(:)];
checkcoords = @(x) size(x,2) == 2; % the coordinate list should be Nx2. 

defaultmethod = 'box';
validmethods = {'box', 'voronoi'};
checkMethods = @(x) any(validatestring(x,validmethods));

defaultextmethod = 'mean';
validextmethods = {'mean', 'median', 'glcm_ent', 'glcm_ent'};
checkExtMethods = @(x) any(validatestring(x,validextmethods));

defaultradius = 2;

addRequired(p,'temporal_data', @isnumeric)
addOptional(p,'Coordinates', defaultcoords, checkcoords);
addOptional(p,'SegmentationMethod', defaultmethod, checkMethods);
addOptional(p,'SegmentationRadius', defaultradius, @isnumeric); %Unused for voronoi.
addOptional(p,'ExtractionMethod', defaultextmethod, checkExtMethods);
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


if strcmp(segmentation_method, 'voronoi')
    [V, C]=voronoin(coordinates);
    [X, Y ] = meshgrid(1:size(ref_image,2), 1:size(ref_image,1));
end
coordinates = round(coordinates);


cellseg_inds = cell(size(coordinates,1),1);

switch segmentation_method
    case 'box'
        for i=1:size(coordinates,1)
            
            if mod(i, size(coordinates,1)/10) == 0
                waitbar(i/size(coordinates,1),wbh, 'Segmenting coordinates...');
            end

            if (coordinates(i,1) - roiradius) > 1 && (coordinates(i,1) + roiradius) < dataset_size(2) &&...
               (coordinates(i,2) - roiradius) > 1 && (coordinates(i,2) + roiradius) < dataset_size(1)

                [R, C ] = meshgrid((coordinates(i,2) - roiradius) : (coordinates(i,2) + roiradius), ...
                                   (coordinates(i,1) - roiradius) : (coordinates(i,1) + roiradius));

                cellseg_inds{i} = sub2ind( dataset_size(1:2), R, C );

                cellseg_inds{i} = cellseg_inds{i}(:);

            end
        end
    case 'voronoi'
        for i=1:size(coordinates,1)

            if mod(i, size(coordinates,1)/10) == 0
                waitbar(i/size(coordinates,1),wbh, 'Segmenting coordinates...');
            end
            vertices = V(C{i},:);

            if (all(C{i}~=1)  && all(vertices(:,1)<dataset_size(2)) && all(vertices(:,2)<dataset_size(1)) ... % [xmin xmax ymin ymax] 
                             && all(vertices(:,1)>=1) && all(vertices(:,2)>=1)) 

                [in, on] = inpolygon(X(:), Y(:), vertices(:,1), vertices(:,2));

                cellseg_inds{i} = sub2ind( dataset_size(1:2), Y(in|on), X(in|on) );

                cellseg_inds{i} = cellseg_inds{i}(:);
            end
        end
end


temporal_profiles = cell( length(cellseg_inds), 1);


waitbar(0,wbh, 'Generating reflectance profiles...');


switch segmentation_method
    case 'box' 
        switch extraction_method
            case 'mean' % Shorthand, but way faster than the naive implementation, for box extraction only. (>2x speedup)
                for i=1:length(cellseg_inds) 

                    if mod(i, length(cellseg_inds)/10) == 0
                        waitbar(i/length(cellseg_inds),wbh, 'Generating reflectance profiles...');
                    end
                    
                    if ~isempty(cellseg_inds{i})

                        temporal_profiles{i} = zeros(1, dataset_size(end));

                        [m,n] = ind2sub(dataset_size(1:2), cellseg_inds{i});

                        thisstack = temporal_data(min(m):max(m), min(n):max(n),:);        

                        thisstack(thisstack == 0) = NaN;

                        thisstack = sum(thisstack,1);
                        thisstack = squeeze(sum(thisstack,2));
                        thisstack = thisstack./ (length(cellseg_inds{i})*ones(size(thisstack)));


                        temporal_profiles{i} = thisstack';
                    end
                end
            case 'median'
                for i=1:length(cellseg_inds)
                    if mod(i, length(cellseg_inds)/10) == 0
                        waitbar(i/length(cellseg_inds),wbh, 'Generating reflectance profiles...');
                    end
                    
                    if ~isempty(cellseg_inds{i})
                    temporal_profiles{i} = zeros(1, dataset_size(end));

                    for t=1:size(temporal_data,3)

                        masked_timepoint = temporal_data(:,:,t); 

                        if all( masked_timepoint(cellseg_inds{i}) ~= 0 )
                            temporal_profiles{i}(t) = median( masked_timepoint(cellseg_inds{i}));
                        else            
                            temporal_profiles{i}(t) =  NaN;
                        end
                    end
                    end
                end
            case 'glcm'
                
        end
    case 'voronoi'
        switch extraction_method
            case 'mean'
                for i=1:length(cellseg_inds)
                    if ~isempty(cellseg_inds{i})
                    temporal_profiles{i} = zeros(1, dataset_size(end));

                    for t=1:size(temporal_data,3)

                        masked_timepoint = temporal_data(:,:,t); 

                        if all( masked_timepoint(cellseg_inds{i}) ~= 0 )
                            temporal_profiles{i}(t) = mean( masked_timepoint(cellseg_inds{i}));
                        else            
                            temporal_profiles{i}(t) =  NaN;
                        end
                    end
                    end
                end
            case 'median'
                for i=1:length(cellseg_inds)
                    if ~isempty(cellseg_inds{i})
                    temporal_profiles{i} = zeros(1, dataset_size(end));

                    for t=1:size(temporal_data,3)

                        masked_timepoint = temporal_data(:,:,t); 

                        if all( masked_timepoint(cellseg_inds{i}) ~= 0 )
                            temporal_profiles{i}(t) = median( masked_timepoint(cellseg_inds{i}));
                        else            
                            temporal_profiles{i}(t) =  NaN;
                        end
                    end
                    end
                end
            case 'glcm'
                
        end
end

% If we didn't supply a progress bar, close it at the end to be a good
% neighbor.
if isempty(p.Results.ProgBarHandle)
   close(wbh) 
end

end