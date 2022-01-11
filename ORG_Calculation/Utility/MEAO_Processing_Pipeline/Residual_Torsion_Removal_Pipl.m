function [aligned_temporal_data, kept_framestamps, tforms]=Residual_Torsion_Removal_Pipl(temporal_data, framestamps, mask_data, reference_frame)
% Robert F Cooper 7-12-2021


    [optimizer, metric]  = imregconfig('monomodal');
    optimizer.GradientMagnitudeTolerance = 1e-5;
    optimizer.MinimumStepLength = 1e-5;
    optimizer.MaximumStepLength = 0.0625;
    optimizer.MaximumIterations = 500;

    tic;
    if ~isempty(mask_data)
        sum_map = sum(mask_data,3);
        average_frm_mask = sum_map >= size(mask_data,3)-2; %ceil(mean(sum_map(:)));
        % Find the largest incribed rectangle in this mask.
        [C, h, w, largest_rect] =FindLargestRectangles(average_frm_mask,[1 1 0], [300 150]);

        % Find the coordinates for each corner of the rectangle, and
        % return them
        cropregion = regionprops(largest_rect,'BoundingBox');
        cropregion = ceil(cropregion.BoundingBox);

        cropregion = [cropregion(1:2), cropregion(1)+cropregion(3), cropregion(2)+cropregion(4)];

        % Bound our crop region to where the sum_map actually exists
        cropregion(cropregion<1) = 1;
        if cropregion(4)>=size(sum_map,1)
            cropregion(4) = size(sum_map,1);
        end
        if cropregion(3)>=size(sum_map,2)
            cropregion(3) = size(sum_map,2);
        end

%         sum_map_crop = sum_map(cropregion(2):cropregion(4),cropregion(1):cropregion(3));
        
        cropped_temporal_data = temporal_data(cropregion(2):cropregion(4), cropregion(1):cropregion(3), :);
    else
        warning(['No mask data supplied. Registration may be inaccurate.']);
        temporal_data = temporal_data;
    end
    
    forward_reg_tform = cell(length(temporal_data),1);
    % Register the image stack forward. It is more stable if we align to the 
    % frame with the largest percentage of the cropped region covered.
    tforms = zeros(3, 3, size(temporal_data,3));
    tforms(:,:,1)=affine2d().T;
    tforms(:,:,reference_frame)=affine2d().T;
    
    cropped_ref_frame= cropped_temporal_data(:,:,reference_frame);
    parfor n=1:size(cropped_temporal_data,3)

        % Register using the cropped frame.
        forward_reg_tform{n}=imregtform(cropped_temporal_data(:,:,n), cropped_ref_frame,'affine',...
                                optimizer, metric,'PyramidLevels',2, 'InitialTransformation', affine2d());%,'DisplayOptimization',true);

        
        tforms(:,:,n) = forward_reg_tform{n}.T;
    end
    toc;

    mean_tforms = mean(tforms,3);

    % Determine the typical distance from the mean xform.
    for t=1:size(tforms,3)
        tfo = tforms(:,:,t);
        frob_dist(t) = abs(tfo(:)'*mean_tforms(:));
    end

    mean_frob_dist = mean(frob_dist);
    std_frob_dist = std(frob_dist);


    max_frob_dist = std_frob_dist+mean_frob_dist;

    %%
    howcorr = ones(size(temporal_data,3) ,1);

    aligned_temporal_data = temporal_data;
    ref_frame= temporal_data(:,:,reference_frame);

    parfor f=1:size(temporal_data,3)    

        % We would NOT expect large changes here- so if there is a big
        % jump, use the surrounding transforms to force some stablility on
        % the registration.
        tfo = tforms(:,:,f);
        t=0;
        theend = false;

        while abs(tfo(:)'*mean_tforms(:)) > max_frob_dist
                
%                 lasttfo = tforms(:,:,f-1);
%                 nexttfo = tforms(:,:,f+1);
                
%                 meantfo=mean(cat(3, lasttfo, nexttfo),3);
                
            tfo = tforms(:,:,f-t);

            t=t+1;

            if (f-t) == 0
               theend = true;
               break;
            end
        end

        if theend % If we reached the end in that direction, check the other direction
            t=1;
            while abs(tfo(:)'*mean_tforms(:)) > max_frob_dist  && (f+t) < size(temporal_data,3)
                tfo = tforms(:,:,f+t);
                t=t+1;
            end
        end
        

        warped_frame = imwarp(temporal_data(:,:,f), affine2d(tfo),'OutputView', imref2d(size(temporal_data(:,:,f))) );
            
        
        warpedmask = imwarp(logical(mask_data(:,:,f)/255), affine2d(tfo),'OutputView', imref2d(size(temporal_data(:,:,f))) );
        
            
        [~, ~, ~, largest_rect] =FindLargestRectangles(warpedmask,[1 1 0], [300 150]);
        % Find the coordinates for each corner of the rectangle, and
        % return them
        cropregion = regionprops(largest_rect,'BoundingBox');
        cropregion = ceil(cropregion.BoundingBox);

        cropregion = [cropregion(1:2), cropregion(1)+cropregion(3), cropregion(2)+cropregion(4)];
        cropregion(cropregion<1) = 1;% Bound our crop region
        if cropregion(4)>=size(warpedmask,1)
            cropregion(4) = size(warpedmask,1);
        end
        if cropregion(3)>=size(warpedmask,2)
            cropregion(3) = size(warpedmask,2);
        end
        
        howcorr(f) = corr2(ref_frame(cropregion(2):cropregion(4), cropregion(1):cropregion(3)),...
                           warped_frame(cropregion(2):cropregion(4), cropregion(1):cropregion(3)));    
        
        aligned_temporal_data(:,:,f) = warped_frame;              
%             if f>=2
%                 imshowpair(masked_temporal_data(:,:,f-1), masked_temporal_data(:,:,f) );
% %                 froby
% %                 tfo
% 
%                 pause(0.1);
%             end
%         imagesc(aligned_temporal_data(:,:,f)); colormap gray; drawnow;pause(1/29.466)
    end

    if sum(howcorr<=0.1)> 0
        warning(['Removed ' num2str( sum(howcorr<=0.1)) ' frames due to low reference correlation.'])
    end
    
    aligned_temporal_data = aligned_temporal_data(:,:,howcorr>0.1);
    kept_framestamps = framestamps(howcorr>0.1);
end
