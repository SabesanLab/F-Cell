function [tforms, ref_im]=Relativize_Trials(ref_images, pName, fNames)


for i=1:length(ref_images)
   
    imsize(i,:) = size(ref_images{i});
end

%%


[optimizer, metric]  = imregconfig('multimodal');
optimizer.InitialRadius = 1.5e-03;
metric.NumberOfHistogramBins = 200;

[monooptimizer, monometric]  = imregconfig('monomodal');
monooptimizer.MaximumIterations = 500;
monooptimizer.RelaxationFactor = 0.6;
monooptimizer.GradientMagnitudeTolerance = 1e-5;

tforms = cell(length(ref_images),1);

% Find the transform from each image to every other

thearea = prod(imsize,2);

[~, ref_im] = max(thearea);


i=ref_im;
for j=1:length(ref_images)
    tforms{j} = affine2d();
    if i~=j
        %%
        tic;

        % First get close via ncc
        [xcorr_map , ~] = normxcorr2_general(ref_images{j}, ref_images{ref_im}, prod(mean([imsize(j,:);imsize(ref_im,:)])/2) );

        [~, ncc_ind] = max(xcorr_map(:));
        [roff, coff]= ind2sub(size(xcorr_map), ncc_ind );
        roff = roff-size(ref_images{j},1);
        coff = coff-size(ref_images{j},2);

        tforms{j} = affine2d([1 0 0; 0 1 0; coff roff 1]);

        ref_images{j}(isnan(ref_images{j})) = 0;
        tforms{j} = imregtform( ref_images{j}, ref_images{ref_im},... % Then tweak for affine
                                 'rigid',monooptimizer,monometric, 'PyramidLevels',1,'InitialTransformation',tforms{j});

        toc;

    end
end
% end

%% Relativized stack

under_indices = regexp(fNames{ref_im},'_');
common_prefix = fNames{ref_im}(1:under_indices(7));

stk_name = [common_prefix 'Relativized_Avg_Ims.tif'];
reg_ims = repmat(ref_images{ref_im},[1 1 length(ref_images)]);
howcorr = ones(length(ref_images),1);


for j=1:length(ref_images)
    if j ~= ref_im
        reg_ims(:,:,j) = imwarp(ref_images{j}, imref2d(size(ref_images{j})), tforms{j}, 'OutputView', imref2d(size(ref_images{ref_im})) );
                
        warpedmask = imwarp(true(size(ref_images{ref_im})), imref2d(size(ref_images{j})), tforms{j}, 'OutputView', imref2d(size(ref_images{ref_im})) );
        
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
        
        howcorr(j) = corr2(reg_ims(cropregion(2):cropregion(4), cropregion(1):cropregion(3),ref_im),...
                        reg_ims(cropregion(2):cropregion(4), cropregion(1):cropregion(3),j));                   

    end

end


% Actually write the data to a stacked tif for review.
imwrite(uint8(reg_ims(:,:,ref_im)), fullfile(pName, stk_name) );
for j=1:length(ref_images)
    if j ~= ref_im && howcorr(j)> quantile(howcorr, 0.1)
        imwrite(uint8(reg_ims(:,:,j)),fullfile(pName, stk_name),'WriteMode','append','Description',num2str(fNames{j}));                
    elseif howcorr(j)<=quantile(howcorr, 0.1)
        tforms{j} = [];
    end
end

% Make a sum map, and create an average image.
sum_map = zeros(size(reg_ims(:,:,1)));
sum_data = zeros(size(reg_ims(:,:,1)));

for f=1:size(reg_ims,3)
    if ~isempty(tforms{f})
        frm_nonzeros = (reg_ims(:,:,f)>0);
        sum_data = sum_data+double(reg_ims(:,:,f));
        sum_map = sum_map+frm_nonzeros;
    end
end


imwrite(uint8(sum_data./sum_map), fullfile(pName,[common_prefix 'ALL_ACQ_AVG.tif']));

