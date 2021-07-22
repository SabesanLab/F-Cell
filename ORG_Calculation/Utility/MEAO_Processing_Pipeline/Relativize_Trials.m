function [tforms, ref_im]=Relativize_Trials(ref_images, pName, fNames)


for i=1:length(ref_images)
   
    imsize(i,:) = size(ref_images{i});
end

%%


[optimizer, metric]  = imregconfig('multimodal');
optimizer.InitialRadius = 1.5e-03;
metric.NumberOfHistogramBins = 200;

[monooptimizer, monometric]  = imregconfig('monomodal');
monooptimizer.MaximumIterations = 250;
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
                                 'affine',monooptimizer,monometric, 'PyramidLevels',1,'InitialTransformation',tforms{j});


        toc;

    end
end
% end

%% Relativized stack

under_indices = regexp(fNames{ref_im},'_');
common_prefix = fNames{ref_im}(1:under_indices(7));

stk_name = [common_prefix 'Relativized_Avg_Ims.tif'];
reg_ims = repmat(ref_images{ref_im},[1 1 length(ref_images)]);
imwrite(uint8(reg_ims(:,:,ref_im)), fullfile(pName, stk_name) );

for j=1:length(ref_images)
    if j ~= ref_im
        reg_ims(:,:,j) = imwarp(ref_images{j}, imref2d(size(ref_images{j})), tforms{j}, 'OutputView', imref2d(size(ref_images{ref_im})) );
        
        imwrite(uint8(reg_ims(:,:,j)),fullfile(pName, stk_name),'WriteMode','append','Description',num2str(j));                

    end
%     imwrite(reg_ims(:,:,j), fullfile(pathname,[confocal_fname{j}(1:end-8) '_piped_AVG.tif']) );
%     delete(confocal_fname{j});
end


% Make a sum map, and create an average image.
sum_map = zeros(size(reg_ims(:,:,1)));

for f=1:size(reg_ims,3)
    frm_nonzeros = (reg_ims(:,:,f)>0); 
    sum_map = sum_map+frm_nonzeros;
end


imwrite(uint8(sum(double(reg_ims),3)./sum_map), fullfile(pName,[common_prefix 'ALL_ACQ_AVG.tif']));

