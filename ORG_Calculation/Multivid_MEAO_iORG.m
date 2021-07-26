% Robert Cooper 2021-07-26
%
% This script extracts and obtains profile data from a MEAO pipelined
% dataset.
clear;
close all force;
clc;

rootDir = uigetdir(pwd, 'Select the folder containing all videos of interest.');

fPaths = read_folder_contents_rec(rootDir, 'avi', '760nm');

[stimFile, stimPath] = uigetfile(fullfile(rootDir, '*.csv'), 'Select the stimulus train file that was used for this dataset.');

stimTrain = dlmread(fullfile(stimPath, stimFile), ',');

wbh = waitbar(0,['Processing dataset 0 of ' num2str(length(fPaths)) '.']);

p = gcp();
for i=1:size(fPaths,1)
    
    [mov_path, ref_image_fname] = getparent(fPaths{i});

    waitbar(i/length(fPaths), wbh, ['Submitted dataset (' num2str(i) ' of ' num2str(length(fPaths)) ').']); 

    try
        f(i) = parfeval( );
    catch ex
       disp([ref_image_fname ' failed to process:']);
       disp([ex.message ': line ' num2str(ex.stack(1).line)] );
    end

end

waitbar(0, wbh, 'Waiting for trials to finish...'); 

results = cell(1,size(fPaths,1));
for i = 1:size(fPaths,1)
    
    try
        % fetchNext blocks until next results are available.
        [completedIdx] = fetchNext(f);
        [mov_path, ref_image_fname] = getparent(fPaths{completedIdx});
        waitbar(i/length(fPaths), wbh, ['Finished trial (' num2str(i) ' of ' num2str(length(fPaths)) ').']);    
    catch ex
       disp([ref_image_fname ' failed to process:']);
       disp([ex.message ': line ' num2str(ex.stack(1).line)] );
    end
end
delete(p);
close(wbh);

