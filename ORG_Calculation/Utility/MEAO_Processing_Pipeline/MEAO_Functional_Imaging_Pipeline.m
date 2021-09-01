% function []=MEAO_Functional_Imaging_Pipeline()
% function []=MEAO_FUNCTIONAL_IMAGING_PIPELINE()
% Robert Cooper 2021-07-21
%
% This script loads all of the MEAO data from a given folder, relativizes
% it, and saves it in a "pipelined" form.
clear;
close all force;
pName = uigetdir(pwd, 'Select the folder containing all videos of interest.');

instr = input('Input the analysis modality string. [760nm]: ','s');

if isempty(instr)
    instr = '760nm';
end

ref_mode = input('Input the *reference* modality string. [760nm]: ','s');

if isempty(ref_mode)
    ref_mode = '760nm';
end

allFiles = read_folder_contents(pName,'avi', instr);


under_indices=cellfun(@(f)regexp(f,'_'),allFiles, 'UniformOutput', false);

locstr = cell(length(allFiles),1);
for i=1:length(allFiles)
    locstr{i} = allFiles{i}( (under_indices{i}(3)+1):(under_indices{i}(4)-1));
end
unique_loc = unique(locstr);

loc_list = cell(length(unique_loc),1);

for loc=1:length(unique_loc)
    loc_list{loc} = allFiles(contains(allFiles, unique_loc{loc}));
end


startloc = 1;
wb = waitbar(0, ['Loading ' strrep(loc_list{loc}{1},'_','\_') ' (Location: ' unique_loc{1} ')...']);

%% Load all of the data, processing each location separately.
for loc=startloc:length(loc_list)

    startloc = loc;
    startind = 1;
    % Grab the filenames for this location.
    fNames = loc_list{loc};
    
    waitbar(0, wb, ['Loading ' strrep(fNames{1},'_','\_') ' (Location: ' unique_loc{1} ')...']);

    temporal_data = cell(length(fNames),1);
    framestamps = cell(length(fNames),1);
    framerate = zeros(length(fNames),1);
    ref_images = cell(length(fNames),1);
    
    for f=startind:length(fNames)
        startind = f;
        if ~contains(fNames{f}, 'mask') && ~contains(fNames{f}, 'piped')
            % Load the MEAO data.
            waitbar(f/length(fNames), wb, ['Loading ' strrep(fNames{f},'_','\_') ' (Location: ' unique_loc{loc} ')...']);
            
            if ~strcmp(instr, ref_mode)
                [temporal_data{f}, framestamps{f}, framerate(f), ~, ~, ana_ref_images{f}] = load_reg_MEAO_data(fullfile(pName,fNames{f}), 'RemoveTorsion', false);
                
                if exist(fullfile(pName,strrep(fNames{f},instr, ref_mode)),'file')
                    [~, ~, ~, ~, ~, ref_images{f}] = load_reg_MEAO_data(fullfile(pName,strrep(fNames{f},instr, ref_mode)), 'RemoveTorsion', false);
                else
                   error(['File: ' strrep(fNames{f},instr, ref_mode) ' doesn''t exist!'] );
                end
            else
                [temporal_data{f}, framestamps{f}, framerate(f), ~, ~, ref_images{f}] = load_reg_MEAO_data(fullfile(pName,fNames{f}));
            end
            
        end
    end

    fNames = fNames(~cellfun(@isempty, temporal_data));
    framestamps = framestamps(~cellfun(@isempty, temporal_data));
    framerate = framerate(~cellfun(@isempty, temporal_data));
    ref_images = ref_images(~cellfun(@isempty, temporal_data));
    if ~strcmp(instr, ref_mode)
        ana_ref_images = ana_ref_images(~cellfun(@isempty, temporal_data));
    end
    temporal_data = temporal_data(~cellfun(@isempty, temporal_data));

    % Relativize all of the videos together, and output the results of that.
    outPath = fullfile(pName, 'Functional Pipeline', unique_loc{loc});
    if ~exist(outPath,'dir')
        mkdir(outPath);
    end

    [rel_tforms, ref_ind] = Relativize_Trials(ref_images, outPath, strrep(fNames,instr, ref_mode));

    
    if ~strcmp(instr, ref_mode)
        sum_map = zeros(size(ana_ref_images{1}));
        sum_data = zeros(size(ana_ref_images{1}));

        for j=1:length(ana_ref_images)
            if ~isempty(rel_tforms{j})
                
                ana_ref_images{j} = imwarp(ana_ref_images{j}, imref2d(size(ana_ref_images{j})), rel_tforms{j}, ...
                                           'OutputView', imref2d(size(ana_ref_images{ref_ind})) );
                
                frm_nonzeros = (ana_ref_images{j}>0);
                sum_data = sum_data+double(ana_ref_images{j});
                sum_map = sum_map+frm_nonzeros;
            end
        end

        under_indices = regexp(fNames{ref_ind},'_');
        common_prefix = fNames{ref_ind}(1:under_indices(7));
        imwrite(uint8(sum_data./sum_map), fullfile(outPath,[common_prefix 'ALL_ACQ_AVG.tif']));
    end  
    
    % Using these xforms, transform each dataset- and its coordinates
    for f=1:length(fNames)    
        if ~isempty(rel_tforms{f})
            waitbar(f/length(fNames), wb, ['Saving pipelined dataset:' strrep([fNames{f}(1:end-4) '_piped.avi'],'_','\_') ' (Location: ' unique_loc{loc} ')...']);

            confocal_vidout = VideoWriter( fullfile(outPath,[fNames{f}(1:end-4) '_piped.avi']), 'Grayscale AVI' );
            confocal_vidout.FrameRate = framerate(f);
            open(confocal_vidout); 
            for t=1:size(temporal_data{f},3)
                writeVideo( confocal_vidout, imwarp(uint8(temporal_data{f}(:,:,t)), imref2d(size(temporal_data{f}(:,:,t))), rel_tforms{f},...
                                                 'OutputView', imref2d(size(temporal_data{ref_ind}(:,:,1))) ) );
            end
            
            pipe_table=table();
            pipe_table.FrameStamps = framestamps{f};

            writetable(pipe_table, fullfile(outPath,[fNames{f}(1:end-4) '_piped.csv']));
       
            close(confocal_vidout);
        else
            warning(['Video: ' fNames{f} ' was removed due to a low final correlation. Please check your data.']);
        end
        
    end

end

close(wb)
