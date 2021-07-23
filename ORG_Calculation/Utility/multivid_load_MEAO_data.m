% Robert Cooper 2021-07-21
%
% This script loads all of the MEAO data from a given folder, relativizes
% it, and returns it for analysis.

pName = uigetdir(pwd, 'Select the folder containing all videos of interest.');

% instr = input('Input the desired modality. [760nm]');

% if isempty(instr)
%     instr = '760nm';
% end

allFiles = read_folder_contents(pName,'avi', '760nm');


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

%% Load all of the data, processing each location separately.
wb = waitbar(0, ['Loading ' strrep(fNames{1},'_','\_') ' (Location: ' unique_loc{1} ')...']);

for loc=1:length(loc_list)

    startind = 1;
    %% Grab the filenames for this location.
    fNames = loc_list{loc};
    
    temporal_data = cell(length(fNames),1);
    framestamps = cell(length(fNames),1);
    mask_data = cell(length(fNames),1);
    
    for f=startind:length(fNames)
        startind = f;
        if ~contains(fNames{f}, 'mask') && ~contains(fNames{f}, 'piped')
            % Load the MEAO data.
            waitbar(f/length(fNames), wb, ['Loading ' strrep(fNames{f},'_','\_') ' (Location: ' unique_loc{loc} ')...']);
            [temporal_data{f}, framestamps{f}, ref_coords{f}, mask_data{f}, ref_images{f}] = load_reg_MEAO_data(fullfile(pName,fNames{f}));

        end
    end

    fNames = fNames(~cellfun(@isempty, temporal_data));
    framestamps = framestamps(~cellfun(@isempty, temporal_data));
    mask_data = mask_data(~cellfun(@isempty, temporal_data));
    ref_images = ref_images(~cellfun(@isempty, temporal_data));
    temporal_data = temporal_data(~cellfun(@isempty, temporal_data));

    %% Relativize all of the videos together, and output the results of that.
    outPath = fullfile(pName, 'Pipelined', unique_loc{loc});
    if ~exist(outPath,'dir')
        mkdir(outPath);
    end

    [rel_tforms, ref_ind] = Relativize_Trials(ref_images, outPath, fNames);

    % Using these xforms, transform each dataset- and its coordinates
    for f=1:length(fNames)
        waitbar(f/length(fNames), wb, ['Saving pipelined dataset:' strrep([fNames{f}(1:end-4) '_piped.avi'],'_','\_') ' (Location: ' unique_loc{loc} ')...']);

        confocal_vidout = VideoWriter( fullfile(outPath,[fNames{f}(1:end-4) '_piped.avi']), 'Grayscale AVI' );
        open(confocal_vidout); 
        for t=1:size(temporal_data{f},3)
            writeVideo( confocal_vidout, imwarp(uint8(temporal_data{f}(:,:,t)), imref2d(size(temporal_data{f}(:,:,t))), rel_tforms{f},...
                                             'OutputView', imref2d(size(temporal_data{ref_ind}(:,:,1))) ) );
        end
        close(confocal_vidout);
    end

    % After pipeline, transfer stimulus info file.
    
end

close(wb)
