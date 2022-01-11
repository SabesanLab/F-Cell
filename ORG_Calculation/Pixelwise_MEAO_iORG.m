% 2021-07-13
% Robert F Cooper
% This code uses the F(Cell) toolkit to obtain pixelwise iORGs from
% any MEAO dataset.
% It outputs an avi that is the same duration and size as the original, but
% Where each pixel value is the iORG for that pixel location.
% Assumes the data has been registered previously.

close all force;
clear;

[fName, pName]=uigetfile('*.avi','Pick the video you wish to analyze.');

wbh = waitbar(0,['Loading dataset: ' strrep(fName,'_','\_')]);

% Load the MEAO data.
[temporal_data, framestamps, ~, mask_data]=load_reg_MEAO_data(fullfile(pName,fName));

% Save the data to a torsion-removed tmp file.
vidObj = VideoWriter('torsion_begone.avi','Uncompressed AVI');
open(vidObj);
for i=1:size(temporal_data, 3)
    writeVideo(vidObj, uint8(temporal_data(:,:,i)));
end
close(vidObj);

% [temporal_data, framestamps, ~, mask_data]=load_pipelined_MEAO_data(fullfile(pName,fName));

% Extract temporal profiles at each pixel (no coordinate list supplied)
[temporal_profiles, ref_coords]=extract_temporal_profiles(temporal_data,'SegmentationRadius',1, 'ProgBarHandle', wbh );

% Normalize the temporal profiles to each frame's mean
[norm_temporal_profiles]=framewise_normalize_temporal_profiles(temporal_profiles, 'ProgBarHandle', wbh);

% Standardize the temporal profiles to their *pre stimulus* behavior
% (frames: 1-58), 29.4Hz
[stdz_temporal_profiles]=standardize_temporal_profiles(norm_temporal_profiles, framestamps', [1 50], 10,'Method', 'linear_vast', 'ProgBarHandle', wbh);

% Take a block of profiles, and output them to a video.
profiles_to_video(stdz_temporal_profiles, size(temporal_data), fullfile(pName,[fName(1:end-4) '_pixelwise.avi']));

close(wbh) 
