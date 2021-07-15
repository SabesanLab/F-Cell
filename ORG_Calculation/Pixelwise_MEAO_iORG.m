
close all force;
clear;

[fName, pName]=uigetfile('*.avi','Pick the video you wish to analyze.');

wbh = waitbar(0,['Loading dataset: ' strrep(fName,'_','\_')]);

[temporal_data, framestamps, ~, mask_data]=load_reg_MEAO_data(fullfile(pName,fName));

[temporal_profiles, ref_coords]=extract_temporal_profiles(temporal_data,'SegmentationRadius',1, 'ProgBarHandle', wbh );

[norm_temporal_profiles]=framewise_normalize_temporal_profiles(temporal_profiles, 'ProgBarHandle', wbh);

[stdz_temporal_profiles]=standardize_temporal_profiles(norm_temporal_profiles, framestamps, [1 58], 29.4,'Method', 'linear_stddev', 'ProgBarHandle', wbh);

profiles_to_video(stdz_temporal_profiles, size(temporal_data), fullfile(pName,[fName(1:end-4) '_pixelwise.avi']));

close(wbh) 
