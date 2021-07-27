% 2021-07-26
% Robert F Cooper
% This code uses the F(Cell) toolkit to obtain coordinate-based iORGs from
% any MEAO dataset.
%
% It outputs an avi that is the same duration and size as the original, but
% Where each pixel value is the iORG for that pixel location.
% Assumes the data has been registered previously.

close all force;
clear;

rootDir = uigetdir(pwd, 'Select the folder containing all videos of interest.');

fNames = read_folder_contents(rootDir, 'avi', '760nm');

% Load our reference image.
all_acq_avg_im = read_folder_contents(rootDir, 'tif', 'ALL_ACQ_AVG');

if length(all_acq_avg_im) > 1
    warning('More than one pipeline average file found. Results may be inaccurate.');
end
ref_coords=[];
if ~isempty(all_acq_avg_im) && exist(fullfile(rootDir, all_acq_avg_im{1}), 'file')
    ref_im = imread(fullfile(rootDir, all_acq_avg_im{1}));
    
    % Load our reference coordinates.
    if exist(fullfile(rootDir, [all_acq_avg_im{1}(1:end-4) '_coords.csv'] ), 'file')
        ref_coords = dlmread(fullfile(rootDir, [all_acq_avg_im{1}(1:end-4) '_coords.csv'] ));
    end
else
    error('No reference image found. Please re-run pipeline to fix issue.');
end

[stimFile, stimPath] = uigetfile(fullfile(rootDir, '*.csv'), 'Select the stimulus train file that was used for this dataset.');

stimTrain = dlmread(fullfile(stimPath, stimFile), ',');

wbh = waitbar(0,['Loading dataset: ' strrep(fNames{1},'_','\_')]);

finalized_temporal_profiles = cell(length(fNames),1);
framestamps = cell(length(fNames),1);
framerate = zeros(length(fNames),1);

startind = 1;
%% Load all of the temporal profiles from each pipelined dataset.
for f=startind:length(fNames)
    
    % Load the pipelined MEAO data.
    waitbar(0, wbh, ['Loading dataset: ' strrep(fNames{f},'_','\_')]);    
    [temporal_data, framestamps{f}, framerate(f)] = load_pipelined_MEAO_data(fullfile(rootDir, fNames{f}));
    
    % Extract temporal profiles at each pixel
    [temporal_profiles, ref_coords]=extract_temporal_profiles(temporal_data,'SegmentationRadius',2, 'Coordinates', ref_coords, 'ProgBarHandle', wbh );

    % Normalize the temporal profiles to each frame's mean
    [norm_temporal_profiles]=framewise_normalize_temporal_profiles(temporal_profiles, 'ProgBarHandle', wbh);

    % Standardize the temporal profiles to their *pre stimulus* behavior    
    [finalized_temporal_profiles{f}]=standardize_temporal_profiles(norm_temporal_profiles, framestamps{f}', [1 stimTrain(1)], framerate(f),'Method', 'linear_vast', 'ProgBarHandle', wbh);

    figure(707); hold on;
    Population_iORG(finalized_temporal_profiles{f},framestamps{f});
    
end
hold off;

iORG_Map(ref_coords, finalized_temporal_profiles(2:end), framestamps(2:end));

close(wbh) 
