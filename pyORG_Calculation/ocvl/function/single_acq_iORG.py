import os
from os import walk
from os.path import splitext
from pathlib import Path
from tkinter import Tk, filedialog, ttk, HORIZONTAL

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from scipy.spatial import Voronoi, voronoi_plot_2d
from skimage.feature import peak_local_max
from ssqueezepy.experimental import scale_to_freq

from ocvl.function.analysis.cell_profile_extraction import extract_profiles, norm_profiles, standardize_profiles, \
    refine_coord, refine_coord_to_stack, exclude_profiles
from ocvl.function.analysis.iORG_profile_analyses import signal_power_iORG, wavelet_iORG, extract_texture_profiles, \
    filtered_absolute_difference
from ocvl.function.preprocessing.improc import norm_video
from ocvl.function.utility.generic import PipeStages
from ocvl.function.utility.meao import MEAODataset
from ocvl.function.utility.pycoordclip import coordclip
from ocvl.function.utility.resources import save_video, save_tiff_stack
from ocvl.function.utility.temporal_signal_utils import reconstruct_profiles


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]



if __name__ == "__main__":
    root = Tk()
    root.lift()
    w = 1
    h = 1
    x = root.winfo_screenwidth() / 4
    y = root.winfo_screenheight() / 4
    root.geometry(
        '%dx%d+%d+%d' % (
            w, h, x, y))  # This moving around is to make sure the dialogs appear in the middle of the screen.

    pName = filedialog.askdirectory(title="Select the folder containing all videos of interest.", parent=root)

    if not pName:
        quit()

    stimtrain_fName = filedialog.askopenfilename(title="Select the stimulus train file.", parent=root)

    if not stimtrain_fName:
        quit()

    x = root.winfo_screenwidth() / 2 - 128
    y = root.winfo_screenheight() / 2 - 128
    root.geometry(
        '%dx%d+%d+%d' % (
            w, h, x, y))  # This moving around is to make sure the dialogs appear in the middle of the screen.
    root.update()

    allFiles = dict()

    totFiles = 0
    # Parse out the locations and filenames, store them in a hash table.
    controlpath = None
    searchpath = Path(pName)
    for path in searchpath.rglob("*.avi"):
        if "piped" in path.name:
            splitfName = path.name.split("_")

            if path.parent not in allFiles:
                allFiles[path.parent] = []
                allFiles[path.parent].append(path)

                if "control" in path.parent.name:
                    print("DETECTED CONTROL DATA AT: " + str(path.parent))
                    controlpath = path.parent
            else:
                allFiles[path.parent].append(path)

            totFiles += 1

    pb = ttk.Progressbar(root, orient=HORIZONTAL, length=512)
    pb.grid(column=0, row=0, columnspan=2, padx=3, pady=5)
    pb_label = ttk.Label(root, text="Initializing setup...")
    pb_label.grid(column=0, row=1, columnspan=2)
    pb.start()
    # Resize our root to show our progress bar.
    w = 512
    h = 64
    x = root.winfo_screenwidth() / 2 - 256
    y = root.winfo_screenheight() / 2 - 64
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))
    root.update()

    first = True
    outputcsv = True

    og_framestamps = []
    cell_framestamps = []
    mean_cell_profiles = []
    texture_cell_profiles = []
    full_cell_profiles = []

    segmentation_radius = 3

    # Before we start, get an estimate of the "noise" from the control signals.
    sig_threshold_im = None
   # controlpath = None # TEMP!
    if controlpath is not None:
        print("Processing control data to find noise floor...")


    # [ 0:"bob"  1:"moe" 2:"larry" 3:"curly"]
    # Loops through all locations in allFiles
    for l, loc in enumerate(allFiles):
        #if loc == controlpath:
         #   continue

        first = True
        res_dir = loc.joinpath(
            "Results")  # creates a results folder within loc ex: R:\00-23045\MEAOSLO1\20220325\Functional\Processed\Functional Pipeline\(1,0)\Results
        res_dir.mkdir(exist_ok=True)  # actually makes the directory if it doesn't exist. if it exists it does nothing.

        this_dirname = res_dir.parent.name

        r = 0
        pb["maximum"] = len(allFiles[loc])
        mapper = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis", len(allFiles[loc])).reversed())
        max_frmstamp = 0

        # Loops through all the files within this location
        for file in allFiles[loc]:

            # Ignores the All_ACQ_AVG tif while running through the files in this location
            if "ALL_ACQ_AVG" not in file.name:
                # Waitbar stuff
                pb["value"] = r
                pb_label["text"] = "Processing " + file.name + "..."
                pb.update()
                pb_label.update()
                print("Processing " + file.name + "...")

                # Loading in the pipelined data (calls the load_pipelined_data() fxn
                dataset = MEAODataset(file.as_posix(), stimtrain_path=stimtrain_fName,
                                      analysis_modality="760nm", ref_modality="760nm", stage=PipeStages.PIPELINED)
                dataset.load_pipelined_data()

                # Initialize the dict for individual cells.
                if first:
                    reference_coord_data = dataset.coord_data
                    framerate = dataset.framerate
                    stimulus_train = dataset.stimtrain_frame_stamps
                    simple_amp = np.empty((len(allFiles), len(reference_coord_data)))
                    simple_amp[:] = np.nan
                    ref_im = dataset.reference_im

                    reference_coord_data = refine_coord(ref_im, dataset.coord_data)

                    for c in range(len(dataset.coord_data)):
                        og_framestamps.append([])
                        cell_framestamps.append([])
                        mean_cell_profiles.append([])
                        texture_cell_profiles.append([])
                        full_cell_profiles.append([])

                    first = False

                dataset.coord_data = refine_coord_to_stack(dataset.video_data, ref_im, reference_coord_data)

                norm_video_data = norm_video(dataset.video_data, norm_method="mean", rescaled=True)

                full_profiles = extract_profiles(norm_video_data, dataset.coord_data, seg_radius=segmentation_radius, summary="none", sigma=1)

                temp_profiles = extract_profiles(norm_video_data, dataset.coord_data, seg_radius=segmentation_radius-1, summary="median", sigma=1)

                temp_profiles, good_profiles = exclude_profiles(temp_profiles, dataset.framestamps,
                                                 critical_region=np.arange(stimulus_train[0] - int(0.1 * framerate),
                                                                           stimulus_train[1] + int(0.2 * framerate)),
                                                 critical_fraction=0.4)

                full_profiles[:, :, :, ~good_profiles] = np.nan

                texture_dict = extract_texture_profiles(full_profiles, ("homogeneity", "energy"), 32,
                                                        framestamps=dataset.framestamps, display=False)

                stdize_profiles, reconst_framestamps, nummissed = reconstruct_profiles(temp_profiles,
                                                                                        dataset.framestamps)

                homogeneity, reconst_framestamps, nummissed = reconstruct_profiles(texture_dict["homogeneity"],
                                                                                       dataset.framestamps)

                # reconst_framestamps = dataset.framestamps
                # stdize_profiles = temp_profiles
                # homogeneity = texture_dict["homogeneity"]


                # Put the profile of each cell into its own array
                for c in range(len(dataset.coord_data)):
                    og_framestamps[c].append(dataset.framestamps)
                    cell_framestamps[c].append(reconst_framestamps)  # HERE IS THE PROBLEM, YO - appends extra things to the same cell, despite not being long enough
                    mean_cell_profiles[c].append(stdize_profiles[c, :])
                    texture_cell_profiles[c].append(homogeneity[c, :])
                    full_cell_profiles[c].append(full_profiles[:, :, :, c])

                r += 1

                if dataset.framestamps[-1] > max_frmstamp:
                    max_frmstamp = dataset.framestamps[-1]

        del dataset, temp_profiles, stdize_profiles

        # Rows: Acquisitions
        # Cols: Framestamps
        # Depth: Coordinates
        all_cell_mean_iORG = np.full((len(allFiles[loc]), max_frmstamp + 1, len(reference_coord_data)), np.nan)
        all_cell_texture_iORG = np.full((len(allFiles[loc]), max_frmstamp + 1, len(reference_coord_data)), np.nan)

        # This nightmare fuel of a matrix is arranged:
        # Dim 1: Profile number
        # Dim 2/3: Intensity profile
        # Dim 4: Temporal framestamp
        # Dim 5: Cell number
        all_full_cell_iORG = np.full((len(allFiles[loc]), segmentation_radius*2+1, segmentation_radius*2+1,
                                      max_frmstamp + 1, len(reference_coord_data)), np.nan)

        full_framestamp_range = np.arange(max_frmstamp + 1)
        cell_power_iORG = np.full((len(reference_coord_data), max_frmstamp + 1), np.nan)

        cwt_window_start = int(-0.25*framerate)
        cwt_window_end = int(1*framerate)
        # Make 3D matricies, where:
        # The first dimension (rows) is individual acquisitions, where NaN corresponds to missing data
        # The second dimension (columns) is time
        # The third dimension is each tracked coordinate
        cell_amp = np.full( (len(reference_coord_data), len(allFiles[loc])), np.nan)
        peak_scale = np.full_like(cell_amp, np.nan)
        fad = np.full_like(cell_amp, np.nan)

        for c in range(len(reference_coord_data)):
            for i, profile in enumerate(mean_cell_profiles[c]):
                all_cell_mean_iORG[i, cell_framestamps[c][i], c] = profile
                all_cell_texture_iORG[i, cell_framestamps[c][i], c] = texture_cell_profiles[c][i]
                #all_full_cell_iORG[i, :, :, og_framestamps[c][i], c] = full_cell_profiles[c][i].reshape(all_full_cell_iORG[i, :, :, og_framestamps[c][i], c].shape, order="F")

                save_tiff_stack(res_dir.joinpath(allFiles[loc][i].name[0:-4] + "cell(" + str(reference_coord_data[c][0]) + "," +
                                                  str(reference_coord_data[c][1]) + ")_vid_" + str(i) + ".tif"),
                                                  full_cell_profiles[c][i])
            #
            # plt.figure(11)
            # plt.clf()
            # plt.imshow(ref_im)
            # plt.plot(reference_coord_data[c][0], reference_coord_data[c][1], "r*")

            # What about a temporal histogram?
            fad[c, :] = filtered_absolute_difference(all_cell_mean_iORG[:, :, c], full_framestamp_range, filter_type="trunc_sinc")

            cell_power_iORG[c, :], numincl = signal_power_iORG(all_cell_mean_iORG[:, :, c], full_framestamp_range,
                                                               summary_method="rms", window_size=3)

        plt.figure(11)
        plt.clf()
        plt.hist(np.nanmean(fad, axis=1), 50)
        plt.show(block=False)
        plt.waitforbuttonpress()

        for c in range(len(reference_coord_data)):


            prestim_amp = np.nanmedian(cell_power_iORG[c, 0:stimulus_train[0]])
            poststim_amp = np.nanmedian(cell_power_iORG[c, stimulus_train[1]:(stimulus_train[1] + 10)])

            simple_amp[l, c] = poststim_amp - prestim_amp

        # TODO: Calling the coordclip fxn to return the simple_amp that corresponds to a 100 cone ROI
        # clippedcoords = coordclip(coord_data, 10, 100, 'i')

        # plt.figure(0)
        # plt.clf()
        # for a in range(all_cell_iORG.shape[0]):
        #     plt.plot(full_framestamp_range, all_cell_iORG[a, :, c])
        #     plt.plot(stimulus_train[0], poststim_amp, "rD")
        # plt.hist(simple_amp)
        # plt.plot(cell_power_iORG[c, :])
        #   plt.show(block=False)
        #   plt.waitforbuttonpress()
        # plt.savefig(res_dir.joinpath(this_dirname +  + "_allcell_iORG_amp.png"))

        # find the cells with the min, med, and max amplitude
        min_amp = np.nanmin(simple_amp[0, :])
        [min_amp_row, min_amp_col] = np.where(simple_amp == min_amp)
        # print('min_amp ',min_amp)
        med_amp = np.nanmedian(simple_amp[0, :])
        near_med_amp = find_nearest(simple_amp[0, :], med_amp)
        [med_amp_row, med_amp_col] = np.where(simple_amp == near_med_amp)

        # print('med_amp ', med_amp)
        max_amp = np.nanmax(simple_amp[0, :])
        [max_amp_row, max_amp_col] = np.where(simple_amp == max_amp)
        # print('max_amp ', max_amp)

        plt.figure(1)
        histbins = np.arange(start=-0.2, stop=1.5, step=0.025)
        plt.hist(simple_amp[l, :], bins=histbins)
        # plt.plot(cell_power_iORG[c, :], "k-", alpha=0.05)
        plt.show(block=False)
        plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_amp.png"))
        # plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_amp.svg"))
        plt.close(plt.gcf())

        hist_normie = Normalize(vmin=histbins[0], vmax=histbins[-1])
        hist_mapper = plt.cm.ScalarMappable(cmap=plt.get_cmap("magma"), norm=hist_normie)

        # simple_amp_norm = (simple_amp-histbins[0])/(histbins[-1] - histbins[0])

        plt.figure(2)
        vor = Voronoi(reference_coord_data)
        voronoi_plot_2d(vor, show_vertices=False, show_points=False)
        for c, cell in enumerate(vor.regions[1:]):
            if not -1 in cell:
                poly = [vor.vertices[i] for i in cell]
                plt.fill(*zip(*poly), color=hist_mapper.to_rgba(simple_amp[l, c]))
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="box")
        plt.show(block=False)
        plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_voronoi.png"))
        # plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_voronoi.svg"))
        plt.close(plt.gcf())

        # plotting the cells with the min/med/max amplitude
        plt.figure(300)
        # plt.plot(np.reshape(full_framestamp_range,(1,176)).astype('float64'),cell_power_iORG[min_amp_col,:])
        plt.plot(np.reshape(full_framestamp_range, (176, 1)).astype('float64'),
                 np.transpose(cell_power_iORG[min_amp_col, :]))
        plt.plot(np.reshape(full_framestamp_range, (176, 1)).astype('float64'),
                 np.transpose(cell_power_iORG[med_amp_col, :]))
        plt.plot(np.reshape(full_framestamp_range, (176, 1)).astype('float64'),
                 np.transpose(cell_power_iORG[max_amp_col, :]))
        # This also works...
        # plt.plot(full_framestamp_range.astype('float64'),
        #         np.ravel(cell_power_iORG[min_amp_col, :]))

        # should really be the cell_framestamps that correspond to the cells on the x axis
        # need to fix the bug with the framstamps being empty first though
        # plt.plot(cell_framestamps[min_amp_col, :],cell_power_iORG[min_amp_col, :])
        plt.savefig(res_dir.joinpath(this_dirname + "_MinMedMax_amp_cones.png"))
        plt.show(block=False)
        plt.close(plt.gcf())

        # output cell_power_iORG to csv (optional)
        if outputcsv:
            import csv

            csv_dir = res_dir.joinpath(this_dirname + "_cell_power_iORG.csv")
            print(csv_dir)
            f = open(csv_dir, 'w', newline="")
            writer = csv.writer(f, delimiter=',')
            writer.writerows(cell_power_iORG)
            f.close

        print("Done!")
        print(stimulus_train)
