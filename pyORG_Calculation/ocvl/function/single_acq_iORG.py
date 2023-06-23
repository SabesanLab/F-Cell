import os
from multiprocessing import Pool
from pathlib import Path
from tkinter import Tk, filedialog, ttk, HORIZONTAL

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from numpy import random

from scipy.spatial.distance import pdist, squareform

from ocvl.function.analysis.cell_profile_extraction import extract_profiles, norm_profiles, standardize_profiles, \
    refine_coord, refine_coord_to_stack, exclude_profiles
from ocvl.function.analysis.iORG_profile_analyses import signal_power_iORG, wavelet_iORG, extract_texture_profiles, \
    iORG_signal_metrics, pooled_variance
from ocvl.function.preprocessing.improc import norm_video
from ocvl.function.utility.generic import PipeStages
from ocvl.function.utility.meao import MEAODataset
from ocvl.function.utility.resources import save_tiff_stack
from ocvl.function.utility.temporal_signal_utils import reconstruct_profiles, densify_temporal_matrix, trim_video


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def fast_rms_avg(all_iORG, framestamp_range, stiminds):

    max_num_sig = all_iORG.shape[0]
    cell_rms_amp = np.full((all_iORG.shape[-1], max_num_sig), np.nan)

    for cellind in range(all_iORG.shape[-1]):
        this_cell_iORG = all_iORG[:, :, cellind]

        rngesus = np.random.default_rng()  # Shuffle our seed, and reorder the signals.
        avg_order = rngesus.permutation(max_num_sig)
        this_cell_iORG = this_cell_iORG[avg_order, :]

        valid = np.any(np.isfinite(this_cell_iORG), axis=1)

        if np.sum(valid) > 5:
            this_cell_iORG = this_cell_iORG[valid, :]  # Only analyze valid signals- lets not waste our time with nans.
            cell_rms_iORG = np.full_like(this_cell_iORG, np.nan)

            # After the rando reorder, then determine the RMS signals of ever increasing numbers of samples
            for ind in range(0, this_cell_iORG.shape[0]):

                cell_rms_iORG[ind, :] = signal_power_iORG(this_cell_iORG[0:ind+1, :], framestamp_range,
                                                          summary_method="rms", window_size=1)[0]

                cell_rms_amp[cellind, ind] = iORG_signal_metrics(cell_rms_iORG[ind, :].reshape((1, cell_rms_iORG.shape[1])),
                                                              framestamp_range,
                                                              filter_type="none", notch_filter=None, display=False,
                                                              prestim_idx=stiminds[0], poststim_idx=stiminds[1])[0]
    return cell_rms_amp


def fast_acq_avg(fad_data):
    max_num_avg = fad_data.shape[1]
    fad_avg = np.full((fad_data.shape[0], max_num_avg), np.nan)
    rng = np.random.default_rng()  # Shuffle our seed.
    avg_order = rng.permutation(max_num_avg)

    for c in range(fad_data.shape[0]):
        fad_data[c, :] = fad_data[c, avg_order]  # reshuffle the order of the data.

        valid = np.isfinite(fad_data[c, :])
        valid_fads = fad_data[c, valid]
        if np.sum(valid) > 5: # Only include cells with more than 10 samples- otherwise drop them as they're probably noisy.
            valid_range = np.arange(1, len(valid_fads) + 1)
            fad_avg[c, valid_range - 1] = np.cumsum(valid_fads, axis=0)
            fad_avg[c, valid_range - 1] = (fad_avg[c, valid_range - 1] / valid_range)
    return fad_avg


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

            if (path.parent.parent == searchpath or path.parent == searchpath):
                if path.parent not in allFiles:
                    allFiles[path.parent] = []
                    allFiles[path.parent].append(path)

                    if "control" in path.parent.name:
                        # print("DETECTED CONTROL DATA AT: " + str(path.parent))
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

    outputcsv = False

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
        segmentation_radius = None # If set to None, then try and autodetect from the data.

        res_dir = loc.joinpath("Results")  # creates a results folder within loc ex: R:\00-23045\MEAOSLO1\20220325\Functional\Processed\Functional Pipeline\(1,0)\Results
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
                    og_framestamps = []
                    cell_framestamps = []
                    mean_cell_profiles = []
                    texture_cell_profiles = []
                    full_cell_profiles = []

                    reference_coord_data = dataset.coord_data
                    framerate = dataset.framerate
                    stimulus_train = dataset.stimtrain_frame_stamps
                    ref_im = dataset.reference_im

                    # reference_coord_data = refine_coord(ref_im, dataset.coord_data) # REINSTATE MEEEEE
                    reference_coord_data = dataset.coord_data-1

                    coorddist = pdist(reference_coord_data, "euclidean")
                    coorddist = squareform(coorddist)
                    coorddist[coorddist == 0] = np.amax(coorddist.flatten())
                    mindist = np.amin( coorddist, axis=-1)

                    if not segmentation_radius:
                        segmentation_radius = np.round(np.nanmean(mindist) / 4) if np.round(np.nanmean(mindist) / 4) >= 1 else 1

                        segmentation_radius = int(segmentation_radius)
                        print("Detected segmentation radius: " + str(segmentation_radius))

                    for c in range(len(dataset.coord_data)):
                        og_framestamps.append([])
                        cell_framestamps.append([])
                        mean_cell_profiles.append([])
                        texture_cell_profiles.append([])
                        full_cell_profiles.append([])

                    first = False

                dataset.coord_data = refine_coord_to_stack(dataset.video_data, ref_im, reference_coord_data)

                # full_profiles = extract_profiles(dataset.video_data, dataset.coord_data, seg_radius=segmentation_radius+1,
                #                                  summary="none", sigma=1)

                norm_video_data = norm_video(dataset.video_data, norm_method="mean", rescaled=True)

                temp_profiles = extract_profiles(norm_video_data, dataset.coord_data, seg_radius=segmentation_radius,
                                                 seg_mask="disk", summary="mean")

                temp_profiles = standardize_profiles(temp_profiles, dataset.framestamps,
                                                     stimulus_stamp=stimulus_train[0], method="mean_sub")

                temp_profiles, good_profiles = exclude_profiles(temp_profiles, dataset.framestamps,
                                                 critical_region=np.arange(stimulus_train[0] - int(0.1 * framerate),
                                                                           stimulus_train[1] + int(0.2 * framerate)),
                                                 critical_fraction=0.5)

                # full_profiles[:, :, :, ~good_profiles] = np.nan

                stdize_profiles, reconst_framestamps, nummissed = reconstruct_profiles(temp_profiles,
                                                                                       dataset.framestamps,
                                                                                       method="L1",
                                                                                       threshold=0.3)

                # proffft = np.fft.fft(stdize_profiles, axis=1)
                # avgfft = np.nanmean(proffft, axis=0)
                # plt.plot((reconst_framestamps / (reconst_framestamps[-1]+1)) / (1 / framerate), (np.abs(avgfft) ** 2))
                # plt.waitforbuttonpress()

                # Put the profile of each cell into its own array
                for c in range(len(dataset.coord_data)):
                    og_framestamps[c].append(dataset.framestamps)
                    cell_framestamps[c].append(reconst_framestamps)
                    mean_cell_profiles[c].append(stdize_profiles[c, :])
                    # texture_cell_profiles[c].append(homogeneity[c, :])
                    # full_cell_profiles[c].append(full_profiles[:, :, :, c])

                r += 1

                if dataset.framestamps[-1] > max_frmstamp:
                    max_frmstamp = dataset.framestamps[-1]

        del dataset, temp_profiles, stdize_profiles

        # Rows: Acquisitions
        # Cols: Framestamps
        # Depth: Coordinates
        all_cell_iORG = np.full((len(allFiles[loc]), max_frmstamp + 1, len(reference_coord_data)), np.nan)
        all_cell_texture_iORG = np.full((len(allFiles[loc]), max_frmstamp + 1, len(reference_coord_data)), np.nan)

        # This nightmare fuel of a matrix is arranged:
        # Dim 1: Profile number
        # Dim 2/3: Intensity profile
        # Dim 4: Temporal framestamp
        # Dim 5: Cell number
        all_full_cell_iORG = np.full((len(allFiles[loc]), segmentation_radius*2+1, segmentation_radius*2+1,
                                      max_frmstamp + 1, len(reference_coord_data)), np.nan)

        full_framestamp_range = np.arange(max_frmstamp+1)
        cell_power_iORG = np.full((len(reference_coord_data), max_frmstamp + 1), np.nan)
        cell_power_fad = np.full((len(reference_coord_data)), np.nan)
        cell_power_amp = np.full((len(reference_coord_data)), np.nan)
        indiv_fad_iORG = np.full((len(reference_coord_data), max_frmstamp + 1), np.nan)

        # Make 3D matricies, where:
        # The first dimension (rows) is individual acquisitions, where NaN corresponds to missing data
        # The second dimension (columns) is time
        # The third dimension is each tracked coordinate
        cell_amp = np.full( (len(reference_coord_data), len(allFiles[loc])), np.nan)
        peak_scale = np.full_like(cell_amp, np.nan)
        indiv_fad = np.full_like(cell_amp, np.nan)
        prestim_mean = np.full_like(cell_amp, np.nan)

        prestim_ind = np.flatnonzero(np.logical_and(full_framestamp_range < stimulus_train[0],
                                                    full_framestamp_range >= (stimulus_train[0] - int(0.5 * framerate))))
        poststim_ind = np.flatnonzero(np.logical_and(full_framestamp_range >= stimulus_train[0],
                                                     full_framestamp_range <= (stimulus_train[0] + int(0.5 * framerate))))

        # For Ram comparison 05/10/23
        opl_vals = pd.read_csv("P:\\RFC_Projects\\SabLab_Collab\\03_May_2023_LSO&OCT_ORG\\OPL_final_vals.csv",
                               delimiter=',', encoding="utf-8-sig", header=None).to_numpy()
        opl_vals=opl_vals.flatten()
        low_responders = (opl_vals<150)
        #
        # opl_full = pd.read_csv("P:\\RFC_Projects\\SabLab_Collab\\03_May_2023_LSO&OCT_ORG\\OPL_full.csv",
        #                        delimiter=',', encoding="utf-8-sig", header=None).to_numpy()


        # plt.figure(43)
        # # plt.clf()
        # plt.imshow(ref_im, cmap="gray")
        # plt.plot(reference_coord_data[low_responders,0], reference_coord_data[low_responders,1], "*")
        # plt.show(block=False)

        mapper = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis", all_cell_iORG.shape[0]))
        for c in range(len(reference_coord_data)):
            for i, profile in enumerate(mean_cell_profiles[c]):
                all_cell_iORG[i, cell_framestamps[c][i], c] = profile

                prestim_mean[c, i] = np.nanmean(all_cell_iORG[i, prestim_ind, c])

            if np.sum(np.any(np.isfinite(all_cell_iORG[:, :, c]), axis=1)) >= (all_cell_iORG[:, :, c].shape[0]/2):
                if ~low_responders[c]:
                    todisp = False
                else:
                    todisp = True
                # todisp = False

                indiv_fad[c, :], _, _, fad_profiles = iORG_signal_metrics(all_cell_iORG[:, :, c], full_framestamp_range,
                                                            framerate, filter_type="MS1", notch_filter=None, display=todisp, fwhm_size=18,
                                                            prestim_idx=prestim_ind, poststim_idx=poststim_ind)
                # Have used 1-2 before.
                indiv_fad[indiv_fad == 0] = np.nan
                fad_profiles[fad_profiles == 0] = np.nan



            # if todisp and np.sum(np.any(np.isfinite(all_cell_iORG[:, :, c]), axis=1)) >= 1 and np.nanmean(indiv_fad[c, :], axis=-1) < 30:
                # print("OPL: "+ str(opl_vals[c]))
                # mapper = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis", 60))
                # print(c)
                # plt.figure(43)
                # # plt.clf()
                # # plt.imshow(ref_im, cmap="gray")
                # plt.plot(reference_coord_data[c,0], reference_coord_data[c,1], "r*",
                #          color=mapper.to_rgba(int(np.nanmean(indiv_fad[c, :], axis=-1)), norm=False))
                # plt.waitforbuttonpress()

                #save_tiff_stack("ex.tif", full_cell_profiles[c][0][:,:,:])

                # print("iORG: " + str(np.nanmean(indiv_fad[c, :], axis=-1)))
                #full_cell_profiles[:,:,:,c]

                # plt.waitforbuttonpress()

                # proffft = np.fft.fft(all_cell_iORG[:, :, c], axis=1)
                # avgfft = np.nanmean(proffft, axis=0)
                # plt.figure(9000)
                # plt.clf()
                # plt.plot((full_framestamp_range / (full_framestamp_range[-1]+1)) / (1 / framerate), (np.abs(avgfft) ** 2))
                # plt.waitforbuttonpress()


            # if np.sum(np.any(np.isfinite(fad_profiles), axis=1)) >= 8: # Only include cells that have data for each acquisition.
            #     prestim_avg = np.nanmedian(fad_profiles[:, 0:prestim_ind[-1]], axis=0)
            #     prestim_deriv = np.abs(np.gradient(prestim_avg))
            #
            #     prestim_gradient = np.nancumsum(np.repeat(np.nanmean(prestim_deriv), fad_profiles.shape[1]))
            #
            #     prestim_indiv_deriv = np.abs(np.gradient(fad_profiles[:, 0:prestim_ind[-1]-4], axis=1))
            #     prestim_indiv_med_derv = np.nanmean(prestim_indiv_deriv, axis=1)
            #     prestim_indiv_gradient = np.repeat(prestim_indiv_med_derv[:,None], fad_profiles.shape[1], axis=1)
            #     prestim_indiv_gradient = np.nancumsum(prestim_indiv_gradient, axis=1)
            #
            #     tmp = prestim_indiv_gradient[:, (prestim_ind[0] - 4)]
            #     prestim_indiv_gradient -= np.repeat(tmp[:,None], fad_profiles.shape[1], axis=1)
            #     tmp = fad_profiles[:, (prestim_ind[0] - 4)]
            #     prestim_indiv_gradient += np.repeat(tmp[:,None], fad_profiles.shape[1], axis=1)
            #
            #     test_iORG = np.squeeze(np.nanmean((fad_profiles - prestim_indiv_gradient), axis=0))
            #
            #     indiv_fad_iORG[c, :] = np.squeeze(np.nanmean(fad_profiles, axis=0)) - prestim_gradient
            #
            #     # For usurping everything and running the tweaked approach on whole datasets.
            #     # indiv_fad[c, :] = np.nan
            #     # indiv_fad[c, 0] = np.nanmax(test_iORG[poststim_ind[-1]-4]) - test_iORG[poststim_ind[0]-4]
            #
            #     # if np.sum(np.any(np.isfinite(fad_profiles), axis=1)) >= fad_profiles.shape[0]/2:
            #     if todisp and (np.nanmean(indiv_fad[c, :]) <= 15):
            #         plt.figure(43)
            #         plt.clf()
            #         for i in range(fad_profiles.shape[0]):
            #             plt.subplot(2, 2, 1)
            #             plt.plot(fad_profiles[i,:],color=mapper.to_rgba(i, norm=False))
            #             plt.plot(prestim_avg, color="k")
            #             plt.plot(prestim_gradient, color="k", alpha=0.3)
            #             plt.subplot(2, 2, 2)
            #             plt.plot((fad_profiles[i,:]- prestim_gradient),color=mapper.to_rgba(i, norm=False))
            #             plt.plot(indiv_fad_iORG[c, :], color="k")
            #             plt.subplot(2, 2, 3)
            #             plt.plot(fad_profiles[i,:],color=mapper.to_rgba(i, norm=False))
            #             plt.plot(prestim_indiv_gradient[i,:], color="k", alpha=0.3)
            #             plt.subplot(2, 2, 4)
            #             plt.plot((fad_profiles[i,:] - prestim_indiv_gradient[i,:]),color=mapper.to_rgba(i, norm=False))
            #             plt.plot(test_iORG, color="k")
            #             plt.show(block=False)
            #             print("iORG: " + str(np.nanmean(indiv_fad[c, :])))
            #             print(indiv_fad[c, :])
            #         plt.waitforbuttonpress()


            cell_power_iORG[c, :], numincl = signal_power_iORG(all_cell_iORG[:, :, c], full_framestamp_range,
                                                               summary_method="rms", window_size=1, display=False)

            cell_power_fad[c], cell_power_amp[c] = iORG_signal_metrics(cell_power_iORG[c, :].reshape((1, cell_power_iORG.shape[1])),
                                                    full_framestamp_range, framerate,
                                                    filter_type="none", notch_filter=None, display=False,
                                                    prestim_idx=prestim_ind, poststim_idx=poststim_ind)[0:2]

        cell_power_fad[cell_power_fad == 0] = np.nan



        plt.figure(42)
        plt.clf()
        # plt.plot(indiv_fad_iORG[:, prestim_ind[0]:poststim_ind[-1]].transpose())
        plt.hist(np.nanmedian(indiv_fad[~low_responders,:], axis=-1),color="orange",bins=100)
        plt.hist(np.nanmedian(indiv_fad[low_responders, :], axis=-1), color="b",bins=100)
        plt.show(block=False)
        plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_mad_wlowresp.png"))

        plt.figure(44)
        plt.clf()
        plt.hist(np.nanmean(indiv_fad, axis=-1),bins=100)
        plt.show(block=False)
        plt.waitforbuttonpress()



        # *** MAKE THIS A PARAM ***
        enough_data = np.sum(np.isfinite(indiv_fad), axis=1) >= np.floor( len(allFiles[loc])/2 )

        # If we have more than 1 dataset we're analyzing, then we'll need to squeeze it after removing low-dataset counts
        if len(cell_power_fad.shape) > 2:
            cell_power_fad = np.squeeze(cell_power_fad[enough_data])
            indiv_fad = np.squeeze(indiv_fad[enough_data, :])
            prestim_mean = np.squeeze(prestim_mean[enough_data, :])

        # Log transform the data... or don't.
        log_cell_power_fad = np.log(cell_power_fad)
        log_indiv_fad = np.log(indiv_fad)

        #plt.figure(69)
        #plt.plot(indiv_fad[c, :], np.abs(1 - prestim_mean[c, :]), "*")
        # twodee_histbins = np.arange(start=0, stop=255, step=10.2)
        # plt.hist2d(prestim_mean[np.isfinite(log_indiv_fad)].flatten(), log_indiv_fad[np.isfinite(log_indiv_fad)].flatten(), bins=twodee_histbins)

        histbins = np.arange(start=0.9, stop=2.0, step=0.025)

        # plt.figure(10)
        # plt.hist((cell_power_fad), bins=np.arange(0, 255, 5), density=True)
        # plt.title("RMS power MAD")
        # plt.show(block=False)
        # plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_mad_3coordprofiles.svg"))

        # plt.figure(111)
        # plt.hist(np.log(cell_power_fad), bins=np.arange(2.25, 5.25, 0.05), density=True)
        # plt.title("RMS power logMAD")
        # plt.show(block=False)
        # plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_rms_logmad.svg"))
        #
        # plt.figure(112)
        # plt.hist(np.log(cell_power_fad), bins=np.arange(2.25, 5.25, 0.05), density=True, histtype="step",
        #          cumulative=True, label=loc.name)
        # plt.title("RMS power logMAD cumulative")
        # plt.legend(loc="best")
        # plt.show(block=False)
        # plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_rms_logmad_cumulative.svg"))

        plt.figure(111)
        plt.hist(np.log(cell_power_amp), bins=np.arange(0, 5.5, 0.1), density=True)
        plt.title("RMS power log amplitude")
        plt.show(block=False)
        plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_rms_logamp.svg"))

        plt.figure(112)
        plt.hist(np.log(cell_power_amp), bins=np.arange(0, 5.5, 0.1), density=True, histtype="step",
                 cumulative=True, label=loc.name)
        plt.title("RMS power log amplitude cumulative")
        plt.legend(loc="best")
        plt.show(block=False)
        plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_rms_logamp_cumulative.svg"))


        plt.figure(12)
        plt.hist((np.nanmean(indiv_fad, axis=-1)), bins=np.arange(-500, 800, 10), density=True)
        plt.title("Maximum absolute deviation Median:" + str(np.nanmedian(indiv_fad.flatten())) )
        #plt.xlim((0, 500))
        plt.show(block=False)
        plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_mad.svg"))

        plt.figure(131)
        plt.hist(np.log(np.nanmean(indiv_fad, axis=-1)), bins=np.arange(-3, 6, 0.1), density=True, label=loc.name)
        plt.title("Log Maximum absolute deviation: Median:" + str(np.log(np.nanmedian(indiv_fad.flatten()))))
        plt.show(block=False)
        plt.legend(loc="best")
        plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_logmad.svg"))

        plt.figure(132)
        plt.hist(np.log(np.nanmean(indiv_fad, axis=-1)), bins=np.arange(0, 6, 0.05), density=True, histtype="step",
                 cumulative=True, label=loc.name)
        plt.title("Log Maximum absolute deviation cumulative: Median:" + str(np.log(np.nanmedian(indiv_fad.flatten()))) )
        plt.show(block=False)
        plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_logmad_cumulative.svg"))

        # plt.figure(14)
        # plt.plot(np.nanmean(log_indiv_fad, axis=-1),
        #          np.nanstd(log_indiv_fad, axis=-1),".")
        # plt.title("logFAD mean vs logFAD std dev")
        # plt.show(block=False)
        # plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_logamp_vs_stddev.svg"))

        plt.figure(15)
        plt.plot(np.nanmean(indiv_fad, axis=-1),
                 np.nanstd(indiv_fad, axis=-1),".")
        plt.title("FAD vs std dev")
        plt.show(block=False)
        plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_amp_vs_stddev.svg"))

        overoneforty = np.flatnonzero(np.nanmean(indiv_fad, axis=-1) > 140)

        pvar, pmean = pooled_variance(log_indiv_fad)

        pstddev = np.sqrt(pvar)
        antilog_stddev = np.exp(pstddev)
        antilog_2stddev = np.exp(2*pstddev) # Or antilog_stddev ** 2
        print("Geometric Standard Deviation: " + str(antilog_stddev-1))
        print("Geometric Coefficient of Variation: %" + str(100 * np.sqrt(np.exp(pvar)-1)) )

        # This bit is to output data for ISER.
        numsamples = np.nansum(np.isfinite(log_indiv_fad), axis=-1)
        weighted_var = np.nanvar(log_indiv_fad, axis=-1) * numsamples

        iserdata = {"Pooled Variance": [pvar], "Geometric Coefficient of Variation:" : [np.sqrt(np.exp(pvar)-1)],
                    "Weighted Sum of Variances": [np.nansum(weighted_var.flatten())],
                   "Total Samples": [np.nansum(np.isfinite(log_indiv_fad.flatten()))],
                    "Num Cells": [np.nansum(numsamples>1)]}

        # outdata = pd.DataFrame(iserdata)
        # outdata.to_csv(res_dir.joinpath(this_dirname + "_iser2023_data.csv"))

        outdata = pd.DataFrame(log_indiv_fad)
        outdata.to_csv(res_dir.joinpath(this_dirname + "_allcell_iORG_logFAD.csv"), index=False)
        outdata = pd.DataFrame(np.log(cell_power_fad))
        outdata.to_csv(res_dir.joinpath(this_dirname + "_allcell_iORG_RMS_logFAD.csv"), index=False)
        outdata = pd.DataFrame(np.log(cell_power_amp))
        outdata.to_csv(res_dir.joinpath(this_dirname + "_allcell_iORG_RMS_logamp.csv"), index=False)

        monte = False
        plt.waitforbuttonpress()

        # Monte carlo section- attempting to determine point at which there isn't much of a change between the value as
        # a function of randomly included numbers.
        if monte:

            # Do the Monte Carlo of the individual acquisition data.
            numiter = 200
            max_num_avg = indiv_fad.shape[1]
            log_fad_avg = np.full((indiv_fad.shape[0], max_num_avg, numiter), np.nan)

            with Pool(processes=10) as pool:
                thread_res = []
                for i in range(numiter):
                    #print("Submitting iteration: " + str(i))
                    thread_res.append( pool.apply_async(fast_acq_avg, args=(indiv_fad,)))

                for i in range(numiter):
                    #print("Recieving iteration: "+str(i))
                    log_fad_avg[:, :, i] = thread_res[i].get()

            plt.figure(16)
            plt.gcf()
            intra_cell_fad_GCV = np.full((indiv_fad.shape[0], max_num_avg), np.nan)
            for c in range(indiv_fad.shape[0]):
                cellreps = np.log(log_fad_avg[c, :, :])
                cellreps = cellreps[~np.all(np.isnan(cellreps), axis=-1), :] # Remove all nans- we don't have data here.
                if cellreps.size != 0:
                    intra_cell_fad_GCV[c, 0:cellreps.shape[0] - 1] = np.sqrt( np.exp(np.nanvar(cellreps[0:-1,:], axis=-1))-1)
                    plt.plot( np.sqrt( np.exp(np.nanvar(cellreps, axis=-1))-1) )
                # plt.plot( np.nanmean(cellreps, axis=-1)+ 2*np.sqrt( np.exp(np.nanvar(cellreps, axis=-1))-1), color="black", linewidth=3)
            plt.draw()

            intra_cell_fad_GCV = intra_cell_fad_GCV.transpose().tolist()
            for j in range(len(intra_cell_fad_GCV)):
                intra_cell_fad_GCV[j] = np.array(intra_cell_fad_GCV[j])
                intra_cell_fad_GCV[j] = 100*intra_cell_fad_GCV[j][~np.isnan(intra_cell_fad_GCV[j])]

            outdata = pd.DataFrame(intra_cell_fad_GCV)
            outdata.to_csv(res_dir.joinpath(this_dirname + "_intracell_fad_GCV_monte.csv"), index=False)

            plt.figure(17)
            plt.clf()
            plt.boxplot(intra_cell_fad_GCV)
            plt.ylim((0, 150))
            plt.savefig(res_dir.joinpath(this_dirname + "_iORG_monte_fad_intra_gcv.png"))
            plt.savefig(res_dir.joinpath(this_dirname + "_iORG_monte_fad_intra_gcv.svg"))

            # ***************************************
            # * Do the Monte Carlo of the RMS data. *
            # ***************************************
            log_rms_avg = np.full((all_cell_iORG.shape[-1], max_num_avg, numiter), np.nan)
            intra_cell_rms_GCV = np.full((all_cell_iORG.shape[-1], max_num_avg), np.nan)

            print("Heya, heyaaaah")
            with Pool(processes=10) as pool:
                thread_res = []
                for i in range(numiter):
                    print("Submitting iteration: " + str(i))
                    thread_res.append(pool.apply_async(fast_rms_avg, args=(all_cell_iORG,
                                                                           full_framestamp_range,
                                                                           (prestim_ind, poststim_ind),) ))

                for i in range(numiter):
                    print("Recieving iteration: "+str(i))
                    log_rms_avg[:, :, i] = thread_res[i].get()

            plt.figure(18)
            plt.gcf()
            for c in range(indiv_fad.shape[0]):
                cellreps = np.log(log_rms_avg[c, :, :])
                cellreps = cellreps[~np.all(np.isnan(cellreps), axis=-1), :] # Remove all nans- we don't have data here.
                if cellreps.size != 0:
                    intra_cell_rms_GCV[c, 0:cellreps.shape[0] - 1] = np.sqrt( np.exp(np.nanvar(cellreps[0:-1,:], axis=-1))-1)
                    plt.plot( np.sqrt( np.exp(np.nanvar(cellreps, axis=-1))-1) )
                # plt.plot( np.nanmean(cellreps, axis=-1)+ 2*np.sqrt( np.exp(np.nanvar(cellreps, axis=-1))-1), color="black", linewidth=3)
            plt.draw()

            intra_cell_rms_GCV = intra_cell_rms_GCV.transpose().tolist()
            for j in range(len(intra_cell_rms_GCV)):
                intra_cell_rms_GCV[j] = np.array(intra_cell_rms_GCV[j])
                intra_cell_rms_GCV[j] = 100*intra_cell_rms_GCV[j][~np.isnan(intra_cell_rms_GCV[j])]

            outdata = pd.DataFrame(intra_cell_rms_GCV)
            outdata.to_csv(res_dir.joinpath(this_dirname + "_intracell_rms_GCV_monte.csv"), index=False)

            plt.figure(19)
            plt.clf()
            plt.boxplot(intra_cell_rms_GCV)
            plt.ylim((0, 150))
            plt.savefig(res_dir.joinpath(this_dirname + "_iORG_monte_rms_intra_gcv.png"))
            plt.savefig(res_dir.joinpath(this_dirname + "_iORG_monte_rms_intra_gcv.svg"))
            plt.draw()

            # plt.waitforbuttonpress()