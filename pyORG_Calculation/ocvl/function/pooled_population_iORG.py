import os
from os import walk
from os.path import splitext
from pathlib import Path
from tkinter import Tk, filedialog, ttk, HORIZONTAL

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ocvl.function.analysis.cell_profile_extraction import extract_profiles, norm_profiles, standardize_profiles, \
    refine_coord, refine_coord_to_stack, exclude_profiles
from ocvl.function.analysis.iORG_profile_analyses import signal_power_iORG
from ocvl.function.utility.generic import PipeStages
from ocvl.function.utility.meao import MEAODataset
from ocvl.function.utility.resources import save_tiff_stack
from ocvl.function.utility.temporal_signal_utils import reconstruct_profiles

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
    searchpath = Path(pName)
    for path in searchpath.rglob("*.avi"):
        if "piped" in path.name:
            splitfName = path.name.split("_")

            if path.parent not in allFiles:
                allFiles[path.parent] = []
                allFiles[path.parent].append(path)
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
    reference_coord_data = None
    maxnum_cells = None
    skipnum = 0

    for loc in allFiles:
        res_dir = loc.joinpath("Results")
        res_dir.mkdir(exist_ok=True)

        this_dirname = res_dir.parent.name

        r = 0
        pb["maximum"] = len(allFiles[loc])
        pop_iORG = []
        pop_iORG_implicit = np.empty((len(allFiles[loc])-skipnum+1))
        pop_iORG_implicit[:] = np.nan
        pop_iORG_recover = np.empty((len(allFiles[loc]) - skipnum + 1))
        pop_iORG_recover[:] = np.nan
        pop_iORG_amp = np.empty((len(allFiles[loc]) - skipnum + 1))
        pop_iORG_amp[:] = np.nan
        pop_iORG_num = []
        framestamps = []
        max_frmstamp = 0
        plt.figure(0)
        plt.clf()

        mapper = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis", len(allFiles[loc])))

        for file in allFiles[loc]:

            if "ALL_ACQ_AVG" not in file.name and r >= skipnum:
                pb["value"] = r
                pb_label["text"] = "Processing " + file.name + "..."
                pb.update()
                pb_label.update()

                dataset = MEAODataset(file.as_posix(), analysis_modality="760nm", ref_modality="760nm",
                                      stimtrain_path=stimtrain_fName, stage=PipeStages.PIPELINED)
                dataset.load_pipelined_data()

                if first:
                    reference_coord_data = refine_coord(dataset.reference_im, dataset.coord_data, search_radius=1)
                    full_profiles = []
                    first = False

                #dataset.coord_data = reference_coord_data
                dataset.coord_data = refine_coord_to_stack(dataset.video_data, dataset.reference_im, reference_coord_data, search_radius=3)
                
                if maxnum_cells is not None:
                    perm = np.random.permutation(len(dataset.coord_data))
                    perm = perm[0:maxnum_cells]
                else:
                    perm = np.arange(len(dataset.coord_data))
                print("Analyzing " + str(len(perm)) + " cells.")

                # full_profiles = extract_profiles(dataset.video_data, dataset.coord_data, seg_radius=2, summary="none")

                # for c in range(len(dataset.coord_data)):
                #      save_tiff_stack(
                #          res_dir.joinpath(file.name[0:-4] + "cell(" + str(dataset.coord_data[c][0]) + "," +
                #                           str(dataset.coord_data[c][1]) + ".tif"), full_profiles[:, :, :, c])

                temp_profiles = extract_profiles(dataset.video_data, dataset.coord_data[perm, :], seg_radius=1, display=False)

                temp_profiles, num_removed = exclude_profiles(temp_profiles, dataset.framestamps,
                                                              critical_region=np.arange(
                                                                  dataset.stimtrain_frame_stamps[0] - int(0.1 * dataset.framerate),
                                                                  dataset.stimtrain_frame_stamps[1] + int(0.2 * dataset.framerate)),
                                                              critical_fraction=0.4)

                norm_temporal_profiles = norm_profiles(temp_profiles, norm_method="mean", video_ref=dataset.video_data)
                stdize_profiles = standardize_profiles(norm_temporal_profiles, dataset.framestamps,
                                                       dataset.stimtrain_frame_stamps[0], method="mean_sub", display=False)


                # indiv_resp = pd.DataFrame(np.concatenate((np.reshape(dataset.framestamps/dataset.framerate, (1, len(dataset.framestamps))),
                #                                          stdize_profiles), axis=0))
                # indiv_resp.to_csv(res_dir.joinpath(file.name + "_cell_profiles.csv"), header=False, index=False)

                tmp_iorg, tmp_incl = signal_power_iORG(stdize_profiles, dataset.framestamps, summary_method="rms", window_size=1)

                prestim_ind = np.logical_and(dataset.framestamps < dataset.stimtrain_frame_stamps[0],
                                             dataset.framestamps >= (dataset.stimtrain_frame_stamps[0] - int(
                                                 1 * dataset.framerate)))
                poststim_ind = np.logical_and(dataset.framestamps >= dataset.stimtrain_frame_stamps[1],
                                              dataset.framestamps < (dataset.stimtrain_frame_stamps[1] + int(
                                                  0.5 * dataset.framerate)))
                poststim_loc = dataset.framestamps[poststim_ind]
                prestim_amp = np.nanmedian(tmp_iorg[prestim_ind])
                poststim = tmp_iorg[poststim_ind]

                if poststim.size == 0:
                    poststim_amp = np.NaN
                    prestim_amp = np.NaN
                    pop_iORG_amp[r] = np.NaN
                    pop_iORG_implicit[r] = np.NaN
                    pop_iORG_recover[r] = np.NaN
                else:
                    poststim_amp = np.quantile(poststim, [0.95])
                    max_frmstmp = poststim_loc[np.argmax(poststim)] - dataset.stimtrain_frame_stamps[0]
                    print(str(max_frmstmp))
                    final_val = np.mean(tmp_iorg[-5:])
                    pop_iORG_implicit[r] = 1000*max_frmstmp / dataset.framerate
                    pop_iORG_amp[r] = (poststim_amp - prestim_amp)
                    pop_iORG_recover[r] = 1-((final_val-prestim_amp) / pop_iORG_amp[r])

                    framestamps.append(dataset.framestamps)
                    pop_iORG.append(tmp_iorg)
                    pop_iORG_num.append(tmp_incl)

                    print("iORG Amplitude: " + str(pop_iORG_amp[r]) + " Implicit time (ms): " + str(pop_iORG_implicit[r]) +
                          " Recovery fraction: " + str(pop_iORG_recover[r]))

                    plt.figure(0)

                    plt.xlabel("Time (seconds)")
                    plt.ylabel("Response")
                    plt.plot(dataset.framestamps/dataset.framerate, pop_iORG[r - skipnum], color=mapper.to_rgba(r - skipnum, norm=False),
                             label=file.name)
                    plt.show(block=False)
                    #plt.savefig(res_dir.joinpath(file.name[0:-4] + "_pop_iORG.png"))
                    r += 1

                if dataset.framestamps[-1] > max_frmstamp:
                    max_frmstamp = dataset.framestamps[-1]

        plt.vlines(dataset.stimtrain_frame_stamps[0] / dataset.framerate, -1, 10, color="red")
        plt.xlim([0,  max_frmstamp/dataset.framerate])
        plt.ylim([0, 1.5])
        #plt.legend()
        plt.savefig( res_dir.joinpath(this_dirname + "_pop_iORG.svg"))
        plt.savefig( res_dir.joinpath(this_dirname + "_pop_iORG.png") )


        # Grab all of the
        all_iORG = np.empty((len(pop_iORG), max_frmstamp+1))
        all_iORG[:] = np.nan
        all_incl = np.empty((len(pop_iORG), max_frmstamp + 1))
        all_incl[:] = np.nan
        for i, iorg in enumerate(pop_iORG):
            all_incl[i, framestamps[i]] = pop_iORG_num[i]
            all_iORG[i, framestamps[i]] = iorg

        # Pooled variance calc
        pooled_iORG = np.nansum( all_incl*all_iORG, axis=0 ) / np.nansum(all_incl, axis=0)
        #pooled_stddev_iORG = np.sqrt(pooled_var_iORG)
        all_frmstamps = np.arange(max_frmstamp+1)

        prestim_ind = np.logical_and(all_frmstamps < dataset.stimtrain_frame_stamps[0],
                                     all_frmstamps >= (dataset.stimtrain_frame_stamps[0] - int(1 * dataset.framerate)))
        poststim_ind = np.logical_and(all_frmstamps >= dataset.stimtrain_frame_stamps[1],
                                      all_frmstamps < (dataset.stimtrain_frame_stamps[1] + int(0.5 * dataset.framerate)))
        poststim_loc = all_frmstamps[poststim_ind]
        prestim_amp = np.nanmedian(pooled_iORG[prestim_ind])
        poststim = pooled_iORG[poststim_ind]

        if poststim.size == 0:
            poststim_amp = np.NaN
            prestim_amp = np.NaN
            pop_iORG_amp[r] = np.NaN
            pop_iORG_implicit[r] = np.NaN
            pop_iORG_recover[r] = np.NaN
        else:
            poststim_amp = np.quantile(poststim, [0.95])
            max_frmstmp = poststim_loc[np.argmax(poststim)] - dataset.stimtrain_frame_stamps[0]

            final_val = np.mean(pooled_iORG[-5:])
            pop_iORG_implicit[r] = 1000 * max_frmstmp / dataset.framerate
            pop_iORG_amp[r] = (poststim_amp - prestim_amp)
            pop_iORG_recover[r] = 1 - ((final_val - prestim_amp) / pop_iORG_amp[r])

        print("Pooled iORG Avg Amplitude: " + str(pop_iORG_amp[r]) + " Implicit time (ms): " + str(pop_iORG_implicit[r]) +
              " Recovery fraction: " + str(pop_iORG_recover[r]))

        pop_dFrame = pd.DataFrame(np.concatenate((all_iORG,
                                                  np.reshape(pooled_iORG, (1, len(pooled_iORG)))), axis=0))
        pop_dFrame.to_csv(res_dir.joinpath(this_dirname + "_pop_iORG.csv"), header=False)

        pop_amp_dFrame = pd.DataFrame( np.concatenate((np.array(pop_iORG_amp, ndmin=2).transpose(),
                                                       np.array(pop_iORG_implicit, ndmin=2).transpose(),
                                                       np.array(pop_iORG_recover, ndmin=2).transpose()), axis=1),
                                       columns=["Amplitude", "Implicit time", "Recovery %"] )
        pop_amp_dFrame.to_csv(res_dir.joinpath(this_dirname + "_pop_iORG_stats.csv"))

        plt.figure(1)
        plt.clf()
        plt.plot(all_frmstamps / dataset.framerate, pooled_iORG)
        plt.vlines(dataset.stimtrain_frame_stamps[0] / dataset.framerate, -1, 10, color="red")
        plt.xlim([0, max_frmstamp / dataset.framerate])
        plt.ylim([0, 1.5])
        plt.xlabel("Time (seconds)")
        plt.ylabel("Response")
        plt.show(block=False)
        plt.savefig(res_dir.joinpath(this_dirname + "_pooled_pop_iORG.png"))
        plt.savefig(res_dir.joinpath(this_dirname + "_pooled_pop_iORG.svg"))
        print("Done!")

