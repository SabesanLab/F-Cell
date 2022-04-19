import os
from os import walk
from os.path import splitext
from tkinter import Tk, filedialog, ttk, HORIZONTAL

import numpy as np
from matplotlib import pyplot as plt

from ocvl.function.analysis.cell_profile_extraction import extract_profiles, norm_profiles, standardize_profiles
from ocvl.function.analysis.iORG_profile_analyses import signal_power_iORG
from ocvl.function.utility.generic import PipeStages
from ocvl.function.utility.meao import MEAODataset
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
    for (dirpath, dirnames, filenames) in walk(pName):
        for fName in filenames:
            if splitext(fName)[1] == ".avi" and "piped" in fName:
                splitfName = fName.split("_")

                if dirpath not in allFiles:
                    allFiles[dirpath] = []
                    allFiles[dirpath].append(fName)
                else:
                    allFiles[dirpath].append(fName)

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

    for loc in allFiles:
        res_dir = os.path.join(loc, "Results")
        os.makedirs(res_dir, exist_ok=True)

        r = 0
        pb["maximum"] = len(allFiles[loc])

        max_frmstamp = 0
        cell_framestamps = []
        cell_profiles = []
        first = True
        for file in allFiles[loc]:

            if "ALL_ACQ_AVG" not in file:
                pb["value"] = r
                pb_label["text"] = "Processing " + file + "..."
                pb.update()
                pb_label.update()

                dataset = MEAODataset(os.path.join(loc, file), stimtrain_path=stimtrain_fName,
                                      analysis_modality="760nm", ref_modality="760nm", stage=PipeStages.PIPELINED)
                dataset.load_pipelined_data()

                # Initialize the dict for individual cells.
                if first:
                    for c in range(len(dataset.coord_data)):
                        cell_framestamps.append([])
                        cell_profiles.append([])
                        coord_data = dataset.coord_data
                        framerate = dataset.framerate
                        stimulus_train = dataset.stimtrain_frame_stamps

                    first = False

                temp_profiles = extract_profiles(dataset.video_data, dataset.coord_data, seg_radius=1)
                norm_temporal_profiles = norm_profiles(temp_profiles, norm_method="mean")
                stdize_profiles = standardize_profiles(norm_temporal_profiles, dataset.framestamps, stimulus_train[0],
                                                       method="mean_sub")
                #stdize_profiles, dataset.framestamps, nummissed = reconstruct_profiles(stdize_profiles, dataset.framestamps)

                # Put the profile of each cell into its own array
                for c in range(len(dataset.coord_data)):
                    cell_framestamps[c].append(dataset.framestamps)
                    cell_profiles[c].append(stdize_profiles[c, :])
                r += 1

                if dataset.framestamps[-1] > max_frmstamp:
                    max_frmstamp = dataset.framestamps[-1]

        del dataset, temp_profiles, norm_temporal_profiles, stdize_profiles

        # Rows: Acquisitions
        # Cols: Framestamps
        # Depth: Coordinate
        all_cell_iORG = np.empty((len(allFiles[loc]), max_frmstamp+1, len(coord_data)))
        all_cell_iORG[:] = np.nan
        full_framestamp_range = np.arange(max_frmstamp+1)
        cell_power_iORG = np.empty((len(coord_data), max_frmstamp + 1))
        cell_power_iORG[:] = np.nan

        # Make 3D matricies
        for c in range(len(coord_data)):
            for i, profile in enumerate(cell_profiles[c]):
                all_cell_iORG[i, cell_framestamps[c][i], c] = profile

        print("Yay!")

        simple_amp = np.empty((len(coord_data)))
        simple_amp[:] = np.nan
        for c in range(len(coord_data)):
            cell_power_iORG[c, :], numincl = signal_power_iORG(all_cell_iORG[:, :, c], full_framestamp_range,
                                                      summary_method="std", window_size=1)

            prestim_amp = np.nanmedian(cell_power_iORG[c, 0:stimulus_train[0]])
            poststim_amp = np.nanmedian(cell_power_iORG[c, stimulus_train[1]:(stimulus_train[1]+10)])

            simple_amp[c] = poststim_amp-prestim_amp

            plt.figure(0)
            plt.clf()
            for a in range(all_cell_iORG.shape[0]):
                plt.plot(full_framestamp_range, all_cell_iORG[a, :, c])
                plt.plot(stimulus_train[0], poststim_amp, "rD")
          # plt.hist(simple_amp)
          # plt.plot(cell_power_iORG[c, :])
            plt.show(block=False)
            plt.waitforbuttonpress()
           # plt.savefig(os.path.join(res_dir, file[0:-4] + "_allcell_iORG_amp.png"))



        plt.figure(1)
        plt.hist(simple_amp, bins=np.arange(start=-10, stop=100, step=2.5))
        # plt.plot(cell_power_iORG[c, :], "k-", alpha=0.05)
        plt.show(block=False)
        plt.savefig(os.path.join(res_dir, file[0:-4] + "_allcell_iORG_amp.png"))
        plt.savefig(os.path.join(res_dir, file[0:-4] + "_allcell_iORG_amp.svg"))
        plt.clf()

        print("Done!")

