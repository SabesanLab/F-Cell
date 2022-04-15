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
        pop_iORG = []
        pop_iORG_num=[]
        framestamps = []
        max_frmstamp = 0

        for file in allFiles[loc]:

            if "ALL_ACQ_AVG" not in file:
                pb["value"] = r
                pb_label["text"] = "Processing " + file + "..."
                pb.update()
                pb_label.update()

                dataset = MEAODataset(os.path.join(loc, file), analysis_modality="760nm", ref_modality="760nm",
                                      stage=PipeStages.PIPELINED)
                dataset.load_pipelined_data()

                temp_profiles = extract_profiles(dataset.video_data, dataset.coord_data)
                norm_temporal_profiles = norm_profiles(temp_profiles, norm_method="mean")
                stdize_profiles = standardize_profiles(norm_temporal_profiles, dataset.framestamps, 55, method="mean_sub")
                #stdize_profiles, dataset.framestamps, nummissed = reconstruct_profiles(stdize_profiles, dataset.framestamps)

                framestamps.append(dataset.framestamps)
                tmp_iorg, tmp_incl = signal_power_iORG(stdize_profiles, dataset.framestamps, summary_method="var", window_size=1)
                pop_iORG.append(tmp_iorg)
                pop_iORG_num.append(tmp_incl)
                plt.figure(0)
                plt.plot(dataset.framestamps, pop_iORG[r])
                plt.show(block=False)
                plt.savefig(os.path.join(res_dir, file[0:-4] + "_pop_iORG.png"))
                r += 1
                if dataset.framestamps[-1] > max_frmstamp:
                    max_frmstamp = dataset.framestamps[-1]

        del dataset
        # Grab all of the
        all_iORG = np.empty((len(allFiles[loc]), max_frmstamp+1))
        all_iORG[:] = np.nan
        all_incl = np.empty((len(allFiles[loc]), max_frmstamp + 1))
        all_incl[:] = np.nan
        for i, iorg in enumerate(pop_iORG):
            all_incl[i, framestamps[i]] = pop_iORG_num[i]
            all_iORG[i, framestamps[i]] = iorg

        # Pooled variance calc
        pooled_var_iORG = np.nansum( (all_incl-1)*all_iORG, axis=0 ) / np.nansum(all_incl-1, axis=0)
        pooled_stddev_iORG = np.sqrt(pooled_var_iORG)

        plt.figure(1)
        plt.clf()
        plt.plot(pooled_stddev_iORG)
        plt.show(block=False)
        plt.savefig(os.path.join(res_dir, file[0:-4] + "_pooled_pop_iORG.png"))
        print("Done!")

