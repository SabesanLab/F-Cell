import os
from os import walk
from os.path import splitext
from tkinter import Tk, filedialog

from matplotlib import pyplot as plt

from ocvl.function.analysis.cell_profile_extraction import extract_profiles, norm_profiles, standardize_profiles
from ocvl.function.analysis.iORG_profile_analyses import signal_power_iORG
from ocvl.function.utility.generic import PipeStages
from ocvl.function.utility.meao import MEAODataset
from ocvl.function.utility.temporal_signal_utils import reconstruct_profiles

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


for loc in allFiles:
    res_dir = os.path.join(loc, "Results")
    os.makedirs(res_dir, exist_ok=True)

    for file in allFiles[loc]:

        dataset = MEAODataset(os.path.join(loc, file), analysis_modality="760nm", ref_modality="760nm",
                              stage=PipeStages.PIPELINED)
        dataset.load_pipelined_data()

        temp_profiles = extract_profiles(dataset.video_data, dataset.coord_data)
        norm_temporal_profiles = norm_profiles(temp_profiles, norm_method="mean")
        stdize_profiles = standardize_profiles(norm_temporal_profiles, dataset.framestamps, 55, method="mean_sub")
        stdize_profiles, dataset.framestamps, nummissed = reconstruct_profiles(stdize_profiles, dataset.framestamps)


        pop_iORG = signal_power_iORG(stdize_profiles, dataset.framestamps, summary_method="std", window_size=0)
        plt.figure(0)
        plt.clf()
        plt.plot(dataset.framestamps, pop_iORG)
        plt.show(block=False)
        plt.savefig(os.path.join(res_dir, file[0:-4] + "_pop_iORG.png"))