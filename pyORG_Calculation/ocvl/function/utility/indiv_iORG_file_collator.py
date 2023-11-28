from pathlib import Path
from tkinter import filedialog, Tk

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == "__main__":

    metric_header = "Amplitude"

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
    resultdir = Path(pName)

    allFiles = dict()

    totFiles = 0
    # Parse out the locations and filenames, store them in a hash table.
    searchpath = Path(pName)
    for path in searchpath.rglob("*.csv"):
        if "log_amplitude_cumhist" in path.name:
            splitfName = path.name.split("_")

            firstcoord = str(abs(float(splitfName[0].split(",")[0][1:])))

            if (path.parent.parent == searchpath or path.parent == searchpath):
                if firstcoord not in allFiles:
                    allFiles[firstcoord] = []
                    allFiles[firstcoord].append(path)
                else:
                    allFiles[firstcoord].append(path)

            totFiles += 1

    all_data = pd.DataFrame()

    # NEVER grow a dataframe rowwise, or baby Jesus will cry
    # https://stackoverflow.com/questions/13784192/creating-an-empty-pandas-dataframe-and-then-filling-it
    for locid in allFiles:

        all_cumhist = np.full((len(allFiles[locid]), 110), np.nan)
        # The above may change depending on how many bins we go with.
        f = 0
        subID = []

        plt.figure(0)
        plt.clf()
        for locfile in allFiles[locid]:

            stat_table = pd.read_csv(locfile, delimiter=",", header=0, encoding="utf-8-sig")
            bins = stat_table.loc[1, :]

            subID.append(locfile.parent.name)

            cumhist = np.nancumsum(stat_table.loc[0, :])
            cumhist /= np.amax(cumhist.flatten()) # Normalize to 1 as its a probability density, not a probability mass, function

            all_cumhist[f, :] = cumhist
            f+=1

            plt.plot(bins, cumhist)

        # all_cumhist=np.nancumsum(all_cumhist, axis=1)
        # all_cumhist /= np.amax(all_cumhist.flatten())
        plt.show(block=False)
        plt.legend(subID)
        plt.savefig(resultdir.joinpath(resultdir.name + "_" + locid + "_all_cumulative_histograms.svg"))

        mean_cumhist = np.mean(all_cumhist, axis=0)
        stddev_cumhist = np.std(all_cumhist, axis=0)
        conf_interval = 2 * stddev_cumhist / np.sqrt(len(allFiles[locid]))

        plt.figure(locid)
        plt.fill_between(bins, mean_cumhist + conf_interval, color=[0, 0, 0, 0.3])
        plt.fill_between(bins, mean_cumhist - conf_interval, color=[1, 1, 1, 1])
        plt.plot(bins, mean_cumhist)
        plt.title(locid)
        plt.savefig(resultdir.joinpath(resultdir.name+ "_"+locid+"_mean_n_conf_cumulative_histogram.svg"))

        plt.show(block=False)
        #plt.waitforbuttonpress()

        indices = pd.MultiIndex.from_product([[locid], subID],
                                             names=["Location", "ID"])
        subframe = pd.DataFrame(all_cumhist, index=indices)

        all_data=pd.concat([all_data, subframe])
        print("wut")


print("Concatenate this, bitch")

all_data.to_csv(resultdir.joinpath(resultdir.name+"_collated_"+metric_header+"_data.csv"))
plt.waitforbuttonpress()