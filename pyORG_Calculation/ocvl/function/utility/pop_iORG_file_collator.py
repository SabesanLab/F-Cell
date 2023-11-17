from pathlib import Path
from tkinter import filedialog, Tk

import numpy as np
import pandas as pd

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

    allFiles = dict()

    totFiles = 0
    # Parse out the locations and filenames, store them in a hash table.
    searchpath = Path(pName)
    for path in searchpath.rglob("*.csv"):
        if "pop_iORG_stats" in path.name:
            splitfName = path.name.split("_")

            firstcoord = str(abs(float(splitfName[0].split(",")[0][1:])))

            if (path.parent.parent == searchpath or path.parent == searchpath):
                if path.parent not in allFiles:
                    allFiles[path.parent] = []
                    allFiles[path.parent].append(path)
                else:
                    allFiles[path.parent].append(path)

            totFiles += 1

    all_data = pd.DataFrame()
    all_pooled_data = pd.DataFrame()




    # NEVER grow a dataframe rowwise
    # https://stackoverflow.com/questions/13784192/creating-an-empty-pandas-dataframe-and-then-filling-it
    for subid in allFiles:

        subframe = pd.DataFrame()
        pooled_data = np.full((1, len(allFiles[subid])), np.nan)
        loc = []
        f = 0
        for file in allFiles[subid]:

            stat_table = pd.read_csv(file, delimiter=",", header=0,encoding="utf-8-sig", index_col=0)
            stat_table = stat_table.dropna(how="all")  # Drop rows with all nans- no point in taking up space.

            splitfName = file.name.split("_")

            firstcoord = str(abs(float(splitfName[0].split(",")[0][1:])))

            loc.append(firstcoord)
            pooled_data[0, f] = (stat_table[metric_header].iloc[-1])

            allcoi = stat_table[metric_header][0:-1]

            indices = pd.MultiIndex.from_product([[file.parent.name], np.arange(0, len(allcoi))],
                                                 names=["ID", "Acquisition Number"])
            allcoi.index = indices
            allcoi = allcoi.rename(firstcoord)
            allcoi = allcoi.to_frame()

            subframe=pd.concat([subframe, allcoi], axis=1)

            f+=1

        all_data=pd.concat([all_data, subframe])
        pooled_data = pd.DataFrame(pooled_data, columns=loc, index=[subid.name])
        all_pooled_data = pd.concat([all_pooled_data, pooled_data])
        print("wut")


print("Concatenate this, bitch")
resultdir=Path(pName)
all_data.to_csv(resultdir.joinpath(resultdir.name+"_collated_"+metric_header+"_data.csv"))
all_pooled_data.to_csv(resultdir.joinpath(resultdir.name+"_collated_POOLED_"+metric_header+"_data.csv"))
