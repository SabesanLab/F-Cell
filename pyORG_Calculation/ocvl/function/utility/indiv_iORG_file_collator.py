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

        for locfile in allFiles[locid]:

            stat_table = pd.read_csv(locfile, delimiter=",", header=0, encoding="utf-8-sig")
            bins = stat_table.loc[1,:]

            subID.append(locfile.parent.name)

            all_cumhist[f, :] = stat_table.loc[0, :]

            f+=1

        all_cumhist=np.nancumsum(all_cumhist, axis=1)
        all_cumhist /= np.amax(all_cumhist.flatten()) # Normalize to 1 as its a probability density, not a probability mass, function
        indices = pd.MultiIndex.from_product([[locid], subID],
                                             names=["Location", "ID"])
        subframe = pd.DataFrame(all_cumhist, index=indices)

        all_data=pd.concat([all_data, subframe])
        print("wut")


print("Concatenate this, bitch")
resultdir=Path(pName)
all_data.to_csv(resultdir.joinpath(resultdir.name+"_collated_"+metric_header+"_data.csv"))