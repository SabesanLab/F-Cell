#  Copyright (c) 2021. Robert F Cooper
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
from enum import Enum

import cv2 as cv
import numpy as np
import os
from os import walk
from os.path import splitext
from tkinter import *
from tkinter import filedialog, simpledialog
from tkinter import ttk

from ocvl.function.preprocessing.improc import flat_field
from ocvl.function.utility.generic import GenericDataset, PipeStages
from ocvl.function.utility.meao import MEAODataset





root = Tk()
root.lift()
w = 1
h = 1
x = root.winfo_screenwidth() / 4
y = root.winfo_screenheight() / 4
root.geometry(
    '%dx%d+%d+%d' % (w, h, x, y))  # This moving around is to make sure the dialogs appear in the middle of the screen.

pName = filedialog.askdirectory(title="Select the folder containing all videos of interest.", parent=root)

if not pName:
    quit()

x = root.winfo_screenwidth() / 2 - 128
y = root.winfo_screenheight() / 2 - 128
root.geometry(
    '%dx%d+%d+%d' % (w, h, x, y))  # This moving around is to make sure the dialogs appear in the middle of the screen.
root.update()

# a_mode = simpledialog.askstring(title="Input the analysis modality string: ",
#                                prompt="Input the analysis modality string:",
#                                initialvalue="760nm", parent=root)
# if not a_mode:
#     a_mode = "760nm"
#
# ref_mode = simpledialog.askstring(title="Input the *alignment reference* modality string. ",
#                                   prompt="Input the *alignment reference* modality string:", initialvalue=a_mode, parent=root)
# if not ref_mode:
#     ref_mode = "760nm"

# For debugging.
a_mode = "760nm"
ref_mode = "Confocal"

print("Selected analysis modality name of: " + a_mode + ", and a reference modality of: " + ref_mode)

allFiles = dict()
allFiles["Unknown"] = [] # Prep an empty list for all the locations we can't parse.
totFiles = 0
# Parse out the locations and filenames, store them in a hash table.
for (dirpath, dirnames, filenames) in walk(pName):
    for fName in filenames:
        if a_mode in fName and splitext(fName)[1] == ".avi":
            splitfName = fName.split("_")

            if splitfName[3][0] == "(" and splitfName[3][-1] == ")":
                loc = splitfName[3]
                if loc not in allFiles:
                    allFiles[loc] = []
                    allFiles[loc].append(os.path.join(pName, fName))
                else:
                    allFiles[loc].append(os.path.join(pName, fName))
            else:
                allFiles["Unknown"].append(os.path.join(pName, fName))

            totFiles += 1

    break  # Break after the first run so we don't go recursive.

if not allFiles:
    pass  # Handle this for non-MEAO data.

pb = ttk.Progressbar(root, orient=HORIZONTAL, length=378)
pb.grid(column=0, row=0, columnspan=2, padx=3, pady=5)
pb_label = ttk.Label(root, text="Initializing setup...")
pb_label.grid(column=0, row=1, columnspan=2)
pb.start()

# Resize our root to show our progress bar.
w = 384
h = 64
x = root.winfo_screenwidth() / 2 - 192
y = root.winfo_screenheight() / 2 - 64
root.geometry('%dx%d+%d+%d' % (w, h, x, y))
root.update()

curFile = 0
for loc in allFiles:

    for file in allFiles[loc]:
        pb["value"] = curFile / totFiles
        pb_label["text"] = "Processing " + file + "..."
        curFile += 1

        if "extract_reg_cropped.avi" in file and "_mask.avi" not in file:
            # Here is where we'll place the options. For now, just MEAO...
            dataset = MEAODataset(file, analysis_modality=a_mode, ref_modality=ref_mode, stage=PipeStages.PROCESSED)

            dataset.load_unpipelined_data()
        else:
            pass
           # dataset = GenericDataset(file, stage=PipeStages.RAW)

          #  dataset.load_data()

            #dataset.video_data = flat_field(dataset.video_data)

           # dataset.save_data("_ff")



pb.stop()
