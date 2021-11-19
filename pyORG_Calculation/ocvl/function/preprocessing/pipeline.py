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
from itertools import repeat
from pathlib import Path

import cv2
import numpy as np
import multiprocessing as mp
import os
from os import walk
from os.path import splitext
from tkinter import *
from tkinter import filedialog, simpledialog
from tkinter import ttk

import pandas as pd

from ocvl.function.preprocessing.improc import flat_field, weighted_z_projection, relativize_image_stack, \
    im_dist_to_stk, pairwise_stack_alignment
from ocvl.function.utility.generic import GenericDataset, PipeStages
from ocvl.function.utility.meao import MEAODataset
from ocvl.function.utility.resources import save_video, load_video


def initialize_and_load_meao(file, a_mode, ref_mode):
    print(file)
    dataset = MEAODataset(file, analysis_modality=a_mode, ref_modality=ref_mode, stage=PipeStages.PROCESSED)

    dataset.load_unpipelined_data()

    imp, wp = weighted_z_projection(dataset.video_data, dataset.mask_data)

    ref_imp = weighted_z_projection(dataset.ref_video_data, dataset.mask_data)

    return dataset, imp, wp, ref_imp[0]


def run_pipeline():

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

    # If Unknown is empty (implying we have a location for everything), then remove it from the dict
    if not allFiles["Unknown"]:
        del allFiles["Unknown"]

    # Filter through the list, ensuring we only have paths pertaining to our analysis mode.
    # (The MEAODataset will take care of the rest)
    for loc in allFiles:
        allFiles[loc] = [file for file in allFiles[loc] if
                                        "mask.avi" not in file and "extract_reg_cropped.avi" in file and a_mode in file]

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

    # Create a pool of threads for processing.

    with mp.Pool(processes=int(np.round(mp.cpu_count()/2))) as pool:

        for loc in allFiles:

            # pb["value"] = curFile / totFiles
            # pb_label["text"] = "Processing " + os.path.basename(os.path.realpath(file)) + "..."
            # pb_label.update()

            multiproc_res = pool.starmap_async(initialize_and_load_meao, zip(allFiles[loc], repeat(a_mode), repeat(ref_mode)) )

            loc_results = multiproc_res.get()
            del multiproc_res

            # Instantiate the memory we'll need to store all this. (Datasets / zprojected data)
            dataset = np.empty((len(loc_results)), dtype=type(loc_results[0][0]))
            im_proj = np.empty((loc_results[0][1].shape[0], loc_results[0][1].shape[1], len(loc_results)),
                               dtype=loc_results[0][1].dtype)
            weight_proj = np.empty((loc_results[0][2].shape[0], loc_results[0][2].shape[1], len(loc_results)),
                               dtype=loc_results[0][2].dtype)
            ref_im_proj = np.empty((loc_results[0][3].shape[0], loc_results[0][3].shape[1], len(loc_results)),
                               dtype=loc_results[0][3].dtype)
            for r in range(len(loc_results)):
                dataset[r] = loc_results[r][0]
                im_proj[..., r] = loc_results[r][1]
                weight_proj[..., r] = loc_results[r][2]
                ref_im_proj[..., r] = loc_results[r][3]


            num_vid_proj = ref_im_proj.shape[-1]
            print("Selecting ideal central frame...")
            dist_res = pool.starmap_async(im_dist_to_stk, zip(range(num_vid_proj),
                                                              repeat(ref_im_proj.astype("uint8")),
                                                              repeat(np.ceil(weight_proj).astype("uint8"))) )
            avg_loc_dist = dist_res.get()
            del dist_res
            avg_loc_dist = np.argsort(avg_loc_dist)
            dist_ref_idx = avg_loc_dist[0]
            print("Determined it to be frame " + str(dist_ref_idx) + ".")

            # Begin writing our results to disk.
            writepath = os.path.join(pName, "Functional Pipeline", loc)
            Path(writepath).mkdir(parents=True, exist_ok=True)

            ref_im_proj, xform, inliers = relativize_image_stack(ref_im_proj.astype("uint8"),
                                                                 np.ceil(weight_proj).astype("uint8"),
                                                                 dist_ref_idx,
                                                                 numkeypoints=20000,
                                                                 method="rigid",
                                                                 dropthresh=0.4)

            for f in range(len(xform)):
                if inliers[f]:
                    for i in range(dataset[f].num_frames): # Make all of the data in our dataset relative as well.
                        dataset[f].video_data[..., i] = cv2.warpAffine(dataset[f].video_data[..., i], xform[f],
                                                                       dataset[f].video_data[..., i].shape,
                                                                       flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP)
                        dataset[f].ref_video_data[..., i] = cv2.warpAffine(dataset[f].ref_video_data[..., i], xform[f],
                                                                           dataset[f].ref_video_data[..., i].shape,
                                                                           flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP)
                        dataset[f].mask_data[..., i] = cv2.warpAffine(dataset[f].mask_data[..., i], xform[f],
                                                                      dataset[f].mask_data[..., i].shape,
                                                                      flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP)
                    # Save the pipelined dataset.
                    metadata = pd.DataFrame(dataset[f].framestamps, columns=["FrameStamps"])
                    metadata.to_csv(os.path.join(writepath, dataset[f].filename[:-4]+"_piped.csv"), index=False)
                    save_video(os.path.join(writepath, dataset[f].filename[:-4]+"_piped.avi"), dataset[f].video_data, 29.4)

                    im_proj[..., f] = cv2.warpAffine(im_proj[..., f], xform[f],
                                                     im_proj[..., f].shape,
                                                     flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP)
                    weight_proj[..., f] = cv2.warpAffine(weight_proj[..., f], xform[f],
                                                         weight_proj[..., f].shape,
                                                         flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP)

            weight_proj = weight_proj[..., inliers]
            im_proj = im_proj[..., inliers]

            # Z Project each of our image types
            ref_zproj, weight_zproj = weighted_z_projection(ref_im_proj.astype("float64") * weight_proj, weight_proj)
            analysis_zproj, weight_zproj = weighted_z_projection(im_proj.astype("float64") * weight_proj, weight_proj)

            # After we z-project everything, dump it to disk.
            base_ref_frame = os.path.basename(os.path.realpath(dataset[dist_ref_idx].video_path))
            common_prefix = base_ref_frame.split("_")
            analysis_zproj_fname = "_".join(common_prefix[0:6]) + "_" + dataset[dist_ref_idx].analysis_modality + "_" + \
                                   "ALL_ACQ_AVG.tif"
            ref_zproj_fname = "_".join(common_prefix[0:6]) + "_" + dataset[dist_ref_idx].reference_modality + "_" + \
                                   "ALL_ACQ_AVG.tif"
            ref_vid_fname = "_".join(common_prefix[0:6]) + "_" + dataset[dist_ref_idx].reference_modality + "_" + \
                              "ALL_ACQ_STK.avi"

            cv2.imwrite(os.path.join(writepath, ref_zproj_fname), ref_zproj.astype("uint8"))
            cv2.imwrite(os.path.join(writepath, analysis_zproj_fname), analysis_zproj.astype("uint8"))
            save_video(os.path.join(writepath, ref_vid_fname), ref_im_proj, 29.4)

            del dataset
            del im_proj
            del weight_proj
            del ref_im_proj
            print("Completed processing of location "+loc)
        pb.stop()


if __name__ == "__main__":
    vid = load_video("\\\\134.48.93.176\\Raw Study Data\\00-64774\\MEAOSLO1\\20210824\\Processed\\Functional Pipeline\\pre_selected_stk.avi")
    mask = np.ones(vid.data.shape, dtype="uint8")
    mask[vid.data == 0] = 0
    numfrm = vid.data.shape[-1]

    relativize_image_stack(vid.data, mask, 34)
    # avg_loc_dist = np.zeros( (numfrm) )
    # for f in range(numfrm):
    #     avg_loc_dist[f] = im_dist_to_stk(f, vid.data, mask)
    #     print(str(avg_loc_dist[f]))
    #
    # avg_loc_dist = np.argsort(avg_loc_dist)

    print("wtfbbq")
    #run_pipeline()
