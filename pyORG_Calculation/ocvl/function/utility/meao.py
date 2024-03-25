import glob
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pandas as pd
from os import path

from ocvl.function.preprocessing.improc import dewarp_2D_data, optimizer_stack_align
from ocvl.function.utility.generic import PipeStages
from ocvl.function.utility.resources import load_video


class MEAODataset:
    def __init__(self, video_path="", image_path=None, coord_path=None, stimtrain_path=None,
                 analysis_modality="760nm", ref_modality="760nm", stage=PipeStages.RAW):

        self.analysis_modality = analysis_modality
        self.reference_modality = ref_modality

        # Paths to the data used here.
        self.video_path = video_path
        self.ref_video_path = video_path.replace(analysis_modality, ref_modality)
        self.metadata_path = self.video_path[0:-3] + "csv"
        self.mask_path = self.video_path[0:-4] + "_mask.avi"
        p_name = os.path.dirname(os.path.realpath(self.video_path))
        self.filename = os.path.basename(os.path.realpath(self.video_path))
        common_prefix = self.filename.split("_")
        common_prefix = "_".join(common_prefix[0:6])

        if not image_path:
            imname = None
            if stage is PipeStages.PROCESSED:
                for filename in glob.glob( path.join(p_name, common_prefix + "_" + self.analysis_modality + "?_extract_reg_avg.tif") ):
                    # print(filename)
                    imname = filename
            elif stage is PipeStages.PIPELINED:
                # First look for an image associated with this dataset
                for filename in glob.glob( path.join(p_name, common_prefix + "_" + self.analysis_modality + "?_extract_reg_cropped_piped_avg.tif") ):
                    imname = filename

                # If we don't have an image specific to this dataset, search for the all acq avg
                if not imname:
                    for filename in glob.glob(path.join(p_name, "*"+self.analysis_modality + "_ALL_ACQ_AVG.tif")):
                        # print(filename)
                        imname = filename

                if not imname:
                    for filename in glob.glob(path.join(p_name, "*_ALL_ACQ_AVG.tif")):
                        # print(filename)
                        imname = filename
            else:
                imname = path.join(p_name, common_prefix + "_" + self.analysis_modality + "1_extract_reg_avg.tif")

        if not imname:
            warnings.warn("Unable to detect viable average image file. Dataset functionality may be limited.")
            self.image_path = None
        else:
            self.image_path = path.join(p_name, imname)

        if not coord_path:
            coordname = None
            if stage is PipeStages.PROCESSED:
                for filename in glob.glob(
                        path.join(p_name, common_prefix + "_" + self.analysis_modality + "?_extract_reg_avg_coords.csv")):
                    # print(filename)
                    coordname = filename
            elif stage is PipeStages.PIPELINED:
                # First look for an image associated with this dataset
                for filename in glob.glob(path.join(p_name,
                                                    common_prefix + "_" + self.analysis_modality + "?_extract_reg_cropped_piped_avg_coords.csv")):
                    coordname = filename

                # If we don't have an image specific to this dataset, search for the all acq avg
                if not coordname:
                    for filename in glob.glob(path.join(p_name, "*_ALL_ACQ_AVG_coords.csv")):
                        # print(filename)
                        coordname = filename
            else:
                coordname = path.join(p_name, common_prefix + "_" + self.analysis_modality + "1_extract_reg_avg_coords.csv")

            if not coordname:
                #warnings.warn("Unable to detect viable coordinate file. Dataset functionality may be limited.")
                self.coord_path = None
            else:
                self.coord_path = path.join(p_name, coordname)
        else:
            self.coord_path = coord_path

        self.stimtrain_path = stimtrain_path

        # Information about the dataset
        self.stage = stage
        self.framerate = -1
        self.num_frames = -1
        self.width = -1
        self.height = -1
        self.framestamps = np.empty([1])
        self.reference_frame_idx = []
        self.stimtrain_frame_stamps = np.empty([1])

        # The data are roughly grouped by the following:
        # Base data
        self.coord_data = np.empty([1])
        self.reference_im = np.empty([1])
        self.metadata_data = np.empty([1])
        # Video data (processed or pipelined)
        self.video_data = np.empty([1])
        self.ref_video_data = np.empty([1])
        self.mask_data = np.empty([1])
        # Extracted data (temporal profiles
        self.raw_profile_data = np.empty([1])
        self.postproc_profile_data = np.empty([1])

    def clear_video_data(self):
        print("Deleting video data from "+self.video_path)
        del self.video_data
        del self.ref_video_data
        del self.mask_data

    def load_data(self):
        if self.stage is PipeStages.RAW:
            self.load_raw_data()
        elif self.stage is PipeStages.PROCESSED:
            self.load_processed_data()
        elif self.stage is PipeStages.PIPELINED:
            self.load_pipelined_data()
        elif self.stage is PipeStages.ANALYSIS_READY:
            self.load_analysis_ready_data()

    def load_analysis_ready_data(self, raw_profiles, postprocessed_profiles=np.empty([1])):
        if self.stage is PipeStages.ANALYSIS_READY:
            self.raw_profile_data = raw_profiles
            self.postproc_profile_data = postprocessed_profiles

    def load_pipelined_data(self):
        if self.stage is PipeStages.PIPELINED:
            res = load_video(self.video_path)

            self.framerate = res.metadict["framerate"]
            self.num_frames = res.data.shape[-1]
            self.width = res.data.shape[1]
            self.height = res.data.shape[0]
            self.video_data = res.data

            if os.path.exists(self.mask_path):
                res = load_video(self.mask_path)
                self.mask_data = res.data / 255
                self.mask_data[self.mask_data < 0] = 0
                self.video_data = (self.video_data * self.mask_data).astype("uint8")
            else:
                pass
                # warnings.warn("No pipelined mask data detected.")

            # Load the reference video data.
            if os.path.exists(self.ref_video_path):
                res = load_video(self.ref_video_path)
                self.ref_video_data = (res.data * self.mask_data).astype("uint8")
            else:
                pass
                # warnings.warn("No pipelined reference video data detected.")

            # Load our text data, if we can.
            if Path(self.metadata_path).is_file():
                metadata = pd.read_csv(self.metadata_path, delimiter=',', encoding="utf-8-sig")
                metadata.columns = metadata.columns.str.strip()

                self.framestamps = metadata["FrameStamps"].to_numpy()-1 # Subtract one, because they're stored where 1 is the first index.
                #self.reference_frame_idx = min(range(len(ncc)), key=ncc.__getitem__) # Should probably carry this forward
            elif Path(self.metadata_path[0:-4]+"_acceptable_frames.csv").is_file():
                self.framestamps = np.squeeze(pd.read_csv(self.metadata_path[0:-4]+"_acceptable_frames.csv", delimiter=',', header=None,
                                              encoding="utf-8-sig").to_numpy())
            else:
                self.framestamps = np.arange(0, self.num_frames)

            if self.image_path:
                self.reference_im = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

            if self.coord_path:
                self.coord_data = pd.read_csv(self.coord_path, delimiter=',', header=None,
                                              encoding="utf-8-sig").to_numpy()
                # print(self.coord_data)

            if self.stimtrain_path: # [ 58 2 106 ] (176?) -> [ 58 60 176 ]
                self.stimtrain_frame_stamps = np.cumsum(np.squeeze(pd.read_csv(self.stimtrain_path, delimiter=',', header=None,
                                                          encoding="utf-8-sig").to_numpy()))
            else:
                self.stimtrain_frame_stamps = self.num_frames-1



    def load_processed_data(self, force=False, clip_top=0):

        # Establish our unpipelined filenames
        if self.stage is not PipeStages.RAW or force:

            # Load the video data.
            res = load_video(self.video_path)

            self.framerate = res.metadict["framerate"]
            self.num_frames = res.data.shape[-1]
            self.width = res.data.shape[1]
            self.height = res.data.shape[0]
            self.video_data = res.data

            if os.path.exists(self.mask_path):
                res = load_video(self.mask_path)
                self.mask_data = res.data / 255
                self.mask_data[self.mask_data < 0] = 0

                if clip_top != 0:
                    kern = np.zeros((clip_top*2+1, clip_top*2+1), dtype=np.uint8)
                    kern[:, clip_top] = 1

                    for f in range(self.num_frames):
                        # plt.figure(0)
                        # plt.subplot(2, 1, 1)
                        # plt.imshow(self.mask_data[:, :, f])
                        self.mask_data[:,:,f] = cv2.erode(self.mask_data[:,:,f].astype("uint8"), kernel=kern, borderType=cv2.BORDER_CONSTANT, borderValue=0)
                        # plt.subplot(2, 1, 2)
                        # plt.imshow(self.mask_data[:, :, f])
                        # plt.show()
                        # plt.waitforbuttonpress()

                self.video_data = (self.video_data * self.mask_data).astype("uint8")
            else:
                warnings.warn("No processed mask data detected.")

            # Load the reference video data.
            if os.path.exists(self.ref_video_path) and self.ref_video_path != self.mask_path:
                res = load_video(self.ref_video_path)
                self.ref_video_data = (res.data * self.mask_data).astype("uint8")
            elif self.ref_video_path == self.video_path:
                self.ref_video_data = self.video_data
            else:
                warnings.warn("No processed reference video data detected.")


            # Load our text data.
            metadata = pd.read_csv(self.metadata_path, delimiter=',', encoding="utf-8-sig")
            metadata.columns = metadata.columns.str.strip()

            self.framestamps = metadata["OriginalFrameNumber"].to_numpy()
            ncc = 1 - metadata["NCC"].to_numpy(dtype=float)
            self.reference_frame_idx = min(range(len(ncc)), key=ncc.__getitem__)

            # Dewarp our data.
            # First find out how many strips we have.
            numstrips = 0
            for col in metadata.columns.tolist():
                if "XShift" in col:
                    numstrips += 1

            xshifts = np.zeros([ncc.shape[0], numstrips])
            yshifts = np.zeros([ncc.shape[0], numstrips])

            for col in metadata.columns.tolist():
                shiftrow = col.strip().split("_")[0][5:]
                npcol = metadata[col].to_numpy()
                if npcol.dtype == "object":
                    npcol[npcol == " "] = np.nan
                if col != "XShift" and "XShift" in col:
                    xshifts[:, int(shiftrow)] = npcol
                if col != "YShift" and "YShift" in col:
                    yshifts[:, int(shiftrow)] = npcol

            # Determine the residual error in our dewarping, and obtain the maps
            self.video_data, map_mesh_x, map_mesh_y = dewarp_2D_data(self.video_data, yshifts, xshifts)

            # Dewarp our other two datasets as well.
            warp_mask = np.zeros(self.video_data.shape)
            ref_vid = np.zeros(self.video_data.shape)
            for f in range(self.num_frames):
                warp_mask[..., f] = cv2.remap(self.mask_data[..., f].astype("float64"), map_mesh_x,
                                              map_mesh_y, interpolation=cv2.INTER_NEAREST)

                ref_vid[..., f] = cv2.remap(self.ref_video_data[..., f].astype("float64") / 255,
                                            map_mesh_x, map_mesh_y,
                                            interpolation=cv2.INTER_CUBIC)
            # Clamp our values.
            warp_mask[warp_mask < 0] = 0
            warp_mask[warp_mask >= 1] = 1
            ref_vid[ref_vid < 0] = 0
            ref_vid[ref_vid >= 1] = 1

            self.mask_data = warp_mask.astype("uint8")
            self.ref_video_data = (255 * ref_vid).astype("uint8")

            self.ref_video_data, xforms, inliers = optimizer_stack_align(self.ref_video_data, self.mask_data,
                                                                         reference_idx=self.reference_frame_idx,
                                                                         dropthresh=0.0)

            print( "Keeping " + str(np.sum(inliers)) + " of " + str(self.num_frames)+"...")

            # Update everything with what's an inlier now.
            self.ref_video_data = self.ref_video_data[..., inliers].astype("uint8")
            self.framestamps = self.framestamps[inliers]
            self.video_data = self.video_data[..., inliers]
            self.mask_data = self.mask_data[..., inliers]
            self.num_frames = self.video_data.shape[-1] # Need to do this before loop below (video data has been filtered)

            (rows, cols) = self.video_data.shape[0:2]

            for f in range(self.num_frames):
                if xforms[f] is not None:
                    self.video_data[..., f] = cv2.warpAffine(self.video_data[..., f], xforms[f],
                                                             (cols, rows),
                                                             flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP)
                    self.mask_data[..., f] = cv2.warpAffine(self.mask_data[..., f], xforms[f],
                                                            (cols, rows),
                                                            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)

            self.video_data = self.video_data.astype("uint8")
            self.ref_video_data = self.ref_video_data.astype("uint8")

            # save_video("//134.48.93.176/Raw Study Data/00-64774/MEAOSLO1/20210824/Processed/Functional Pipeline/", dataset[f].video_data, 29.4)
            # for i in range(this_data.shape[-1]):
            #     # Display the resulting frame
            #
            #     cv2.imshow('Frame', this_data[...,i]*this_mask[..., i])
            #
            #     # Press Q on keyboard to  exit
            #     if cv2.waitKey(25) & 0xFF == ord('q'):
            #         break
