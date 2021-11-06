import cv2
import numpy as np
import os.path
import pandas as pd
from numpy.polynomial import Polynomial
from os import path

from ocvl.function.preprocessing.improc import dewarp_2D_data, relativize_image_stack, flat_field
from ocvl.function.utility.generic import PipeStages
from ocvl.function.utility.resources import ResourceLoader, load_video, save_video


class MEAODataset():
    def __init__(self, video_path="", analysis_modality="760nm", ref_modality="760nm", stage=PipeStages.RAW):

        self.analysis_modality = analysis_modality
        self.reference_modality = ref_modality

        # Paths to the data used here.
        self.video_path = video_path
        self.ref_video_path = video_path.replace(analysis_modality, ref_modality)
        self.metadata_path = self.video_path[0:-3] + "csv"
        self.mask_path = self.video_path[0:-4] + "_mask.avi"
        p_name = os.path.dirname(os.path.realpath(self.video_path))
        f_name = os.path.basename(os.path.realpath(self.video_path))
        common_prefix = f_name.split("_")
        common_prefix = "_".join(common_prefix[0:6])
        self.image_path = path.join(p_name, common_prefix + "_" + self.analysis_modality + "1_extract_reg_avg.tif")
        self.coord_path = path.join(p_name,
                                    common_prefix + "_" + self.analysis_modality + "1_extract_reg_avg_coords.csv")

        # Information about the dataset
        self.stage = stage
        self.framerate = -1
        self.num_frames = -1
        self.width = -1
        self.height = -1
        self.framestamps = np.empty([1])
        self.reference_frame_idx = []

        # The data itself.
        self.video_data = np.empty([1])
        self.ref_video_data = np.empty([1])
        self.mask_data = np.empty([1])
        self.coord_data = np.empty([1])
        self.reference_im = np.empty([1])
        self.metadata_data = np.empty([1])

    def load_unpipelined_data(self, force=False):

        # Establish our unpipelined filenames
        if self.stage is not (PipeStages.RAW or PipeStages.PIPELINED) or force:

            # Load the video data.
            res = load_video(self.video_path)

            self.framerate = res.metadict["framerate"]
            self.num_frames = res.data.shape[-1]
            self.width = res.data.shape[1]
            self.height = res.data.shape[0]
            self.video_data = res.data

            res = load_video(self.mask_path)
            self.mask_data = res.data / 255
            self.video_data = (self.video_data * self.mask_data).astype("uint8")

            if self.ref_video_path != self.video_path:
                # Load the reference video data.
                res = load_video(self.ref_video_path)
                self.ref_video_data = res.data * self.mask_data

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
                if col != "XShift" and "XShift" in col:
                    shiftrow = col.split("_")[0][5:]
                    xshifts[:, int(shiftrow)] = metadata[col].to_numpy()
                if col != "YShift" and "YShift" in col:
                    shiftrow = col.split("_")[0][5:]
                    yshifts[:, int(shiftrow)] = metadata[col].to_numpy()

            self.video_data, map_mesh_x, map_mesh_y = dewarp_2D_data(self.video_data, yshifts, xshifts)

            warp_mask = np.zeros(self.video_data.shape)
            ref_vid = np.zeros(self.video_data.shape)
            for f in range(self.num_frames):
                warp_mask[..., f] = cv2.remap(self.mask_data[..., f].astype("float64"), map_mesh_x,
                                              map_mesh_y, interpolation=cv2.INTER_NEAREST)

                ref_vid[..., f] = cv2.remap(self.ref_video_data[..., f].astype("float64") / 255,
                                            map_mesh_x, map_mesh_y,
                                            interpolation=cv2.INTER_LANCZOS4)
            # Clamp our values.
            warp_mask[warp_mask < 0] = 0
            warp_mask[warp_mask >= 1] = 1
            ref_vid[ref_vid < 0] = 0
            ref_vid[ref_vid >= 1] = 1

            self.mask_data = warp_mask.astype("uint8")
            self.ref_video_data = (255 * ref_vid).astype("uint8")

            self.ref_video_data, xforms, inliers = relativize_image_stack(self.ref_video_data, self.mask_data,
                                                                          reference_idx=self.reference_frame_idx)

            self.framestamps = self.framestamps[np.where(inliers)[0]]

            for f in range(self.num_frames):
                if xforms[f] is not None:
                    self.video_data[..., f] = cv2.warpAffine(self.video_data[..., f], xforms[f],
                                                             self.video_data[..., f].shape,
                                                             flags=cv2.INTER_LANCZOS4)
                    self.mask_data[..., f] = cv2.warpAffine(self.mask_data[..., f], xforms[f],
                                                            self.mask_data[..., f].shape,
                                                            flags=cv2.INTER_NEAREST)



            # for i in range(this_data.shape[-1]):
            #     # Display the resulting frame
            #
            #     cv2.imshow('Frame', this_data[...,i]*this_mask[..., i])
            #
            #     # Press Q on keyboard to  exit
            #     if cv2.waitKey(25) & 0xFF == ord('q'):
            #         break
