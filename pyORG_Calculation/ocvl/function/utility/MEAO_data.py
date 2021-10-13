import os.path
from os import path
import numpy as np
import cv2
import pandas as pd
from numpy.polynomial.polynomial import polyval
from numpy.polynomial import Polynomial



class MEAODataset():
    def __init__(self, video_path="", ref_modality="760nm", pipelined=False):

        self.reference_modality = ref_modality


        # Paths to the data used here.
        self.video_path = video_path
        self.metadata_path = self.video_path[0:-3] + "csv"
        self.mask_path = self.video_path[0:-4] + "_mask.avi"
        p_name = os.path.dirname(os.path.realpath(self.video_path))
        f_name = os.path.basename(os.path.realpath(self.video_path))
        common_prefix = f_name.split("_")
        common_prefix = "_".join(common_prefix[0:6])
        self.image_path = path.join(p_name, common_prefix + "_" + self.reference_modality + "1_extract_reg_avg.tif")
        self.coord_path = path.join(p_name,
                                    common_prefix + "_" + self.reference_modality + "1_extract_reg_avg_coords.csv")

        # Information about the dataset
        self.pipelined = pipelined
        self.framerate = -1
        self.num_frames = -1
        self.width = -1
        self.height = -1
        self.framestamps = np.empty([1])
        self.reference_frame_idx = []

        # The data itself.
        self.video_data = np.empty([1])
        self.mask_data = np.empty([1])
        self.coord_data = np.empty([1])
        self.reference_im = np.empty([1])
        self.metadata_data = np.empty([1])

    def load_unpipelined_data(self, force = False):

        # Establish our unpipelined filenames
        if not self.pipelined and not force:

            # Load the video data.
            vid = cv2.VideoCapture(self.video_path)

            if vid.isOpened():
                # If this video doesn't match our set framerate for the dataset, kick out.
                if self.framerate != -1 and vid.get(cv2.CAP_PROP_FPS) != self.framerate:
                    return
                self.framerate = vid.get(cv2.CAP_PROP_FPS)
                self.num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
                self.width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

                this_data = np.empty([self.height, self.width, self.num_frames], dtype=np.uint8)

            i=0
            while vid.isOpened():
                ret, frm = vid.read()
                if ret:
                    # Only take the first channel.
                    this_data[..., i] = frm[..., 0]
                    i += 1
                else:
                    break

            vid.release()

            # Load the mask data.
            vid = cv2.VideoCapture(self.mask_path)

            if vid.isOpened():
                this_mask = np.empty([self.height, self.width, self.num_frames], dtype=np.uint8)

            i = 0
            while vid.isOpened():
                ret, frm = vid.read()
                if ret:
                    # Only take the first channel.
                    this_mask[..., i] = frm[..., 0]/255
                    i += 1
                else:
                    break

            vid.release()

            self.video_data = this_data*this_mask
            self.mask_data = this_mask

            # Load our text data.
            metadata = pd.read_csv(self.metadata_path, delimiter=',', encoding="utf-8-sig")
            metadata.columns = metadata.columns.str.strip()

            self.framestamps = metadata["OriginalFrameNumber"].to_numpy()
            ncc = 1-metadata["NCC"].to_numpy(dtype=float)
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

            allrows = np.linspace(0, numstrips-1, num=self.height) # Make a linspace for all of our images' rows.
            substrip = np.linspace(0, numstrips-1, num=numstrips)

            indivxshift = np.zeros([self.num_frames, self.height])

            # Fit across rows, in order to capture all strips for a given dataset
            for f in range(self.num_frames):
                row_strip_fit = Polynomial.fit(substrip, xshifts[f, :], deg=8)
                indivxshift[f, :] = row_strip_fit(allrows)

            centered_indivxshift = indivxshift-np.median(indivxshift, axis=0)

            indivyshift = np.zeros([self.num_frames, self.height])

            # Fit across rows, in order to capture all strips for a given dataset
            for f in range(self.num_frames):
                row_strip_fit = Polynomial.fit(substrip, yshifts[f, :], deg=8)
                indivyshift[f, :] = row_strip_fit(allrows)

            centered_indivyshift = indivyshift-np.median(indivyshift, axis=0)

            print(indivxshift)



            # for i in range(this_data.shape[-1]):
            #     # Display the resulting frame
            #
            #     cv2.imshow('Frame', this_data[...,i]*this_mask[..., i])
            #
            #     # Press Q on keyboard to  exit
            #     if cv2.waitKey(25) & 0xFF == ord('q'):
            #         break