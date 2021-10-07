import os.path
from os import path
import numpy as np
import cv2
import pandas as pd
from numpy.polynomial import Polynomial


class MEAODataset():
    def __init__(self, video_paths=[], ref_modality="760nm", pipelined=False):
        if type(video_paths) is not list:
            self.video_paths = [video_paths]
        else:
            self.video_paths = video_paths

        # Paths to the data used here.
        self.reference_modality = ref_modality
        self.image_paths = []
        self.coord_paths = []
        self.metadata_paths = []
        self.mask_paths = []

        # Information about the dataset
        self.pipelined = pipelined
        self.framerate = -1
        self.framestamps = []
        self.reference_frame_idx = []

        # The data itself.
        self.video_data = []
        self.mask_data = []
        self.coord_data = []
        self.reference_im = []
        self.metadata_data =[]

    def load_unpipelined_data(self, force = False):

        # Establish our unpipelined filenames
        if not self.pipelined and not force:
            forremoval = []
            for this_path in self.video_paths:
                if "_mask.avi" not in this_path:
                    p_name = os.path.dirname(os.path.realpath(this_path))
                    f_name = os.path.basename(os.path.realpath(this_path))
                    common_prefix = f_name.split("_")
                    common_prefix = "_".join(common_prefix[0:6])
                    self.image_paths.append(path.join(p_name, common_prefix + "_" + self.reference_modality + "1_extract_reg_avg.tif"))
                    self.coord_paths.append(
                        path.join(p_name, common_prefix + "_" + self.reference_modality + "1_extract_reg_avg_coords.csv"))
                    self.metadata_paths.append(this_path[0:-3] + "csv")
                    self.mask_paths.append(this_path[0:-4] + "_mask.avi")
                else:
                    forremoval.append(this_path)

            if len(self.video_paths) >= 1 and forremoval:
                self.video_paths = [p for p in self.video_paths if p not in forremoval]
            elif forremoval: # If there's nothing to do, exit.
                return


            # Load the video data.
            for this_path in self.video_paths:
                vid = cv2.VideoCapture(this_path)

                if vid.isOpened():
                    # If this video doesn't match our set framerate for the dataset, kick out.
                    if self.framerate != -1 and vid.get(cv2.CAP_PROP_FPS) != self.framerate:
                        return
                    self.framerate = vid.get(cv2.CAP_PROP_FPS)
                    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    this_data = np.empty([height, width, length], dtype=np.uint8)

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

            for mask_path in self.mask_paths:
                vid = cv2.VideoCapture(mask_path)

                if vid.isOpened():
                    this_mask = np.empty([height, width, length], dtype=np.uint8)

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

            self.video_data.append(this_data*this_mask)
            self.mask_data.append(this_mask)

            # Load our text data.
            for meta_path in self.metadata_paths:

                metadata = pd.read_csv(meta_path, delimiter=',', encoding="utf-8-sig")
                metadata.columns=metadata.columns.str.strip()

                self.framestamps.append(metadata["OriginalFrameNumber"].to_numpy())
                ncc = 1-metadata["NCC"].to_numpy(dtype=float)
                self.reference_frame_idx.append( min(range(len(ncc)), key=ncc.__getitem__) )


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


            # Fit across rows, in order to capture all strips for a given dataset
            for i in range(ncc.shape[0]):
                xfit = Polynomial.fit(np.linspace(0, numstrips - 1, num=numstrips), xshifts[i,:], deg=8)

            print(xshifts)
            # for i in range(this_data.shape[-1]):
            #     # Display the resulting frame
            #
            #     cv2.imshow('Frame', this_data[...,i]*this_mask[..., i])
            #
            #     # Press Q on keyboard to  exit
            #     if cv2.waitKey(25) & 0xFF == ord('q'):
            #         break