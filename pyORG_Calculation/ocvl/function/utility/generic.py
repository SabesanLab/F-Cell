import os
from enum import Enum

import numpy as np


from ocvl.function.utility.resources import ResourceLoader, load_video, save_video


class PipeStages(Enum):
    RAW = 0,
    PROCESSED = 1,
    PIPELINED = 2

class GenericDataset:
    def __init__(self, video_path="", metadata_path=None, coord_path=None, stage=PipeStages.RAW):

        # Paths to the data used here.
        self.video_path = video_path
        if metadata_path is None:
            self.metadata_path = self.video_path[0:-3] + "csv"
        else:
            self.metadata_path = metadata_path

        if coord_path is None:
            p_name = os.path.dirname(os.path.realpath(self.video_path))
            f_name = os.path.basename(os.path.realpath(self.video_path))
            self.coord_path = os.path.join(p_name, f_name[0:-4] + "_coords.csv")
        else:
            self.coord_path = coord_path

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
        self.mask_data = np.empty([1])
        self.coord_data = np.empty([1])
        self.reference_im = np.empty([1])
        self.metadata_data = np.empty([1])

    def load_data(self):

        resource = load_video(self.video_path)

        self.video_data = resource.data

        self.framerate = resource.metadict["framerate"]
        self.metadata_data = resource.metadict
        self.height = resource.data.shape[0]
        self.width = resource.data.shape[1]
        self.num_frames = resource.data.shape[2]

    def save_data(self, suffix):
        save_video(self.video_path[0:-4]+suffix+".avi", self.video_data, self.framerate)


