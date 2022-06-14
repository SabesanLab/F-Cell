import warnings

import numpy as np
from matplotlib import pyplot as plt
from numpy.polynomial import Polynomial

from ocvl.function.analysis.iORG_profile_analyses import signal_power_iORG
from ocvl.function.utility.generic import PipeStages
from ocvl.function.utility.meao import MEAODataset
from ocvl.function.utility.temporal_signal_utils import reconstruct_profiles


def pycoordclip(image_stack, coordinates=None, seg_mask="box", seg_radius=1, summary="mean", centroid=None, display=False):
    """
    Original matlab fxn by Robert Cooper
    Adaptation by Mina Gaffney
    Created 06/03/2022

    This function removes all coordinates less than or greater than a specific threshold, in an n-defined polygon around
    the image. A python version of the original matlab fxn coordclip() by Robert Cooper, 2011.

    :param image_stack: a YxXxZ numpy matrix, where there are Y rows, X columns, and Z samples.
    :param coordinates: input as X/Y, these mark locations the locations that will be extracted from all S samples.
    :param thresholdr: pixel row coordinates that are allowed to remain in the list
    :param thresholdc: pixel col coordinates that are allowed to remain in the list
    :param inoutorxor: flag to determine if you want the coordinates inside or outside of your thresholds
                        'o' includes all coordinates outside of your threshold, 'i' includes all coordinates
                        that are inside of your threshold. default = 'i'

    :return: a list of x,y coordinates to be included for further analysis based on an input threshold
    """

