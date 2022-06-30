import warnings

import numpy as np
from matplotlib import pyplot as plt
from numpy.polynomial import Polynomial

from ocvl.function.analysis.iORG_profile_analyses import signal_power_iORG
from ocvl.function.utility.generic import PipeStages
from ocvl.function.utility.meao import MEAODataset
from ocvl.function.utility.temporal_signal_utils import reconstruct_profiles


def coordclip(coordinates, numconesbound):
    """
    Original matlab fxn by Robert Cooper
    Adaptation by Mina Gaffney
    Created 06/03/2022

    This function removes all coordinates less than or greater than a specific threshold, in an n-defined polygon around
    the image. A python version of the original matlab fxn coordclip() by Robert Cooper, 2011.

    :param coordinates: input as X/Y, these mark locations the locations that will be extracted from all S samples.
    :param thresholdr: pixel row coordinates that are allowed to remain in the list
    :param thresholdc: pixel col coordinates that are allowed to remain in the list
    :param inoutorxor: flag to determine if you want the coordinates inside or outside of your thresholds
                        'o' includes all coordinates outside of your threshold, 'i' includes all coordinates
                        that are inside of your threshold. default = 'i'

    :return: a list of x,y coordinates to be included for further analysis based on an input threshold
    """

    # Check that coordinates are present
    if coordinates == []:
        raise Exception("Error: No coordinates were given! Can't run pycoordclip.")

    # check to ensure coords only contain x y info
    if np.size(coordinates, 1) != 2:
        raise Exception("Error: Coordinate list has more than 2 columns. Please only load in X,Y coordinates.")

    # Determine image size based on coordinates
    # (find the max row and col coord value)
    imsizecol = np.max(coordinates[:, 1])
    imsizerow = np.max(coordinates[:, 0])

    immincol = np.min(coordinates[:, 1])
    imminrow = np.min(coordinates[:, 0])
    print(['max col ', imsizecol])
    print(['max row', imsizerow])
    print(['min col ', immincol])
    print(['min row', imminrow])


    # Finding approx center coordinate based on max coord dimensions
    aprxcntrcol = round((imsizecol-immincol)/2)
    aprxcntrrow = round((imsizerow-imminrow)/2)
    print(['approx c col ', aprxcntrcol])
    print(['approx c row ', aprxcntrrow])

    # Testing if the number of coordinates is smaller than numconesbound
    if numconesbound > np.size(coordinates,0):
        numconesbound = np.size(coordinates,0)
        print(['Num bound cones:', numconesbound])

    # growing the ROI from the center







    return []