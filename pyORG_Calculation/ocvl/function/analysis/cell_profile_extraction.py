import warnings

import multiprocessing as mp
from itertools import repeat

import numpy as np
import scipy as sp
from sklearn import linear_model
from matplotlib import pyplot as plt
from numpy.polynomial import Polynomial

from ocvl.function.analysis.iORG_profile_analyses import signal_power_iORG
from ocvl.function.utility.generic import PipeStages
from ocvl.function.utility.meao import MEAODataset


def extract_profiles(image_stack, coordinates=None, seg_mask="box", seg_radius=1, summary="mean", centroid=None):
    """
    This function extracts temporal profiles from a 3D matrix, where the first two dimensions are assumed to
    contain data from a single time point (a single image)

    :param image_stack: a YxXxZ numpy matrix, where there are Y rows, X columns, and Z samples.
    :param coordinates: input as X/Y, these mark locations the locations that will be extracted from all S samples.
    :param seg_mask: the mask shape that will be used to extract temporal profiles.
    :param seg_radius: the radius of the mask shape that will be used.
    :param summary: the method used to summarize the area inside the segmentation radius. Default: "mean",
                    Options: "mean", "median"
    :param centroid: precede extraction at each stage with a centroid using a supplied method. Default: None,
                    Options: "voronoi", "simple"

    :return: an NxM numpy matrix with N cells and M temporal samples of some signal.
    """


    if coordinates is None:
        pass # Todo: create coordinates for every position in the image stack.

    if centroid:
        if centroid == "voronoi":
            pass

    image_stack = image_stack.astype("float64")
    image_stack[image_stack == 0] = np.nan # Anything that is equal to 0 should be excluded from consideration.

    im_size = image_stack.shape

    # Generate an exclusion list for our coordinates- those that are unanalyzable should be excluded before analysis.
    pluscoord = coordinates + seg_radius
    excludelist = pluscoord[:, 0] < im_size[1]
    excludelist |= pluscoord[:, 1] < im_size[0]
    del pluscoord

    minuscoord = coordinates - seg_radius
    excludelist |= minuscoord[:, 0] > 0
    excludelist |= minuscoord[:, 1] > 0
    del minuscoord

    coordinates = np.round(coordinates[excludelist, :]).astype("int")

    profile_data = np.empty((coordinates.shape[0], image_stack.shape[-1]))

    if seg_mask == "box": # Handle more in the future...
        for i in range(coordinates.shape[0]):
            coord = coordinates[i, :]
            coordcolumn = image_stack[(coord[1]-seg_radius):(coord[1]+seg_radius+1),
                                      (coord[0]-seg_radius):(coord[0]+seg_radius+1), :]

            coldims = coordcolumn.shape
            coordcolumn = np.reshape(coordcolumn, (coldims[0]*coldims[1], coldims[2]), order="F")
            if summary == "mean":
                profile_data[i, :] = np.nanmean(coordcolumn, axis=0)
            elif summary == "median":
                profile_data[i, :] = np.nanmedian(coordcolumn, axis=0)

    return profile_data


def norm_profiles(temporal_profiles, norm_method="mean", rescaled=True):
    """
    This function normalizes the columns of the data (a single sample of all cells) using a method supplied by the user.

    :param temporal_profiles: A NxM numpy matrix with N cells and M temporal samples of some signal.
    :param norm_method: The normalization method chosen by the user. Default is "mean". Options: "mean", "median"
    :param rescaled: Whether or not to keep the data at the original scale (only modulate the numbers in place). Useful
                     if you want the data to stay in the same units. Default: True. Options: True/False

    :return: a NxM numpy matrix of normalized temporal profiles.
    """

    if norm_method == "mean":
        all_norm = np.nanmean(temporal_profiles[:])
        framewise_norm = np.nanmean(temporal_profiles, axis=0)
    elif norm_method == "median":
        all_norm = np.nanmedian(temporal_profiles[:])
        framewise_norm = np.nanmedian(temporal_profiles, axis=0)
    else:
        all_norm = np.nanmean(temporal_profiles[:])
        framewise_norm = np.nanmean(temporal_profiles, axis=0)
        warnings.warn("The \"" + norm_method + "\" normalization type is not recognized. Defaulting to mean.")

    if rescaled: # Provide the option to simply scale the data, instead of keeping it in relative terms
        ratio = framewise_norm / all_norm
        return np.divide(temporal_profiles, ratio[None, :])
    else:
        return np.divide(temporal_profiles, framewise_norm[None, :])


def standardize_profiles(temporal_profiles, framestamps, stimulus_stamp, method="linear_std"):
    """
    This function standardizes each temporal profile (here, the rows of the supplied data) according to the provided
    arguments.

    :param temporal_profiles: A NxM numpy matrix with N cells and M temporal samples of some signal.
    :param framestamps: A 1xM numpy matrix containing the associated frame stamps for temporal_profiles.
    :param stimulus_stamp: The framestamp at which to limit the standardization. For functional studies, this is often
                            the stimulus framestamp.
    :param method: The method used to standardize. Default is "linear_std", which subtracts a linear fit to
                    each signal before stimulus_stamp, followed by a standardization based on that pre-stamp linear-fit
                    subtracted data. This was used in Cooper et al 2017/2020.
                    Current options include: "linear_std", "linear_vast", "relative_change", and "mean_sub"

    :return: a NxM numpy matrix of standardized temporal profiles.
    """

    prestimulus_idx = np.where(framestamps <= stimulus_stamp, True, False)
    if len(prestimulus_idx) == 0:
        warnings.warn("Time before the stimulus framestamp doesn't exist in the provided list! No standardization performed.")
        return temporal_profiles

    if method == "linear_std":
        # Standardize using Autoscaling preceded by a linear fit to remove
        # any residual low-frequency changes
        for i in range(temporal_profiles.shape[0]):
            prestim_frmstmp = np.squeeze(framestamps[prestimulus_idx])
            prestim_profile = np.squeeze(temporal_profiles[i, prestimulus_idx])
            goodind = np.isfinite(prestim_profile) # Removes nans, infs, etc.

            thefit = Polynomial.fit(prestim_frmstmp[goodind], prestim_profile[goodind], deg=1)
            fitvals = thefit(prestim_frmstmp[goodind]) # The values we'll subtract from the profile

            prestim_nofit_mean = np.nanmean(prestim_profile[goodind])
            prestim_mean = np.nanmean(prestim_profile[goodind]-fitvals)
            prestim_std = np.nanstd(prestim_profile[goodind]-fitvals)

            temporal_profiles[i, :] = ((temporal_profiles[i, :] - prestim_nofit_mean) / prestim_std)

    elif method == "linear_vast":
        # Standardize using variable stability, or VAST scaling, preceeded by a linear fit:
        # https://www.sciencedirect.com/science/article/pii/S0003267003000941
        # this scaling is defined as autoscaling divided by the CoV.
        for i in range(temporal_profiles.shape[0]):
            prestim_frmstmp = np.squeeze(framestamps[prestimulus_idx])
            prestim_profile = np.squeeze(temporal_profiles[i, prestimulus_idx])
            goodind = np.isfinite(prestim_profile) # Removes nans, infs, etc.

            thefit = Polynomial.fit(prestim_frmstmp[goodind], prestim_profile[goodind], deg=1)
            fitvals = thefit(prestim_frmstmp[goodind]) # The values we'll subtract from the profile

            prestim_nofit_mean = np.nanmean(prestim_profile[goodind])
            prestim_mean = np.nanmean(prestim_profile[goodind]-fitvals)
            prestim_std = np.nanstd(prestim_profile[goodind]-fitvals)

            temporal_profiles[i, :] = ((temporal_profiles[i, :] - prestim_nofit_mean) / prestim_std) / \
                                      (prestim_std / prestim_nofit_mean)

    elif method == "relative_change":
        # Make our output a representation of the relative change of the signal
        for i in range(temporal_profiles.shape[0]):
            prestim_frmstmp = np.squeeze(framestamps[prestimulus_idx])
            prestim_profile = np.squeeze(temporal_profiles[i, prestimulus_idx])
            goodind = np.isfinite(prestim_profile) # Removes nans, infs, etc.

            prestim_mean = np.nanmean(prestim_profile[goodind])
            temporal_profiles[i, :] -= prestim_mean
            temporal_profiles[i, :] /= prestim_mean
            temporal_profiles[i, :] *= 100

    elif method == "mean_sub":
        # Make our output just a prestim mean-subtracted signal.
        for i in range(temporal_profiles.shape[0]):
            prestim_frmstmp = np.squeeze(framestamps[prestimulus_idx])
            prestim_profile = np.squeeze(temporal_profiles[i, prestimulus_idx])
            goodind = np.isfinite(prestim_profile) # Removes nans, infs, etc.

            prestim_mean = np.nanmean(prestim_profile[goodind])
            temporal_profiles[i, :] -= prestim_mean

    return temporal_profiles


def l1_compressive_sensing(temporal_profiles, framestamps, c):

    D = sp.fft.dct(np.eye(framestamps[-1] + 1), norm="ortho", orthogonalize=True)
    fullrange = np.arange(framestamps[-1] + 1)

    naners = np.isnan(temporal_profiles[c, :])
    finers = np.isfinite(temporal_profiles[c, :])

    nummissing = (framestamps[-1] + 1) - np.sum(finers)

    # print("Missing " + str(nummissing[c]) + " datapoints.")

    sigmean = np.mean(temporal_profiles[c, finers])
    sigstd = np.std(temporal_profiles[c, finers])

    A = D[framestamps[finers], :]
    lasso = linear_model.Lasso(alpha=0.001, max_iter=2000)
    lasso.fit(A, temporal_profiles[c, finers])

    # plt.figure(0)
    # plt.subplot(2, 1, 1)
    reconstruction = sp.fft.idct(lasso.coef_.reshape((len(fullrange),)), axis=0,
                                       norm="ortho", orthogonalize=True) + sigmean

    return reconstruction, nummissing


def reconstruct_profiles(temporal_profiles, framestamps, method="L1"):
    """
    This function reconstructs the missing profile data using compressive sensing techniques.

    :param temporal_profiles:
    :param framestamps:
    :param method:
    :return:
    """
    fullrange = np.arange(framestamps[-1] + 1)

    reconstruction = np.empty((temporal_profiles.shape[0], len(fullrange)))
    nummissing = np.empty((temporal_profiles.shape[0], 1))


    if method == "L1":
        # Create a pool of threads for processing.
        with mp.Pool(processes=int(np.round(mp.cpu_count() / 2))) as pool:

            reconst = pool.starmap_async(l1_compressive_sensing, zip(repeat(temporal_profiles), repeat(framestamps),
                                                                     range(temporal_profiles.shape[0])) )
            res = reconst.get()
            for c, result in enumerate(res):
                reconstruction[c, :] = np.array(result[0])
                nummissing[c] = np.array(result[1])


    print(str(100 * np.mean(nummissing) / len(fullrange)) + "% signal reconstructed on average.")

    return reconstruction, fullrange, nummissing

if __name__ == "__main__":
    dataset = MEAODataset(
        "\\\\134.48.93.176\\Raw Study Data\\00-33388\\MEAOSLO1\\20210924\\Functional Processed\\Functional Pipeline\\(-1,-0.2)\\00-33388_20210924_OS_(-1,-0.2)_1x1_939_760nm1_extract_reg_cropped_piped.avi",
                           analysis_modality="760nm", ref_modality="760nm", stage=PipeStages.PIPELINED)

    dataset.load_pipelined_data()

    temp_profiles = extract_profiles(dataset.video_data, dataset.coord_data)
    norm_temporal_profiles = norm_profiles(temp_profiles, norm_method="mean")
    stdize_profiles = standardize_profiles(norm_temporal_profiles, dataset.framestamps, 55, method="mean_sub")
    stdize_profiles, dataset.framestamps, nummissed = reconstruct_profiles(stdize_profiles, dataset.framestamps)


    pop_iORG = signal_power_iORG(stdize_profiles, dataset.framestamps, summary_method="std", window_size=1)
    plt.plot(dataset.framestamps, pop_iORG)
    plt.show()
