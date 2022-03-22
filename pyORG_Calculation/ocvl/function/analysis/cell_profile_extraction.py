import warnings

import numpy as np
from matplotlib import pyplot, pyplot as plt
from numpy.polynomial import Polynomial

from ocvl.function.analysis.iORG_profile_analyses import population_iORG
from ocvl.function.utility.generic import PipeStages
from ocvl.function.utility.meao import MEAODataset


def extract_profiles(image_stack, coordinates=None, seg_mask="box", seg_radius=1, summary="mean", centroid=None):

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


def norm_profiles(profile_data, norm_type="mean", rescaled=True):

    if norm_type == "mean":
        all_norm = np.nanmean(profile_data)
        framewise_norm = np.nanmean(profile_data, axis=0)
    else:
        all_norm = np.nanmean(profile_data)
        framewise_norm = np.nanmean(profile_data, axis=0)
        warnings.warn("The \""+norm_type+"\" normalization type is not recognized. Defaulting to mean.")

    if rescaled: # Provide the option to simply scale the data, instead of keeping it in relative terms
        ratio = framewise_norm / all_norm
        return np.divide(profile_data, ratio[None, :])
    else:
        return np.divide(profile_data, framewise_norm[None, :])


def standardize_profiles(framestamps, profile_data, stimulus_stamp, method="linear_std"):

    prestimulus_idx = np.where(framestamps <= stimulus_stamp, True, False)
    if len(prestimulus_idx) == 0:
        warnings.warn("Time before the stimulus framestamp doesn't exist in the provided list! No standardization performed.")
        return profile_data

    if method == "linear_std":
        # Standardize using Autoscaling preceded by a linear fit to remove
        # any residual low-frequency changes
        for i in range(profile_data.shape[0]):
            prestim_frmstmp = np.squeeze(framestamps[prestimulus_idx])
            prestim_profile = np.squeeze(profile_data[i, prestimulus_idx])
            goodind = np.isfinite(prestim_profile) # Removes nans, infs, etc.

            thefit = Polynomial.fit(prestim_frmstmp[goodind], prestim_profile[goodind], deg=1)
            fitvals = thefit(prestim_frmstmp[goodind]) # The values we'll subtract from the profile

            prestim_nofit_mean = np.nanmean(prestim_profile[goodind])
            prestim_mean = np.nanmean(prestim_profile[goodind]-fitvals)
            prestim_std = np.nanstd(prestim_profile[goodind]-fitvals)

            profile_data[i, :] = ((profile_data[i, :] - prestim_nofit_mean) / prestim_std)

    elif method == "linear_vast":
        # Standardize using variable stability, or VAST scaling, preceeded by a linear fit:
        # https://www.sciencedirect.com/science/article/pii/S0003267003000941
        # this scaling is defined as autoscaling divided by the CoV.
        for i in range(profile_data.shape[0]):
            prestim_frmstmp = np.squeeze(framestamps[prestimulus_idx])
            prestim_profile = np.squeeze(profile_data[i, prestimulus_idx])
            goodind = np.isfinite(prestim_profile) # Removes nans, infs, etc.

            thefit = Polynomial.fit(prestim_frmstmp[goodind], prestim_profile[goodind], deg=1)
            fitvals = thefit(prestim_frmstmp[goodind]) # The values we'll subtract from the profile

            prestim_nofit_mean = np.nanmean(prestim_profile[goodind])
            prestim_mean = np.nanmean(prestim_profile[goodind]-fitvals)
            prestim_std = np.nanstd(prestim_profile[goodind]-fitvals)

            profile_data[i, :] = ((profile_data[i, :] - prestim_nofit_mean) / prestim_std) / \
                                            (prestim_std / prestim_nofit_mean)

    elif method == "relative_change":
        # Make our output a representation of the relative change of the signal
        for i in range(profile_data.shape[0]):
            prestim_frmstmp = np.squeeze(framestamps[prestimulus_idx])
            prestim_profile = np.squeeze(profile_data[i, prestimulus_idx])
            goodind = np.isfinite(prestim_profile) # Removes nans, infs, etc.

            prestim_mean = np.nanmean(prestim_profile[goodind])
            profile_data[i, :] -= prestim_mean
            profile_data[i, :] /= prestim_mean
            profile_data[i, :] *= 100

    elif method == "mean_sub":
        # Make our output just a prestim mean-subtracted signal.
        for i in range(profile_data.shape[0]):
            prestim_frmstmp = np.squeeze(framestamps[prestimulus_idx])
            prestim_profile = np.squeeze(profile_data[i, prestimulus_idx])
            goodind = np.isfinite(prestim_profile) # Removes nans, infs, etc.

            prestim_mean = np.nanmean(prestim_profile[goodind])
            profile_data[i, :] -= prestim_mean

    return profile_data


if __name__ == "__main__":
    dataset = MEAODataset(
        "\\\\134.48.93.176\\Raw Study Data\\00-33388\\MEAOSLO1\\20210924\\Functional Processed\\Functional Pipeline\\(-1,-0.2)\\00-33388_20210924_OS_(-1,-0.2)_1x1_939_760nm1_extract_reg_cropped_piped.avi",
                           analysis_modality="760nm", ref_modality="760nm", stage=PipeStages.PIPELINED)

    dataset.load_pipelined_data()

    temporal_profiles = extract_profiles(dataset.video_data, dataset.coord_data)
    norm_temporal_profiles = norm_profiles(temporal_profiles, norm_type="mean")
    stdize_profiles = standardize_profiles(dataset.framestamps, norm_temporal_profiles, 55, method="mean_sub")
    pop_iORG = population_iORG(stdize_profiles, dataset.framestamps, summary_method="std", window_size=3)

    plt.plot(dataset.framestamps, pop_iORG)
    plt.show()
