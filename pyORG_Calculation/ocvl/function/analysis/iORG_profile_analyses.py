
import numpy as np

from ocvl.function.utility.temporal_signal_utils import densify_temporal_matrix

def signal_power_iORG(temporal_profiles, framestamps, summary_method="var", window_size=1):
    """
    Calculates the iORG on a supplied dataset, using a variety of power based summary methods published in
    Cooper et. al. 2020, and Cooper et. al. 2017.

    :param temporal_profiles: A NxM numpy matrix with N cells and M temporal samples of some signal.
    :param framestamps: A 1xM numpy matrix containing the associated frame stamps for temporal_profiles.
    :param summary_method: The method used to summarize the population at each sample. Current options include:
                            "var", "std", and "moving_rms". Default: "var"
    :param window_size: The window size used to summarize the population at each sample. Can be an odd integer from
                        1 (no window) to M/2. Default: 1

    :return: a 1xM population iORG signal.
    """

    if window_size % 2 < 1:
        raise Exception("Window size must be an odd integer.")
    else:
        window_radius = int((window_size-1)/2)

    if window_radius != 0:
        # If the window radius isn't 0, then densify the matrix, and pad our profiles
        temporal_profiles = densify_temporal_matrix(temporal_profiles, framestamps)
        temporal_profiles = np.pad(temporal_profiles, ((0, 0), (window_radius, window_radius)), "symmetric")

    num_signals = temporal_profiles.shape[0]
    num_samples = temporal_profiles.shape[1]
    iORG = np.empty((num_samples))
    iORG[:] = np.nan

    if summary_method == "var":
        if window_radius == 0:

            iORG = np.nanvar(temporal_profiles, axis=0)
        elif window_size < (num_samples/2):

            for i in range(num_samples):

                samples = temporal_profiles[:, (i - window_radius):(i + window_radius)]
                if np.sum(samples[:] != np.nan) > 10:
                    iORG[i] = np.nanvar(samples[:])

            iORG = iORG[framestamps]
        else:
            raise Exception("Window size must be less than half of the number of samples")

    elif summary_method == "std":
        if window_radius == 0:

            iORG = np.nanstd(temporal_profiles, axis=0)
        elif window_size < (num_samples/2):

            for i in range(num_samples):

                samples = temporal_profiles[:, (i - window_radius):(i + window_radius)]
                if np.sum(samples[:] != np.nan) > 10:
                    iORG[i] = np.nanstd(samples[:])

            iORG = iORG[framestamps]
        else:
            raise Exception("Window size must be less than half of the number of samples")

    elif summary_method == "rms":
        if window_radius == 0:
            iORG = np.nanvar(temporal_profiles, axis=0)
        elif window_size < (num_samples/2):

            temporal_profiles **= 2 # Square first
            for i in range(window_radius, num_samples):

                samples = temporal_profiles[:, (i - window_radius):(i + window_radius)]
                if np.sum(samples[:] != np.nan) > 10:
                    iORG[i] = np.nanmean(samples[:]) # Average second
                    iORG[i] = np.sqrt(iORG[i]) # Sqrt last

            iORG = iORG[framestamps]
        else:
            raise Exception("Window size must be less than half of the number of samples")

    return iORG

def wavelet_iORG(temporal_profiles, framestamps):

    pass
