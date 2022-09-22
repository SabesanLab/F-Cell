import math

import numpy as np
import pywt
from matplotlib import pyplot as plt
from scipy import signal
from ssqueezepy import wavelets, p2up, cwt

from ocvl.function.utility.temporal_signal_utils import densify_temporal_matrix, reconstruct_profiles


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

    if window_size != 0:
        if window_size % 2 < 1:
            raise Exception("Window size must be an odd integer.")
        else:
            window_radius = int((window_size-1)/2)
    else:
        window_radius = 0

    if window_radius != 0:
        # If the window radius isn't 0, then densify the matrix, and pad our profiles
        # Densify our matrix a bit.
        temporal_profiles = densify_temporal_matrix(temporal_profiles, framestamps)
        temporal_profiles = np.pad(temporal_profiles, ((0, 0), (window_radius, window_radius)), "symmetric")

    num_signals = temporal_profiles.shape[0]
    num_samples = temporal_profiles.shape[1]

    num_incl = np.zeros((num_samples))
    iORG = np.empty((num_samples))
    iORG[:] = np.nan

    if summary_method == "var":
        if window_radius == 0:
            iORG = np.nanvar(temporal_profiles, axis=0)
            num_incl = np.sum(np.isfinite(temporal_profiles), axis=0)

        elif window_size < (num_samples/2):

            for i in range(window_radius, num_samples-window_radius):

                samples = temporal_profiles[:, (i - window_radius):(i + window_radius)]
                if np.sum(samples[:] != np.nan) > 10:
                    iORG[i] = np.nanvar(samples[:])
                    num_incl[i] = np.sum(samples[:] != np.nan)

            iORG = iORG[framestamps]
            num_incl = num_incl[framestamps]
        else:
            raise Exception("Window size must be less than half of the number of samples")

    elif summary_method == "std":
        if window_radius == 0:
            iORG = np.nanstd(temporal_profiles, axis=0)
            num_incl = np.sum(np.isfinite(temporal_profiles), axis=0)

        elif window_size < (num_samples/2):

            for i in range(window_radius, num_samples-window_radius):

                samples = temporal_profiles[:, (i - window_radius):(i + window_radius)]
                if np.sum(samples[:] != np.nan) > 10:
                    iORG[i] = np.nanstd(samples[:])
                    num_incl[i] = np.sum(samples[:] != np.nan)

            iORG = iORG[framestamps]
            num_incl = num_incl[framestamps]
        else:
            raise Exception("Window size must be less than half of the number of samples")

    elif summary_method == "rms":
        if window_radius == 0:

            temporal_profiles **= 2 # Square first
            iORG = np.nanmean(temporal_profiles, axis=0)  # Average second
            iORG = np.sqrt(iORG)  # Sqrt last
            num_incl = np.sum(np.isfinite(temporal_profiles), axis=0)

        elif window_size < (num_samples/2):

            temporal_profiles **= 2 # Square first
            for i in range(window_radius, num_samples-window_radius):

                samples = temporal_profiles[:, (i - window_radius):(i + window_radius)]
                if samples[:].size != 0 and np.sum(samples[:] != np.nan) > 10:
                    iORG[i] = np.nanmean(samples[:]) # Average second
                    iORG[i] = np.sqrt(iORG[i]) # Sqrt last
                   # num_incl[i] = np.sum(samples[:] != np.nan)

            iORG = iORG[framestamps]
            #num_incl = num_incl[framestamps]
            num_incl = np.sum(np.isfinite(temporal_profiles), axis=0)
            num_incl = num_incl[framestamps]
        else:
            raise Exception("Window size must be less than half of the number of samples")

    return iORG, num_incl

def wavelet_iORG(temporal_profiles, framestamps, fps):

    #wavelet = "gmw"
    #padtype = "reflect"

    #reconst_profiles, fullrange, nummissing = reconstruct_profiles(temporal_profiles, framestamps)

    #morse = wavelets.Wavelet((wavelet, {"gamma": 3, "beta": 3}))
    #biorwav = pywt.Wavelet("bior1.3")

    #allWx =
    mapper = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis", temporal_profiles.shape[0]))
    plt.figure(11)

    signal = plt.subplot(2, 2, 2)
    waveletd3 = plt.subplot(2, 2, 3)
    waveletd4 = plt.subplot(2, 2, 4)

    for r in range(temporal_profiles.shape[0]):
        nextpowdiff = 2**math.ceil(math.log2(temporal_profiles[r, :].shape[0])) - temporal_profiles[r, :].shape[0]

        signal.plot(framestamps/fps, temporal_profiles[r, :], color=mapper.to_rgba(r, norm=False))
        a4, d4, d3, d2, d1 = pywt.swt(np.pad(temporal_profiles[r, :], (0, nextpowdiff), mode="reflect"),
                                      "bior1.5", level=4, trim_approx=True, norm=False)
        waveletd3.plot(framestamps / fps, d3[0:176], color=mapper.to_rgba(r, norm=False))
        waveletd4.plot(framestamps / fps, d4[0:176], color=mapper.to_rgba(r, norm=False))
        print(np.nansum(temporal_profiles[r, :]**2))
        #plt.waitforbuttonpress()
        #signal.cla()
        #wavelet.imshow(np.abs(Wx), extent=(0, framestamps[-1], scales[0], scales[-1]))

    #wavelet.hist(fifth, bins=10)
    plt.waitforbuttonpress()



