import math

import numpy as np
import pywt
from matplotlib import pyplot as plt
from scipy import signal
from ssqueezepy import wavelets, p2up, cwt
from ssqueezepy.experimental import scale_to_freq

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

def wavelet_iORG(temporal_profiles, framestamps, fps, sig_threshold = None, display=False):

    padtype = "reflect"

    time = framestamps/fps
    the_wavelet = wavelets.Wavelet(("gmw", {"gamma": 2, "beta": 1}))
    #the_wavelet = wavelets.Wavelet(("hhhat", {"mu": 1}))
    #the_wavelet = wavelets.Wavelet(("bump", {"mu": 1, "s": 1.2}))
    #the_wavelet.viz()


    allWx = []
    allScales = []

    if display:
        mapper = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis", temporal_profiles.shape[0]))
        plt.figure(12)
        signal = plt.subplot(2, 2, 2)
        waveletd3 = plt.subplot(2, 2, 3)
        waveletthresh = plt.subplot(2, 2, 4)
        signal.cla()

    for r in range(temporal_profiles.shape[0]):
        if display:
            signal.plot(time, temporal_profiles[r, :], color=mapper.to_rgba(r, norm=False), marker="o", markersize=2)

        Wx, scales = cwt(temporal_profiles[r, :], wavelet=the_wavelet, t=time, padtype=padtype, scales="log",
                         l1_norm=True, nv=64)
        mod_Wx = np.abs(Wx)
        # Converts our scales to samples, and determines the coi.
        #To convert to Hz:
        #wc [(cycles*radians)/samples] / (2pi [radians]) * fs [samples/second]
        #= fc [cycles/second]

        freq_scales = scale_to_freq(scales, the_wavelet, len(temporal_profiles[r, :]), fs=fps, padtype=padtype)
        coi_scales = (scales * the_wavelet.std_t) / the_wavelet.wc_ct
        coi_im = np.ones_like(mod_Wx)

        for s, scale_ind in enumerate(coi_scales):
            scale_ind = int(scale_ind+1)
            coi_im[s, 0:scale_ind] = 0
            coi_im[s, -scale_ind:] = 0

        if display:
            waveletd3.imshow(mod_Wx)

        if sig_threshold is not None:
            overthresh = mod_Wx > sig_threshold
            mod_Wx[mod_Wx <= sig_threshold] = 0
            Wx[mod_Wx <= sig_threshold] = 0

            if display:
                waveletd3.imshow(mod_Wx)
                waveletthresh.imshow(np.reshape(overthresh, mod_Wx.shape))#, extent=(0, framestamps[150], scales[0], scales[-1]))
                plt.show(block=False)
                #plt.waitforbuttonpress()
        else:
            if display:
                waveletd3.imshow(mod_Wx)
                plt.show(block=False)
                #plt.waitforbuttonpress()

        allWx.append(Wx)
        allScales = freq_scales


#        if np.all(~(temporal_profiles[r, 0:150] == 0)):
    # waveletd3.plot(np.nanvar(np.abs(Wx), axis=1))
            #plt.waitforbuttonpress()

    return allWx, allScales, coi_im




