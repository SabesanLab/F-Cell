from itertools import repeat
from multiprocessing import Pool

import numpy as np
import pandas as pd
from joblib._multiprocessing_helpers import mp
from numpy.fft import fftshift
from scipy import signal
from scipy.fft import fft
from scipy.ndimage import center_of_mass, convolve1d
from scipy.signal import savgol_filter, convolve, freqz
from skimage.feature import graycomatrix, graycoprops
from matplotlib import pyplot as plt

from ssqueezepy import wavelets, p2up, cwt
from ssqueezepy.experimental import scale_to_freq

from ocvl.function.utility.resources import save_tiff_stack
from ocvl.function.utility.temporal_signal_utils import densify_temporal_matrix, reconstruct_profiles


def signal_power_iORG(temporal_profiles, framestamps, summary_method="var", window_size=1, fraction_thresh=0.25):
    """
    Calculates the iORG on a supplied dataset, using a variety of power based summary methods published in
    Cooper et. al. 2020, and Cooper et. al. 2017.

    :param temporal_profiles: A NxM numpy matrix with N cells and M temporal samples of some signal.
    :param framestamps: A 1xM numpy matrix containing the associated frame stamps for temporal_profiles.
    :param summary_method: The method used to summarize the population at each sample. Current options include:
                            "var", "std", and "moving_rms". Default: "var"
    :param window_size: The window size used to summarize the population at each sample. Can be an odd integer from
                        1 (no window) to M/2. Default: 1
    :param fraction_thresh: The fraction of the values inside the sample window that must be finite in order for the power
                            to be calculated- otherwise, the value will be considered np.nan.

    :return: a 1xM population iORG signal
    """

    if window_size != 0:
        if window_size % 2 < 1:
            raise Exception("Window size must be an odd integer.")
        else:
            window_radius = int((window_size - 1) / 2)
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

        elif window_size < (num_samples / 2):

            for i in range(window_radius, num_samples - window_radius):

                samples = temporal_profiles[:, (i - window_radius):(i + window_radius + 1)]
                if np.sum(np.isfinite(samples[:])) > np.ceil(samples.size * fraction_thresh):
                    iORG[i] = np.nanvar(samples[:])
                    num_incl[i] = np.sum(samples[:] != np.nan)

            iORG = iORG[window_radius:-window_radius]
            iORG = iORG[framestamps]
            num_incl = num_incl[framestamps]
        else:
            raise Exception("Window size must be less than half of the number of samples")

    elif summary_method == "std":
        if window_radius == 0:
            iORG = np.nanstd(temporal_profiles, axis=0)
            num_incl = np.sum(np.isfinite(temporal_profiles), axis=0)

        elif window_size < (num_samples / 2):

            for i in range(window_radius, num_samples - window_radius):

                samples = temporal_profiles[:, (i - window_radius):(i + window_radius + 1)]
                if np.sum(np.isfinite(samples[:])) > np.ceil(samples.size * fraction_thresh):
                    iORG[i] = np.nanstd(samples[:])
                    num_incl[i] = np.sum(samples[:] != np.nan)

            iORG = iORG[window_radius:-window_radius]
            iORG = iORG[framestamps]
            num_incl = num_incl[framestamps]
        else:
            raise Exception("Window size must be less than half of the number of samples")

    elif summary_method == "rms":
        if window_radius == 0:

            temporal_profiles **= 2  # Square first
            iORG = np.nanmean(temporal_profiles, axis=0)  # Average second
            iORG = np.sqrt(iORG)  # Sqrt last
            num_incl = np.sum(np.isfinite(temporal_profiles), axis=0)

        elif window_size < (num_samples / 2):

            temporal_profiles **= 2  # Square first
            for i in range(window_radius, num_samples - window_radius):

                samples = temporal_profiles[:, (i - window_radius):(i + window_radius + 1)]
                if samples[:].size != 0 and np.sum(np.isfinite(samples[:])) > np.ceil(samples.size * fraction_thresh):
                    iORG[i] = np.nanmean(samples[:])  # Average second
                    iORG[i] = np.sqrt(iORG[i])  # Sqrt last
                # num_incl[i] = np.sum(samples[:] != np.nan)

            iORG = iORG[window_radius:-window_radius]
            iORG = iORG[framestamps]
            # num_incl = num_incl[framestamps]
            num_incl = np.sum(np.isfinite(temporal_profiles), axis=0)
            num_incl = num_incl[framestamps]
        else:
            raise Exception("Window size must be less than half of the number of samples")
    elif summary_method == "avg":
        if window_radius == 0:
            iORG = np.nanmean(temporal_profiles, axis=0)  # Average
            num_incl = np.sum(np.isfinite(temporal_profiles), axis=0)

        elif window_size < (num_samples / 2):

            for i in range(window_radius, num_samples - window_radius):

                samples = temporal_profiles[:, (i - window_radius):(i + window_radius + 1)]
                if samples[:].size != 0 and np.sum(np.isfinite(samples[:])) > np.ceil(samples.size * fraction_thresh):
                    iORG[i] = np.nanmean(samples[:])  # Average

            iORG = iORG[window_radius:-window_radius]
            iORG = iORG[framestamps]
            # num_incl = num_incl[framestamps]
            num_incl = np.sum(np.isfinite(temporal_profiles), axis=0)
            num_incl = num_incl[framestamps]
        else:
            raise Exception("Window size must be less than half of the number of samples")

    return iORG, num_incl


def wavelet_iORG(temporal_profiles, framestamps, fps, sig_threshold=None, display=False):
    padtype = "reflect"

    time = framestamps / fps
    the_wavelet = wavelets.Wavelet(("gmw", {"gamma": 2, "beta": 1}))
    # the_wavelet = wavelets.Wavelet(("hhhat", {"mu": 1}))
    # the_wavelet = wavelets.Wavelet(("bump", {"mu": 1, "s": 1.2}))
    # the_wavelet.viz()

    allWx = []
    allScales = np.nan
    coi_im = np.nan

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

        if np.all(np.isfinite(temporal_profiles[r, :])):
            Wx, scales = cwt(temporal_profiles[r, :], wavelet=the_wavelet, t=time, padtype=padtype, scales="log",
                             l1_norm=True, nv=64)
            mod_Wx = np.abs(Wx)
            # Converts our scales to samples, and determines the coi.
            # To convert to Hz:
            # wc [(cycles*radians)/samples] / (2pi [radians]) * fs [samples/second]
            # = fc [cycles/second]

            freq_scales = scale_to_freq(scales, the_wavelet, len(temporal_profiles[r, :]), fs=fps, padtype=padtype)
            coi_scales = (scales * the_wavelet.std_t) / the_wavelet.wc_ct
            coi_im = np.ones_like(mod_Wx)

            for s, scale_ind in enumerate(coi_scales):
                scale_ind = int(scale_ind + 1)
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
                    waveletthresh.imshow(
                        np.reshape(overthresh, mod_Wx.shape))  # , extent=(0, framestamps[150], scales[0], scales[-1]))
                    plt.show(block=False)
                    # plt.waitforbuttonpress()
            else:
                if display:
                    waveletd3.imshow(mod_Wx)
                    plt.show(block=False)
                    # plt.waitforbuttonpress()

            allWx.append(Wx)
            allScales = freq_scales
        else:
            allWx.append(np.nan)

    #        if np.all(~(temporal_profiles[r, 0:150] == 0)):
    # waveletd3.plot(np.nanvar(np.abs(Wx), axis=1))
    # plt.waitforbuttonpress()

    return allWx, allScales, coi_im


def extract_texture(full_profiles, cellind, numlevels, summary_methods):
    contrast = np.full((1), np.nan)
    dissimilarity = np.full((1), np.nan)
    homogeneity = np.full((1), np.nan)
    energy = np.full((1), np.nan)
    correlation = np.full((1), np.nan)
    glcmmean = np.full((1), np.nan)
    ent = np.full((1), np.nan)
    if "contrast" in summary_methods or "all" in summary_methods:
        contrast = np.full((full_profiles.shape[-2]), np.nan)
    if "dissimilarity" in summary_methods or "all" in summary_methods:
        dissimilarity = np.full((full_profiles.shape[-2]), np.nan)
    if "homogeneity" in summary_methods or "all" in summary_methods:
        homogeneity = np.full((full_profiles.shape[-2]), np.nan)
    if "energy" in summary_methods or "all" in summary_methods:
        energy = np.full((full_profiles.shape[-2]), np.nan)
    if "correlation" in summary_methods or "all" in summary_methods:
        correlation = np.full((full_profiles.shape[-2]), np.nan)
    if "glcmmean" in summary_methods or "all" in summary_methods:
        glcmmean = np.full((full_profiles.shape[-2]), np.nan)
    if "entropy" in summary_methods or "all" in summary_methods:
        ent = np.full((full_profiles.shape[-2]), np.nan)

    avg = np.full((full_profiles.shape[-2], 1), np.nan)

    minlvl = np.nanmin(full_profiles[:, :, :, cellind].flatten())
    maxlvl = np.nanmax(full_profiles[:, :, :, cellind].flatten())

    thisprofile = np.round(((full_profiles[:, :, :, cellind] - minlvl) / (maxlvl - minlvl)) * (numlevels - 1))
    thisprofile[thisprofile >= numlevels] = numlevels - 1
    thisprofile = thisprofile.astype("uint8")
    # com=[]
    for f in range(full_profiles.shape[-2]):
        avg[f] = np.mean(full_profiles[1:-1, 1:-1, f, cellind].flatten())

        if avg[f] != 0:
            grayco = graycomatrix(thisprofile[:, :, f], distances=[1], angles=[0, np.pi / 4, np.pi / 2, np.pi * 3 / 4],
                                  levels=numlevels, normed=True, symmetric=True)

            grayco_invar = np.mean(grayco, axis=-1)
            grayco_invar = grayco_invar[..., None]

            if "contrast" in summary_methods or "all" in summary_methods:
                contrast[f] = graycoprops(grayco_invar, prop="contrast")
            if "dissimilarity" in summary_methods or "all" in summary_methods:
                dissimilarity[f] = graycoprops(grayco_invar, prop="dissimilarity")
            if "homogeneity" in summary_methods or "all" in summary_methods:
                homogeneity[f] = graycoprops(grayco_invar, prop="homogeneity")
            if "energy" in summary_methods or "all" in summary_methods:
                energy[f] = graycoprops(grayco_invar, prop="energy")
            if "correlation" in summary_methods or "all" in summary_methods:
                correlation[f] = graycoprops(grayco_invar, prop="correlation")
            if "glcmmean" in summary_methods or "all" in summary_methods:
                I, J = np.ogrid[0:numlevels, 0:numlevels]
                glcmmean[f] = (np.sum(I * np.squeeze(grayco_invar), axis=(0, 1)) + np.sum(J * np.squeeze(grayco_invar),
                                                                                          axis=(0, 1))) / 2
            if "entropy" in summary_methods or "all" in summary_methods:
                loggray = -np.log(grayco_invar, where=grayco_invar > 0)
                loggray[~np.isfinite(loggray)] = 0
                ent[f] = np.sum(grayco_invar * loggray, axis=(0, 1))

    return contrast, dissimilarity, homogeneity, energy, correlation, glcmmean


def extract_texture_profiles(full_profiles, summary_methods=("all"), numlevels=32, framestamps=None, display=False):
    """
    This function extracts textures using GLCM from a set of cell profiles.

    :param full_profiles: A numpy array of shape NxMxFxC, where NxM are the 2D dimensions of the area surrounding the
                          cell coordinate, F is the number of frames in the video, and C is number of cells.
    :param summary_methods: A tuple contraining "contrast", "dissimilarity", "homogeneity", "energy", "correlation",
                           "gclmmean", "entropy", or "all". (Default: "all")
    :param numlevels: The number of graylevels to quantize the input data to. (Default: 32)
    :param framestamps: The framestamps for each frame; used for data display. (Default: None)
    :param display: Enables display of the texture "profile" for each supplied cell. (Default: False)

    :return: A dictionary where each key is the texture name, and each value contains CxF cell texture profiles.
    """

    # Assumes supplied by extract_profiles, which has the below shapes.
    if "contrast" in summary_methods or "all" in summary_methods:
        contrast = np.empty((full_profiles.shape[-1], full_profiles.shape[-2]))
    if "dissimilarity" in summary_methods or "all" in summary_methods:
        dissimilarity = np.empty((full_profiles.shape[-1], full_profiles.shape[-2]))
    if "homogeneity" in summary_methods or "all" in summary_methods:
        homogeneity = np.empty((full_profiles.shape[-1], full_profiles.shape[-2]))
    if "energy" in summary_methods or "all" in summary_methods:
        energy = np.empty((full_profiles.shape[-1], full_profiles.shape[-2]))
    if "correlation" in summary_methods or "all" in summary_methods:
        correlation = np.empty((full_profiles.shape[-1], full_profiles.shape[-2]))
    if "glcmmean" in summary_methods or "all" in summary_methods:
        glcmmean = np.empty((full_profiles.shape[-1], full_profiles.shape[-2]))
    if "entropy" in summary_methods or "all" in summary_methods:
        entropy = np.empty((full_profiles.shape[-1], full_profiles.shape[-2]))

    retdict = {}

    with Pool(processes=int(np.round(mp.cpu_count() / 2))) as pool:

        reconst = pool.starmap_async(extract_texture,
                                     zip(repeat(full_profiles), range(full_profiles.shape[-1]),
                                         repeat(numlevels), repeat(summary_methods)))

        res = reconst.get()
        for c, result in enumerate(res):
            if "contrast" in summary_methods or "all" in summary_methods:
                contrast[c, :] = np.array(result[0])
            if "dissimilarity" in summary_methods or "all" in summary_methods:
                dissimilarity[c, :] = np.array(result[1])
            if "homogeneity" in summary_methods or "all" in summary_methods:
                homogeneity[c, :] = np.array(result[2])
            if "energy" in summary_methods or "all" in summary_methods:
                energy[c, :] = np.array(result[3])
            if "correlation" in summary_methods or "all" in summary_methods:
                correlation[c, :] = np.array(result[4])
            if "glcmmean" in summary_methods or "all" in summary_methods:
                glcmmean[c, :] = np.array(result[5])
            if "entropy" in summary_methods or "all" in summary_methods:
                entropy[c, :] = np.array(result[5])

    if "contrast" in summary_methods or "all" in summary_methods:
        retdict["contrast"] = contrast
    if "dissimilarity" in summary_methods or "all" in summary_methods:
        retdict["dissimilarity"] = dissimilarity
    if "homogeneity" in summary_methods or "all" in summary_methods:
        retdict["homogeneity"] = homogeneity
    if "energy" in summary_methods or "all" in summary_methods:
        retdict["energy"] = energy
    if "correlation" in summary_methods or "all" in summary_methods:
        retdict["correlation"] = correlation
    if "glcmmean" in summary_methods or "all" in summary_methods:
        retdict["glcmmean"] = glcmmean
    if "entropy" in summary_methods or "all" in summary_methods:
        retdict["entropy"] = entropy

    if display:
        for c in range(full_profiles.shape[-1]):
            plt.figure(0)
            plt.clf()
            if "glcmmean" in summary_methods or "all" in summary_methods:
                plt.subplot(2, 3, 1)
                plt.title("glcmmean")
                plt.plot(framestamps, glcmmean[c, :], "k")
            if "correlation" in summary_methods or "all" in summary_methods:
                plt.subplot(2, 3, 2)
                plt.title("correlation")
                plt.plot(framestamps, correlation[c, :], "k")
            if "homogeneity" in summary_methods or "all" in summary_methods:
                plt.subplot(2, 3, 3)
                plt.title("homogeneity")
                plt.plot(framestamps, homogeneity[c, :], linestyle="-")
                plt.plot(framestamps, savgol_filter(homogeneity[c, :], 9, 2), "b", linewidth=3)
            if "energy" in summary_methods or "all" in summary_methods:
                plt.subplot(2, 3, 4)
                plt.title("energy")
                plt.plot(framestamps, energy[c, :], linestyle="-")
            if "contrast" in summary_methods or "all" in summary_methods:
                plt.subplot(2, 3, 5)
                plt.title("contrast")
                plt.plot(framestamps, contrast[c, :], linestyle="-")
            if "entropy" in summary_methods or "all" in summary_methods:
                plt.subplot(2, 3, 6)
                plt.title("entropy")
                plt.plot(framestamps, entropy[c, :], linestyle="-")
                plt.show(block=False)
            plt.waitforbuttonpress()

    return retdict

    # Temp: for center of mass calculation.
    #         # com.append(center_of_mass(full_profiles[:, :, f, cellind]) - np.round(full_profiles.shape[0]/2))
    #         # glcmmean[f] = np.sqrt(np.sum(com[f]**2))


def filtered_absolute_difference(temporal_profiles, framestamps, filter_type="savgol", filter_size=33, display=True):

    firfilt = signal.firwin(filter_size, (0.75, 2), fs=29.4, pass_zero="bandstop", window=("kaiser", 2))

    if filter_type == "savgol":
        filtered_profiles = savgol_filter(temporal_profiles, window_length=filter_size, polyorder=4, mode="mirror",
                                          axis=1)
        # filter_grad_profiles = savgol_filter(temporal_profiles, window_length=filter_size, polyorder=2, mode="mirror",
        #                                      axis=1, deriv=1)  # np.gradient(filtered_profiles, axis=1)
        filtered_profiles = convolve1d(filtered_profiles, firfilt, mode="reflect", axis=1)

        filter_grad_profiles = np.gradient(filtered_profiles, axis=1)
    elif filter_type == "trunc_sinc":
        # Formulas from Schmid et al- these are MS1 filters.
        alpha = 4
        m = (filter_size - 1)/2
        x = np.linspace(-m, m, filter_size) / (m+1)
        n = 2
        window = np.exp(-alpha * (x ** 2)) + np.exp(-alpha * ((x + 2) ** 2)) + np.exp(-alpha * ((x - 2) ** 2)) \
                 - 2 * np.exp(-alpha) - np.exp(-9 * alpha)

        adj_sinc = np.sin( ((n+2)/2)* np.pi*x ) / (((n+2)/2)*np.pi*x)
        adj_sinc[int(m)] = 0

        if n == 4:
            j = 0
            k = 0.02194 + 0.05028 / (0.76562 - m)**3
            correction = np.zeros_like(adj_sinc)
            for i in range(len(x)):
                correction[i] = np.sum(k * x[i]*np.sin((j+1)*np.pi*x[i]))
            adj_sinc += correction


        trunc_sinc = adj_sinc*window
        trunc_sinc /= np.sum(trunc_sinc)

        #filtered_profiles = convolve1d(temporal_profiles, firfilt, mode="reflect", axis=1)
        filtered_profiles = convolve1d(temporal_profiles, trunc_sinc, mode="reflect", axis=1)
        #filtered_profiles = convolve1d(filtered_profiles, firfilt, mode="reflect", axis=1)

        filter_grad_profiles = np.gradient(filtered_profiles, axis=1)

    abs_diff_profiles = np.nancumsum(np.abs(filter_grad_profiles[:, 48:80]), axis=1)

    fad = np.amax(abs_diff_profiles, axis=1)
    fad[fad == 0] = np.nan
    # if np.nanstd(np.log(fad), axis=-1) > .7:

    if np.any(fad>0):
        plt.figure(42)
        plt.clf()
        for i in range(temporal_profiles.shape[0]):

            plt.subplot(2, 2, 1)
            plt.title("Raw data")
            plt.plot(framestamps, temporal_profiles[i, :])
            plt.plot(framestamps, filtered_profiles[i, :], 'k', linewidth=2)
            # plt.plot(framestamps, filtered_profiles_fir[i, :], "g", linewidth=2)
            plt.subplot(2, 2, 2)
            plt.title("Filtered data")
            plt.plot(framestamps, filtered_profiles[i, :])
            plt.subplot(2, 2, 3)
            plt.title("Filtered Derivative")
            plt.plot(framestamps, filter_grad_profiles[i, :])
            # plt.title("Power spectrum of filtered signal")
            # plt.plot( np.abs(fftshift(fft(filter_grad_profiles[i, :])))**2 )
            plt.subplot(2, 2, 4)
            plt.title("AUC")
            plt.plot(framestamps[48:80], abs_diff_profiles[i, :])
            # plt.waitforbuttonpress()


        plt.waitforbuttonpress()

    return np.amax(abs_diff_profiles, axis=1)
