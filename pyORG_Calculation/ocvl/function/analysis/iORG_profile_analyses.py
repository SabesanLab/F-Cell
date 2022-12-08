from itertools import repeat
from multiprocessing import Pool

import numpy as np
import pandas as pd
from joblib._multiprocessing_helpers import mp
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

                samples = temporal_profiles[:, (i - window_radius):(i + window_radius+1)]
                if np.sum(np.isfinite(samples[:])) > np.ceil(samples.size*fraction_thresh):
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

        elif window_size < (num_samples/2):

            for i in range(window_radius, num_samples-window_radius):

                samples = temporal_profiles[:, (i - window_radius):(i + window_radius+1)]
                if np.sum(np.isfinite(samples[:])) > np.ceil(samples.size*fraction_thresh):
                    iORG[i] = np.nanstd(samples[:])
                    num_incl[i] = np.sum(samples[:] != np.nan)

            iORG = iORG[window_radius:-window_radius]
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

                samples = temporal_profiles[:, (i - window_radius):(i + window_radius+1)] # Fix- window was one element too small.
                if samples[:].size != 0 and np.sum(np.isfinite(samples[:])) > np.ceil(samples.size*fraction_thresh): # Fix- threshold was a hard number instead of a variable.
                    iORG[i] = np.nanmean(samples[:]) # Average second
                    iORG[i] = np.sqrt(iORG[i]) # Sqrt last
                   # num_incl[i] = np.sum(samples[:] != np.nan)

            iORG = iORG[window_radius:-window_radius] # Fix- padding wasn't being cropped off first
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
        else:
            allWx.append(np.nan)


#        if np.all(~(temporal_profiles[r, 0:150] == 0)):
    # waveletd3.plot(np.nanvar(np.abs(Wx), axis=1))
            #plt.waitforbuttonpress()

    return allWx, allScales, coi_im


def extract_texture(full_profiles, cellind, summary_method):

    if summary_method == "contrast" or summary_method == "all":
        contrast = np.empty((1, full_profiles.shape[-2]))
    if summary_method == "dissimilarity" or summary_method == "all":
        dissimilarity = np.empty((1, full_profiles.shape[-2]))
    if summary_method == "homogeneity" or summary_method == "all":
        homogeneity = np.empty((1, full_profiles.shape[-2]))
    if summary_method == "energy" or summary_method == "all":
        energy = np.empty((1, full_profiles.shape[-2]))
    if summary_method == "correlation" or summary_method == "all":
        correlation = np.empty((1, full_profiles.shape[-2]))
    if summary_method == "asm" or summary_method == "all":
        asm = np.empty((1, full_profiles.shape[-2]))

    for f in range(full_profiles.shape[-2]):
        grayco = graycomatrix(full_profiles[:, :, f, cellind], distances=[1], angles=[0, np.pi/2], levels=255)

        if summary_method == "contrast" or summary_method == "all":
            print(graycoprops(grayco, prop="contrast"))
            contrast[f] = graycoprops(grayco, prop="contrast")
        if summary_method == "dissimilarity" or summary_method == "all":
            print(graycoprops(grayco, prop="dissimilarity"))
            dissimilarity[f] = graycoprops(grayco, prop="dissimilarity")
        if summary_method == "homogeneity" or summary_method == "all":
            print(graycoprops(grayco, prop="homogeneity"))
            homogeneity[f] = graycoprops(grayco, prop="homogeneity")
        if summary_method == "energy" or summary_method == "all":
            print(graycoprops(grayco, prop="energy"))
            energy[f] = graycoprops(grayco, prop="energy")
        if summary_method == "correlation" or summary_method == "all":
            print(graycoprops(grayco, prop="correlation"))
            correlation[f] = graycoprops(grayco, prop="correlation")
        if summary_method == "asm" or summary_method == "all":
            print(graycoprops(grayco, prop="ASM"))
            asm[f] = graycoprops(grayco, prop="ASM")

    return contrast, dissimilarity, homogeneity, energy, correlation, asm

def extract_texture_profiles(full_profiles, summary_method="all", numlevels=64,  tmpframestamps=None, outname=None, coords=None):


    for cellind in range(full_profiles.shape[-1]):
        if summary_method == "contrast" or summary_method == "all":
            contrast = np.full((full_profiles.shape[-2]), np.nan)
        if summary_method == "dissimilarity" or summary_method == "all":
            dissimilarity = np.full((full_profiles.shape[-2]), np.nan)
        if summary_method == "homogeneity" or summary_method == "all":
            homogeneity = np.full((full_profiles.shape[-2]), np.nan)
        if summary_method == "energy" or summary_method == "all":
            energy = np.full((full_profiles.shape[-2]), np.nan)
        if summary_method == "correlation" or summary_method == "all":
            correlation = np.full((full_profiles.shape[-2]), np.nan)
        if summary_method == "glcmmean" or summary_method == "all":
            glcmmean = np.full((full_profiles.shape[-2]), np.nan)
        if summary_method == "entropy" or summary_method == "all":
            ent = np.full((full_profiles.shape[-2]), np.nan)

        avg = np.empty((full_profiles.shape[-2], 1))

        minlvl = 0 #np.nanmin(full_profiles[:, :, :, cellind].flatten())
        maxlvl = 255 #np.nanmax(full_profiles[:, :, :, cellind].flatten())

        print("Min: "+str(minlvl)+" Max: "+str(maxlvl))

        thisprofile = np.round(((full_profiles[:, :, :, cellind]-minlvl) / (maxlvl-minlvl)) * (numlevels-1) )
        thisprofile[thisprofile >= numlevels] = numlevels-1
        thisprofile = thisprofile.astype("uint8")

        for f in range(full_profiles.shape[-2]):
            avg[f] = np.mean(full_profiles[:, :, f, cellind].flatten())

            if avg[f] != 0:
                grayco = graycomatrix(thisprofile[:, :, f], distances=[1], angles=[0, np.pi/4, np.pi/2, np.pi*3/4],
                                      levels=numlevels, normed=True, symmetric=True)

                grayco_invar = np.mean(grayco, axis=-1)
                grayco_invar = grayco_invar[..., None]

                if summary_method == "contrast" or summary_method == "all":
                    contrast[f] = graycoprops(grayco_invar, prop="contrast")
                if summary_method == "dissimilarity" or summary_method == "all":
                    dissimilarity[f] = graycoprops(grayco_invar, prop="dissimilarity")
                if summary_method == "homogeneity" or summary_method == "all":
                    homogeneity[f] = graycoprops(grayco_invar, prop="homogeneity")
                if summary_method == "energy" or summary_method == "all":
                    energy[f] = graycoprops(grayco_invar, prop="energy")
                if summary_method == "correlation" or summary_method == "all":
                    correlation[f] = graycoprops(grayco_invar, prop="correlation")
                if summary_method == "glcmmean" or summary_method == "all":
                    I, J = np.ogrid[0:numlevels, 0:numlevels]
                    glcmmean[f] = (np.sum(I*np.squeeze(grayco_invar), axis=(0, 1)) + np.sum(J*np.squeeze(grayco_invar), axis=(0, 1)))/2
                if summary_method == "entropy" or summary_method == "all":
                    loggray = -np.log(grayco_invar)
                    loggray[~np.isfinite(loggray)] = 0
                    ent[f] = np.sum(grayco_invar * loggray, axis=(0, 1))


        plt.figure(0)
        plt.clf()
        plt.subplot(2, 3, 1)
        plt.title("average")
        plt.plot(tmpframestamps, avg, "k", linewidth=3)
        #plt.plot(tmpframestamps, contrast, "r", linestyle="-")
        # plt.plot(tmpframestamps, contrast[1, :], "r", linestyle="-")
        # plt.plot(tmpframestamps, contrast[2, :], "r", linestyle="-")
        # plt.plot(tmpframestamps, contrast[3, :], "r", linestyle="-")
        plt.subplot(2, 3, 2)
        plt.title("glcm mean")
        plt.plot(tmpframestamps, glcmmean, linestyle="-")
        # plt.plot(tmpframestamps, dissimilarity[1, :], linestyle="-")
        # plt.plot(tmpframestamps, dissimilarity[2, :], linestyle="-")
        # plt.plot(tmpframestamps, dissimilarity[3, :], linestyle="-")
        # plt.plot(tmpframestamps, np.nanmean(dissimilarity, axis=0), "k", linewidth=3)
        plt.subplot(2, 3, 3)
        plt.title("homogeneity")
        plt.plot(tmpframestamps, homogeneity, linestyle="-")
        # plt.plot(tmpframestamps, homogeneity[1, :], linestyle="-")
        # plt.plot(tmpframestamps, homogeneity[2, :], linestyle="-")
        # plt.plot(tmpframestamps, homogeneity[3, :], linestyle="-")
        # plt.plot(tmpframestamps, np.nanmean(homogeneity, axis=0), "k", linewidth=3)
        plt.subplot(2, 3, 4)
        plt.title("energy")
        plt.plot(tmpframestamps, energy, linestyle="-")
        # plt.plot(tmpframestamps, energy[1, :], linestyle="-")
        # plt.plot(tmpframestamps, energy[2, :], linestyle="-")
        # plt.plot(tmpframestamps, energy[3, :], linestyle="-")
        # plt.plot(tmpframestamps, np.nanmean(energy, axis=0), "k", linewidth=3)
        plt.subplot(2, 3, 5)
        plt.title("contrast")
        plt.plot(tmpframestamps, contrast, linestyle="-")
        # plt.plot(tmpframestamps, correlation[1, :], linestyle="-")
        # plt.plot(tmpframestamps, correlation[2, :], linestyle="-")
        # plt.plot(tmpframestamps, correlation[3, :], linestyle="-")
        # plt.plot(tmpframestamps.transpose(), np.nanmean(correlation, axis=0), "k", linewidth=3)
        plt.subplot(2, 3, 6)
        plt.title("entropy")
        plt.plot(tmpframestamps, ent, linestyle="-")
        # plt.plot(tmpframestamps, asm[1, :], linestyle="-")
        # plt.plot(tmpframestamps, asm[2, :], linestyle="-")
        # plt.plot(tmpframestamps, asm[3, :], linestyle="-")
        # plt.plot(tmpframestamps.transpose(), np.nanmean(asm, axis=0), "k", linewidth=3)
        plt.show(block=False)

        save_tiff_stack(outname.with_name( outname.name + "cell" + str(coords[cellind][0]) + "," + str(coords[cellind][1]) +".tif"),
                        full_profiles[:, :, :, cellind])

        plt.waitforbuttonpress()



    # if summary_method == "contrast" or summary_method == "all":
    #     contrast = np.empty((full_profiles.shape[-1], full_profiles.shape[-2])) # Assumes supplied by extract_profiles,
    #                                                                             # which has this form.
    # if summary_method == "dissimilarity" or summary_method == "all":
    #     dissimilarity = np.empty((full_profiles.shape[-1], full_profiles.shape[-2]))
    # if summary_method == "homogeneity" or summary_method == "all":
    #     homogeneity = np.empty((full_profiles.shape[-1], full_profiles.shape[-2]))
    # if summary_method == "energy" or summary_method == "all":
    #     energy = np.empty((full_profiles.shape[-1], full_profiles.shape[-2]))
    # if summary_method == "correlation" or summary_method == "all":
    #     correlation = np.empty((full_profiles.shape[-1], full_profiles.shape[-2]))
    # if summary_method == "asm" or summary_method == "all":
    #     asm = np.empty((full_profiles.shape[-1], full_profiles.shape[-2]))
    #
    # with Pool(processes=int(np.round(mp.cpu_count() / 2))) as pool:
    #
    #     reconst = pool.starmap_async(extract_texture, zip(repeat(full_profiles.astype("uint16")), range(full_profiles.shape[-1]),
    #                                             repeat(summary_method)))
    #
    #     res = reconst.get()
    #     for c, result in enumerate(res):
    #         if summary_method == "contrast" or summary_method == "all":
    #             contrast[c, :] = np.array(result[0])
    #         if summary_method == "dissimilarity" or summary_method == "all":
    #             dissimilarity[c, :] = np.array(result[1])
    #         if summary_method == "homogeneity" or summary_method == "all":
    #             homogeneity[c, :] = np.array(result[2])
    #         if summary_method == "energy" or summary_method == "all":
    #             energy[c, :] = np.array(result[3])
    #         if summary_method == "correlation" or summary_method == "all":
    #             correlation[c, :] = np.array(result[4])
    #         if summary_method == "asm" or summary_method == "all":
    #             asm[c, :] = np.array(result[5])

    return contrast, dissimilarity, homogeneity, energy, correlation, glcmmean



