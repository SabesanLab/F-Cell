import multiprocessing as mp
from itertools import repeat

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from sklearn import linear_model
from pynufft import NUFFT


def densify_temporal_matrix(temporal_profiles, framestamps, max_framestamp=None):
    """

    By default, most algorithms in this package work with sparsely sampled data by default (governed by the framestamps)
    to save what little ram we can. This function takes this sparse representation and makes it "dense" for algorithms
    that rely on temporally coupled samples (like those that use windows)

    :param temporal_profiles: A NxM numpy matrix with N cells and M temporal samples of some signal.
    :param framestamps: A 1xM numpy matrix containing the associated frame stamps for temporal_profiles.
    :param max_framestamp: The maximum number of samples we expect in this signal. Defaults to the maximum value in the
                            framestamp array.

    :return: The dense temporal profile matrix, where missing data is filled with "np.nan"s
    """


    if max_framestamp is None:
        max_framestamp = np.max(framestamps)+1

    if len(temporal_profiles.shape) == 1:
        num_signals = 1
        densified_profiles = np.empty(max_framestamp)
        densified_profiles[:] = np.nan
    else:
        num_signals = temporal_profiles.shape[0]
        densified_profiles = np.empty((num_signals, max_framestamp))
        densified_profiles[:] = np.nan

    i=0
    for j in range(max_framestamp):
        if j == framestamps[i]:
            # If j corresponds to the next frame stamp, slot that column in, and increment to the next frame stamp.
            if len(temporal_profiles.shape) == 1:
                densified_profiles[j] = temporal_profiles[i]
            else:
                densified_profiles[:, j] = temporal_profiles[:, i]
            i += 1


    return densified_profiles


def l1_compressed_sensing(temporal_profiles, framestamps, c, threshold=None):
    """
    This function uses

    :param temporal_profiles:
    :param framestamps:
    :param c:
    :return:
    """

    fullrange = np.arange(framestamps[-1] + 1)

    # framegaps = np.abs(np.diff(framestamps))
    # biggest_framegap = np.where(framegaps == np.max(framegaps))[0][0]+1
    #
    # # Remove the frame gap we can find just for fitting, then add it back in at the end.
    # clean_frmstamp = np.concatenate( (framestamps[0:(biggest_framegap+1)],
    #                                   framestamps[(biggest_framegap+1):] - np.max(framegaps)), axis=0)
    # clean_fullrange = np.arange(clean_frmstamp[-1] + 1)

    finers = np.isfinite(temporal_profiles[c, :])

    nummissing = ((framestamps[-1] +1) - len(framestamps)) + np.sum(np.invert(finers))

    if threshold is None or nummissing/framestamps[-1] <= threshold:

        sigmean = np.mean(temporal_profiles[c, finers])
        sigstd = np.std(temporal_profiles[c, finers])

        D = sp.fft.dct(np.eye(framestamps[-1] + 1), norm="ortho", orthogonalize=True)

        A = D[framestamps[finers], :]
        lasso = linear_model.Lasso(alpha=0.001, max_iter=2000)
        lasso.fit(A, (temporal_profiles[c, finers]-sigmean)/sigstd)

        reconstruction = sp.fft.idct(lasso.coef_.reshape((len(fullrange),)), axis=0,
                                     norm="ortho", orthogonalize=True) * sigstd + sigmean

        # filled_recon = densify_temporal_matrix(reconstruction, framestamps)

        # plt.figure(0)
        # plt.subplot(2, 1, 1)
        # plt.plot(framestamps[finers], temporal_profiles[c, finers], "-d")
        # plt.figure(0)
        # plt.subplot(2, 1, 2)
        # plt.plot(fullrange, reconstruction, "-d")
        # plt.waitforbuttonpress()

        return reconstruction, nummissing
    else:
        print( "Missing " + str(100*(nummissing/framestamps[-1])) + "% of data from this profile. Removing...")
        reconstruction = densify_temporal_matrix(temporal_profiles[c, :], framestamps)
        reconstruction[:] = np.nan
        return reconstruction, nummissing


def reconstruct_profiles(temporal_profiles, framestamps, method="L1", threshold=0.2):
    """
    This function reconstructs the missing profile data using compressed sensing techniques.

    :param temporal_profiles: A NxM numpy matrix with N cells and M temporal samples of some signal.
    :param framestamps: A 1xM numpy matrix containing the associated frame stamps for temporal_profiles.
    :param method: The method used for compressive sensing. Defaults to an L1 norm based approach.
    :param threshold: The threshold at which we will simply drop a signal. As we are not doing true compressive sensing,
                      the default is 0.2 (80% of the signal must be present).

    :return: A tuple containing the reconstructed signals, the full framestamp range of the signals, and the number of
             missing framestamps from each signal.
    """
    fullrange = np.arange(framestamps[-1] + 1)

    reconstruction = np.empty((temporal_profiles.shape[0], len(fullrange)))
    nummissing = np.empty((temporal_profiles.shape[0], 1))

    if method == "L1":

        # l1_compressed_sensing( temporal_profiles, framestamps, 0)
        # Create a pool of threads for processing.
        with mp.Pool(processes=int(np.round(mp.cpu_count() / 2))) as pool:

            reconst = pool.starmap_async(l1_compressed_sensing, zip(repeat(temporal_profiles), repeat(framestamps),
                                                                    range(temporal_profiles.shape[0]), repeat(threshold)))
            res = reconst.get()
            for c, result in enumerate(res):
                reconstruction[c, :] = np.array(result[0])
                nummissing[c] = np.array(result[1])

    elif method == "NUFFT":
        Nufft = NUFFT()

        fsamp = np.arange(0 , framestamps[-1])/ framestamps[-1]
        #om = np.random.randn(temporal_profiles.shape[1]*2, 1)

        for c in range(temporal_profiles.shape[0]):
            finers = np.isfinite(temporal_profiles[c, :])
            good_frms = framestamps[finers, np.newaxis]
            Nufft.plan(good_frms, (int(temporal_profiles[c, finers].shape[-1]),) , (int(good_frms[-1]),), (6,))

            freqy = Nufft.forward(temporal_profiles[c, finers])

            resto = Nufft.solve(freqy, "L1TVOLS", maxiter=30, rho=1)

            plt.figure(0)
            plt.subplot(2, 1, 1)
            plt.plot(framestamps[finers], temporal_profiles[c, finers], "-d")
            plt.figure(0)
            plt.subplot(2, 1, 2)
            plt.plot(fullrange, resto, "-d")
            plt.waitforbuttonpress()


    # plt.figure(9001)
    # plt.hist(nummissing, len(fullrange))
    # plt.show(block=False)

    print(str(100 * np.mean(nummissing) / len(fullrange)) + "% signal reconstructed on average.")

    return reconstruction, fullrange, nummissing