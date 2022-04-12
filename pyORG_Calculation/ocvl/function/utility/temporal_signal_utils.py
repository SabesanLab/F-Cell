import multiprocessing as mp
from itertools import repeat

import numpy as np
import scipy as sp
from sklearn import linear_model


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
    num_signals = temporal_profiles.shape[0]

    if max_framestamp is None:
        max_framestamp = np.max(framestamps)

    densified_profiles = np.empty((num_signals, max_framestamp))
    densified_profiles[:] = np.nan

    i=0
    for j in range(max_framestamp):
        if j == framestamps[i]:
            # If j corresponds to the next frame stamp, slot that column in, and increment to the next frame stamp.
            densified_profiles[:, j] = temporal_profiles[:, i]
            i += 1

    return densified_profiles


def l1_compressed_sensing(temporal_profiles, framestamps, c):
    """
    This function uses

    :param temporal_profiles:
    :param framestamps:
    :param c:
    :return:
    """
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
    This function reconstructs the missing profile data using compressed sensing techniques.

    :param temporal_profiles: A NxM numpy matrix with N cells and M temporal samples of some signal.
    :param framestamps: A 1xM numpy matrix containing the associated frame stamps for temporal_profiles.
    :param method: The method used for compressive sensing. Defaults to an L1 norm based approach.

    :return: A tuple containing the reconstructed signals, the full framestamp range of the signals, and the number of
             missing framestamps from each signal.
    """
    fullrange = np.arange(framestamps[-1] + 1)

    reconstruction = np.empty((temporal_profiles.shape[0], len(fullrange)))
    nummissing = np.empty((temporal_profiles.shape[0], 1))


    if method == "L1":
        # Create a pool of threads for processing.
        with mp.Pool(processes=int(np.round(mp.cpu_count() / 2))) as pool:

            reconst = pool.starmap_async(l1_compressed_sensing, zip(repeat(temporal_profiles), repeat(framestamps),
                                                                    range(temporal_profiles.shape[0])))
            res = reconst.get()
            for c, result in enumerate(res):
                reconstruction[c, :] = np.array(result[0])
                nummissing[c] = np.array(result[1])


    print(str(100 * np.mean(nummissing) / len(fullrange)) + "% signal reconstructed on average.")

    return reconstruction, fullrange, nummissing