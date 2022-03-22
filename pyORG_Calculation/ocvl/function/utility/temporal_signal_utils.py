import numpy as np


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
