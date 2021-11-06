import itertools

import cv2
import numpy as np
from cv2 import matchTemplate
from numpy.polynomial import Polynomial

from ocvl.function.utility.resources import save_video


def flat_field(dataset, sigma=20):
    kernelsize = 3 * sigma
    if (kernelsize % 2) == 0:
        kernelsize += 1

    flat_fielded_dataset = np.zeros(dataset.shape, dtype=dataset.dtype)

    for i in range(dataset.shape[-1]):
        dataframe = dataset[..., i]
        dataframe[dataframe == 0] = 1
        blurred_frame = cv2.GaussianBlur(dataframe.astype("float64"), (kernelsize, kernelsize),
                                         sigmaX=sigma, sigmaY=sigma)

        flat_fielded = (dataset[..., i].astype("float64") / blurred_frame)

        flat_fielded -= np.amin(flat_fielded)
        flat_fielded /= np.amax(flat_fielded)
        if dataset.dtype == "uint8":
            flat_fielded *= 255
        elif dataset.dtype == "uint16":
            flat_fielded *= 65535

        # cv2.imshow('Frame', flat_fielded.astype(dataset.dtype))

        flat_fielded_dataset[..., i] = flat_fielded

        # Press Q on keyboard to  exit
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #    return

    return flat_fielded_dataset.astype(dataset.dtype)


# Where the image data is N rows x M cols and F frames
# and the row_shifts and col_shifts are F x N.
# Assumes a row-wise distortion/a row-wise fast scan ("distortionless" along each row)
# Returns a float image (spans from 0-1).
def dewarp_2D_data(image_data, row_shifts, col_shifts, method="median"):
    numstrips = row_shifts.shape[1]
    height = image_data.shape[0]
    width = image_data.shape[1]
    num_frames = image_data.shape[-1]

    allrows = np.linspace(0, numstrips - 1, num=height)  # Make a linspace for all of our images' rows.
    substrip = np.linspace(0, numstrips - 1, num=numstrips)

    indiv_colshift = np.zeros([num_frames, height])
    indiv_rowshift = np.zeros([num_frames, height])

    for f in range(num_frames):
        # Fit across rows, in order to capture all strips for a given dataset
        col_strip_fit = Polynomial.fit(substrip, col_shifts[f, :], deg=8)
        indiv_colshift[f, :] = col_strip_fit(allrows)
        # Fit across rows, in order to capture all strips for a given dataset
        row_strip_fit = Polynomial.fit(substrip, row_shifts[f, :], deg=8)
        indiv_rowshift[f, :] = row_strip_fit(allrows)

    if method == "median":
        centered_col_shifts = -np.median(indiv_colshift, axis=0)
        centered_row_shifts = -np.median(indiv_rowshift, axis=0)

    dewarped = np.zeros(image_data.shape)

    col_base = np.tile(np.arange(width, dtype=np.float32)[np.newaxis, :], [height, 1])
    row_base = np.tile(np.arange(height, dtype=np.float32)[:, np.newaxis], [1, width])

    centered_col_shifts = col_base + np.tile(centered_col_shifts[:, np.newaxis], [1, width]).astype("float32")
    centered_row_shifts = row_base + np.tile(centered_row_shifts[:, np.newaxis], [1, width]).astype("float32")

    for f in range(num_frames):
        dewarped[..., f] = cv2.remap(image_data[..., f].astype("float64") / 255, centered_col_shifts,
                                     centered_row_shifts,
                                     interpolation=cv2.INTER_LANCZOS4)

        # cv2.imshow("diff warped", (image_data[..., f].astype("float64")/255)-dewarped[..., f])
        # cv2.imshow("dewarped", dewarped[..., f])
        # c = cv2.waitKey(1000)
        # if c == 27:
        #     break

    # Clamp our values.
    dewarped[dewarped < 0] = 0
    dewarped[dewarped > 1] = 1

    if image_data.dtype == np.uint8:
        return (dewarped * 255).astype("uint8"), centered_col_shifts, centered_row_shifts
    else:
        return dewarped, centered_col_shifts, centered_row_shifts

    # save_video("C:\\Users\\rober\\Documents\\temp\\test.avi", (dewarped*255).astype("uint8"), 30)


def relativize_image_stack(image_data, mask_data, reference_idx=1):
    num_frames = image_data.shape[-1]

    xform = [0] * num_frames
    corrcoeff = np.empty((num_frames, 1))
    corrcoeff[:] = np.NAN
    corrected_im = np.zeros(image_data.shape)

    # Contrast threshold raise (from 0.05) minimizes features in uniform regions, but is lower than the original
    # paper (0.09)
    sift = cv2.SIFT_create(1000)

    keypoints = []
    descriptors = []

    for f in range(num_frames):
        kp, des = sift.detectAndCompute(image_data[..., f], mask_data[..., f], None)

        keypoints.append(kp)
        descriptors.append(des)


    # Set up FLANN parameters (feature matching)... review these.
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=1000)

    flan = cv2.FlannBasedMatcher(index_params, search_params)

    # Specify the number of iterations.
    for f in range(num_frames):
        matches = flan.knnMatch(descriptors[f], descriptors[reference_idx], k=2)

        good_matches = []
        for f1, f2 in matches:

            if f1.distance < 0.8 * f2.distance:
                good_matches.append(f1)

        if len(good_matches) >= 5:

            src_pts = np.float32([keypoints[f][f1.queryIdx].pt for f1 in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints[reference_idx][f1.trainIdx].pt for f1 in good_matches]).reshape(-1, 1, 2)

            M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)

            h, w = image_data[..., f].shape
            if M is not None and np.sum(inliers) > 2:
                xform[f] = M

                corrected_im[..., f] = cv2.warpAffine(image_data[..., f], xform[f], image_data[..., f].shape,
                                                      flags=cv2.INTER_LANCZOS4)
                warped_mask = cv2.warpAffine(mask_data[..., f], xform[f], mask_data[..., f].shape,
                                             flags=cv2.INTER_NEAREST)

                # Calculate and store the final correlation. It should be decent, if the transform was.
                res = matchTemplate(image_data[..., reference_idx], corrected_im[..., f].astype("uint8"),
                                    cv2.TM_CCOEFF_NORMED, mask=warped_mask)

                corrcoeff[f] = res.max()

                print("Found " + str(np.sum(inliers)) + " matches between frame " + str(f) + " and the reference, for a"
                                                        " normalized correlation of " + str(corrcoeff[f]))
            else:
                pass
                #print("Not enough inliers were found: " + str(np.sum(inliers)))
        else:
            pass
            #print("Not enough matches were found: " + str(len(good_matches)))

    dropthresh = np.nanmean(corrcoeff) - np.nanstd(corrcoeff)
    corrcoeff[np.isnan(corrcoeff)] = 0  # Make all nans into zero for easy tracking.

    inliers = corrcoeff >= dropthresh
    corrected_im = corrected_im[..., np.where(inliers)[0]]

    for i in range(len(inliers)):
        if not inliers[i]:
            xform[i] = None # If we drop a frame, eradicate its xform. It's meaningless anyway.

    print("Kept " + str(np.sum(corrcoeff >= dropthresh)) + " frames. (of " + str(num_frames) + ")")
    # save_video(
    #     "\\\\134.48.93.176\\Raw Study Data\\00-64774\\MEAOSLO1\\20210824\\Processed\\Functional Pipeline\\test.avi",
    #     corrected_im, 29.4)

    return corrected_im, xform, inliers


def weighted_z_projection(image_data, weights, projection_axis=2, type="average"):
    num_frames = image_data.shape[-1]

    image_projection = np.nansum(image_data.astype("float64"), axis=projection_axis)
    weight_projection = np.nansum(weights.astype("float64"), axis=projection_axis)
    weight_projection[weight_projection == 0] = np.nan

    image_projection /= weight_projection

    weight_projection[np.isnan(weight_projection)] = 0

    cv2.imshow("projected", image_projection.astype("uint8"))
    c = cv2.waitKey(1000)
    if c == 27:
        return

    return image_projection, (weight_projection / np.amax(weight_projection))
