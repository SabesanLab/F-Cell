import itertools

import cv2
import numpy as np
from skimage.registration import phase_cross_correlation
from numpy.polynomial import Polynomial

from ocvl.function.utility.resources import save_video


def flat_field_frame(dataframe, sigma):
    kernelsize = 3 * sigma
    if (kernelsize % 2) == 0:
        kernelsize += 1

    mask = np.ones(dataframe.shape, dtype=dataframe.dtype)
    mask[dataframe == 0] = 0

    dataframe[dataframe == 0] = 1
    blurred_frame = cv2.GaussianBlur(dataframe.astype("float64"), (kernelsize, kernelsize),
                                     sigmaX=sigma, sigmaY=sigma)
    flat_fielded = (dataframe.astype("float64") / blurred_frame)

    flat_fielded *= mask
    flat_fielded -= np.amin(flat_fielded)
    flat_fielded = np.divide(flat_fielded, np.amax(flat_fielded), where=flat_fielded != 0)
    if dataframe.dtype == "uint8":
        flat_fielded *= 255
    elif dataframe.dtype == "uint16":
        flat_fielded *= 65535

    return flat_fielded.astype(dataframe.dtype)

def flat_field(dataset, sigma=20):

    if len(dataset.shape) > 2:
        flat_fielded_dataset = np.zeros(dataset.shape, dtype=dataset.dtype)
        for i in range(dataset.shape[-1]):
            flat_fielded_dataset[..., i] = flat_field_frame(dataset[..., i], sigma)
            return flat_fielded_dataset.astype(dataset.dtype)
    else:
        return flat_field_frame(dataset, sigma)





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
        finite = np.isfinite(col_shifts[f, :])
        col_strip_fit = Polynomial.fit(substrip[finite], col_shifts[f, finite], deg=8)
        indiv_colshift[f, :] = col_strip_fit(allrows)
        # Fit across rows, in order to capture all strips for a given dataset
        finite = np.isfinite(row_shifts[f, :])
        row_strip_fit = Polynomial.fit(substrip[finite], row_shifts[f, finite], deg=8)
        indiv_rowshift[f, :] = row_strip_fit(allrows)

    if method == "median":
        centered_col_shifts = -np.nanmedian(indiv_colshift, axis=0)
        centered_row_shifts = -np.nanmedian(indiv_rowshift, axis=0)

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


def im_dist_to_stk(ref_idx, im_stack, mask_stack):
    num_frames = im_stack.shape[-1]
    dists = [10000] * num_frames
    print("Aligning to frame "+str(ref_idx))

    for f2 in range(num_frames):
        dists[f2] = phase_cross_correlation(im_stack[..., ref_idx], im_stack[..., f2],
                                            reference_mask=mask_stack[..., ref_idx], moving_mask=mask_stack[..., f2])

    median_dist = np.nanmedian(dists, axis=0)
    return np.sqrt(median_dist[0] * median_dist[0] + median_dist[1] * median_dist[1])


def relativize_image_stack(image_data, mask_data, reference_idx=0, numkeypoints=5000, method="affine", dropthresh=None):
    num_frames = image_data.shape[-1]

    xform = [None] * num_frames
    corrcoeff = np.empty((num_frames, 1))
    corrcoeff[:] = np.NAN
    corrected_stk = np.zeros(image_data.shape)

    sift = cv2.SIFT_create(numkeypoints, nOctaveLayers=105)

    keypoints = []
    descriptors = []

    for f in range(num_frames):
        kp, des = sift.detectAndCompute(flat_field_frame(image_data[..., f], 20), mask_data[..., f], None)
        if numkeypoints > 8000:
            print("Found "+ str(len(kp)) + " keypoints")
        keypoints.append(kp)
        descriptors.append(des)

    # Set up FLANN parameters (feature matching)... review these.
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=64)

    flan = cv2.FlannBasedMatcher(index_params, search_params)

    # Specify the number of iterations.
    for f in range(num_frames):
        matches = flan.knnMatch(descriptors[f], descriptors[reference_idx], k=2)

        good_matches = []
        for f1, f2 in matches:
            if f1.distance < 0.7 * f2.distance:
                good_matches.append(f1)

        if len(good_matches) >= 5:
            src_pts = np.float32([keypoints[f][f1.queryIdx].pt for f1 in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints[reference_idx][f1.trainIdx].pt for f1 in good_matches]).reshape(-1, 1, 2)

            img_matches = np.empty((max(image_data[..., f].shape[0], image_data[..., f].shape[0]), image_data[..., f].shape[1] + image_data[..., f].shape[1], 3),
                                   dtype=np.uint8)
            cv2.drawMatches( image_data[..., f], keypoints[f], image_data[..., reference_idx], keypoints[reference_idx], good_matches, img_matches)
            cv2.imshow("meh", img_matches)
            cv2.waitKey()

            if method == "affine":
                M, inliers = cv2.estimateAffine2D(dst_pts, src_pts) # More stable- also means we have to set the inverse flag below.
            else:
                M, inliers = cv2.estimateAffinePartial2D(dst_pts, src_pts)

            if M is not None and np.sum(inliers) >= 5:
                xform[f] = M

                corrected_stk[..., f] = cv2.warpAffine(image_data[..., f], xform[f], image_data[..., f].shape,
                                                      flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP)
                warped_mask = cv2.warpAffine(mask_data[..., f], xform[f], mask_data[..., f].shape,
                                             flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)

                # Calculate and store the final correlation. It should be decent, if the transform was.
                res = cv2.matchTemplate(image_data[..., reference_idx], corrected_stk[..., f].astype("uint8"),
                                    cv2.TM_CCOEFF_NORMED, mask=warped_mask)

                corrcoeff[f] = res.max()

                # print("Found " + str(np.sum(inliers)) + " matches between frame " + str(f) + " and the reference, for a"
                #                                         " normalized correlation of " + str(corrcoeff[f]))
            else:
                pass
                #print("Not enough inliers were found: " + str(np.sum(inliers)))
        else:
            pass
            #print("Not enough matches were found: " + str(len(good_matches)))

    if not dropthresh:
        print("No drop threshold detected, auto-generating...")
        dropthresh = np.nanquantile(corrcoeff, 0.01)


    corrcoeff[np.isnan(corrcoeff)] = 0  # Make all nans into zero for easy tracking.

    inliers = np.squeeze(corrcoeff >= dropthresh)
    corrected_stk = corrected_stk[..., inliers]
    # save_video(
    #     "\\\\134.48.93.176\\Raw Study Data\\00-64774\\MEAOSLO1\\20210824\\Processed\\Functional Pipeline\\test_corrected_stk.avi",
    #     corrected_stk, 29.4)
    for i in range(len(inliers)):
        if not inliers[i]:
            xform[i] = None # If we drop a frame, eradicate its xform. It's meaningless anyway.

    print("Using a threshold of "+ str(dropthresh) +", we kept " + str(np.sum(corrcoeff >= dropthresh)) + " frames. (of " + str(num_frames) + ")")


    return corrected_stk, xform, inliers


def weighted_z_projection(image_data, weights, projection_axis=-1, type="average"):
    num_frames = image_data.shape[-1]

    image_projection = np.nansum(image_data.astype("float64"), axis=projection_axis)
    weight_projection = np.nansum(weights.astype("float64"), axis=projection_axis)
    weight_projection[weight_projection == 0] = np.nan

    image_projection /= weight_projection

    weight_projection[np.isnan(weight_projection)] = 0

    #cv2.imshow("projected", image_projection.astype("uint8"))
    #c = cv2.waitKey(1000)
    # if c == 27:
    #     return

    return image_projection, (weight_projection / np.amax(weight_projection))
