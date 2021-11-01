import cv2
import numpy as np
from numpy.polynomial import Polynomial

from ocvl.function.utility.resources import save_video


def flat_field(dataset, sigma=20):

    kernelsize = 3 * sigma
    if (kernelsize % 2) == 0:
        kernelsize += 1

    flat_fielded_dataset = np.zeros(dataset.shape, dtype=dataset.dtype)

    for i in range(dataset.shape[-1]):

        blurred_frame = cv2.GaussianBlur(dataset[..., i].astype("float64"), (kernelsize, kernelsize),
                                         sigmaX=sigma, sigmaY=sigma)

        flat_fielded = (dataset[..., i].astype("float64") / blurred_frame)



        flat_fielded -= np.amin(flat_fielded)
        flat_fielded /= np.amax(flat_fielded)
        if dataset.dtype == "uint8":
            flat_fielded *= 255
        elif dataset.dtype == "uint16":
            flat_fielded *= 65535



        #cv2.imshow('Frame', flat_fielded.astype(dataset.dtype))

        flat_fielded_dataset[..., i] = flat_fielded

        # Press Q on keyboard to  exit
        #if cv2.waitKey(25) & 0xFF == ord('q'):
        #    return

    return flat_fielded_dataset

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
        row_strip_fit = Polynomial.fit(substrip, col_shifts[f, :], deg=8)
        indiv_colshift[f, :] = row_strip_fit(allrows)
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
        dewarped[..., f] = cv2.remap(image_data[..., f].astype("float64")/255, centered_col_shifts, centered_row_shifts,
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
        return (dewarped*255).astype("uint8"), centered_col_shifts, centered_row_shifts
    else:
        return dewarped, centered_col_shifts, centered_row_shifts

    #save_video("C:\\Users\\rober\\Documents\\temp\\test.avi", (dewarped*255).astype("uint8"), 30)

def remove_data_torsion(image_data, mask_data, framestamps=None, reference_indx=None):
    num_frames = image_data.shape[-1]

    if reference_indx is None:
        reference_indx = 1

    xform = [0] * num_frames
    corrected_im = np.zeros(image_data.shape)
    # number_of_iterations = 5000;
    # # Specify the threshold of the increment
    # # in the correlation coefficient between two iterations
    # termination_eps = 1e-5;
    #
    # # Define termination criteria
    # criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
    #
    #
    #
    # for f in range(num_frames):
    #     xform[f] = np.eye(2, 3, dtype=np.float32)
    #     try:
    #         (cc, xform[f]) = cv2.findTransformECC(image_data[..., f], image_data[..., reference_indx], xform[f],
    #                                               cv2.MOTION_AFFINE, criteria, inputMask=mask_data[..., reference_indx])
    #         corrected_im[..., f] = cv2.warpAffine(image_data[..., f], xform[f], image_data[..., f].shape,
    #                                               flags=cv2.INTER_LANCZOS4)
    #         print(cc)
    #     except cv2.error:
    #         print("Frame " + str(f) + " failed to align.")

    sift = cv2.SIFT_create(1000)

    keypoints = []
    descriptors = []

    for f in range(num_frames):
        kp, des = sift.detectAndCompute(image_data[..., f], mask_data[..., f], None)
        keypoints.append(kp)
        descriptors.append(des)

    #Set up FLANN parameters (feature matching)... review these.
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=1000)

    flan = cv2.FlannBasedMatcher(index_params, search_params)
    #Specify the number of iterations.
    for f in range(num_frames):
        matches = flan.knnMatch(descriptors[f], descriptors[reference_indx], k=2)

        good_matches = []
        for f1, f2 in matches:
            #print( str(f1.distance) +" vs "+ str(f2.distance))
            if f1.distance < 0.8*f2.distance:
                good_matches.append(f1)

        if len(good_matches) > 10:
            print("Found " + str(len(good_matches)) + " matches between frame " + str(f) + " and the reference.")
            src_pts = np.float32([keypoints[f][f1.queryIdx].pt for f1 in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints[reference_indx][f1.trainIdx].pt for f1 in good_matches]).reshape(-1, 1, 2)

            M, inliers = cv2.estimateAffine2D(src_pts, dst_pts)

            h, w = image_data[..., f].shape
            if M is not None:
                xform[f] = M

                corrected_im[..., f] = cv2.warpAffine(image_data[..., f], xform[f], image_data[..., f].shape,
                                                      flags=cv2.INTER_LANCZOS4)


            # Need to add double-check and

            # cv2.imshow("aligned", corrected_im[..., f])
            # c = cv2.waitKey(500)
            # if c == 27:
            #     break
        else:
            pass
            print("Not enough matches were found: " + str(len(good_matches)))

    save_video("\\\\134.48.93.176\\Raw Study Data\\00-64774\\MEAOSLO1\\20210824\\Processed\\Functional Pipeline\\test.avi",
               corrected_im, 29.4)

    return corrected_im, xform

