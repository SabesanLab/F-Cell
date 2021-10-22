import cv2
import numpy as np


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
def dewarp_2D_data(image_data, row_shifts, col_shifts, method="median"):
    allrows = np.linspace(0, numstrips - 1, num=self.height)  # Make a linspace for all of our images' rows.
    substrip = np.linspace(0, numstrips - 1, num=numstrips)

    indivxshift = np.zeros([self.num_frames, self.height])

    # Fit across rows, in order to capture all strips for a given dataset
    for f in range(self.num_frames):
        row_strip_fit = Polynomial.fit(substrip, xshifts[f, :], deg=8)
        indivxshift[f, :] = row_strip_fit(allrows)

    indivyshift = np.zeros([self.num_frames, self.height])

    # Fit across rows, in order to capture all strips for a given dataset
    for f in range(self.num_frames):
        row_strip_fit = Polynomial.fit(substrip, yshifts[f, :], deg=8)
        indivyshift[f, :] = row_strip_fit(allrows)

    if method == "median":
        centered_col_shifts = -np.median(col_shifts, axis=0)
        centered_row_shifts = -np.median(row_shifts, axis=0)