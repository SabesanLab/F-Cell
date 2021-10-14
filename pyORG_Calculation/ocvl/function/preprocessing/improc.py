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

