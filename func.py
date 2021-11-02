import numpy as np

def trim0rows(image):
    trimmed = image[~(image==0).all(1)]     #Use with np.array type
    return trimmed