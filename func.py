import numpy as np

def trim0rows(image):
    vert = image[~(image<0.2).all(1)]     #Use with np.array type
    trimmed = vert[:,~(vert<0.2).all(0)]
    return trimmed