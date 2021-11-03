import numpy as np
import matplotlib.pyplot as plt
import glob
import func as myf
import skimage.io as si
import sklearn as sl

imagelist = glob.glob('testimport/*.jpg')
images = np.array([si.imread(image, as_gray=True) for image in imagelist])

# si.imshow(trimmed)
# si.show()