import numpy as np
import matplotlib.pyplot as plt
import func as myf
import skimage.io as si
import sklearn as sl

testimage = si.imread('image/IMAGE_0000.jpg', as_gray=True)
trimmed = myf.trim0rows(testimage)
print(trimmed[1,:])
si.imshow(trimmed)
si.show()