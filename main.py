import numpy as np
import glob
import func as myf
import skimage.io as si
from skimage.transform import resize
import sklearn as sl

imagelist = glob.glob('testimport/*.jpg')  # Creates list of names of JPEG files in specified folder
images = np.array(
    [si.imread(image, as_gray=True) for image in imagelist])  # import images, ensuring grayscale, into a 3D ndarray

print('images array shape: ', images.shape)
print('No. of images: ', len(imagelist))

trimmedimages = []
for i in range(0, len(imagelist)):
    trimmedimages.append(myf.trim0rows(images[i][:][:]))  # Creates a list of the imported images after being cropped

segimg = []
for j in trimmedimages:
    segimg.append(resize(j, np.array([150,100]), anti_aliasing=True)) # Resizes images after trimming to normalize all sizes

print('length of segmented images list: ', len(segimg))

si.imshow(segimg[0])
si.show()

