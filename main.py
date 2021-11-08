import numpy as np
import glob
import func as myf
import skimage.io as si
from skimage import img_as_ubyte
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
import matplotlib.pyplot as plt
import csv
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import AdaBoostClassifier

## IMPORTING IMAGES ##
imagelist = glob.glob('image/*.jpg')  # Creates list of names of JPEG files in specified folder
images = np.array(
    [si.imread(image, as_gray=True) for image in imagelist])  # import images, ensuring grayscale, into a 3D ndarray
for image in images:
    image[image > 0.98] = 0.99

print('images array shape: ', images.shape)
print('No. of images: ', len(imagelist))

## READING LABELS ##
csvfile = open('labelapp.csv')
labelsfile = csv.reader(csvfile, delimiter=',')
imgname = []
imglabel = []
imgpov = []
labelsfile.__next__()
for row in labelsfile:
    imgname.append(row[0])
    imglabel.append(row[1])
    imgpov.append(row[2])
csvfile.close()

## TRIMMING AND SEGMENTING IMAGES ##
trimmedimages = []
for i in range(0, len(imagelist)):
    trimmedimages.append(myf.trimimg(images[i][:][:]))  # Creates a list of the imported images after being cropped
# for j in range(0, len(imagelist)):
#     si.imsave('trimmed/trimmedIMG_{}.jpg'.format(j), img_as_ubyte(trimmedimages[j]))

segimg = []  # A list of images after normalizing to a standard resolution
for k in trimmedimages:
    segimg.append(
        resize(k, np.array([250, 200]), anti_aliasing=True))  # Resizes images after trimming to normalize all sizes
# for l in range(0, len(segimg)):
#     si.imsave('normalized/normIMG_{}.jpg'.format(l), img_as_ubyte(segimg[l]))
print('length of segmented images list: ', len(segimg))

## CREATING ARRAY OF FEATURES FOR IMAGE POV DETECTION ##
selectimgset = 1  # Select which img set to use in model (0 = before trimming/segmenting, 1 = after)
povfeatures = np.zeros(203)
if selectimgset == 0:
    for k in range(0, len(imagelist)):  # Makes an array of features, first column is mean of pic's 6th
        selectedimage = images[k][:][:]  # column values and second column is mean of pic's last row values
        ProportionDark = np.divide(np.count_nonzero(selectedimage[149, :] < np.mean(selectedimage)),
                                   len(selectedimage[149, :]))
        tempmeans = np.append(np.array([np.mean(selectedimage[:, 5]), np.mean(selectedimage[-1, :]), ProportionDark]),
                              np.diag(selectedimage))
        povfeatures = np.vstack([povfeatures, tempmeans])
elif selectimgset == 1:  # Uses first column and bottom row if image is already trimmed
    for k in segimg:
        ProportionDark = np.divide(np.count_nonzero(k[49, :] < np.mean(k)), len(k[49, :]))  # Proportion of dark pixels
        # in row 50
        tempmeans = np.append(np.array([np.mean(k[:, 0]), np.mean(k[-1, :]),ProportionDark]), np.diag(k))
        povfeatures = np.vstack([povfeatures, tempmeans])
povfeatures = np.delete(povfeatures, 0, 0)  # Removes the initialized value at the top of features array

## *****FOR TESTING ONLY***** ##
testlabelpov = imgpov[:len(imagelist)]  # Change according to sample size

xtrain, xtest, ytrain, ytest = train_test_split(povfeatures, testlabelpov, random_state=0)

param_grid = {
    'n_estimators' : range(5,155,5),
    'learning_rate' : [rate/10 for rate in range(2,20,2)]
}
base_model = AdaBoostClassifier(algorithm='SAMME.R',random_state=0)
POVmodel = HalvingRandomSearchCV(base_model, param_grid, cv=5, factor=2, n_jobs=-1,
                               random_state=0, refit=True).fit(xtrain,ytrain)
print('Best parameters: ', POVmodel.best_params_)
print('Best score: ', POVmodel.best_score_)
ypredict = POVmodel.predict(xtest)

print('Accuracy on test set: ' + str(accuracy_score(ytest, ypredict)))
print(classification_report(ytest, ypredict))

# si.imshow(segimg[1])
# si.show()
