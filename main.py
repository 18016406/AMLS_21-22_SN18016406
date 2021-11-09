import numpy as np
import glob
import func as myf
import skimage.io as si
import csv
import matplotlib.pyplot as plt
from skimage import img_as_ubyte, filters
from skimage.transform import resize
from skimage.exposure import adjust_gamma
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, HalvingRandomSearchCV
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
povfeatures = myf.MakePOVfeaturesarray(segimg,len(imgpov))

## USING ADABOOST CLASSIFIER TO PREDICT IMAGE POV ##
# POVLabel = imgpov[:len(imagelist)]  # For if we are not importing all images listed in CSV file
xtrain, xtest, ytrain, ytest = train_test_split(povfeatures, imgpov, random_state=0)  # Split features & labels
param_grid = {
    'n_estimators': range(5, 155, 5),  # Create a dictionary of parameters to try out for optimization
    'learning_rate': [rate / 10 for rate in range(2, 20, 2)]
}
base_model = AdaBoostClassifier(algorithm='SAMME.R', random_state=0)  # Using AdaBoost Classifier with SAMME.R algorithm
POVmodel = HalvingRandomSearchCV(base_model, param_grid, cv=5, factor=2, n_jobs=-1,
                                 random_state=0, refit=True).fit(xtrain, ytrain)
print('Best parameters: ', POVmodel.best_params_)
print('Best score: ', POVmodel.best_score_)
ypredict = POVmodel.predict(xtest)
print('Image POV prediction accuracy: ' + str(accuracy_score(ytest, ypredict)))
print(classification_report(ytest, ypredict))

if len(segimg) > len(imgpov):                       # If there are images that do not have labels (i.e. new test images)
    numofextraimgs = len(segimg) - len(imgpov)      # Use model to predict labels and add them into the label list
    extraimgs = segimg[-numofextraimgs:]
    newimgfeatures = myf.MakePOVfeaturesarray(extraimgs,numofextraimgs)
    newimgPOV = POVmodel.predict(newimgfeatures)
    imgpov.append(newimgPOV)

## USING EDGE DETECTION ON ALL TEST IMAGES FOR FEATURE CONTRAST ##
edgeimg = []
for i in segimg:
    edgeimg.append(adjust_gamma(filters.sobel(i), gamma=1)) # Perform sobel edge detection and allow adjustable gamma
# for j in range(0, len(edgeimg)):
#     si.imsave('edged/EdgeDetectIMG_{}.jpg'.format(j), img_as_ubyte(edgeimg[j]))