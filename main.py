import numpy as np
import glob
import func as myf
import csv
import skimage.io as si
from skimage import img_as_ubyte, feature
from skimage.exposure import adjust_gamma
from sklearn.ensemble import AdaBoostClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, HalvingRandomSearchCV
from sklearn.metrics import classification_report, accuracy_score

flagtotest = int(input('Include testing on unseen image set? 0 for no, 1 for yes\n'))
if flagtotest == 1:
    flagtasktotest = int(input('Which task to test?\n 1. Binary classification of presence of tumor\n 2. Multiclass '
                               'classification of presence and type of tumor\n 3. Both\n'))
else:
    flagtasktotest = 0

## IMPORTING IMAGES ##
imagelist = glob.glob('image/*.jpg')  # Creates list of names of JPEG files in specified folder
images = np.array(
    [si.imread(image, as_gray=True) for image in imagelist])  # import images, ensuring grayscale, into a 3D ndarray
for image in images:
    image[image > 0.98] = 0.99
print('images array shape: ', images.shape)
print('Please wait...')
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
segimg = myf.trimandsegment(images)
# print('length of segmented images list: ', len(segimg))
# if flagtotest == 1:
#     # This section is for using program to classify on unseen data images in '/testimage/' folder
#     # Process testing images to trim and normalize size as we do with training set
#     testingimagelist = glob.glob('testimage/*.jpg')
#     testingimages = np.array(
#         [si.imread(image, as_gray=True) for image in testingimagelist])  # Import testing images as grayscale
#     for image in testingimages:
#         image[image > 0.98] = 0.99  # Make sure no pixels are above value 1
#     processedtestimgs = myf.trimandsegment(testingimages)  # Trim and normalize image size
#
# ## TASK 1, BINARY CLASSIFICATION: PRESENCE/ABSENCE OF TUMOR ##
# t1labels = [x != 'no_tumor' for x in imglabel]  # Creates a separate labels list classifying only presence of tumor
# # Creating feature vector and using feature selection#
# t1numoffeaturerows = 11  # Number of rows to use as feature
# t1numoffeaturecolumns = 9  # Number of columns to use as feature
# t1totalfeatures = (t1numoffeaturerows * 200) + (t1numoffeaturecolumns * 250)
# t1features = np.zeros(t1totalfeatures)
# for k in segimg[:len(t1labels)]:
#     t1temp = np.concatenate(
#         [k[10, :], k[30, :], k[40, :], k[50, :], k[80, :], k[90, :], k[100, :], k[120, :], k[130, :], k[140, :],
#          k[150, :],
#          k[:, 60], k[:, 70], k[:, 75], k[:, 80], k[:, 90], k[:, 100], k[:, 120], k[:, 125], k[:, 130]])
#     t1features = np.vstack([t1features, t1temp])
# t1features = np.delete(t1features, 0, 0)  # Removes the initialized value at the top of features array
#
# # print('Shape of features array for Task 1: ', t1features.shape)
#
# # Choose top 3% of features based on chi-squared scoring
# t1selectfeatures = SelectPercentile(chi2, percentile=3)
# t1reducedfeatures = t1selectfeatures.fit_transform(t1features, t1labels)
# print('Number of features used for dataset in Task 1: ', t1reducedfeatures.shape[1])
#
# # Training model#
# t1xtrain, t1xtest, t1ytrain, t1ytest = train_test_split(t1reducedfeatures, t1labels,
#                                                         random_state=0)  # Split features & labels
# t1logreg = LogisticRegression(solver='sag', n_jobs=-1, max_iter=500)
# t1logreg.fit(t1xtrain, t1ytrain)
# t1ypredict = t1logreg.predict(t1xtest)
# print('-------------------------------------------------------')
# print('Binary prediction accuracy: ' + str(accuracy_score(t1ytest, t1ypredict)))
# print(classification_report(t1ytest, t1ypredict))
# print('-------------------------------------------------------')
#
# if flagtasktotest == 1 or flagtasktotest == 3:
#     # Creating feature vector for testing set#
#     t1testfeatures = np.zeros(t1totalfeatures)
#     for k in processedtestimgs:
#         t1testtemp = np.concatenate(
#             [k[10, :], k[30, :], k[40, :], k[50, :], k[80, :], k[90, :], k[100, :], k[120, :], k[130, :], k[140, :],
#              k[150, :],
#              k[:, 60], k[:, 70], k[:, 75], k[:, 80], k[:, 90], k[:, 100], k[:, 120], k[:, 125], k[:, 130]])
#         t1testfeatures = np.vstack([t1testfeatures, t1testtemp])
#     t1testfeatures = np.delete(t1testfeatures, 0, 0)  # Removes the initialized value at the top of features array
#     t1testreducedfeatures = t1selectfeatures.transform(t1testfeatures)  # Select the same features as with training set
#     t1testingpredict = t1logreg.predict(t1testreducedfeatures)  # Use trained model to predict
#     np.savetxt('Binary classification test results.csv', [i for i in zip(testingimagelist, t1testingpredict)],
#                delimiter=',', fmt='%s')

## CREATING ARRAY OF FEATURES FOR IMAGE POV DETECTION ##
povfeatures = myf.MakePOVfeaturesarray(segimg, len(imgpov))
povselectfeatures = SelectPercentile(chi2, percentile=40)  # Choose top 40% features to use in model
reducedpovfeatures = povselectfeatures.fit_transform(povfeatures, imgpov)
print('Number of features for POV classification: ', reducedpovfeatures.shape[1])

## USING ADABOOST CLASSIFIER TO PREDICT IMAGE POV ##
povxtrain, povxtest, povytrain, povytest = train_test_split(reducedpovfeatures, imgpov, random_state=0)

param_grid = {
    'n_estimators': range(5, 155, 5),  # Create a dictionary of parameters to try out for optimization
    'learning_rate': [rate / 10 for rate in range(2, 20, 2)]
}
base_model = AdaBoostClassifier(algorithm='SAMME.R', random_state=0)  # Using AdaBoost Classifier with SAMME.R algorithm
POVmodel = HalvingRandomSearchCV(base_model, param_grid, cv=5, factor=2, n_jobs=-1,
                                 random_state=0, refit=True).fit(povxtrain, povytrain)
print('Best parameters: ', POVmodel.best_params_)
print('Best score: ', POVmodel.best_score_)

povypredict = POVmodel.predict(povxtest)
print('-------------------------------------------------------')
print('Image POV prediction accuracy: ' + str(accuracy_score(povytest, povypredict)))
print(classification_report(povytest, povypredict))
print('-------------------------------------------------------')

## SEPARATE DATASET INTO 3 LISTS ACCORDING TO POV ##
povsortingmask = np.full(len(segimg), False, dtype=bool)  # Create a mask of boolean values to filter and take
segimgarray = np.array(segimg)  # only the values corresponding to required POV
imglabelarray = np.array(imglabel)

for i in range(0, len(imgpov)):
    if imgpov[i] == 't':
        povsortingmask[i] = True
topviewimg = segimgarray[povsortingmask]
topviewlabels = imglabelarray[povsortingmask]
print('Number of top view images: ', len(topviewlabels))

povsortingmask[:] = False
for i in range(0, len(imgpov)):
    if imgpov[i] == 's':
        povsortingmask[i] = True
sideviewimg = segimgarray[povsortingmask]
sideviewlabels = imglabelarray[povsortingmask]
print('Number of side view images: ', len(sideviewlabels))

povsortingmask[:] = False
for i in range(0, len(imgpov)):
    if imgpov[i] == 'b':
        povsortingmask[i] = True
backviewimg = segimgarray[povsortingmask]
backviewlabels = imglabelarray[povsortingmask]
print('Number of back view images: ', len(backviewlabels))

## USING EDGE DETECTION ON ALL TEST IMAGES FOR FEATURE CONTRAST ##
# edgeimg = []
# for i in segimg:
#     edgeimg.append(
#         adjust_gamma(feature.canny(i, sigma=1.5), gamma=1))  # Perform sobel edge detection and allow adjustable gamma
# for j in range(0, len(edgeimg)):
#     si.imsave('edged/EdgeDetectIMG_{}.jpg'.format(j), img_as_ubyte(edgeimg[j]))
