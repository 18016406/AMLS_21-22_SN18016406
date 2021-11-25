import numpy as np
import glob
import func as myf
import csv
import matplotlib.pyplot as plt
import skimage.io as si
from skimage import img_as_ubyte, feature
from skimage.exposure import adjust_gamma
from skimage.transform import hough_ellipse
from sklearn.ensemble import AdaBoostClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split, HalvingRandomSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import minmax_scale

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
print('length of processed images list: ', len(segimg))
if flagtotest == 1:
    # This section is for using program to classify on unseen data images in '/testimage/' folder
    # Process testing images to trim and normalize size as we do with training set
    testingimagelist = glob.glob('testimage/*.jpg')
    testingimages = np.array(
        [si.imread(image, as_gray=True) for image in testingimagelist])  # Import testing images as grayscale
    for image in testingimages:
        image[image > 0.98] = 0.99  # Make sure no pixels are above value 1
    processedtestimgs = myf.trimandsegment(testingimages)  # Trim and normalize image size

## TASK 1, BINARY CLASSIFICATION: PRESENCE/ABSENCE OF TUMOR ##
t1labels = [x != 'no_tumor' for x in imglabel]  # Creates a separate labels list classifying only presence of tumor
# Creating feature vector and using feature selection#
t1numoffeaturerows = 11  # Number of rows to use as feature
t1numoffeaturecolumns = 9  # Number of columns to use as feature
t1totalfeatures = (t1numoffeaturerows * 200) + (t1numoffeaturecolumns * 250)
t1features = np.zeros(t1totalfeatures)
for k in segimg[:len(t1labels)]:
    t1temp = np.concatenate(
        [k[10, :], k[30, :], k[40, :], k[50, :], k[80, :], k[90, :], k[100, :], k[120, :], k[130, :], k[140, :],
         k[150, :],
         k[:, 60], k[:, 70], k[:, 75], k[:, 80], k[:, 90], k[:, 100], k[:, 120], k[:, 125], k[:, 130]])
    t1features = np.vstack([t1features, t1temp])
t1features = np.delete(t1features, 0, 0)  # Removes the initialized value at the top of features array

# print('Shape of features array for Task 1: ', t1features.shape)

# Choose top 3% of features based on chi-squared scoring
t1selectfeatures = SelectPercentile(chi2, percentile=3)

t1reducedfeatures = t1selectfeatures.fit_transform(t1features, t1labels)
print('Number of features used for dataset in Task 1: ', t1reducedfeatures.shape[1])
# Training model#
t1xtrain, t1xtest, t1ytrain, t1ytest = train_test_split(t1reducedfeatures, t1labels,
                                                        random_state=0)  # Split features & labels
t1logreg = LogisticRegression(solver='sag', n_jobs=-1, max_iter=500)
t1logreg.fit(t1xtrain, t1ytrain)
t1ypredict = t1logreg.predict(t1xtest)
print('-------------------------------------------------------')
print('Binary prediction accuracy: ' + str(accuracy_score(t1ytest, t1ypredict)))
print(classification_report(t1ytest, t1ypredict))
# conf = confusion_matrix(t1ytest, t1ypredict, labels=t1logreg.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=t1logreg.classes_)
# disp.plot()
# plt.show()
print('-------------------------------------------------------')

if flagtotest == 1:
    # Creating feature vector for testing set#
    t1testfeatures = np.zeros(t1totalfeatures)
    for k in processedtestimgs:
        t1testtemp = np.concatenate(
            [k[10, :], k[30, :], k[40, :], k[50, :], k[80, :], k[90, :], k[100, :], k[120, :], k[130, :], k[140, :],
             k[150, :],
             k[:, 60], k[:, 70], k[:, 75], k[:, 80], k[:, 90], k[:, 100], k[:, 120], k[:, 125], k[:, 130]])
        t1testfeatures = np.vstack([t1testfeatures, t1testtemp])
    t1testfeatures = np.delete(t1testfeatures, 0, 0)  # Removes the initialized value at the top of features array
    t1testreducedfeatures = t1selectfeatures.transform(t1testfeatures)  # Select the same features as with training set
    t1testingpredict = t1logreg.predict(t1testreducedfeatures)  # Use trained model to predict
    if flagtasktotest == 1 or flagtasktotest == 3:
        np.savetxt('Binary classification test results.csv', [i for i in zip(testingimagelist, t1testingpredict)],
                   delimiter=',', fmt='%s')
        print('**Task 1 prediction saved to file!**')

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

numericalpov = []
for i in imgpov:  # Change char label of POV into a numerical value to use in a numpy array
    if i == 't':
        numericalpov.append(0)
    elif i == 's':
        numericalpov.append(0.5)
    else:
        numericalpov.append(1)

# Features to use consists of every single pixel value of the image, a numerical value representing POV of the image
# and a boolean value of whether the image has a tumor or not. This boolean value will be obtained from task 1
t2bigfeatures = np.append(np.append(segimg[0].flatten(), numericalpov[0]), t1labels[0])
for i in range(1, len(numericalpov)):
    t2bigfeatures = np.vstack([t2bigfeatures, np.append(np.append(segimg[i].flatten(), numericalpov[i]), t1labels[i])])
t2selectfeatures = SelectPercentile(chi2, percentile=3)  # Select only the top 5% performing features
t2reducedfeatures = t2selectfeatures.fit_transform(t2bigfeatures, imglabel)
print('Number of features for Task 2 classification: ', t2reducedfeatures.shape[1])

## USING ADABOOST CLASSIFIER TO PREDICT TYPE OF TUMOR ##
t2xtrain, t2xtest, t2ytrain, t2ytest = train_test_split(t2reducedfeatures, imglabel, random_state=0)

t2_base_model = AdaBoostClassifier(algorithm='SAMME.R',
                                   random_state=0)  # Using AdaBoost Classifier with SAMME.R algorithm
t2model = t2_base_model.fit(t2xtrain, t2ytrain)
t2ypredict = t2model.predict(t2xtest)
print('-------------------------------------------------------')
print('Task 2 prediction accuracy: ' + str(accuracy_score(t2ytest, t2ypredict)))
print(classification_report(t2ytest, t2ypredict))
conf = confusion_matrix(t2ytest, t2ypredict, labels=t2model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=t2model.classes_)
disp.plot()
plt.show()
print('-------------------------------------------------------')

if flagtasktotest == 2 or flagtasktotest == 3:
    t2testpovfeatures = myf.MakePOVfeaturesarray(processedtestimgs, len(processedtestimgs))
    t2testreducedpovfeatures = povselectfeatures.transform(t2testpovfeatures)
    t2testpovpredict = POVmodel.predict(t2testreducedpovfeatures)
    t2testnumericalpov = []
    for i in t2testpovpredict:  # Change char label of POV into a numerical value to use in a numpy array
        if i == 't':
            t2testnumericalpov.append(0)
        elif i == 's':
            t2testnumericalpov.append(0.5)
        else:
            t2testnumericalpov.append(1)
    t2testbigfeatures = np.append(np.append(processedtestimgs[0].flatten(), t2testnumericalpov[0]), t1testingpredict[0])
    for i in range(1, len(processedtestimgs)):
        t2testbigfeatures = np.vstack(
            [t2testbigfeatures,
             np.append(np.append(processedtestimgs[i].flatten(), t2testnumericalpov[i]), t1testingpredict[i])])
    t2testfeatures = t2selectfeatures.transform(t2testbigfeatures)
    t2testresults = t2model.predict(t2testfeatures)
    np.savetxt('Multiclass classification test results.csv', [i for i in zip(testingimagelist, t2testresults)],
               delimiter=',', fmt='%s')
    print('**Task 2 prediction saved to file!**')