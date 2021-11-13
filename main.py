import numpy as np
import glob
import func as myf
import csv
import skimage.io as si
from skimage import img_as_ubyte, filters, feature
from skimage.exposure import adjust_gamma
from sklearn.ensemble import AdaBoostClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, HalvingRandomSearchCV
from sklearn.metrics import classification_report, accuracy_score

t1flagtotest = int(input('Include testing on unseen image set? 0 for no, 1 for yes\n'))

## IMPORTING IMAGES ##
imagelist = glob.glob('image/*.jpg')  # Creates list of names of JPEG files in specified folder
images = np.array(
    [si.imread(image, as_gray=True) for image in imagelist])  # import images, ensuring grayscale, into a 3D ndarray
for image in images:
    image[image > 0.98] = 0.99
print('images array shape: ', images.shape)

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

## TASK 1, BINARY CLASSIFICATION: PRESENCE/ABSENCE OF TUMOR ##
t1labels = [x != 'no_tumor' for x in imglabel]      #Creates a separate labels list classifying only presence of tumor
#Creating feature vector and using feature selection#
t1numoffeaturerows = 11         #Number of rows to use as feature
t1numoffeaturecolumns = 9       #Number of columns to use as feature
t1totalfeatures = (t1numoffeaturerows*200) + (t1numoffeaturecolumns*250)
t1features = np.zeros(t1totalfeatures)
for k in segimg[:len(t1labels)]:
    t1temp = np.concatenate([k[10,:],k[30,:],k[40,:],k[50,:],k[80,:],k[90,:],k[100,:],k[120,:],k[130,:],k[140,:],k[150,:]\
                     ,k[:,60],k[:,70],k[:,75],k[:,80],k[:,90],k[:,100],k[:,120],k[:,125],k[:,130]])
    t1features = np.vstack([t1features, t1temp])
t1features = np.delete(t1features, 0, 0)  # Removes the initialized value at the top of features array

# print('Shape of features array for Task 1: ', t1features.shape)

t1selectfeatures = SelectPercentile(chi2,percentile=3)
t1reducedfeatures=t1selectfeatures.fit_transform(t1features,t1labels)  #Chooses top 3% of features
                                                                        #based on chi-squared scoring
# print('New reduced features array shape for Task 1: ', t1reducedfeatures.shape)

#Training model#
t1xtrain, t1xtest, t1ytrain, t1ytest = train_test_split(t1reducedfeatures, t1labels, random_state=0)  # Split features & labels
t1logreg = LogisticRegression(solver='sag', n_jobs=-1, max_iter=500)
t1logreg.fit(t1xtrain, t1ytrain)
t1ypredict = t1logreg.predict(t1xtest)
print('Binary prediction accuracy: ' + str(accuracy_score(t1ytest, t1ypredict)))
# print(classification_report(t1ytest, t1ypredict))

if t1flagtotest == 1:       #This section is for using to classify on unseen data images in '/testimage/' folder
                            #Will save results as a CSV file in root folder (same directory as main.py)
    #Importing images not used in training#
    t1testingimagelist = glob.glob('testimage/*.jpg')
    t1testingimages = np.array(
        [si.imread(image, as_gray=True) for image in t1testingimagelist])  #Import testing images as grayscale
    for image in t1testingimages:
        image[image > 0.98] = 0.99                                         #Make sure no pixels are above value 1
    t1processedimgs = myf.trimandsegment(t1testingimages)                  #Trim and normalize image size
    #Creating feature vector for testing set#
    t1testfeatures = np.zeros(t1totalfeatures)
    for k in t1processedimgs:
        t1testtemp = np.concatenate([k[10,:],k[30,:],k[40,:],k[50,:],k[80,:],k[90,:],k[100,:],k[120,:],k[130,:],k[140,:],k[150,:]\
                         ,k[:,60],k[:,70],k[:,75],k[:,80],k[:,90],k[:,100],k[:,120],k[:,125],k[:,130]])
        t1testfeatures = np.vstack([t1testfeatures, t1testtemp])
    t1testfeatures = np.delete(t1testfeatures, 0, 0)  # Removes the initialized value at the top of features array
    t1testreducedfeatures = t1selectfeatures.transform(t1testfeatures)  #Select the same features as with training set
    t1testingpredict = t1logreg.predict(t1testreducedfeatures)          #Use trained model to predict
    np.savetxt('Binary classification test results.csv', [i for i in zip(t1testingimagelist, t1testingpredict)],
               delimiter=',', fmt='%s')

# ## CREATING ARRAY OF FEATURES FOR IMAGE POV DETECTION ##
# povfeatures = myf.MakePOVfeaturesarray(segimg,len(imgpov))
#
# ## USING ADABOOST CLASSIFIER TO PREDICT IMAGE POV ##
# # POVLabel = imgpov[:len(imagelist)]  # For if we are not importing all images listed in CSV file
# xtrain, xtest, ytrain, ytest = train_test_split(povfeatures, imgpov, random_state=0)  # Split features & labels
# param_grid = {
#     'n_estimators': range(5, 155, 5),  # Create a dictionary of parameters to try out for optimization
#     'learning_rate': [rate / 10 for rate in range(2, 20, 2)]
# }
# base_model = AdaBoostClassifier(algorithm='SAMME.R', random_state=0)  # Using AdaBoost Classifier with SAMME.R algorithm
# POVmodel = HalvingRandomSearchCV(base_model, param_grid, cv=5, factor=2, n_jobs=-1,
#                                  random_state=0, refit=True).fit(xtrain, ytrain)
# print('Best parameters: ', POVmodel.best_params_)
# print('Best score: ', POVmodel.best_score_)
# ypredict = POVmodel.predict(xtest)
# print('Image POV prediction accuracy: ' + str(accuracy_score(ytest, ypredict)))
# print(classification_report(ytest, ypredict))
#
# if len(segimg) > len(imgpov):                       # If there are images that do not have labels (i.e. new test images)
#     numofextraimgs = len(segimg) - len(imgpov)      # Use model to predict labels and add them into the label list
#     extraimgs = segimg[-numofextraimgs:]
#     newimgfeatures = myf.MakePOVfeaturesarray(extraimgs,numofextraimgs)
#     newimgPOV = POVmodel.predict(newimgfeatures)
#     imgpov.append(newimgPOV)
#
# ## SEPARATE DATASET INTO 3 LISTS ACCORDING TO POV ##
# povsortingmask = np.full(len(segimg), False, dtype=bool)    # Create a mask of boolean values to filter and take
# segimgarray = np.array(segimg)                              # only the values corresponding to required POV
# imglabelarray = np.array(imglabel)
#
# for i in range(0, len(imgpov)):
#     if imgpov[i] == 't':
#         povsortingmask[i] = True
# topviewimg = segimgarray[povsortingmask]
# topviewlabels = imglabelarray[povsortingmask]
# print('Number of top view images: ', len(topviewlabels))
#
# povsortingmask[:] = False
# for i in range(0, len(imgpov)):
#     if imgpov[i] == 's':
#         povsortingmask[i] = True
# sideviewimg = segimgarray[povsortingmask]
# sideviewlabels = imglabelarray[povsortingmask]
# print('Number of side view images: ', len(sideviewlabels))
#
# povsortingmask[:] = False
# for i in range(0, len(imgpov)):
#     if imgpov[i] == 'b':
#         povsortingmask[i] = True
# backviewimg = segimgarray[povsortingmask]
# backviewlabels = imglabelarray[povsortingmask]
# print('Number of back view images: ', len(backviewlabels))
#
# ## USING EDGE DETECTION ON ALL TEST IMAGES FOR FEATURE CONTRAST ##
# edgeimg = []
# for i in segimg:
#     edgeimg.append(adjust_gamma(feature.canny(i,sigma=1.5), gamma=1)) # Perform sobel edge detection and allow adjustable gamma
# for j in range(0, len(edgeimg)):
#     si.imsave('edged/EdgeDetectIMG_{}.jpg'.format(j), img_as_ubyte(edgeimg[j]))