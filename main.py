import numpy as np
import glob
import func as myf
import skimage.io as si
from skimage import img_as_ubyte
import csv
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier

## IMPORTING IMAGES ##
imagelist = glob.glob('testimport/*.jpg')  # Creates list of names of JPEG files in specified folder
images = np.array(
    [si.imread(image, as_gray=True) for image in imagelist])  # import images, ensuring grayscale, into a 3D ndarray

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
for j in range(0, len(imagelist)):
    si.imsave('trimmed/trimmedIMG_{}.jpg'.format(j), img_as_ubyte(trimmedimages[j]))

segimg = []
for k in trimmedimages:
    segimg.append(
        resize(k, np.array([250, 200]), anti_aliasing=True))  # Resizes images after trimming to normalize all sizes
for l in range(0, len(segimg)):
    si.imsave('normalized/normIMG_{}.jpg'.format(l), img_as_ubyte(segimg[l]))
print('length of segmented images list: ', len(segimg))

## CREATING ARRAY OF FEATURES FOR IMAGE POV DETECTION ##
selectimgset = 1        #Select which img set to use in model (0 = before trimming/segmenting, 1 = after)
povfeatures = np.array([0, 0])
if selectimgset == 0:
    for k in range(0, len(imagelist)):
        selectedimage = images[k][:][:]
        tempmeans = np.array([np.mean(selectedimage[:,5]), np.mean(selectedimage[-1, :])])
        povfeatures = np.vstack([povfeatures, tempmeans])  # Makes an array of features, first column is mean of pic's 6th
                                                        # column values and second column is mean of pic's last row values
elif selectimgset == 1:
    for k in segimg:
        tempmeans = np.array([np.mean(k[:, 0]), np.mean(k[-1, :])])
        povfeatures = np.vstack([povfeatures, tempmeans])  # Makes an array of features, first column is mean of pic's left
                                                        # column values and second column is mean of pic's last row values
povfeatures = np.delete(povfeatures, 0, 0)  # Removes the initialized value [0,0] at the top of features array


## *****FOR TESTING ONLY***** ##
testlabelpov = imgpov[:len(imagelist)]                #Change according to sample size

xtrain, xtest, ytrain, ytest = train_test_split(povfeatures, testlabelpov)
model = AdaBoostClassifier(n_estimators=100)
model.fit(xtrain,ytrain)
ypredict=model.predict(xtest)

# ypredict = myf.LogisticRegressionPredict(xtrain,ytrain,xtest)
print('Accuracy on test set: '+str(accuracy_score(ytest,ypredict)))
print(classification_report(ytest,ypredict))

# si.imshow(segimg[1])
# si.show()
