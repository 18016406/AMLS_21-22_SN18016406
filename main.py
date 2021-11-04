import numpy as np
import glob
import func as myf
import skimage.io as si
import csv
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

## IMPORTING IMAGES ##
imagelist = glob.glob('testimport/*.jpg')  # Creates list of names of JPEG files in specified folder
images = np.array(
    [si.imread(image, as_gray=True) for image in imagelist])  # import images, ensuring grayscale, into a 3D ndarray

print('images array shape: ', images.shape)
print('No. of images: ', len(imagelist))

## TRIMMING AND SEGMENTING IMAGES ##
trimmedimages = []
for i in range(0, len(imagelist)):
    trimmedimages.append(myf.trim0rows(images[i][:][:]))  # Creates a list of the imported images after being cropped

segimg = []
for j in trimmedimages:
    segimg.append(
        resize(j, np.array([150, 100]), anti_aliasing=True))  # Resizes images after trimming to normalize all sizes

print('length of segmented images list: ', len(segimg))

## CREATING ARRAY OF FEATURES FOR IMAGE POV DETECTION ##
povfeatures = np.array([0, 0])
for k in segimg:
    tempmeans = np.array([np.mean(k[:, 0]), np.mean(k[-1, :])])
    povfeatures = np.vstack([povfeatures, tempmeans])  # Makes an array of features, first column is mean of pic's left
                                                # column values and second column is mean of pic's last row values
povfeatures = np.delete(povfeatures, 0, 0)  # Removes the initialized value [0,0] at the top of features array

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

## *****FOR TESTING ONLY***** ##
testlabelpov = imgpov[:1000]

xtrain, xtest, ytrain, ytest = train_test_split(povfeatures, testlabelpov)
ypredict = myf.LogisticRegressionPredict(xtrain,ytrain,xtest)
print('Accuracy on test set: '+str(accuracy_score(ytest,ypredict)))
print(classification_report(ytest,ypredict))
print(confusion_matrix(ytest,ypredict))

# si.imshow(segimg[1])
# si.show()
