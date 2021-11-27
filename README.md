# ELEC0134
Applied Machine Learning Systems 2021/22 Assignment to use machine learning techniques to catagorize MRI scan images and identify tumors

This assignment project aims to use machine learning techniques to classify input MRI brain scans into 2 categories based on the presence/absence of a tumor as Task number 1. 
For Task number 2, if a tumor is present, it is further categorized into 3 subcategories of either pituitary, glioma or meningioma tumor.

***How to use:***

Image dataset for training models should be located in a folder named *'image'*. The corresponding appended labels CSV file for training is located in the root folder of the program (same folder as main.py).

Run program with main.py

Running the program should first prompt you if you want to test on unseen data. Input 1 if you have MRI scan images in the folder *'testimage'* in the root folder. Input 0 if you don't want to test on unseen data and
only want to see training/testing results on the initial 3000 images dataset.

With task 1, the output of the program is a CSV file in the root directory called *'Binary classification test results.csv'*. Inside the file will be 2 columns; the first column is the testing image names and the second is the classification result. **False** means no tumor is detected, **True** means a tumor is detected

Task 2's output from the program is a CSV file, also in the root directory called *'Multiclass classification test results.csv'* and follows the same format as in task 1 except that the second column is a string that indicates the tumor type rather than a boolean value.

***Python environment should have the following additional libraries installed along with any prerequisites:***

numpy

skimage

sklearn

matplotlib

***Folder root directory should have:***

  main.py             -----Main script

  func.py             -----Additional self-defined functions

  labelapp.csv        -----CSV file containing image names and tumor labels, appended with POV of image

  image               -----Folder containing all images for testing (images given by the zip files in 3 parts)

  normalized          -----Empty folder where normalized images will be saved into

  testimage	          -----Empty folder to put in images not in training set to use model(s) on
