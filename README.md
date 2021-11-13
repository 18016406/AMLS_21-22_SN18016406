# ELEC0134
Applied Machine Learning Systems 2021/22 Assignment to use machine learning techniques to catagorize MRI scan images and identify tumors

This assignment project aims to use machine learning techniques to classify input MRI brain scans into 2 categories based on the presence/absence of a tumor. If a tumor is present, it is further categorized into 4 subcategories of either pituitary, glioma or meningioma tumor.

Python environment should have the following additional libraries installed along with any prerequisites:
numpy
skimage
sklearn


Folder directory should have:
main.py             -Main script
func.py             -Additional self-defined functions
labelapp.csv        -CSV file containing image names and tumor labels, appended with POV of image
image               -Folder containing all images for testing (images given by the zip files in 3 parts)
normalized          -Empty folder where normalized images will be saved into
edged               -Empty folder where edge detected images will be saved into
testimage	    -Empty folder to put in images not in training set to use model(s) on