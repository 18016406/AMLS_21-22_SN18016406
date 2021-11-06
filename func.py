import numpy as np
from sklearn.linear_model import LogisticRegression


def trimimg(image):
    bottomthresh = np.mean(image) + 0.1
    topthresh = np.max(image) * 0.3  # Define a top and bottom threshold to remove rows/columns
    previmg = image
    while True:             #Perform multiple passes of trimming each side sequentially until there are no changes
        i = k = 0
        j = len(previmg[:, 0])-1
        l = len(previmg[0, :])-1

        while (previmg[i, :] < bottomthresh).all() or (previmg[i, :] > topthresh).all():
            i += 1  # If all pixels in row i (starting from top) is less than lower threshold
            # or more than upper threshold (indicating white border) then stop iterating, remembering the index
        while (previmg[j, :] < bottomthresh).all() or (previmg[j, :] > topthresh).all():
            j -= 1  # Same as above but starting from bottom row going up
        while (previmg[:, k] < bottomthresh).all() or (previmg[:, k] > topthresh).all():
            k += 1  # Same check but for column instead of row going from left to right
        while (previmg[:, l] < bottomthresh).all() or (previmg[:, l] > topthresh).all():
            l -= 1  # Same check but going right to left
        trimmed = previmg[i:j+1, k:l+1]
        if previmg.shape == trimmed.shape:
            break
        previmg = trimmed
    return trimmed


def LogisticRegressionPredict(x_train, y_train, xtest):
    # Build Logistic Regression Model
    logreg = LogisticRegression(solver='lbfgs')
    # Train the model using the training sets
    logreg.fit(x_train, y_train)
    y_pred = logreg.predict(xtest)
    return y_pred
