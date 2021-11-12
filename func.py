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

def MakePOVfeaturesarray(segimg, numofsamples):
    povfeatures = np.zeros((len(np.diag(segimg[0]))+3))
    for k in segimg[:numofsamples]:
        ProportionDark50 = np.divide(np.count_nonzero(k[49, :] < np.mean(k)),
                                     len(k[49, :]))  # Proportion of dark pixels in row 50
        features = np.append(np.array([np.mean(k[:, 0]), np.mean(k[-1, :]), ProportionDark50]), np.diag(k))
        povfeatures = np.vstack([povfeatures, features])
    povfeatures = np.delete(povfeatures, 0, 0)  # Removes the initialized value at the top of features array
    return povfeatures

def LogisticRegressionPredict(x_train, y_train, xtest):
    # Build Logistic Regression Model
    logreg = LogisticRegression(solver='sag',n_jobs=-1,max_iter=500)
    # Train the model using the training sets
    logreg.fit(x_train, y_train)
    y_pred = logreg.predict(xtest)
    return y_pred
