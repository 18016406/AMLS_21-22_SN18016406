import numpy as np
from sklearn.linear_model import LogisticRegression


def trim0rows(image):
    vert = image[~(image < 0.2).all(1)]  # Use with np.array type
    trimmed = vert[:, ~(vert < 0.2).all(0)]
    return trimmed


def LogisticRegressionPredict(x_train, y_train, xtest):
    # Build Logistic Regression Model
    logreg = LogisticRegression(solver='lbfgs')
    # Train the model using the training sets
    logreg.fit(x_train, y_train)
    y_pred = logreg.predict(xtest)
    return y_pred
