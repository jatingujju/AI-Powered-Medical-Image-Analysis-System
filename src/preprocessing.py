import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_data(X, y):

    X = X / 255.0

    encoder = LabelEncoder()

    y = encoder.fit_transform(y)

    return X, y, encoder