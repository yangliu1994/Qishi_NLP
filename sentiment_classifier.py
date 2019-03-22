from collections import Counter
from scipy.sparse import dok_matrix
from sklearn.linear_model import LogisticRegression


import pandas as pd
import numpy as np

from tools import load_data, save_prediction

def dumb_featurize(text):
    feats = {}
    words = text.split(" ")

    for word in words:
        if word == "love" or word == "like" or word == "best":
            feats["contains_positive_word"] = 1
        if word == "hate" or word == "dislike" or word == "worst" or word == "awful":
            feats["contains_negative_word"] = 1

    return feats


def better_featurize(text):
    raise NotImplementedError


class SentimentClassifier:

    def __init__(self, feature_method=dumb_featurize, min_feature_ct=1, L2_reg=1.0):
        """
        :param feature_method: featurize function
        :param min_feature_count: int, ignore the features that appear less than this number to avoid overfitting
        """
        self.feature_vocab = {}
        self.feature_method = feature_method
        self.min_feature_ct = min_feature_ct
        self.L2_reg = L2_reg

    def featurize(self, X):
        """
        # Featurize input text

        :param X: list of texts
        :return: list of featurized vectors
        """
        featurized_data = []
        for text in X:
            feats = self.feature_method(text)
            featurized_data.append(feats)
        return featurized_data

    def pipeline(self, X, training=False):
        """
        Data processing pipeline to translate raw data input into sparse vectors
        :param X: featurized input
        :return: 2d sparse vectors

        Implement the pipeline method that translate the dictionary like feature vectors into homogeneous numerical
        vectors, for example:
        [{"fea1": 1, "fea2": 2},
         {"fea2": 2, "fea3": 3}]
         -->
         [[1, 2, 0],
          [0, 2, 3]]

        Hints:
        1. How can you know the length of the feature vector?
        2. When should you use sparse matrix?
        3. Have you treated non-seen features properly?
        4. Should you treat training and testing data differently?
        """
        # Have to build feature_vocab during training

        if training:
            keys = []
            for dct in X:
                keys += dct.keys()
            keys = list(set(keys))
            for i, key in enumerate(keys):
                self.feature_vocab[key] = i

        # Translate raw texts into vectors
        vectors = dok_matrix((len(X), len(self.feature_vocab)), dtype=int)
        for i, dct in enumerate(X):
            for key in dct:
                if key in self.feature_vocab:
                    vectors[i, self.feature_vocab[key]] = dct[key]
        return vectors

    def fit(self, X, y):
        X = self.pipeline(self.featurize(X), training=True)

        D, F = X.shape
        self.model = LogisticRegression(C=self.L2_reg)
        self.model.fit(X, y)

        return self

    def predict(self, X):
        X = self.pipeline(self.featurize(X))
        return self.model.predict(X)

    def score(self, X, y):
        X = self.pipeline(self.featurize(X))
        return self.model.score(X, y)

    # Write learned parameters to file
    def save_weights(self, filename='weights.csv'):
        weights = [["__intercept__", self.model.intercept_[0]]]
        for feat, idx in self.feature_vocab.items():
            weights.append([feat, self.model.coef_[0][idx]])

        weights = pd.DataFrame(weights)
        weights.to_csv(filename, header=False, index=False)

        return weights


"""
Run this to test your model implementation
"""

cls = SentimentClassifier()
X_train = [{"fea1": 1, "fea2": 2}, {"fea2": 2, "fea3": 3}]

X = cls.pipeline(X_train, True)
assert X.shape[0] == 2 and X.shape[1] >= 3, "Fail to vectorize training features"

X_test = [{"fea1": 1, "fea2": 2}, {"fea2": 2, "fea3": 3}]
X = cls.pipeline(X_test)
assert X.shape[0] == 2 and X.shape[1] >= 3, "Fail to vectorize testing features"

X_test = [{"fea1": 1, "fea2": 2}, {"fea2": 2, "fea4": 3}]
try:
    X = cls.pipeline(X_test)
    assert X.shape[0] == 2 and X.shape[1] >= 3
except:
    print("Fail to treat un-seen features")
    raise Exception

print(cls.feature_vocab)
print(X)

print("Success!!")


"""
Run this cell to test your model performance
"""

from sklearn.model_selection import train_test_split

data = load_data("train.txt")
X, y = data.text, data.target
X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.3)
cls = SentimentClassifier(feature_method=dumb_featurize, min_feature_ct=1)
cls = cls.fit(X_train, y_train)
print("Training set accuracy: ", cls.score(X_train, y_train))
print("Dev set accuracy: ", cls.score(X_dev, y_dev))


"""
Run this cell to save weights and the prediction
"""
weights = cls.save_weights()

X_test = load_data("test.txt").text
save_prediction(cls.predict(X_test))
