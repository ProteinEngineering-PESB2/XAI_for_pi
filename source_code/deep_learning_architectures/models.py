from deep_learning_architectures.demo_architecture import demo_architecture

import numpy as np
from sklearn.metrics import (
    precision_score, accuracy_score, recall_score,
    f1_score, matthews_corrcoef, mean_squared_error,
    mean_absolute_error, r2_score, roc_auc_score, confusion_matrix
)
from scipy.stats import (kendalltau, pearsonr, spearmanr)
from keras.utils.layer_utils import count_params

class Models:
    
    """Organize CNN objects, train and validation process"""
    def __init__(self, x_train, y_train, x_test, y_test, labels, mode, arquitecture):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.labels = labels
        self.mode = mode
        self.arquitecture = arquitecture

        self.cnn = demo_architecture(x_train=self.x_train, labels = self.labels, mode=self.mode)
        
    def __reshape(self):
        dim = self.x_train.shape[1]
        sq_dim = sqrt(dim)
        square_side = ceil(sq_dim)
        resized_x_train = np.resize(self.x_train, (self.x_train.shape[0], square_side*square_side))
        resized_x_test = np.resize(self.x_test, (self.x_test.shape[0], square_side*square_side))
        squared_x_train = np.reshape(resized_x_train, (-1, square_side, square_side))
        squared_x_test = np.reshape(resized_x_test, (-1, square_side, square_side))
        return squared_x_train, squared_x_test

    def fit_models(self, epochs, verbose):
        """Fit model"""
        self.cnn.fit(self.x_train, self.y_train, epochs = epochs, verbose = verbose)

    def save_model(self, folder, prefix = ""):
        """
        Save model in .h5 format, in 'folder' location
        """
        self.cnn.save(f"{folder}/{prefix}-{self.arquitecture}-{self.mode}.h5")

    def get_metrics(self):
        
        """
        Returns classification performance metrics.
        Accuracy, recall, precision, f1_score, mcc.
        """
        trainable_count = count_params(self.cnn.trainable_weights)
        non_trainable_count = count_params(self.cnn.non_trainable_weights)
        result = {}
        result["arquitecture"] = self.arquitecture
        result["trainable_params"] = trainable_count
        result["non_trainable_params"] = non_trainable_count
        if self.mode == "binary":
            y_train_predicted = np.round_(self.cnn.predict(self.x_train))
            y_test_score = self.cnn.predict(self.x_test)
            y_test_predicted = np.round_(y_test_score)
        if self.mode == "classification":
            y_train_predicted = np.argmax(self.cnn.predict(self.x_train), axis = 1)
            y_test_score = self.cnn.predict(self.x_test)
            y_test_predicted = np.argmax(y_test_score, axis = 1)
        if self.mode == "regression":
            y_train_predicted = self.cnn.predict(self.x_train)
            y_test_predicted = self.cnn.predict(self.x_test)
        if self.mode in ("binary", "classification"):
            result["labels"] = self.labels.tolist()
            train_metrics = {
                "accuracy": accuracy_score(y_true = self.y_train, y_pred = y_train_predicted),
                "recall": recall_score(
                    y_true = self.y_train, y_pred = y_train_predicted, average="micro"),
                "precision": precision_score(
                    y_true = self.y_train, y_pred = y_train_predicted, average="micro"),
                "f1_score": f1_score(
                    y_true = self.y_train, y_pred = y_train_predicted, average="micro"),
                "mcc": matthews_corrcoef(y_true = self.y_train, y_pred = y_train_predicted),
                "confusion_matrix": confusion_matrix(
                    y_true = self.y_train, y_pred = y_train_predicted).tolist()
            }
            test_metrics = {
                "accuracy": accuracy_score(y_true = self.y_test, y_pred = y_test_predicted),
                "recall": recall_score(
                    y_true = self.y_test, y_pred = y_test_predicted, average="micro"),
                "precision": precision_score(
                    y_true = self.y_test, y_pred = y_test_predicted, average="micro"),
                "f1_score": f1_score(
                    y_true = self.y_test, y_pred = y_test_predicted, average="micro"),
                "mcc": matthews_corrcoef(y_true = self.y_test, y_pred = y_test_predicted),
                "confusion_matrix": confusion_matrix(
                    y_true = self.y_test, y_pred = y_test_predicted).tolist()
            }
            if self.mode == "binary":
                test_metrics["roc_auc_score"] = roc_auc_score(
                    y_true = self.y_test, y_score = y_test_score, average="micro")
            else:
                test_metrics["roc_auc_score"] = roc_auc_score(
                    y_true = self.y_test, y_score = y_test_score, multi_class = 'ovr')
        else:
            train_metrics = {
                "mse": mean_squared_error(y_true = self.y_train, y_pred = y_train_predicted),
                "mae": mean_absolute_error(y_true = self.y_train, y_pred = y_train_predicted),
                "r2_score": r2_score(y_true = self.y_train, y_pred = y_train_predicted),
                "kendalltau": kendalltau(self.y_train, y_train_predicted),
                "pearsonr": pearsonr(self.y_train, y_train_predicted),
                "spearmanr": spearmanr(self.y_train, y_train_predicted)
            }
            test_metrics = {
                "mse": mean_squared_error(y_true = self.y_test, y_pred = y_test_predicted),
                "mae": mean_absolute_error(y_true = self.y_test, y_pred = y_test_predicted),
                "r2": r2_score(y_true = self.y_test, y_pred = y_test_predicted),
                "kendalltau": kendalltau(self.y_test, y_test_predicted),
                "pearsonr": pearsonr(self.y_test, y_test_predicted),
                "spearmanr": spearmanr(self.y_test, y_test_predicted)
            }
        result["train_metrics"] = train_metrics
        result["test_metrics"] = test_metrics
        return result