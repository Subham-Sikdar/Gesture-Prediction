import warnings
import sys

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from urllib.parse import urlparse
import mlflow.sklearn

from make_dataset import create_csv

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    #auc = roc_auc_score(actual, pred, multi_class='ovr')
    return accuracy#, auc


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    df = create_csv()
    df.dropna(inplace=True)
    df = df[df['class'] != 0]

    features = ['channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6', 'channel7', 'channel8']
    X = df[features]
    y = df['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

    # max_depth = float(sys.argv[1]) if len(sys.argv) > 1 else None
    # min_samples_split = float(sys.argv[2]) if len(sys.argv) > 2 else 2

    with mlflow.start_run():
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)

        predicted_qualities = clf.predict(X_test)

        #(accuracy, auc) = eval_metrics(y_test, predicted_qualities)
        accuracy = eval_metrics(y_test, predicted_qualities)

        # print("DecisionTree model (max_depth=%f, min_samples_split=%f):" % (max_depth, min_samples_split))
        print("  accuracy: %s" % accuracy)
        #print("  auc: %s" % auc)

        # mlflow.log_param("max_depth", max_depth)
        # mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_metric("accuracy", accuracy)
        #mlflow.log_metric("auc", auc)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(clf, "model", registered_model_name="DecisionTreeClassifier")
        else:
            mlflow.sklearn.log_model(clf, "model")