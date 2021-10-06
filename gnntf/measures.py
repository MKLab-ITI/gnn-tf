import tensorflow as tf
import random
import numpy as np
from sklearn import metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def acc(predictions, labels):
    return 1-tf.math.count_nonzero(predictions-labels).numpy()/predictions.shape[0]


def auc(labels, predictions):
    fpr, tpr, thresholds = metrics.roc_curve(labels, predictions, pos_label=1)
    return metrics.auc(fpr, tpr)


def prec(labels, predictions, k=5):
    top = np.argsort(predictions)[-k:]
    predictions = predictions*0
    predictions[top] = 1
    return metrics.precision_score(labels, predictions, pos_label=1)


def rec(labels, predictions, k=5):
    top = np.argsort(predictions)[-k:]
    predictions = predictions*0
    predictions[top] = 1
    return metrics.recall_score(labels, predictions, pos_label=1)


def f1(labels, predictions, k=5):
    precision = prec(labels, predictions, k)
    recall = rec(labels, predictions, k)
    if precision+recall == 0:
        return 0
    return 2*precision*recall/(precision+recall)
