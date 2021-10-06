import tensorflow as tf
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def acc(predictions, labels):
    return 1-tf.math.count_nonzero(predictions-labels).numpy()/predictions.shape[0]


def auc(labels, predictions):
    from sklearn import metrics
    fpr, tpr, thresholds = metrics.roc_curve(labels, predictions, pos_label=1)
    return metrics.auc(fpr, tpr)
