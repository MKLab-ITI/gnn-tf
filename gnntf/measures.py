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


def avprec(labels, predictions, k=5):
    nom = 0
    top = np.argsort(predictions)[-k:]
    for pos, i in enumerate(reversed(top)):
        nom += labels[i]/(pos+1)
    return 0 if nom == 0 else nom/np.sum(np.array(labels)[top])


def rec(labels, predictions, k=5):
    top = np.argsort(predictions)[-k:]
    return np.sum(np.array(labels)[top])/np.sum(labels)


def prec(labels, predictions, k=5):
    top = np.argsort(predictions)[-k:]
    return np.mean(np.array(labels)[top])


def f1(labels, predictions, k=5):
    precision = prec(labels, predictions, k)
    recall = rec(labels, predictions, k)
    if precision+recall == 0:
        return 0
    return 2*precision*recall/(precision+recall)
