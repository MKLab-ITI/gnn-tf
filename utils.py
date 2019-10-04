import sklearn.metrics


def evaluate(labels, predicted_labels):
    fpr, tpr, _ = sklearn.metrics.roc_curve(labels, predicted_labels)
    return sklearn.metrics.auc(fpr, tpr)