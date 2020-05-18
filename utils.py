import sklearn.metrics
import tensorflow as tf

def evaluate(labels, predicted_labels, measure, edges=None):
    if measure == 'AUC':
        fpr, tpr, _ = sklearn.metrics.roc_curve(labels, predicted_labels)
        return sklearn.metrics.auc(fpr, tpr)
    elif measure == 'HR3':
        node_ids = dict()
        i = -1
        for u, v in edges:
            i += 1
            if u not in node_ids:
                node_ids[u] = list()
            if v not in node_ids:
                node_ids[v] = list()
            node_ids[u].append(i)
            node_ids[v].append(i)
        k = 3
        HRs = list()
        for u in node_ids:
            if sum(predicted_labels[i] for i in node_ids[u])==0:
                continue
            HR = 0
            for num, i in enumerate(sorted(node_ids[u], key=lambda i: predicted_labels[i], reverse=True)):
                if num == k:
                    break
                HR += labels[i]
            HRs.append(min(HR, 1))
        return sum(HRs) / len(HRs)
    elif measure == 'HR1':
        best_score = dict()
        true_prediction = dict()
        i = -1
        for u, v in edges:
            i += 1
            if best_score.get(u,0) <= predicted_labels[i]:
                true_prediction[u] = labels[i]
                best_score[u] = predicted_labels[i]
            if best_score.get(v,0) <= predicted_labels[i]:
                true_prediction[v] = labels[i]
                best_score[v] = predicted_labels[i]
        return sum(true_prediction.values())/len(true_prediction)
    else:
        raise Exception('No such measure: ', measure)