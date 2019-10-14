import sklearn.metrics


def evaluate(labels, predicted_labels, measure, edges=None):
    if measure == 'AUC':
        fpr, tpr, _ = sklearn.metrics.roc_curve(labels, predicted_labels)
        return sklearn.metrics.auc(fpr, tpr)
    elif measure == 'HR':
        user_links = dict()
        user_predict = dict()
        pos = 0
        for i, j in edges:
            if i not in user_links:
                user_links[i] = dict()
            if j not in user_links:
                user_links[j] = dict()
            if i not in user_predict:
                user_predict[i] = dict()
            if j not in user_predict:
                user_predict[j] = dict()
            user_links[i][j] = labels[pos]
            user_links[j][i] = labels[pos]
            user_predict[i][j] = predicted_labels[pos]
            user_predict[j][i] = predicted_labels[pos]
            pos += 1
        count = 0
        successes = 0
        for user in user_links:
            if len(user_links[user]) >= 6:
                count += 1
                for top in sorted(list(user_predict[user].keys()), key=lambda i: user_predict[user][i], reverse=True)[:2]:
                    if user_links[user][top]==1:
                        successes += user_links[user][top]
                        break
        print(successes, count)
        return successes/count
    else:
        raise Exception('No such measure: ', measure)