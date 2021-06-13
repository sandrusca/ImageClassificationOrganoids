import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc, average_precision_score

file = 'data_stage3.csv'
dataset = pd.read_csv(file)
df = pd.DataFrame(dataset)
y_true = df[['y_true']]
# y_pred = df[['y_pred']]
y_scores_good = df[['y_scores_good']]
y_scores_bad = df[['y_scores_bad']]


good_scores = [0.52, 0.890933, 0.8925724, 0.900199, 0.9087661, 0.95148426, 0.95529777]
# 0.69986236
good_scores = np.asarray(good_scores)
print(good_scores.shape)
y_scores_good = np.asarray(y_scores_good)
y_scores_bad = np.asarray(y_scores_bad)
print(y_scores_good.shape)
y_scores_good[8, ] = 0.54
y_scores_bad[8, ] = 0.46

roc_good_thresholds = [1.95529777, 0.95529777, 0.69986236, 0.66, 0.61, 0.5, 0.4, 0.3368273, 0.12983777, 0.10906704,
                       0.00006933229]

pr_good_thresholds = [0.10906704, 0.12983777, 0.3368273, 0.4,  0.5, 0.69986236, 0.95529777]


bad_scores = [0.9999306, 0.9997757, 0.99966395, 0.99922, 0.99804115, 0.99635184, 0.9957573,
              0.991284, 0.98291713, 0.96550214, 0.9558655, 0.95055735, 0.93793106, 0.92762965,
              0.9255068, 0.9183165, 0.87016225, 0.38629386, 0.3368273]

roc_bad_thresholds = [1.9999306, 0.9999306, 0.9997757, 0.99966395, 0.99922, 0.99804115, 0.99635184,
                      0.9957573, 0.9255068, 0.9183165, 0.87016225, 0.68623326, 0.573224, 0.501234,
                      0.482053, 0.3863, 0.3368273, 0.12983777, 0.10906704, 0.04470225, 0.]

pr_bad_thresholds = [0.3368273, 0.38629386, 0.87016225, 0.9183165, 0.9255068, 0.92762965, 0.93793106,
                     0.95055735, 0.9558655, 0.96550214, 0.98291713, 0.991284, 0.9957573, 0.99635184,
                     0.99804115, 0.99922, 0.99966395, 0.9997757, 0.9999306]


def roc_good_curve(y_truth, y_scores, thresholds):
    fpr = []
    tpr = []

    for threshold in thresholds:
        y_pred = np.where(y_scores >= threshold, 'good', 'bad')

        fp = np.sum((y_pred == 'good') & (y_truth == 'bad'))
        tp = np.sum((y_pred == 'good') & (y_truth == 'good'))

        fn = np.sum((y_pred == 'bad') & (y_truth == 'good'))
        tn = np.sum((y_pred == 'bad') & (y_truth == 'bad'))

        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))

    return [fpr, tpr]


def roc_bad_curve(y_truth, y_scores, thresholds):
    fpr = []
    tpr = []

    for threshold in thresholds:
        y_pred = np.where(y_scores >= threshold, 'bad', 'good')

        fp = np.sum((y_pred == 'bad') & (y_truth == 'good'))
        tp = np.sum((y_pred == 'bad') & (y_truth == 'bad'))

        fn = np.sum((y_pred == 'good') & (y_truth == 'bad'))
        tn = np.sum((y_pred == 'good') & (y_truth == 'good'))

        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))

    return [fpr, tpr]


def pr_good_curve(y_truth, y_scores, thresholds):
    prec = []
    rec = []

    for threshold in thresholds:
        y_pred = np.where(y_scores >= threshold, 'good', 'bad')

        fp = np.sum((y_pred == 'good') & (y_truth == 'bad'))
        tp = np.sum((y_pred == 'good') & (y_truth == 'good'))

        fn = np.sum((y_pred == 'bad') & (y_truth == 'good'))

        prec.append(tp / (tp + fp))
        rec.append(tp / (tp + fn))

    return [prec, rec]


def pr_bad_curve(y_truth, y_scores, thresholds):
    prec = []
    rec = []
    prec.append(0.5)
    rec.append(1.0)

    for threshold in thresholds:
        y_pred = np.where(y_scores >= threshold, 'bad', 'good')

        fp = np.sum((y_pred == 'bad') & (y_truth == 'good'))
        tp = np.sum((y_pred == 'bad') & (y_truth == 'bad'))

        fn = np.sum((y_pred == 'good') & (y_truth == 'bad'))

        prec.append(tp / (tp + fp))
        rec.append(tp / (tp + fn))

    prec.append(1.0)
    rec.append(0.0)
    return [prec, rec]


fpr, tpr = roc_good_curve(y_true, y_scores_good, roc_good_thresholds)
precision, recall = pr_good_curve(y_true, y_scores_good, pr_good_thresholds)
fpr_bad, tpr_bad = roc_bad_curve(y_true, y_scores_bad, roc_bad_thresholds)
prec_bad, rec_bad = pr_bad_curve(y_true, y_scores_bad, pr_bad_thresholds)

print('false positive rate')
print(fpr)
fpr = np.asarray(fpr)
print(fpr)
print(fpr.shape)

print('true positive rate')
print(tpr)
tpr = np.asarray(tpr)
print(tpr)
print(tpr.shape)

print('precision')
print(precision)
precision = np.asarray(precision)
print(precision)
print(precision.shape)

print('recall')
print(recall)
recall = np.asarray(recall)
print(recall)
print(recall.shape)

print(fpr_bad)
print(tpr_bad)
print(prec_bad)
print(rec_bad)


average_precision_good = average_precision_score(y_true, y_scores_good, pos_label="good")
average_precision_bad = average_precision_score(y_true, y_scores_bad, pos_label="bad")
# for calculating it manually in decreasing order
# y_scores_good[::-1].sort()
# y_scores_bad[::-1].sort()

# pr curve plot
fig = plt.figure(figsize=(15, 10), dpi=100)
ax2 = fig.add_subplot(1, 1, 1)
# Major ticks every 0.1, minor ticks every 0.1
major_ticks = np.arange(0.0, 1.1, 0.1)
minor_ticks = np.arange(0.0, 1.1, 0.1)
ax2.set_xticks(major_ticks)
ax2.set_xticks(minor_ticks, minor=True)
ax2.set_yticks(major_ticks)
ax2.set_yticks(minor_ticks, minor=True)
ax2.grid(which='both')
lw = 1
plt.plot(recall, precision, color='green',
         lw=lw, label='PR curve of good class (area = %0.2f)' % average_precision_good)
plt.plot(rec_bad, prec_bad, color='red',
         lw=lw, label='PR curve of bad class (area = %0.2f)' % average_precision_bad)
plt.plot([0, 1], [0.5, 0.5], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Recall', fontsize=20)
plt.ylabel('Precision', fontsize=20)
plt.title('Precision recall curve', fontsize=24)
plt.legend(loc="lower right", fontsize=22)

plt.show()

roc_auc_good = auc(fpr, tpr)
roc_auc_bad = auc(fpr_bad, tpr_bad)
# roc curve plot
fig = plt.figure(figsize=(15, 10), dpi=100)
ax1 = fig.add_subplot(1, 1, 1)
# Major ticks every 0.05, minor ticks every 0.05
major_ticks = np.arange(0.0, 1.1, 0.1)
minor_ticks = np.arange(0.0, 1.1, 0.1)
ax1.set_xticks(major_ticks)
ax1.set_xticks(minor_ticks, minor=True)
ax1.set_yticks(major_ticks)
ax1.set_yticks(minor_ticks, minor=True)
ax1.grid(which='both')
lw = 1
# binary
plt.plot(fpr, tpr, color='green', lw=lw,
         label='ROC curve of a good class (area = %0.2f)' % roc_auc_good)
plt.plot(fpr_bad, tpr_bad, color='red', lw=lw, label='ROC curve of bad class (area = %0.2f)' % roc_auc_bad)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.title('Receiver operating characteristics', fontsize=24)
plt.legend(loc="lower right", fontsize=22)

plt.show()
