import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score
from sklearn.metrics import classification_report

from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

from sklearn.metrics import roc_curve, auc

'''
# data distribution
data = ((40, 38), (78, 85), (67, 194))

labels = ['stage 1 (day 6-8)', 'stage 2 (day 9-11)', 'stage 3 (day 12-15)']
good = [40, 78, 67]
bad = [38, 85, 194]

x = np.arange(len(labels))
width = 0.40

fig, ax2 = plt.subplots()
goods1 = ax2.bar(x - width / 2, good, width, label='good grown organoids')
bads1 = ax2.bar(x + width / 2, bad, width, label='bad grown organoids')

ax2.set_ylabel('Number of images')
ax2.set_title('Dataset distribution')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.legend()

ax2.bar_label(goods1, padding=3)
ax2.bar_label(bads1, padding=3)

fig.tight_layout()
'''

# rest of plots

# binary
class_labels = ['good', 'bad']
# multi
# class_labels = ['good1', 'bad1', 'good2', 'bad2', 'good3', 'bad3']

file = 'data_stage1.csv'
dataset = pd.read_csv(file)
df = pd.DataFrame(dataset)
y_true = df[['y_true']]
y_pred = df[['y_pred']]
# binary
y_scores_good = df[['y_scores_good']]
y_scores_bad = df[['y_scores_bad']]
# multi
'''
file_names = df[['file_names']]
y_scores1_good = df[['y_scores1_good']]
y_scores1_bad = df[['y_scores1_bad']]
y_scores2_good = df[['y_scores2_good']]
y_scores2_bad = df[['y_scores2_bad']]
y_scores3_good = df[['y_scores3_good']]
y_scores3_bad = df[['y_scores3_bad']]
'''

y_true = np.asarray(y_true)
y_pred = np.asarray(y_pred)
y_scores_good = np.asarray(y_scores_good)
y_scores_bad = np.asarray(y_scores_bad)

# multiclass
'''
file_names = np.asarray(file_names)
y_scores1_good = np.asarray(y_scores1_good)
y_scores1_bad = np.asarray(y_scores1_bad)
y_scores2_good = np.asarray(y_scores2_good)
y_scores2_bad = np.asarray(y_scores2_bad)
y_scores3_good = np.asarray(y_scores3_good)
y_scores3_bad = np.asarray(y_scores3_bad)
class_labels = np.asarray(class_labels)
'''


def performance_measure(y_act, y_predict):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(y_predict)):
        if y_act[i] == y_predict[i] == ['good']:
            tp += 1
        if y_predict[i] == ['good'] and y_act[i] != y_predict[i]:
            fp += 1
        if y_act[i] == y_predict[i] == ['bad']:
            tn += 1
        if y_predict[i] == ['bad'] and y_act[i] != y_predict[i]:
            fn += 1

    return tp, fp, tn, fn


# plot confusion matrix from results
conf_mat = confusion_matrix(y_true, y_pred, labels=class_labels)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh or cm[i, j] == 6 else "black")
    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(conf_mat, classes=class_labels,
                      title='Confusion matrix, without normalization')
plt.show()

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(conf_mat, classes=class_labels, normalize=True,
                      title='Normalized confusion matrix')
plt.show()

# print classification report
print(classification_report(y_true, y_pred, labels=class_labels))

# ROC-curve plot
# binary
fpr_good, tpr_good, _1 = roc_curve(y_true, y_scores_good, pos_label="good")
roc_auc_good = auc(fpr_good, tpr_good)

fpr_bad, tpr_bad, _2 = roc_curve(y_true, y_scores_bad, pos_label="bad")
roc_auc_bad = auc(fpr_bad, tpr_bad)

print('fpr_good')
print(fpr_good)
print('tpr_good')
print(tpr_good)
print('thresholds:')
print(_1)
print('fpr_bad')
print(fpr_bad)
print('tpr_bad')
print(tpr_bad)
print('thresholds:')
print(_2)

# multiclass
'''
fpr_good1, tpr_good1, _good1 = roc_curve(y_true, y_scores1_good, pos_label="good1")
roc_auc_good1 = auc(fpr_good1, tpr_good1)
fpr_bad1, tpr_bad1, _bad1 = roc_curve(y_true, y_scores1_bad, pos_label="bad1")
roc_auc_bad1 = auc(fpr_bad1, tpr_bad1)
print('Threshold:')
print(_good1)
print(_bad1)
print('false-positive-rate aka 1-specificity')
print(fpr_good1)
print('true positive rate aka sensitivity')
print(tpr_good1)
print(roc_auc_good1)

fpr_good2, tpr_good2, _good2 = roc_curve(y_true, y_scores2_good, pos_label="good2")
roc_auc_good2 = auc(fpr_good2, tpr_good2)
fpr_bad2, tpr_bad2, _ = roc_curve(y_true, y_scores2_bad, pos_label="bad2")
roc_auc_bad2 = auc(fpr_bad2, tpr_bad2)

fpr_good3, tpr_good3, _ = roc_curve(y_true, y_scores3_good, pos_label="good3")
roc_auc_good3 = auc(fpr_good3, tpr_good3)
fpr_bad3, tpr_bad3, _ = roc_curve(y_true, y_scores3_bad, pos_label="bad3")
roc_auc_bad3 = auc(fpr_bad3, tpr_bad3)
'''

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
plt.plot(fpr_good, tpr_good, color='green', lw=lw,
         label='ROC curve of a good class (area = %0.2f)' % roc_auc_good)
plt.plot(fpr_bad, tpr_bad, color='red', lw=lw, label='ROC curve of bad class (area = %0.2f)' % roc_auc_bad)
# multiclass
'''
plt.plot(fpr_good1, tpr_good1, color='green', lw=lw, label='ROC curve of good1 class (area = %0.2f)' % roc_auc_good1)
plt.plot(fpr_bad1, tpr_bad1, color='red', lw=lw, label='ROC curve of bad1 class (area = %0.2f)' % roc_auc_bad1)
plt.plot(fpr_good2, tpr_good2, color='blue', lw=lw, label='ROC curve of good2 class (area = %0.2f)' % roc_auc_good2)
plt.plot(fpr_bad2, tpr_bad2, color='black', lw=lw, label='ROC curve of bad2 class (area = %0.2f)' % roc_auc_bad2)
plt.plot(fpr_good3, tpr_good3, color='yellow', lw=lw, label='ROC curve of good3 class (area = %0.2f)' % roc_auc_good3)
plt.plot(fpr_bad3, tpr_bad3, color='orange', lw=lw, label='ROC curve of bad3 class (area = %0.2f)' % roc_auc_bad3)
'''
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
# plt.fill_between(fpr_bad, tpr_bad, alpha=0.30)
plt.title('Receiver operating characteristics', fontsize=24)
plt.legend(loc="lower right", fontsize=22)
plt.show()

# balanced accuracy score
bal_acc_score = balanced_accuracy_score(y_true, y_pred)
print('balanced_acc_score')
print(bal_acc_score)
print('all other accuracies')
print(accuracy_score(y_true, y_pred, normalize=True))

# PR-curve plot
average_precision_good = average_precision_score(y_true, y_scores_good, pos_label="good")
print('Average precision-recall score _good_: {0:0.2f}'.format(average_precision_good))
average_precision_bad = average_precision_score(y_true, y_scores_bad, pos_label="bad")
print('Average precision-recall score _bad_: {0:0.2f}'.format(average_precision_bad))

prec, rec, thresholds = precision_recall_curve(y_true, y_scores_good, pos_label="good")
print('recall_good:')
print(rec)
print('precision_good:')
print(prec)
print('threshold_good:')
print(thresholds)

prec_bad, recall_bad, _ = precision_recall_curve(y_true, y_scores_bad, pos_label="bad")
print('recall_bad:')
print(recall_bad)
print('precision_bad:')
print(prec_bad)
print('threshold_bad:')
print(_)

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
plt.plot(rec, prec, color='green',
         lw=lw, label='PR curve of good class (area = %0.2f)' % average_precision_good)
plt.plot(recall_bad, prec_bad, color='red',
          lw=lw, label='PR curve of bad class (area = %0.2f)' % average_precision_bad)
plt.plot([0, 1], [0.5, 0.5], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Recall', fontsize=20)
plt.ylabel('Precision', fontsize=20)
plt.title('Precision recall curve', fontsize=24)
plt.legend(loc="lower right", fontsize=22)
plt.show()

# skplt.metrics.plot_roc_curve(y_true, scores, title="ROC Curves")
# fpr_good1, tpr_good1, _ = roc_curve(y_true, y_scores_good, pos_label="good")
# roc_display1 = RocCurveDisplay(fpr=fpr_good1, tpr=tpr_good1).plot()
# fpr_bad1, tpr_bad1, _ = roc_curve(y_true, y_scores_bad, pos_label="bad")
# roc_display_bad1 = RocCurveDisplay(fpr=fpr_bad1, tpr=tpr_bad1).plot()
