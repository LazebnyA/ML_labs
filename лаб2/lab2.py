import math

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, auc, roc_curve

# 1st. task
df = pd.read_csv('KM-12-2.csv')
# print(df.to_string())

# =====================================================================================================

# 2nd. task
gt_counts = df['GT'].value_counts()

print(f"Розподіл об'єктів відповідних класів: {gt_counts.to_string()}")
print("Оскільки кількість об'єктів однакова для кожного класу — набір даних збалансований.")


# =====================================================================================================

# 3rd. task

def get_x_values(start, step):
    x_values = []

    while start <= 1.0:
        x_values.append(start)
        start += step

    x_values = [round(el, 3) for el in x_values]

    if x_values[-1] != 1.0:
        x_values.append(1.0)

    return x_values


def get_predictions(actual, probabilities, threshold_val):
    predictions_df = pd.DataFrame({'GT': actual})

    dfs = []

    for val in threshold_val:
        predictions = [1 if probability >= val else 0 for probability in probabilities]

        df = pd.DataFrame({str(val): predictions})

        dfs.append(df)

    predictions_df = pd.concat([predictions_df] + dfs, axis=1)

    return predictions_df


def get_accuracy(predictions, threshold):
    accuracy = (predictions[threshold] == predictions['GT']).mean()
    return accuracy


def get_confusion_map(predictions, threshold):
    confusion_map = {
        'TP': ((predictions[threshold] == 1) & (predictions['GT'] == 1)).sum(),
        'FP': ((predictions[threshold] == 1) & (predictions['GT'] == 0)).sum(),
        'FN': ((predictions[threshold] == 0) & (predictions['GT'] == 1)).sum(),
        'TN': ((predictions[threshold] == 0) & (predictions['GT'] == 0)).sum()
    }

    return confusion_map


def get_precision(predictions, threshold):
    confusion_map = get_confusion_map(predictions, threshold)

    precision = confusion_map['TP'] / (confusion_map['TP'] + confusion_map['FP'])

    return precision


def get_recall_1(predictions, threshold):
    confusion_map = get_confusion_map(predictions, threshold)

    recall = confusion_map['TP'] / (confusion_map['TP'] + confusion_map['FN'])

    return recall


def get_recall_0(predictions, threshold):
    confusion_map = get_confusion_map(predictions, threshold)

    recall = confusion_map['TN'] / (confusion_map['TN'] + confusion_map['FP'])

    return recall


def get_f_scores(precision, recall):
    return 2 * precision * recall / (precision + recall)


def get_MCC(predictions, threshold):
    confusion_map = get_confusion_map(predictions, threshold)

    numerator = confusion_map['TP'] * confusion_map['TN'] - confusion_map['FP'] * confusion_map['FN']
    denominator = math.sqrt((confusion_map['TP'] + confusion_map['FP']) *
                            (confusion_map['TP'] + confusion_map['FN']) *
                            (confusion_map['TN'] + confusion_map['FP']) *
                            (confusion_map['TN'] + confusion_map['FN'])
                            )

    if denominator == 0:
        mcc = 0
    else:
        mcc = numerator / denominator

    return mcc


def get_sensitivity_specificity(predictions, threshold):
    confusion_map = get_confusion_map(predictions, threshold)

    sensitivity = confusion_map['TP'] / (confusion_map['TP'] + confusion_map['FN'])
    specificity = confusion_map['TN'] / (confusion_map['TN'] + confusion_map['FP'])

    return sensitivity, specificity


def get_balanced_acc(predictions, threshold):
    sensitivity, specificity = get_sensitivity_specificity(predictions, threshold)

    balanced_accuracy = (sensitivity + specificity) / 2

    return balanced_accuracy


def get_youden_j(predictions, threshold):
    sensitivity, specificity = get_sensitivity_specificity(predictions, threshold)

    youden_j = sensitivity + specificity - 1

    return youden_j


step = 0.05
start_threshold = 0

x_values = get_x_values(start_threshold, step)
print(x_values)

# 3rd. task pt. a

predictions_df_m1 = get_predictions(df['GT'], df['Model_1_1'], x_values)
predictions_df_m2 = get_predictions(df['GT'], df['Model_2_1'], x_values)

# print(predictions_df_m1.to_string())

# Accuracy

accuracy_m1 = list(map(lambda x: get_accuracy(predictions_df_m1, x),
                       map(str, x_values)))

accuracy_m2 = list(map(lambda x: get_accuracy(predictions_df_m2, x),
                       map(str, x_values)))

print(f"Accuracy model 1: {accuracy_m1}")
print(f"Accuracy model 2: {accuracy_m2}")

# Precision


precision_m1 = list(map(lambda x: get_precision(predictions_df_m1, x),
                        map(str, x_values)))

precision_m2 = list(map(lambda x: get_precision(predictions_df_m2, x),
                        map(str, x_values)))

print(f"Precision model 1: {precision_m1}")
print(f"Precision model 2: {precision_m2}")

# Recall
recall_m1 = list(map(lambda x: get_recall_1(predictions_df_m1, x),
                     map(str, x_values)))

recall_m2 = list(map(lambda x: get_recall_1(predictions_df_m2, x),
                     map(str, x_values)))

print(f"Recall model 1: {recall_m1}")
print(f"Recall model 2: {recall_m2}")

# F-Scores
f_scores_m1 = list(map(lambda x, y: get_f_scores(x, y),
                       precision_m1, recall_m1))

f_scores_m2 = list(map(lambda x, y: get_f_scores(x, y),
                       precision_m2, recall_m2))

print(f"F-Scores model 1: {f_scores_m1}")
print(f"F-Scores model 2: {f_scores_m2}")

# Matthew Correlation Coefficient
mcc_m1 = list(map(lambda x: get_MCC(predictions_df_m1, x),
                  map(str, x_values)))

mcc_m2 = list(map(lambda x: get_MCC(predictions_df_m2, x),
                  map(str, x_values)))

print(f"Matthew Correlation Coefficient model 1: {mcc_m1}")
print(f"Matthew Correlation Coefficient model 2: {mcc_m2}")

# Balanced Accuracy
balanced_acc_m1 = list(map(lambda x: get_balanced_acc(predictions_df_m1, x),
                           map(str, x_values)))

balanced_acc_m2 = list(map(lambda x: get_balanced_acc(predictions_df_m2, x),
                           map(str, x_values)))

print(f"Balanced accuracy model 1: {balanced_acc_m1}")
print(f"Balanced accuracy model 2: {balanced_acc_m2}")

# Youden J

youden_j_m1 = list(map(lambda x: get_youden_j(predictions_df_m1, x),
                       map(str, x_values)))

youden_j_m2 = list(map(lambda x: get_youden_j(predictions_df_m2, x),
                       map(str, x_values)))

print(f"Youden's J Statistics model 1: {youden_j_m1}")
print(f"Youden's J Statistics model 2: {youden_j_m2}")

# Area under curve for Precision-Recall Curve

precision_for_curve_m1, recall_for_curve_m1, thresholds_m1 = precision_recall_curve(df['GT'], df['Model_1_1'])
precision_for_curve_m2, recall_for_curve_m2, thresholds_m2 = precision_recall_curve(df['GT'], df['Model_2_1'])

print(len(thresholds_m1))
print(len(thresholds_m2))

auc_pr_m1 = auc(recall_for_curve_m1, precision_for_curve_m1)
auc_pr_m2 = auc(recall_for_curve_m2, precision_for_curve_m2)

print("Area Under Curve for Precision-Recall Curve model 1:", auc_pr_m1)
print("Area Under Curve for Precision-Recall Curve model 2:", auc_pr_m2)

# Area under curve for Receiver Operation Curve


fpr_m1, tpr_m1, thresholds_m1 = roc_curve(df['GT'], df['Model_1_1'])
fpr_m2, tpr_m2, thresholds_m2 = roc_curve(df['GT'], df['Model_2_1'])

print(len(thresholds_m1))
print(len(thresholds_m2))

auc_roc_m1 = auc(fpr_m1, tpr_m1)
auc_roc_m2 = auc(fpr_m2, tpr_m2)

print("Area Under Curve for Receiver Operation Curve model 1:", auc_roc_m1)
print("Area Under Curve for Receiver Operation Curve model 2:", auc_roc_m2)



# 3rd. task pt. b

plt.figure(figsize=(10, 6))
ticks = get_x_values(0, 0.1)
plt.xticks(ticks)
plt.yticks(ticks)

plt.plot(x_values, accuracy_m1, label='Accuracy', alpha=1)

plt.plot(x_values, precision_m1, label='Precision')

plt.plot(x_values, recall_m1, label='Recall')

plt.plot(x_values, f_scores_m1, label='F-Score')

plt.plot(x_values, mcc_m1, label='MCC')

plt.plot(x_values, balanced_acc_m1, label='Balanced Accuracy', alpha = 0.5)

plt.plot(x_values, youden_j_m1, label="Youden's J")

plt.scatter(x_values[accuracy_m1.index(max(accuracy_m1))], max(accuracy_m1), marker='o', alpha = 1)
plt.scatter(x_values[precision_m1.index(max(precision_m1))], max(precision_m1), marker='o')
plt.scatter(x_values[recall_m1.index(max(recall_m1))], max(recall_m1), marker='o')
plt.scatter(x_values[f_scores_m1.index(max(f_scores_m1))], max(f_scores_m1), marker='o')
plt.scatter(x_values[mcc_m1.index(max(mcc_m1))], max(mcc_m1), marker='o')
plt.scatter(x_values[balanced_acc_m1.index(max(balanced_acc_m1))], max(balanced_acc_m1), marker='o', alpha = 0.5)
plt.scatter(x_values[youden_j_m1.index(max(youden_j_m1))], max(youden_j_m1), marker='o')

plt.xlabel('Threshold')
plt.ylabel('Metric Value')
plt.title('Metrics vs Threshold for Model 1')
plt.legend()

plt.grid(True)
plt.savefig('metrics_plot_model1.png', dpi=300)
plt.show()

####################################

plt.figure(figsize=(10, 6))
ticks = get_x_values(0, 0.1)
plt.xticks(ticks)
plt.yticks(ticks)

plt.plot(x_values, accuracy_m2, label='Accuracy', alpha = 1)

plt.plot(x_values, precision_m2, label='Precision')

plt.plot(x_values, recall_m2, label='Recall')

plt.plot(x_values, f_scores_m2, label='F-Score')

plt.plot(x_values, mcc_m2, label='MCC')

plt.plot(x_values, balanced_acc_m2, label='Balanced Accuracy', alpha = 0.5)

plt.plot(x_values, youden_j_m2, label="Youden's J")

plt.scatter(x_values[accuracy_m2.index(max(accuracy_m2))], max(accuracy_m2), marker='o', alpha = 1)
plt.scatter(x_values[precision_m2.index(max(precision_m2))], max(precision_m2), marker='o')
plt.scatter(x_values[recall_m2.index(max(recall_m2))], max(recall_m2), marker='o')
plt.scatter(x_values[f_scores_m2.index(max(f_scores_m2))], max(f_scores_m2), marker='o')
plt.scatter(x_values[mcc_m2.index(max(mcc_m2))], max(mcc_m2), marker='o')
plt.scatter(x_values[balanced_acc_m2.index(max(balanced_acc_m2))], max(balanced_acc_m2), marker='o', alpha = 0.5)
plt.scatter(x_values[youden_j_m2.index(max(youden_j_m2))], max(youden_j_m2), marker='o')

plt.xlabel('Threshold')
plt.ylabel('Metric Value')
plt.title('Metrics vs Threshold for Model 2')
plt.legend()

plt.grid(True)
plt.savefig('metrics_plot_model2.png', dpi=300)
plt.show()
