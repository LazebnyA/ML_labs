import math

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, auc, roc_curve


# ---------------------- func block ---------------------- #

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


# ---------------------- func block ---------------------- #

# =====================================================================================================


# 1st. task
df = pd.read_csv('KM-12-2.csv')
print(f'''
1. Відкрити та зчитати дані з наданого файлу. Файл містить п’ять стовпчиків: \n{df.head()}\n
a. Фактичне значення цільової характеристики. \n{df["GT"].head()}\n
b. Результат передбачення моделі No1 у вигляді ймовірності
приналежності об’єкту до класу 0. \n{df["Model_1_0"].head()}\n
c. Результат передбачення моделі No1 у вигляді ймовірності
приналежності об’єкту до класу 1. \n{df["Model_1_1"].head()}\n
d. Результат передбачення моделі No2 у вигляді ймовірності
приналежності об’єкту до класу 0. \n{df["Model_2_0"].head()}\n
e. Результат передбачення моделі No2 у вигляді ймовірності
приналежності об’єкту до класу 1. \n{df["Model_2_1"].head()}\n
''')

# =====================================================================================================

# 2nd. task
print('''
2. Визначити збалансованість набору даних. Вивести кількість об’єктів
кожного класу.
''')

gt_counts = df['GT'].value_counts()

print(f"Розподіл об'єктів відповідних класів: {gt_counts.to_string()}")
print("Оскільки кількість об'єктів однакова для кожного класу — набір даних збалансований.\n")

# =====================================================================================================

# 3rd. task
print("3. Для зчитаного набору даних виконати наступні дії:")

step = 0.05
start_threshold = 0

x_values = get_x_values(start_threshold, step)

# 3rd. task pt. a
predictions_df_m1 = get_predictions(df['GT'], df['Model_1_1'], x_values)
predictions_df_m2 = get_predictions(df['GT'], df['Model_2_1'], x_values)

# Accuracy
accuracy_m1 = list(map(lambda x: get_accuracy(predictions_df_m1, x),
                       map(str, x_values)))

accuracy_m2 = list(map(lambda x: get_accuracy(predictions_df_m2, x),
                       map(str, x_values)))

print("a. Обчислити всі метрики.\n")

print(f"Accuracy model 1: {accuracy_m1}")
print(f"Accuracy model 2: {accuracy_m2}\n")

# Precision


precision_m1 = list(map(lambda x: get_precision(predictions_df_m1, x),
                        map(str, x_values)))

precision_m2 = list(map(lambda x: get_precision(predictions_df_m2, x),
                        map(str, x_values)))

print(f"Precision model 1: {precision_m1}")
print(f"Precision model 2: {precision_m2}\n")

# Recall
recall_m1 = list(map(lambda x: get_recall_1(predictions_df_m1, x),
                     map(str, x_values)))

recall_m2 = list(map(lambda x: get_recall_1(predictions_df_m2, x),
                     map(str, x_values)))

print(f"Recall model 1: {recall_m1}")
print(f"Recall model 2: {recall_m2}\n")

# F-Scores
f_scores_m1 = list(map(lambda x, y: get_f_scores(x, y),
                       precision_m1, recall_m1))

f_scores_m2 = list(map(lambda x, y: get_f_scores(x, y),
                       precision_m2, recall_m2))

print(f"F-Scores model 1: {f_scores_m1}")
print(f"F-Scores model 2: {f_scores_m2}\n")

# Matthew Correlation Coefficient
mcc_m1 = list(map(lambda x: get_MCC(predictions_df_m1, x),
                  map(str, x_values)))

mcc_m2 = list(map(lambda x: get_MCC(predictions_df_m2, x),
                  map(str, x_values)))

print(f"Matthew Correlation Coefficient model 1: {mcc_m1}")
print(f"Matthew Correlation Coefficient model 2: {mcc_m2}\n")

# Balanced Accuracy
balanced_acc_m1 = list(map(lambda x: get_balanced_acc(predictions_df_m1, x),
                           map(str, x_values)))

balanced_acc_m2 = list(map(lambda x: get_balanced_acc(predictions_df_m2, x),
                           map(str, x_values)))

print(f"Balanced accuracy model 1: {balanced_acc_m1}")
print(f"Balanced accuracy model 2: {balanced_acc_m2}\n")

# Youden J
youden_j_m1 = list(map(lambda x: get_youden_j(predictions_df_m1, x),
                       map(str, x_values)))

youden_j_m2 = list(map(lambda x: get_youden_j(predictions_df_m2, x),
                       map(str, x_values)))

print(f"Youden's J Statistics model 1: {youden_j_m1}")
print(f"Youden's J Statistics model 2: {youden_j_m2}\n")

# Area under curve for Precision-Recall Curve
precision_for_curve_m1, recall_for_curve_m1, thresholds_pr_m1 = precision_recall_curve(df['GT'], df['Model_1_1'])
precision_for_curve_m2, recall_for_curve_m2, thresholds_pr_m2 = precision_recall_curve(df['GT'], df['Model_2_1'])

auc_pr_m1 = auc(recall_for_curve_m1, precision_for_curve_m1)
auc_pr_m2 = auc(recall_for_curve_m2, precision_for_curve_m2)

print("Area Under Curve for Precision-Recall Curve model 1:", auc_pr_m1)
print("Area Under Curve for Precision-Recall Curve model 2:", auc_pr_m2)
print()

# Area under curve for Receiver Operation Curve
fpr_m1, tpr_m1, thresholds_roc_m1 = roc_curve(df['GT'], df['Model_1_1'])
fpr_m2, tpr_m2, thresholds_roc_m2 = roc_curve(df['GT'], df['Model_2_1'])

auc_roc_m1 = auc(fpr_m1, tpr_m1)
auc_roc_m2 = auc(fpr_m2, tpr_m2)

print("Area Under Curve for Receiver Operation Curve model 1:", auc_roc_m1)
print("Area Under Curve for Receiver Operation Curve model 2:", auc_roc_m2)
print()

# 3rd. task pt. b
print("\nb. Графіки. Metrics vs Threshold for Model 1/2.")

colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']

plt.figure(figsize=(10, 6))
ticks = get_x_values(0, 0.1)
plt.xticks(ticks)
plt.yticks(ticks)

plt.plot(x_values, accuracy_m1, color=colors[0], label='Accuracy', alpha=1)
plt.plot(x_values, precision_m1, color=colors[1], label='Precision')
plt.plot(x_values, recall_m1, color=colors[2], label='Recall')
plt.plot(x_values, f_scores_m1, color=colors[3], label='F-Score')
plt.plot(x_values, mcc_m1, color=colors[4], label='MCC')
plt.plot(x_values, balanced_acc_m1, color=colors[5], label='Balanced Accuracy', alpha=0.5)
plt.plot(x_values, youden_j_m1, color=colors[6], label="Youden's J")

max_accuracy_m1_value = max(accuracy_m1)
max_accuracy_m1_threshold = x_values[accuracy_m1.index(max_accuracy_m1_value)]
plt.scatter(max_accuracy_m1_threshold, max_accuracy_m1_value, marker='o', alpha=1, color=colors[0])

max_precision_m1_value = max(precision_m1)
max_precision_m1_threshold = x_values[precision_m1.index(max_precision_m1_value)]
plt.scatter(max_precision_m1_threshold, max_precision_m1_value, marker='o', color=colors[1])

max_recall_m1_value = max(recall_m1)
max_recall_m1_threshold = x_values[recall_m1.index(max_recall_m1_value)]
plt.scatter(max_recall_m1_threshold, max_recall_m1_value, marker='o', color=colors[2])

max_f_scores_m1_value = max(f_scores_m1)
max_f_scores_m1_threshold = x_values[f_scores_m1.index(max_f_scores_m1_value)]
plt.scatter(max_f_scores_m1_threshold, max_f_scores_m1_value, marker='o', color=colors[3])

max_mcc_m1_value = max(mcc_m1)
max_mcc_m1_threshold = x_values[mcc_m1.index(max_mcc_m1_value)]
plt.scatter(max_mcc_m1_threshold, max_mcc_m1_value, marker='o', color=colors[4])

max_balanced_acc_m1_value = max(balanced_acc_m1)
max_balanced_acc_m1_threshold = x_values[balanced_acc_m1.index(max_balanced_acc_m1_value)]
plt.scatter(max_balanced_acc_m1_threshold, max_balanced_acc_m1_value, marker='o', alpha=0.5, color=colors[5])

max_youden_j_m1_value = max(youden_j_m1)
max_youden_j_m1_threshold = x_values[youden_j_m1.index(max_youden_j_m1_value)]
plt.scatter(max_youden_j_m1_threshold, max_youden_j_m1_value, marker='o', color=colors[6])

m1_optimal_thresholds = {
    'Accuracy': max_accuracy_m1_threshold,
    'Precision': max_precision_m1_threshold,
    'Recall': max_recall_m1_threshold,
    'F-Scores': max_f_scores_m1_threshold,
    'MCC': max_mcc_m1_threshold,
    'Balanced Accuracy': max_balanced_acc_m1_threshold,
    'Youden`s J': max_youden_j_m1_threshold
}

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

plt.plot(x_values, accuracy_m2, color=colors[0], label='Accuracy', alpha=1)
plt.plot(x_values, precision_m2, color=colors[1], label='Precision')
plt.plot(x_values, recall_m2, color=colors[2], label='Recall')
plt.plot(x_values, f_scores_m2, color=colors[3], label='F-Score')
plt.plot(x_values, mcc_m2, color=colors[4], label='MCC')
plt.plot(x_values, balanced_acc_m2, color=colors[5], label='Balanced Accuracy', alpha=0.5)
plt.plot(x_values, youden_j_m2, color=colors[6], label="Youden's J")

max_accuracy_m2_value = max(accuracy_m2)
max_accuracy_m2_threshold = x_values[accuracy_m2.index(max_accuracy_m2_value)]
plt.scatter(max_accuracy_m2_threshold, max_accuracy_m2_value, marker='o', alpha=1, color=colors[0])

max_precision_m2_value = max(precision_m2)
max_precision_m2_threshold = x_values[precision_m2.index(max_precision_m2_value)]
plt.scatter(max_precision_m2_threshold, max_precision_m2_value, marker='o', color=colors[1])

max_recall_m2_value = max(recall_m2)
max_recall_m2_threshold = x_values[recall_m2.index(max_recall_m2_value)]
plt.scatter(max_recall_m2_threshold, max_recall_m2_value, marker='o', color=colors[2])

max_f_scores_m2_value = max(f_scores_m2)
max_f_scores_m2_threshold = x_values[f_scores_m2.index(max_f_scores_m2_value)]
plt.scatter(max_f_scores_m2_threshold, max_f_scores_m2_value, marker='o', color=colors[3])

max_mcc_m2_value = max(mcc_m2)
max_mcc_m2_threshold = x_values[mcc_m2.index(max_mcc_m2_value)]
plt.scatter(max_mcc_m2_threshold, max_mcc_m2_value, marker='o', color=colors[4])

max_balanced_acc_m2_value = max(balanced_acc_m2)
max_balanced_acc_m2_threshold = x_values[balanced_acc_m2.index(max_balanced_acc_m2_value)]
plt.scatter(max_balanced_acc_m2_threshold, max_balanced_acc_m2_value, marker='o', alpha=0.5, color=colors[5])

max_youden_j_m2_value = max(youden_j_m2)
max_youden_j_m2_threshold = x_values[youden_j_m2.index(max_youden_j_m2_value)]
plt.scatter(max_youden_j_m2_threshold, max_youden_j_m2_value, marker='o', color=colors[6])

m2_optimal_thrasholds = {
    'Accuracy': max_accuracy_m2_threshold,
    'Precision': max_precision_m2_threshold,
    'Recall': max_recall_m2_threshold,
    'F-Scores': max_f_scores_m2_threshold,
    'MCC': max_mcc_m2_threshold,
    'Balanced Accuracy': max_balanced_acc_m2_threshold,
    'Youden`s J': max_youden_j_m2_threshold
}

plt.xlabel('Threshold')
plt.ylabel('Metric Value')
plt.title('Metrics vs Threshold for Model 2')
plt.legend()

plt.grid(True)
plt.savefig('metrics_plot_model2.png', dpi=300)
plt.show()

# 3rd. task pt. c
print("\nc. Графіки. Object Counts vs Classifier Score for Model 1/2.")

fig, ax = plt.subplots(figsize=(10, 6))

class_0_objects_m1 = df.loc[df['GT'] == 0, 'Model_1_1']
class_1_objects_m1 = df.loc[df['GT'] == 1, 'Model_1_1']

ax.hist(class_0_objects_m1, bins=20, color='blue', alpha=0.5, label='Class 0 Objects')

ax.hist(class_1_objects_m1, bins=20, color='red', alpha=0.5, label='Class 1 Objects')

ax.set_title('Object Counts vs Classifier Score for Model 1')
ax.set_xlabel('Classifier Score')
ax.set_ylabel('Object Count')

for el, color in zip(m1_optimal_thresholds.items(), colors):
    ax.axvline(x=el[1], linestyle='--', label=el[0], color=color)

ax.legend()
plt.savefig('count_objects_by_score_m1.png', dpi=300)
plt.show()

############################

fig, ax = plt.subplots(figsize=(10, 6))

class_0_objects_m2 = df.loc[df['GT'] == 0, 'Model_2_1']
class_1_objects_m2 = df.loc[df['GT'] == 1, 'Model_2_1']

ax.hist(class_0_objects_m2, bins=20, color='blue', alpha=0.5, label='Class 0 Objects')

ax.hist(class_1_objects_m2, bins=20, color='red', alpha=0.5, label='Class 1 Objects')

ax.set_title('Object Counts vs Classifier Score for Model 2')
ax.set_xlabel('Classifier Score')
ax.set_ylabel('Object Count')

for el, color in zip(m2_optimal_thrasholds.items(), colors):
    ax.axvline(x=el[1], linestyle='--', label=el[0], color=color)

ax.legend()
plt.savefig('count_objects_by_score_m2.png', dpi=300)
plt.show()

# 3rd. task pt. d

print('''
d. Збудувати для кожного класифікатору PR-криву та ROC-криву,
показавши графічно на них значення оптимального порогу. (PR/ROC curve)''')

# RV

# Building graphs
plt.plot(recall_for_curve_m1, precision_for_curve_m1, label='Model_1_1')
ideal_points_m1 = np.ones(len(recall_for_curve_m1))
optimal_idx_m1 = np.argmin(
    np.sqrt((recall_for_curve_m1 - ideal_points_m1) ** 2 + (precision_for_curve_m1 - ideal_points_m1) ** 2))
optimal_threshold_m1 = thresholds_pr_m1[optimal_idx_m1]
plt.scatter(recall_for_curve_m1[optimal_idx_m1],
            precision_for_curve_m1[optimal_idx_m1])
plt.annotate(round(optimal_threshold_m1, 2),
             (recall_for_curve_m1[optimal_idx_m1],
              precision_for_curve_m1[optimal_idx_m1]))

plt.plot(recall_for_curve_m2, precision_for_curve_m2, label='Model_2_1')
ideal_points_m2 = np.ones(len(recall_for_curve_m2))  # Создаем массив единиц такой же длины, как и recall_for_curve_m2
optimal_idx_m2 = np.argmin(
    np.sqrt((recall_for_curve_m2 - ideal_points_m2) ** 2 + (precision_for_curve_m2 - ideal_points_m2) ** 2))
optimal_threshold_m2 = thresholds_pr_m2[optimal_idx_m2]
plt.scatter(recall_for_curve_m2[optimal_idx_m2],
            precision_for_curve_m2[optimal_idx_m2])
plt.annotate(round(optimal_threshold_m2, 2),
             (recall_for_curve_m2[optimal_idx_m2],
              precision_for_curve_m2[optimal_idx_m2]))

plt.title('Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.savefig('PR-curves.png', dpi=300)
plt.show()

# ROC
plt.plot(fpr_m1, tpr_m1, label='Model_1_1')
ideal_x_m1 = np.zeros(len(fpr_m1))
ideal_y_m1 = np.ones(len(fpr_m1))
optimal_idx_m1 = np.argmin(np.sqrt((fpr_m1 - ideal_x_m1) ** 2 + (tpr_m1 - ideal_y_m1) ** 2))
optimal_threshold_m1 = thresholds_roc_m1[optimal_idx_m1]
plt.scatter(fpr_m1[optimal_idx_m1],
            tpr_m1[optimal_idx_m1])
plt.annotate(round(optimal_threshold_m1, 2),
             (fpr_m1[optimal_idx_m1],
              tpr_m1[optimal_idx_m1]))

plt.plot(fpr_m2, tpr_m2, label='Model_1_1')
ideal_x_m2 = np.zeros(len(fpr_m2))
ideal_y_m2 = np.ones(len(fpr_m2))
optimal_idx_m2 = np.argmin(np.sqrt((fpr_m2 - ideal_x_m2) ** 2 + (tpr_m2 - ideal_y_m2) ** 2))
optimal_threshold_m2 = thresholds_roc_m2[optimal_idx_m2]
plt.scatter(fpr_m2[optimal_idx_m2],
            tpr_m2[optimal_idx_m2])
plt.annotate(round(optimal_threshold_m2, 2),
             (fpr_m2[optimal_idx_m2],
              tpr_m2[optimal_idx_m2]))

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-curve')
plt.legend()
plt.grid(True)
plt.savefig('ROC-curves.png', dpi=300)
plt.show()


# Task 4
print("\n4. Перша модель є кращою, це можна прослідкувати на графіків пунктів 3с. і 3d.\n")

# Task 5
print('''
5. Створити новий набір даних, прибравши з початкового набору (50 +
5К)% об’єктів класу 1, вибраних випадковим чином. Параметр К
представляє собою залишок від ділення дня народження студента на дев’ять
та має визначатися в програмі на основі дати народження студента, яка
задана в програмі у вигляді текстової змінної формату ‘DD-MM’.
''')
