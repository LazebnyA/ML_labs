import pandas as pd

import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

import graphviz

import json


def visualize_decision_tree(tree, name='gini', to_return=False):
    dot_data = export_graphviz(tree, class_names=['class_0', 'class_1'], feature_names=X.columns, impurity=False,
                               filled=True)

    graph = graphviz.Source(dot_data)

    if not to_return:
        graph.render(f"decision_tree_{name}", format="png", view=True)
    return graph


def get_metrics_for_pred(y_pred, y_actual, to_print=True):
    confusion_matrix_test = metrics.confusion_matrix(y_actual, y_pred)
    tn, fp, fn, tp = confusion_matrix_test.ravel()
    sensitivity_test = tp / (tp + fn)
    specificity_test = tn / (tn + fp)

    accuracy = metrics.accuracy_score(y_actual, y_pred)
    precision = metrics.precision_score(y_actual, y_pred, average='weighted')
    recall = metrics.recall_score(y_actual, y_pred, average='weighted')
    f1_score = metrics.f1_score(y_actual, y_pred, average='weighted')
    mcc = metrics.matthews_corrcoef(y_actual, y_pred)
    balanced_accuracy = metrics.balanced_accuracy_score(y_actual, y_pred)
    yodens_j = sensitivity_test + specificity_test - 1

    if to_print:
        print("Confusion Matrix:\n", confusion_matrix_test)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1_score)
        print("MCC: ", mcc)
        print("Balanced accuracy: ", balanced_accuracy)
        print("Youden's J statistics: ", yodens_j)

    return (
        accuracy,
        precision,
        recall,
        f1_score,
        mcc,
        balanced_accuracy,
        yodens_j,
        list(confusion_matrix_test)
    )


def get_results(clf, criterion_name):
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    print(f"\nРезультати, отриманні при застосуванні критерію розщеплення: {criterion_name}")

    print("\nМетрики для тренувальної вибірки: ")
    get_metrics_for_pred(y_pred_train, y_train)
    print("\n" + "-" * 50 + "\n")

    print("Метрики для тестової вибірки: ")
    get_metrics_for_pred(y_pred_test, y_test)

    # Візуалізація роботи моделі за допомогою Confusion Matrix.
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    conf_matrix_train = metrics.confusion_matrix(y_train, y_pred_train)
    sns.heatmap(conf_matrix_train, annot=True, cmap='Blues', fmt='g', ax=axes[0])
    axes[0].set_title(f'Train Confusion Matrix. {criterion_name}')
    axes[0].set_xlabel('Predicted labels')
    axes[0].set_ylabel('True labels')

    conf_matrix_test = metrics.confusion_matrix(y_test, y_pred_test)
    sns.heatmap(conf_matrix_test, annot=True, cmap='Blues', fmt='g', ax=axes[1])
    axes[1].set_title(f'Test Confusion Matrix. {criterion_name}')
    axes[1].set_xlabel('Predicted labels')
    axes[1].set_ylabel('True labels')

    plt.tight_layout()

    # plt.savefig(f'conf_matrix_train_test_{criterion_name}.png')

    plt.show()


def get_feature_importances(clf, criterion_name):
    feature_importances = clf.feature_importances_

    feature_names = X.columns

    indices = np.argsort(feature_importances)

    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importances with criterion: {criterion_name}")
    plt.bar(range(X.shape[1]), feature_importances[indices], color="b", align="center")
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.show()

# 1st. task
df = pd.read_csv('dataset_2.txt', header=None)
df.columns = ["Num", "Date", "Feature1", "Feature2", "Feature3", "Feature4", "Feature5", "Target"]

# 2nd. task
num_records, num_fields = df.shape
print("\n\n2. Визначити та вивести кількість записів та кількість полів у завантаженому наборі даних.")
print("Кількість записів:", num_records)
print("Кількість полів у кожному записі:", num_fields)

# 3rd. task
print("\n\n3. Вивести перші 10 записів набору даних.")
print(df.iloc[:10].to_string())

# 4th. task Scikit Learn

print("\n\n4. Розділити набір даних на навчальну (тренувальну) та тестову вибірки.")

X = df.iloc[:, 2:-1]
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nНавчальна вибірка: \n", X_train, y_train)
print("_" * 50)

print("\nТестова вибірка: \n", X_test, y_test)

# 4th. task. Вручну.

# print("\n\n4. Розділити набір даних на навчальну (тренувальну) та тестову вибірки.")
#
# split_idx = num_records - ((num_records // 2) // 2)
#
# df_train = df.iloc[:split_idx]
# df_test = df.iloc[split_idx:]
#
# X_train = df_train.iloc[:, 2:-1]
# y_train = df_train.iloc[:, -1]
#
# X_test = df_test.iloc[:, 2:-1]
# y_test = df_test.iloc[:, -1]
#
# print("\nНавчальна вибірка: \n", X_train, y_train)
# print("_"*50)
#
# print("\nТестова вибірка: \n", X_test, y_test)

# 5th. task

clf_gini = DecisionTreeClassifier(max_depth=5, random_state=0, criterion="gini")
clf_gini.fit(X_train, y_train)

clf_entropy = DecisionTreeClassifier(max_depth=5, random_state=0, criterion="entropy")
clf_entropy.fit(X_train, y_train)

# 6th. task

visualize_decision_tree(clf_gini, "gini")
visualize_decision_tree(clf_entropy, "entropy")

# 7th. task

get_results(clf_gini, "gini")
get_results(clf_entropy, "entropy")

# 8th. task

max_depth_values = range(1, 12)
min_samples_leaf_values = range(1, 61)

metrics_dict_list = []

for i, max_depth in enumerate(max_depth_values):
    for j, min_samples_leaf in enumerate(min_samples_leaf_values):
        # Create and train model
        clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        clf.fit(X_train, y_train)

        # visualize
        # graph = visualize_decision_tree(clf, to_return=True)
        # graph.render(f"metrics_for_different_tree_params/vis_trees/depth_{max_depth}/decision_tree_depth_{max_depth}_{min_samples_leaf}", format="png", view=False)

        # Predict target values
        y_pred = clf.predict(X_test)

        # Metrics
        metrics_for_pred = get_metrics_for_pred(y_pred, y_test, False)
        metrics_dict_list.append(
            {
                "Max depth": max_depth,
                "Min amount of leaf elements": min_samples_leaf,
                "Metrics":
                    {
                        "Accuracy": metrics_for_pred[0],
                        "Precision": metrics_for_pred[1],
                        "Recall": metrics_for_pred[2],
                        "F1 Score": metrics_for_pred[3],
                        "MCC": metrics_for_pred[4],
                        "Balanced Accuracy": metrics_for_pred[5],
                        "Youden's J": metrics_for_pred[6],
                        "Confusion matrix": metrics_for_pred[7]
                    }
            }
        )

# for d in metrics_dict_list:
#     print(json.dumps(d, indent=4))

for max_depth in max_depth_values:
    plt.figure(figsize=(10, 6))
    plt.xlabel("Мінімальна кількість елементів в листі дерева")
    plt.ylabel("Максимальна глибина дерева")
    plt.title(f"Metrics for different min leaf elements if max depth is fixed. Max depth = {max_depth}")

    metrics_for_depth_lst = list(filter(lambda d: d["Max depth"] == max_depth, metrics_dict_list))
    metrics_for_min_samples_leaf = []
    lst_of_conf_matrix = []
    for d in metrics_for_depth_lst:
        metrics_for_min_samples_leaf.append(list((d["Metrics"].values()))[:-1])
        lst_of_conf_matrix.append(d["Metrics"]["Confusion matrix"])
    metrics_for_min_samples_leaf = np.array(metrics_for_min_samples_leaf)

    # Accuracy
    plt.plot(min_samples_leaf_values, metrics_for_min_samples_leaf[:, 0], label='Accuracy')
    plt.plot(min_samples_leaf_values, metrics_for_min_samples_leaf[:, 1], label='Precision')
    plt.plot(min_samples_leaf_values, metrics_for_min_samples_leaf[:, 2], label='Recall')
    plt.plot(min_samples_leaf_values, metrics_for_min_samples_leaf[:, 3], label='F1 Score')
    plt.plot(min_samples_leaf_values, metrics_for_min_samples_leaf[:, 4], label='MCC')
    plt.plot(min_samples_leaf_values, metrics_for_min_samples_leaf[:, 5], label='Balanced Accuracy')
    plt.plot(min_samples_leaf_values, metrics_for_min_samples_leaf[:, 6], label="Youden's J stats")

    plt.legend()

    # plt.savefig(f'metrics_for_different_tree_params/graphs/max_depth_{max_depth}.png')

    plt.show()

    # for i, matrix in enumerate(lst_of_conf_matrix):
    #     plt.figure(figsize=(6, 6))
    #     plt.title(f"Confusion matrix for max_depth = {max_depth}, min_leaf_samples = {i+1}")
    #
    #     sns.heatmap(np.array(matrix), annot=True, cmap='Blues', fmt='g')
    #     plt.xlabel('Predicted labels')
    #     plt.ylabel('True labels')
    #
    #     plt.savefig(f'metrics_for_different_tree_params/conf_matrices/max_depth_{max_depth}_{i+1}.png')
    #     plt.close()

# 9th. task

get_feature_importances(clf_gini, "gini")
get_feature_importances(clf_entropy, "entropy")
