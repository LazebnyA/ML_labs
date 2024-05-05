import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

import seaborn as sns


def get_metrics_for_pred(y_pred, y_actual, to_print=True):
    confusion_matrix_test = metrics.confusion_matrix(y_actual, y_pred)

    accuracy = metrics.accuracy_score(y_actual, y_pred)
    precision = metrics.precision_score(y_actual, y_pred, average='weighted')
    recall = metrics.recall_score(y_actual, y_pred, average='weighted')
    f1_score = metrics.f1_score(y_actual, y_pred, average='weighted')
    mcc = metrics.matthews_corrcoef(y_actual, y_pred)
    balanced_accuracy = metrics.balanced_accuracy_score(y_actual, y_pred)

    if to_print:
        print("Confusion Matrix:\n", confusion_matrix_test)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1_score)
        print("MCC: ", mcc)
        print("Balanced accuracy: ", balanced_accuracy)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score,
        "MCC": mcc,
        "Balanced Accuracy": balanced_accuracy,
        "Confusion Matrix": list(confusion_matrix_test)
    }


# Task 1.
print("\n\n1. Відкрити та зчитати наданий файл з даними.")
df = pd.read_csv('dataset2_l4.txt', header=0)
print(df.head)

# Task 2.
num_records, num_fields = df.shape
print("\n\n2. Визначити та вивести кількість записів та кількість полів у завантаженому наборі даних.")
print("Кількість записів:", num_records)
print("Кількість полів у кожному записі:", num_fields)

# Task 3.
print("\n\n3. Виведення атрибутів набору даних:")
print("Атрибути набору даних:")
print(list(df.columns))

# Task 4.
print("\n\n4. З’ясувати збалансованість набору даних.\n")
class_distribution = df['Class'].value_counts()
print(class_distribution)

print("\nНабір даних не є збалансованим.")

# Task 5.
print("\n\n5. Отримати двадцять варіантів перемішування набору даних \n"
      "та розділення його на навчальну (тренувальну) та тестову вибірки.\n"
      "Сформувати начальну та тестові вибірки на основі обраного користувачем варіанту.\n")

n_splits = 20

shuffle_split = ShuffleSplit(n_splits=n_splits, random_state=32)

variants = []
for train_index, test_index in shuffle_split.split(df):
    train_set = df.iloc[train_index]
    test_set = df.iloc[test_index]
    variants.append((train_set, test_set))

chosen_variant = int(input("Введіть варіант від 1 до 20: "))

train_set, test_set = variants[chosen_variant - 1]

print("\nРозмір навчальної вибірки:", train_set.shape)
print("Навчальна вибірка: \n", train_set.head)

print("\nРозмір тестової вибірки:", test_set.shape)
print("Тестова вибірка: \n", test_set.head)

train_features = train_set.iloc[:, :-1]
train_labels = train_set.iloc[:, -1]

test_features = test_set.iloc[:, :-1]
test_labels = test_set.iloc[:, -1]

# Task 6
print("\n\n6. Збудувати класифікаційну модель на основі методу k найближчих\n"
      "сусідів (кількість сусідів обрати самостійно, вибір аргументувати) та\n"
      "навчити її на тренувальній вибірці.")

# --------------- Знаходимо k за допомогою крос-валідації ---------------- #

# k_values = [i for i in range(1, 10)]
# scores = []
#
# for k in k_values:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     score = cross_val_score(knn, df.iloc[:, :-1], df.iloc[:, -1])
#     scores.append(np.mean(score))
#
# plt.plot(k_values, scores)
# plt.xlabel("k value")
# plt.ylabel("Accuracy")
# plt.show()

# ----------- Маємо два кандидати k=1 і k=3 ------------------- #

# k = 1

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(train_features, train_labels)


# Task 7
print("\n\n7. Обчислити класифікаційні метрики збудованої моделі для тренувальної"
      "та тестової вибірки. Представити результати роботи моделі на тестовій"
      "вибірці графічно.")

y_pred = knn.predict(test_features)

print("\n\nМетрики для моделі 1-го найближчого сусіда для тестової вибірки")
metrics_k_1_test = get_metrics_for_pred(y_pred, test_labels, True)

y_pred = knn.predict(train_features)

print("\n\nМетрики для моделі 1-го найближчого сусіда для тренувальної вибірки")
metrics_k_1_train = get_metrics_for_pred(y_pred, train_labels, True)

# k = 3
#
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(train_features, train_labels)
#
# test_labels_pred = knn.predict(test_features)
#
# print("\n\nМетрики для моделі 3-х найближчих сусідів")
# metrics_k_2 = get_metrics_for_pred(test_labels_pred, test_labels, True)



# --------------- Візуалізація Confusion Matrix -------------------- #

class_labels = list(df["Class"].unique())

plt.figure(figsize=(10, 8))
heatmap_test = sns.heatmap(metrics_k_1_test["Confusion Matrix"], annot=True, fmt='d', cmap='Blues', xticklabels=class_labels,
                           yticklabels=class_labels)

plt.show()
# Для тестової вибірки є помилки.

# ------------- Візуалізація метрик --------------- #


plt.figure(figsize=(10, 6))
plt.bar(list(metrics_k_1_test.keys())[:-1], list(metrics_k_1_test.values())[:-1], color='skyblue')

plt.show()

# --------------- Візуалізація Confusion Matrix -------------------- #

plt.figure(figsize=(10, 8))
heatmap_train = sns.heatmap(metrics_k_1_train["Confusion Matrix"], annot=True, fmt='d', cmap='Blues', xticklabels=class_labels,
                            yticklabels=class_labels)

plt.show()

# ------------- Візуалізація метрик --------------- #

plt.figure(figsize=(10, 6))
plt.bar(list(metrics_k_1_train.keys())[:-1], list(metrics_k_1_train.values())[:-1], color='skyblue')

plt.show()

# Task 8

print('''
8. Обрати алгоритм KDTree та з’ясувати вплив розміру листа (від 20 до
200 з кроком 5) на результати класифікації. Результати представити
графічно.
''')

leaf_sizes = range(20, 201, 5)
knn_metrics = {'leaf_size': list(leaf_sizes), 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}

for leaf_size in leaf_sizes:
    model = KNeighborsClassifier(n_neighbors=1, algorithm='kd_tree', leaf_size=leaf_size)

    model.fit(train_features, train_labels)
    model_pred = model.predict(test_features)

    model_metrics = get_metrics_for_pred(model_pred, test_labels, False)
    knn_metrics['accuracy'].append(model_metrics['Accuracy'])
    knn_metrics['precision'].append(model_metrics['Precision'])
    knn_metrics['recall'].append(model_metrics['Recall'])
    knn_metrics['f1_score'].append(model_metrics['F1 Score'])

    # print(f"\n\n{leaf_size}")
    # print(np.array(model_metrics['Confusion Matrix']))


plt.figure(figsize=(10, 6))

plt.plot(knn_metrics['leaf_size'], knn_metrics['accuracy'], label='Accuracy', alpha=0.5)
plt.plot(knn_metrics['leaf_size'], knn_metrics['precision'], label='Precision', alpha=0.5)
plt.plot(knn_metrics['leaf_size'], knn_metrics['recall'], label='Recall', alpha=0.5)
plt.plot(knn_metrics['leaf_size'], knn_metrics['f1_score'], label='F1 Score', alpha=0.5)

plt.title('Metrics vs Leaf Size')
plt.xlabel('Leaf Size')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.show()

