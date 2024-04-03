import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz

import graphviz

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

clf = DecisionTreeClassifier(max_depth=5, random_state=0)
clf.fit(X_train, y_train)

# 6th. task

dot_data = export_graphviz(clf, class_names=['class_0', 'class_1'], feature_names=X.columns, impurity=False, filled=True)

graph = graphviz.Source(dot_data)

graph.render("decision_tree", format="png", view=True)
