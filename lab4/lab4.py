import pandas as pd

from sklearn.model_selection import ShuffleSplit

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

shuffle_split = ShuffleSplit(n_splits=n_splits)

variants = []
for train_index, test_index in shuffle_split.split(df):
    train_set = df.iloc[train_index]
    test_set = df.iloc[test_index]
    variants.append((train_set, test_set))

chosen_variant = int(input("Введіть варіант від 1 до 20: "))

train_set, test_set = variants[chosen_variant-1]

print("\nРозмір навчальної вибірки:", train_set.shape)
print("Навчальна вибірка: \n", train_set.head)

print("\nРозмір тестової вибірки:", test_set.shape)
print("Тестова вибірка: \n", test_set.head)


