import pandas as pd
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

K = 30

# 1st task.
df = pd.read_csv('Top100-2007.csv')

# =====================================================================================================

# 2nd task.
num_records, num_fields = df.shape

print("\n\n2. Визначити та вивести кількість записів та кількість полів у кожному записі.")
print("Кількість записів:", num_records)
print("Кількість полів у кожному записі:", num_fields)

# =====================================================================================================

# 3d task.
print(f"\n\n3.1 Вивести 5 записів, починаючи з {K}-ого.")
print(f"{df.iloc[K-1:K+4]}")

print(f"\n3.2 Вивести 3 * {K} + 2 (= {3 * K + 2}) останніх записів.")
print(f"{df.iloc[-(3 * K + 2):]}")

# =====================================================================================================

# 4th task.
print(f"\n\n4. Визначити та вивести тип полів кожного запису.")
print(df.dtypes)

# =====================================================================================================
# 5th task.
print("\n\n5. Очистити текстові поля від зайвих пробілів. ")

print("\nДо очищення: ")
print(df[['Name', 'Country']].iloc[:5])

df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

print("\nПісля очищення: ")
print(df[['Name', 'Country']].iloc[:5])


# with open("Top100-2007.csv", "r") as file:
#     records_lst = file.readlines()
#
#     # 2nd task
#
#     records_num = len(records_lst) - 1
#     fields_num = len(records_lst[0])
#
#     print("\n2. Визначити та вивести кількість записів та кількість полів у кожному записі. ")
#     print(f"Кількість записів: {records_num}")
#     print(f"Кількість полів: {len(records_lst[0].split(','))}")
#
#     # =====================================================================================================
#
#     # 3d task
#     print(f"\n3.1 Вивести 5 записів, починаючи з {K}-ого.")
#
#     for i in range(K, K + 6):
#         print(records_lst[i].rstrip())
#
#     print("\n" + "_" * 50 + "\n\n")
#     print(f"3.2 Вивести 3 * {K} + 2 (= {3 * K + 2}) останніх записів.")
#
#     start_val = records_num - (3 * K + 2)
#     for i in range(start_val, len(records_lst)):
#         print(records_lst[i].rstrip())
#
#     # =====================================================================================================
#
#     # 4th task
#     print("4. Визначити та вивести тип полів кожного запису. ")
#
