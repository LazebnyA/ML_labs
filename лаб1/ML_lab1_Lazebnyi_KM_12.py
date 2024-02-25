import pandas as pd
import matplotlib.pyplot as plt

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
print(df.iloc[K - 1:K + 4].to_string())

print(f"\n3.2 Вивести 3 * {K} + 2 (= {3 * K + 2}) останніх записів.")
print(df.iloc[-(3 * K + 2):].to_string())

# =====================================================================================================

# 4th task.
print(f"\n\n4. Визначити та вивести тип полів кожного запису.")
print(df.dtypes.to_string())

# =====================================================================================================

# 5th task.
print("\n\n5. Очистити текстові поля від зайвих пробілів. ")

print("\nДо очищення: ")
print(df[['Name', 'Country']].iloc[:5].to_string())

df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

print("\nПісля очищення: ")
print(df[['Name', 'Country']].iloc[:5].to_string())

# =====================================================================================================

# 6th task. Fields: Winning Percentage -> float, Career Earnings -> int (Singles Record (Career) after task 8 -> int)

print("\n\n6. Визначити поля, які потрібно привести до числового вигляду та зробити це (продемонструвати "
      "підтвердження). ")

df['Winning Percentage'] = df['Winning Percentage'].map(lambda x: float(x.rstrip('%')) if isinstance(x, str) else x)  # removing % from the end of a string & casting each cell to the float type

print("\nПоле 'Winning Percentage' було переведено до числового вигляду. Попередньо був видалений знак '%' в кінці.")
print(df['Winning Percentage'])

df['Career Earnings'] = df['Career Earnings'].map(lambda x: int(x.lstrip('$')) if isinstance(x, str) else x)  # removing $ from the start of a string & casting each cell to the int type

print("\nПоле 'Career Earnings' було переведено до числового вигляду. Попередньо був видалений знак '$' в кінці.")
print(df['Career Earnings'])

# =====================================================================================================

# 7th task.

print("\n\n7. Визначити записи із пропущеними даними та вивести їх на екран, після чого видалити з датафрейму. ")

print(f"\nЗаписи із пропущеними даними: \n{df[df.isnull().any(axis=1)].to_string()}")

df = df.dropna(how='any')

print(f"\nЗаписи із пропущеними даними після очищення: \n{df[df.isnull().any(axis=1)].to_string()}")

# =====================================================================================================

# 8th task.

print("\n\n8. На основі поля Singles Record (Career) ввести нові поля: "
      "\n\ta. Загальна кількість зіграних матчів Total; "
      "\n\tb. Кількість виграних матчів Win; "
      "\n\tc. Кількість програних матчів Lose.")


df['Win'] = df['Singles Record (Career)'].map(lambda x: int(x.split('-')[0]) if isinstance(x, str) else x)
df['Lose'] = df['Singles Record (Career)'].map(lambda x: int(x.split('-')[1]) if isinstance(x, str) else x)
df['Total'] = df['Win'] + df['Lose']

print(f"\nВведені поля мають вигляд: \n{df[['Win', 'Lose', 'Total']]}")

# =====================================================================================================

# 9th task.

print("\n\n9. Видалити з датафрейму поля Singles Record (Career) та Link to Wikipedia. ")

df = df.drop(columns=['Singles Record (Career)', 'Link to Wikipedia'])

print(f"\nСписок полів після видалення: \n{list(df.columns)}")

# =====================================================================================================

# 10th task.

print("\n\n10. Змінити порядок розташування полів таким чином: Rank, Name, Country, Pts, Total, Win, Lose, "
      "Winning Percentage.")

df = df.reindex(columns=['Rank', 'Name', 'Country', 'Pts', 'Total', 'Win', 'Lose', 'Winning Percentage', 'Career '
                                                                                                         'Earnings'])

print(f"\nСписок полів після зміни порядку: \n{list(df.columns)}")

# =====================================================================================================

# 11th task.

print("\n\n11. Визначити та вивести:"
      "\n\ta. Відсортований за абеткою перелік країн, тенісисти з яких входять у Топ-100;"
      "\n\tb. Гравця та кількість його очок із найменшою сумою призових; "
      "\n\tc. Гравців та країну, яку вони представляють, кількість виграних матчів у яких дорівнює кількості "
      "програних. ")

print(f"\na. \n{sorted(df['Country'].unique())}")

min_earnings_idx = df['Career Earnings'].idxmin()
print(f"\nb. \n{df[['Name', 'Pts']].loc[min_earnings_idx].to_string()}")

same_wins_loses_records = df[(df['Win'] == df['Lose'])]
print(f"\nc. \n{same_wins_loses_records.to_string()}")

# =====================================================================================================

# 12th task.

print("\n\n12. Визначити та вивести: "
      "\n\ta. Кількість тенісистів з кожної країни у Топ-100;"
      "\n\tb. Середній рейтинг тенісистів з кожної країни. ")

print(f"\na. \n{df.groupby('Country').size().reset_index(name='Quantity of players by country')}")

print(f"\nb. \n{df.groupby('Country')['Rank'].mean().reset_index(name='Mean rating by country')}")

# =====================================================================================================

# 13th task.

print("\n\n13. Побудувати діаграму кількості програних матчів по кожній десятці гравців з Топ-100.")

categories_labels = ['1-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
categories_by_10 = pd.cut(df['Rank'], bins=[i for i in range(0, 101, 10)], labels=categories_labels)

grouped_by_loses_df = df.groupby(categories_by_10, observed=True)['Lose'].sum().reset_index(name='Total Matches Lost')

print(f"\nДатафрейм на основі якого будувалась діаграма: \n{grouped_by_loses_df.to_string()}")

plt.bar(grouped_by_loses_df['Rank'], grouped_by_loses_df['Total Matches Lost'])
plt.xlabel("Десятка гравців")
plt.ylabel("Кількість програних матчів")
plt.show()

# =====================================================================================================

# 14th task.

print("\n\n14. Побудувати кругову діаграму сумарної величини призових для кожної країни. ")

countries_by_earnings = df.groupby(df['Country'])['Career Earnings'].sum().reset_index(name='Total Earnings by country')

print(f"\nДатафрейм на основі якого будувалась кругова діаграма: \n{countries_by_earnings.to_string()}")

plt.pie(countries_by_earnings['Total Earnings by country'], labels=countries_by_earnings['Country'], textprops={'fontsize': 6})
plt.show()

# =====================================================================================================

# 15th task.

print("\n\n15. Побудувати на одному графіку (тип графіка обрати самостійно!):"
      "\n\ta. Середню кількість очок для кожної країни;"
      "\n\tb. Середню кількість зіграних матчів тенісистами кожної країни.")

mean_pts_by_country = df.groupby(df['Country'])['Pts'].mean().reset_index(name='Mean points by country')
print(f"\nДатафрейм на основі якого будувався графік а. : \n{mean_pts_by_country.to_string()}")

mean_played_matches_by_country = df.groupby(df['Country'])['Total'].mean().reset_index(name='Mean quantity of played matches by country')
print(f"\nДатафрейм на основі якого будувався графік b. : \n{mean_played_matches_by_country.to_string()}")

plt.figure(figsize=(10, 6))
plt.plot(mean_pts_by_country['Country'], mean_pts_by_country['Mean points by country'], color='blue', marker='o', label='Mean points')

plt.plot(mean_played_matches_by_country['Country'], mean_played_matches_by_country['Mean quantity of played matches by country'],
         color='red', marker='o', label='Mean played matches')

plt.xlabel('Country')
plt.ylabel('Mean Value')
plt.title('Mean Points and Mean Played Matches by Country')
plt.xticks(rotation=90)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()


