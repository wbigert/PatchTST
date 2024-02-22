import pandas as pd

csv_data = pd.read_csv('./data/weather.csv')
column_names = csv_data.columns

to_remove = ['wv (m/s)', 'rain (mm)', 'wd (deg)', 'raining (s)']
column_names = column_names.drop(to_remove)
column_names_list = column_names.tolist()
csv_data = csv_data.drop(columns=to_remove)

print(column_names_list)
print(csv_data.head())

# save to csv ./data/weather_trimmed.csv
csv_data.to_csv('./data/weather_trimmed.csv', index=False)