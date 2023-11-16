import matplotlib.pyplot as plt
import pandas as pd

# Pct a
dataset = pd.read_csv('Train.csv', parse_dates=['Datetime'], dayfirst=True)

dateTime = dataset.Datetime.values.tolist()
count = dataset.Count.values.tolist()

plt.figure(figsize=(12, 6))
plt.plot(dateTime, count)
plt.xlabel('Timp')
plt.ylabel('Numar de masini')
plt.grid(True)
plt.show()

start_time = dataset['Datetime'].iloc[0]
end_time = dataset['Datetime'].iloc[-1]
diff = end_time - start_time

total_samples = len(dataset)
sampling_rate = total_samples / diff.total_seconds()

# Pentru acest set de date fs este:
fs = 1 / (dataset['Datetime'].iloc[1] -
          dataset['Datetime'].iloc[0]).total_seconds()
print(f'Frecvența de eșantionare este {fs} Hz')
