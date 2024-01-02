import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import calendar

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

# Pct b
print(f'Intervalul de timp acoperit de setul de date este {diff}')

# Pct c
max_frequency = fs / 2
print(f'Frecvența maximă este {max_frequency} Hz')

# Pct d
N = len(count)
fft_result = np.fft.fft(count)
fft_result = np.abs(fft_result / N)
fft_result = fft_result[:N // 2]
signal = np.fft.ifft(fft_result)

freq = fs * np.linspace(0, N // 2, N // 2) / N

plt.figure(figsize=(12, 6))
plt.stem(freq, fft_result)
plt.xlabel('Frecventa')
plt.ylabel('Amplitudine')
plt.title("FFT")
plt.xlim((0, 0.00008))
plt.ylim((0, 140))
plt.grid(True)

plt.show()

# Pct e
# Semnalul prezinta o componenta continua pentru frecventa 0 Hz
freq_0 = freq[np.argmax(fft_result)]
componenta_continua = freq_0 == 0
print(
    f'Semnalul prezinta o componenta continua? {componenta_continua} la frecventa {freq_0} Hz si anume {np.max(fft_result)}')

fft_result[np.argmax(fft_result)] = 0

plt.figure(figsize=(12, 6))
plt.stem(freq, fft_result)
plt.xlabel('Frecventa')
plt.ylabel('Amplitudine')
plt.title("FFT")
plt.xlim((0, 0.00008))
plt.ylim((0, 140))
plt.grid(True)

plt.show()

# Pct f
max_index = np.argsort(fft_result)[::-1][:4]

greatest_values = fft_result[max_index]
freq_for_greatest_values = freq[max_index]

for i in range(4):
    print(
        f'Valoare {i + 1}: {greatest_values[i]}, Frecvență: {freq_for_greatest_values[i]} Hz')

# Pct g

start_index = dataset.loc[dataset.index > 1000,
                          'Datetime'][dataset.Datetime.dt.day_name() == 'Monday'].index[0]
numar_zile = calendar.monthrange(
    dataset['Datetime'][start_index].year, dataset['Datetime'][start_index].month)[1]
end_index = start_index + numar_zile * 24  # type: ignore
print(
    f'Intervalul ales este {dataset.Datetime[start_index]} - {dataset.Datetime[end_index]}')

plt.figure(figsize=(15, 6))
plt.plot(count[start_index:end_index])
plt.xlabel('Timp')
plt.ylabel('Numar masini')
plt.grid(True)

plt.show()

# Pct i
thresh_hold = freq.mean() + freq.std()

freq_copy = freq[freq < thresh_hold]
fft_result_copy = fft_result[freq < thresh_hold]

plt.figure(figsize=(12, 6))
plt.stem(freq_copy, fft_result_copy)
plt.xlabel('Frecventa')
plt.ylabel('Amplitudine')
plt.grid(True)

plt.show()

filtered_signal = np.fft.ifft(fft_result)
plt.figure(figsize=(12, 6))
plt.plot(np.real(filtered_signal), label='Semnal filtrat')
plt.plot(np.real(signal), label='Semnal original')
plt.legend()
plt.grid(True)

plt.show()
