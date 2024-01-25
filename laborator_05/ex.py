import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import calendar


def load_data(file_path):
    return pd.read_csv(file_path, parse_dates=['Datetime'], dayfirst=True)


def plot_time_series(dateTime, count):
    plt.figure(figsize=(12, 6))
    plt.plot(dateTime, count)
    plt.xlabel('Timp')
    plt.ylabel('Numar de masini')
    plt.grid(True)
    plt.show()


def calculate_sampling_rate(dataset):
    start_time, end_time = dataset['Datetime'].iloc[0], dataset['Datetime'].iloc[-1]
    diff = end_time - start_time
    total_samples = len(dataset)
    sampling_rate = total_samples / diff.total_seconds()
    fs = 1 / (dataset['Datetime'].iloc[1] -
              dataset['Datetime'].iloc[0]).total_seconds()
    print(f'Frecvența de eșantionare este {fs} Hz')


def plot_fft_results(freq, fft_result):
    plt.figure(figsize=(12, 6))
    plt.stem(freq, fft_result)
    plt.xlabel('Frecventa')
    plt.ylabel('Amplitudine')
    plt.title("FFT")
    plt.xlim((0, 0.000004))
    plt.ylim((0, 100))
    plt.grid(True)
    plt.show()


def analyze_continuous_component(freq, fft_result):
    freq_0 = freq[np.argmax(fft_result)]
    componenta_continua = freq_0 == 0
    print(f'Semnalul prezinta o componenta continua: {componenta_continua}')
    print(f'Frecventa componentei: {freq_0} Hz')
    print(f'Valoarea componentei: {np.max(fft_result)}')


def eliminate_continuous_component(fft_result, freq):
    fft_result[np.argmax(fft_result)] = 0
    plot_fft_results(freq, fft_result)


def identify_top_frequencies(freq, fft_result):
    max_index = np.argsort(fft_result)[::-1][:4]
    greatest_values = fft_result[max_index]
    freq_for_greatest_values = freq[max_index]
    for i in range(4):
        print(
            f'Valoare {i + 1}: {greatest_values[i]}, Frecvență: {freq_for_greatest_values[i]} Hz')


def analyze_time_interval(dataset):
    start_index = dataset.loc[dataset.index > 1000,
                              'Datetime'][dataset.Datetime.dt.day_name() == 'Monday'].index[0]
    numar_zile = calendar.monthrange(
        dataset['Datetime'][start_index].year, dataset['Datetime'][start_index].month)[1]
    end_index = start_index + numar_zile * 24  # type: ignore
    print(
        f'Intervalul este {dataset.Datetime[start_index]} - {dataset.Datetime[end_index]}')
    plt.plot(count[start_index:end_index])
    plt.xlabel('Timp')
    plt.ylabel('Numar masini')
    plt.grid(True)
    plt.show()


def filter_frequency_threshold(freq, fft_result):
    thresh_hold = freq.mean() + freq.std()
    freq_copy = freq[freq < thresh_hold]
    fft_result_copy = fft_result[freq < thresh_hold]
    plot_fft_results(freq_copy, fft_result_copy)


def compare_filtered_and_original_signals(signal, filtered_signal):
    plt.plot(np.real(filtered_signal), label='Semnal filtrat')
    plt.plot(np.real(signal), label='Semnal original')
    plt.legend()
    plt.grid(True)
    plt.show()


# Punctul a
dataset = load_data('Train.csv')
dateTime = dataset.Datetime.values.tolist()
count = dataset.Count.values.tolist()
plot_time_series(dateTime, count)

# Punctul b
calculate_sampling_rate(dataset)

# Punctul c
fs = 1 / (dataset['Datetime'].iloc[1] -
          dataset['Datetime'].iloc[0]).total_seconds()
max_frequency = fs / 2
print(f'Frecvența maximă este {max_frequency} Hz')

# Punctul d
N = len(count)
fft_result = np.fft.fft(count)
fft_result = np.abs(fft_result / N)
fft_result = fft_result[:N // 2]
signal = np.fft.ifft(fft_result)
freq = fs * np.linspace(0, N // 2, N // 2) / N
plot_fft_results(freq, fft_result)

# Punctul e
analyze_continuous_component(freq, fft_result)
eliminate_continuous_component(fft_result, freq)

# Punctul f
identify_top_frequencies(freq, fft_result)

# Punctul g
analyze_time_interval(dataset)

# Punctul i
filter_frequency_threshold(freq, fft_result)

filtered_signal = np.fft.ifft(fft_result)
compare_filtered_and_original_signals(signal, filtered_signal)
