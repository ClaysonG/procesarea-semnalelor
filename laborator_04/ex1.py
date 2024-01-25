import numpy as np
import matplotlib.pyplot as plt
import time


def calculate_dft(x):
    """Calculate the Discrete Fourier Transform (DFT) of a signal."""
    n = len(x)
    X = np.zeros(n, dtype=np.complex128)

    # Compute the DFT using the formula
    for m in range(n):
        for i in range(n):
            X[m] += x[i] * np.exp(-2j * np.pi * m * i / n)

    return X


def measure_execution_time(func, *args):
    """Measure the execution time of a function."""
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time


# Signal sizes to test
signal_sizes = [128, 256, 512, 1024, 2048, 4096, 8192]

# Lists to store execution times for DFT and FFT
dft_times = []
fft_times = []

# Test each signal size
for size in signal_sizes:
    # Generate a sinusoidal signal
    time_values = np.linspace(0, 1, size)
    signal = np.sin(2 * np.pi * 5 * time_values)

    # Measure DFT execution time
    _, dft_time = measure_execution_time(calculate_dft, signal)
    dft_times.append(dft_time)

    # Measure FFT execution time using NumPy's built-in function
    _, fft_time = measure_execution_time(np.fft.fft, signal)
    fft_times.append(fft_time)

# Plotting the results
plt.plot(signal_sizes, dft_times, label='DFT Time')
plt.plot(signal_sizes, fft_times, label='FFT Time')
plt.xlabel('Signal Size')
plt.ylabel('Execution Time')
plt.yscale('log')  # Logarithmic scale on the y-axis
plt.legend()
plt.grid()

# Display the plot
plt.show()
