import numpy as np
import matplotlib.pyplot as plt


def rectangular_window(N):
    return np.ones(N)


def hanning_window(N):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / (N - 1))


def sinusoid(A, f, t):
    return A * np.sin(2 * np.pi * f * t)


def main():
    f = 100
    fs = 1000
    N = 200
    t = np.arange(N) / fs

    # Generate the original sinusoidal signal
    original_signal = sinusoid(1, f, t)

    # Apply rectangular window to the signal
    rectangular_window_signal = original_signal * rectangular_window(N)

    # Apply Hanning window to the signal
    hanning_window_signal = original_signal * hanning_window(N)

    # Plotting signals
    fig, axs = plt.subplots(3, 1, figsize=(10, 7))
    axs[0].plot(original_signal)
    axs[0].set_title('Semnalul original')

    axs[1].plot(rectangular_window_signal, color='green')
    axs[1].set_title('Semnalul cu fereastra dreptunghiulara')

    axs[2].plot(hanning_window_signal, color='red')
    axs[2].set_title('Semnalul cu fereastra Hanning')

    for ax in axs:
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    # Comparison of windows

    plt.plot(t, rectangular_window_signal,
             label='Sinusoida cu fereastra dreptunghiulara', color='green')
    plt.plot(t, hanning_window_signal,
             label='Sinusoida cu fereastra Hanning', color='red')
    plt.title('Comparare ferestre')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
