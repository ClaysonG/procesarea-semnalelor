import numpy as np
import matplotlib.pyplot as plt


def generate_trend(time_points, a=2, b=1, c=10):
    return a * np.sin(2 * np.pi * b * time_points) + c


def generate_seasonal_pattern(time_points, freq1=5, freq2=3):
    return 3 * np.cos(2 * np.pi * freq1 * time_points) + np.sin(4 * np.pi * freq2 * time_points)


def generate_noise(time_points, scale=0.2):
    return np.random.normal(0, scale, len(time_points))


def generate_time_series(time_points):
    trend_series = generate_trend(time_points)
    seasonal_pattern = generate_seasonal_pattern(time_points)
    noise = generate_noise(time_points)
    return trend_series + seasonal_pattern + noise


def plot_signals(time_points, signals, plot_titles, line_colors):
    fig, axs = plt.subplots(len(signals), 1)
    for i, signal in enumerate(signals):
        axs[i].plot(time_points, signal, color=line_colors[i])
        axs[i].set_title(plot_titles[i])
        axs[i].grid()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    num_points = 1000
    time_points = np.linspace(0, 1, num_points)

    generated_signals = [
        generate_time_series(time_points),
        generate_trend(time_points),
        generate_seasonal_pattern(time_points),
        generate_noise(time_points)
    ]

    plot_titles = [
        "Seria de timp generata",
        "Trend",
        "Seasonal Pattern",
        "Noise"
    ]
    line_colors = ['purple', 'orange', 'brown', 'gray']

    plot_signals(time_points, generated_signals, plot_titles, line_colors)
