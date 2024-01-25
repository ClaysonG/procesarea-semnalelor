import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt


def generate_time_series():
    np.random.seed(42)
    N = 1000
    t = np.arange(N)
    trend = 0.001 * t**2 + 0.1 * t + 10
    sezon = 5 * np.sin(2 * np.pi * 0.02 * t) + 3 * np.cos(2 * np.pi * 0.05 * t)
    variabilitate = np.random.normal(0, 1, N)
    serie_timp = trend + sezon + variabilitate
    return t, serie_timp, trend, sezon, variabilitate


def plot_time_series(t, serie_timp, trend, sezon, variabilitate):
    plt.figure(figsize=(12, 6))
    plt.subplot(4, 1, 1)
    plt.plot(t, serie_timp, label='Seria de timp')
    plt.legend()
    plt.title('Seria de timp')

    plt.subplot(4, 1, 2)
    plt.plot(t, trend, label='Trend')
    plt.legend()
    plt.title('Trend')

    plt.subplot(4, 1, 3)
    plt.plot(t, sezon, label='Sezon')
    plt.legend()
    plt.title('Sezon')

    plt.subplot(4, 1, 4)
    plt.plot(t, variabilitate, label='Variabilitate mică')
    plt.legend()
    plt.title('Variabilitate mică')

    plt.tight_layout()
    plt.show()


def calculate_autocorrelation(serie_timp):
    autocorrelation = np.correlate(serie_timp, serie_timp, mode='full')
    autocorrelation /= np.max(autocorrelation)
    return autocorrelation


def plot_autocorrelation(autocorrelation):
    plt.figure(figsize=(12, 6))
    plt.plot(autocorrelation, label='Autocorelație')
    plt.title('Vectorul de autocorelație')
    plt.legend()
    plt.show()


def fit_arima_model_and_predict(serie_timp, p=2):
    model = sm.tsa.ARIMA(serie_timp, order=(p, 0, 0))
    result = model.fit(method='yule_walker')
    predictions = result.predict(start=p, end=len(serie_timp) - 1)
    return predictions


def plot_original_and_predictions(serie_timp, predictions, p):
    plt.figure(figsize=(12, 6))
    plt.plot(serie_timp, label='Seria de timp originală')
    plt.plot(predictions, label=f'Predicții AR(p={p})')
    plt.title(f'Model AR(p={p}) și Predicții')
    plt.xlabel('Timp')
    plt.ylabel('Valoare')
    plt.legend()
    plt.show()


def find_best_arima_order(serie_timp, min_p=1, max_p=10):
    best_p = None
    best_rmse = float('inf')

    for p in range(min_p, max_p + 1):
        model = sm.tsa.ARIMA(serie_timp, order=(p, 0, 0))
        result = model.fit(method='yule_walker')
        predictions = result.predict(start=p, end=len(serie_timp) - 1)
        rmse = sqrt(mean_squared_error(serie_timp[p:], predictions))

        if rmse < best_rmse:
            best_rmse = rmse
            best_p = p

    return best_p, best_rmse


def main():
    t, serie_timp, trend, sezon, variabilitate = generate_time_series()
    plot_time_series(t, serie_timp, trend, sezon, variabilitate)

    autocorrelation = calculate_autocorrelation(serie_timp)
    plot_autocorrelation(autocorrelation)

    p = 2
    predictions = fit_arima_model_and_predict(serie_timp, p)
    plot_original_and_predictions(serie_timp, predictions, p)

    best_p, best_rmse = find_best_arima_order(serie_timp)
    print(f"Cel mai bun ordin (p): {best_p}")
    print(f"Eroarea asociată (RMSE): {best_rmse}")


if __name__ == "__main__":
    main()
