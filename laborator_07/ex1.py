import numpy as np
import matplotlib.pyplot as plt

# 1 - 1


def func1(n1, n2):
    return np.sin(2 * np.pi * n1 + 3 * np.pi * n2)

# 1 - 2


def func2(n1, n2):
    return np.sin(4 * np.pi * n1) + np.cos(6 * np.pi * n2)

# 1 - 3, 4, 5


def generate_signal_spectrum(dim, coordinates):
    Y = np.zeros((dim, dim), dtype=complex)
    for coord in coordinates:
        Y[coord[0], coord[1]] = 1
    X = np.real(np.fft.ifft2(Y))
    return X, np.abs(np.fft.fftshift(Y)) + 1e-15


def visualize_signal_and_spectrum(title, X, Y):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title)

    img = axs[0].imshow(X)
    axs[0].set_title("Semnalul")
    fig.colorbar(img, ax=axs[0])

    img = axs[1].imshow(Y)
    axs[1].set_title("Spectrul semnalului")
    fig.colorbar(img, ax=axs[1])

    plt.tight_layout()
    plt.show()


def main():
    dim = 512

    # 1 - 1
    n1, n2 = np.meshgrid(range(dim), range(dim))
    X1 = func1(n1, n2)
    Y1 = np.fft.fftshift(np.fft.fft2(X1))
    freq_db1 = 20 * np.log10(abs(Y1))
    visualize_signal_and_spectrum('Functia 1', X1, freq_db1)

    # 1 - 2
    X2 = func2(n1, n2)
    Y2 = np.fft.fftshift(np.fft.fft2(X2))
    freq_db2 = 20 * np.log10(abs(Y2 + 1e-20))
    visualize_signal_and_spectrum('Functia 2', X2, freq_db2)

    # 1 - 3
    coordinates3 = [(0, 5), (0, dim - 5)]
    X3, Y3 = generate_signal_spectrum(dim, coordinates3)
    visualize_signal_and_spectrum('Functia 3', X3, Y3)

    # 1 - 4
    coordinates4 = [(5, 0), (dim - 5, 0)]
    X4, Y4 = generate_signal_spectrum(dim, coordinates4)
    visualize_signal_and_spectrum('Functia 4', X4, Y4)

    # 1 - 5
    coordinates5 = [(5, 5), (dim - 5, dim - 5)]
    X5, Y5 = generate_signal_spectrum(dim, coordinates5)
    visualize_signal_and_spectrum('Functia 5', X5, Y5)


if __name__ == "__main__":
    main()
