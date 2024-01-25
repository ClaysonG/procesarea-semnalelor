import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig


def compute_polynomial_product_convolution(p, q):
    return sig.convolve(p, q)


def compute_polynomial_product_fft(p, q, n):
    return np.real(np.fft.ifft(np.fft.fft(p, n) * np.fft.fft(q, n)))


def main():
    N = 100

    # Generate random coefficients for polynomials
    p_coef = np.random.randint(0, 10, size=N + 1)
    q_coef = np.random.randint(0, 10, size=N + 1)

    # Create polynomials
    p = np.poly1d(p_coef)
    q = np.poly1d(q_coef)

    # Compute polynomial product using convolution
    convolution_result = compute_polynomial_product_convolution(p, q)

    # Compute polynomial product using FFT
    fft_result = compute_polynomial_product_fft(p, q, n=2 * N + 1)

    # Plotting results
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(convolution_result)
    axs[0].set_title('Produsul folosind convolutia: inmultirea polinoamelor')

    axs[1].plot(fft_result)
    axs[1].set_title('Produsul folosind convolutia: fft')

    for ax in axs:
        ax.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
