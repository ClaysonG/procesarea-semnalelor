import numpy as np
import matplotlib.pyplot as plt
from scipy import misc


def high_freq_attenuation(image, freq_cutoff):
    fft_image = np.fft.fft2(image)
    fft_image_cutoff = fft_image.copy()

    freq_db = 20 * np.log10(abs(fft_image + 1e-15))
    fft_image_cutoff[freq_db > freq_cutoff] = 0

    return np.real(np.fft.ifft2(fft_image_cutoff))


def calculate_snr(original, compressed):
    return np.sum(original ** 2) / np.sum(np.abs(original - compressed) ** 2)


def main():
    # Load the original image
    original_image = misc.face(gray=True)

    # Copy the original image for compression
    compressed_image = original_image.copy()

    # Set initial SNR to infinity and threshold
    snr = np.inf
    snr_threshold = 0.0072

    # Initial frequency cutoff
    freq_cutoff = 300

    # Iterate until SNR reaches the threshold
    while snr > snr_threshold:
        compressed_image = high_freq_attenuation(compressed_image, freq_cutoff)
        snr = calculate_snr(original_image, compressed_image)
        freq_cutoff -= 10
        print(snr)

    # Compute FFTs of the original and compressed images
    fft_original = np.fft.fft2(original_image)
    fft_compressed = np.fft.fft2(compressed_image)

    # Plotting
    fig, axs = plt.subplots(2, 2)
    freq_db_original = 20 * np.log10(abs(fft_original + 1e-15))
    freq_db_compressed = 20 * np.log10(abs(fft_compressed + 1e-15))

    # Original Image and Spectrum
    axs[0, 0].imshow(original_image, cmap='gray')
    axs[0, 0].set_title("Original Image")
    axs[1, 0].imshow(np.fft.fftshift(freq_db_original), cmap='gray')
    axs[1, 0].set_title("Spectrum of Original Image")

    # Compressed Image and Spectrum
    axs[0, 1].imshow(compressed_image, cmap='gray')
    axs[0, 1].set_title("Compressed Image")
    axs[1, 1].imshow(np.fft.fftshift(freq_db_compressed), cmap='gray')
    axs[1, 1].set_title("Spectrum of Compressed Image")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
