import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy.ndimage import gaussian_filter


def add_pixel_noise(image, pixel_noise):
    noise = np.random.randint(-pixel_noise,
                              high=pixel_noise+1, size=image.shape)
    return image + noise


def main():
    # Load the original image
    original_image = misc.face(gray=True)

    # Set pixel noise level
    pixel_noise = 300

    # Add pixel noise to the original image
    noisy_image = add_pixel_noise(original_image, pixel_noise)

    # Denoise the noisy image using Gaussian filter
    denoised_image = gaussian_filter(noisy_image, sigma=1)

    # Print SNR before and after denoising
    snr_before = np.sum(original_image**2) / \
        np.sum((original_image - noisy_image)**2)
    snr_after = np.sum(original_image**2) / \
        np.sum((original_image - denoised_image)**2)  # type: ignore
    print(f"SNR before denoising: {snr_before}")
    print(f"SNR after denoising: {snr_after}")

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(10, 6))
    axs[0].imshow(original_image, cmap=plt.cm.gray)  # type: ignore
    axs[0].set_title("Original Image")

    axs[1].imshow(noisy_image, cmap=plt.cm.gray)  # type: ignore
    axs[1].set_title("Noisy Image")

    axs[2].imshow(denoised_image, cmap=plt.cm.gray)  # type: ignore
    axs[2].set_title("Denoised Image")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
