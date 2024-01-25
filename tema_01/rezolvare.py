import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2ycbcr, ycbcr2rgb
from skimage.util import img_as_float
from sklearn.metrics import mean_squared_error
from scipy import datasets
from scipy.fft import dctn, idctn
from typing import List

# Constants
block_size = 8
Q_jpeg = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 28, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float64)


def show_images(images: List[np.ndarray], titles: List[str], cmap=plt.cm.gray) -> None:  # type: ignore
    """
    Display a grid of images with corresponding titles.

    Parameters:
    - images (List[np.ndarray]): List of images to be displayed.
    - titles (List[str]): List of titles corresponding to each image.
    - cmap: Colormap for displaying the images. Default is plt.cm.gray.

    Raises:
    - ValueError: If the number of images does not match the number of titles.

    Returns:
    - None
    """

    num_images = len(images)

    if num_images != len(titles):
        raise ValueError(
            "Number of images must match number of titles.")

    plt.figure(figsize=(10, 5))

    for i in range(num_images):
        plt.subplot(1, 3, i + 1).imshow(images[i], cmap=cmap)
        plt.title(titles[i])

    plt.tight_layout()
    plt.show()


def compress_jpeg(image: np.ndarray, Q_matrix: np.ndarray) -> np.ndarray:
    """
    Compress an image using the JPEG algorithm.

    Parameters:
    - image (np.ndarray): Input image to be compressed.
    - Q_matrix (np.ndarray): Quantization matrix used for compression.

    Returns:
    - np.ndarray: Compressed image.

    Notes:
    - This function uses the discrete cosine transform (DCT) for compression.
    - The input image should have dimensions divisible by the block size (8x8).
    - The Q_matrix determines the level of quantization applied to the DCT coefficients.
    """

    rows, cols = image.shape[:2]

    jpeg_img = np.zeros_like(image)
    # Iterate over image blocks
    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            # Extract block from image
            block = image[i: i + block_size, j: j + block_size]
            # Apply DCT to block
            dct_block = dctn(block)
            # Quantize DCT coefficients
            quantized_block = Q_matrix * np.round(dct_block / Q_matrix)
            # Apply inverse DCT to quantized block
            jpeg_img[i: i + block_size, j: j +
                     block_size] = idctn(quantized_block)

    return jpeg_img


def compress_ycbcr(image: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Compress an RGB image in the YCbCr color space using the JPEG algorithm.

    Parameters:
    - image (np.ndarray): Input RGB image to be compressed.
    - Q (np.ndarray): Quantization matrix used for compression.

    Returns:
    - np.ndarray: Compressed image in YCbCr color space.

    Notes:
    - This function converts the input RGB image to the YCbCr color space before compression.
    - The input image and quantization matrix should have dimensions divisible by the block size (8x8).
    - The Q matrix determines the level of quantization applied to the DCT coefficients in each channel.
    """

    # Convert RGB image to YCbCr color space
    image = rgb2ycbcr(image)
    # Initialize compressed image
    compressed_img = np.zeros_like(image)

    # Iterate over channels
    for channel in range(3):
        # Extract the dimensions
        rows, cols = image[:, :, channel].shape
        # Iterate over image blocks
        for i in range(0, rows, block_size):
            for j in range(0, cols, block_size):
                # Extract block from image
                block = image[i: i + block_size, j: j + block_size, channel]
                # Apply DCT to block
                dct_block = dctn(block)
                # Quantize DCT coefficients
                quantized_block = Q * np.round(dct_block / Q)
                # Apply inverse DCT to quantized block
                compressed_img[i: i + block_size, j: j +
                               block_size, channel] = idctn(quantized_block)

    return compressed_img


def sarcina_1():
    """
    Completati algoritmul JPEG incluzand toate blocurile din imagine

    Parameters:
    - None

    Returns:
    - None
    """

    # Load image
    image = datasets.ascent()

    # Perform JPEG compression
    jpeg_image = compress_jpeg(image, Q_jpeg)

    # Display original and compressed images
    show_images(images=[image, jpeg_image], titles=[
                'Originala', 'Comprimata'])


def sarcina_2():
    """
    Extindeti la imagini color (incluzand transformarea din RGB in Y'CbCr)

    Parameters:
    - None

    Returns:
    - None
    """

    # Load image
    image_rgb = datasets.face()

    # Perform Y'CbCr compression
    compressed_ycbcr = compress_ycbcr(image_rgb, Q_jpeg)

    # Display original and compressed images
    show_images(
        images=[image_rgb, np.clip(compressed_ycbcr, 0, 255).astype(
            np.uint8), np.clip(ycbcr2rgb(compressed_ycbcr), 0, 1)],
        titles=['Originala', 'Comprimata Y\'CbCr',
                'Comprimata RGB'],
        cmap=plt.cm.viridis  # type: ignore
    )


def sarcina_3():
    """
    Extindeti algoritmul pentru compresia imaginii pana la un prag MSE impus de utilizator

    Parameters:
    - None

    Returns:
    - None
    """

    # Load image
    image_rgb = datasets.face()

    target_mse = 0.01
    scale = 1
    mse = 0
    compressed_ycbcr = np.copy(image_rgb)

    #  Compress image until MSE is below target
    while mse < target_mse:  # type: ignore
        # Scale quantization matrix
        Q_scaled = Q_jpeg * scale
        # Compress image
        compressed_ycbcr = compress_ycbcr(image_rgb, Q_scaled)
        # Compute MSE
        mse = mean_squared_error(img_as_float(
            image_rgb.flatten()), ycbcr2rgb(compressed_ycbcr).flatten())
        print(f"MSE: {mse}")
        # Increase scale
        scale += 50

    # Display original and compressed images
    show_images(
        images=[image_rgb, np.clip(compressed_ycbcr, 0, 255).astype(
            np.uint8), np.clip(ycbcr2rgb(compressed_ycbcr), 0, 1)],
        titles=['Originală', 'Comprimată Y\'CbCr',
                'Comprimată RGB'],
        cmap=plt.cm.viridis  # type: ignore
    )


def sarcina_4():
    """
    Extindeti algoritmul pentru compresie video

    Parameters:
    - None

    Returns:
    - None
    """

    # Set the path to the video file
    video_path = os.path.join(os.getcwd(), 'tema_01', 'video.mp4')

    # Create a video reader
    reader = imageio.get_reader(video_path)

    for i, image_rgb in enumerate(reader):  # type: ignore
        compressed_ycbcr = compress_ycbcr(image_rgb, Q_jpeg)

        # Display every 10th frame
        if i % 10 == 0:
            # Display original and compressed images
            show_images(
                images=[image_rgb, np.clip(compressed_ycbcr, 0, 255).astype(
                    np.uint8), np.clip(ycbcr2rgb(compressed_ycbcr), 0, 1)],
                titles=['Originala',
                        'Comprimata Y\'CbCr', 'Comprimata RGB'],
                cmap=plt.cm.viridis  # type: ignore
            )

    reader.close()


if __name__ == "__main__":
    sarcina_1()
    sarcina_2()
    sarcina_3()
    sarcina_4()
