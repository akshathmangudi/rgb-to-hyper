import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from typing import List
import tensorflow as tf
from scipy.io import loadmat
from PIL import Image
from model import Generator, Discriminator
from config import HSI_CHANNELS, IMG_WIDTH, IMG_HEIGHT, OUT_DIR_PATH
from tensorflow.keras.optimizers import Adam # type: ignore
import plotly.graph_objects as go # type: ignore
from sklearn.decomposition import PCA

def save_mask_overlay(image_path: str, mask: np.ndarray, output_path: str):
    """
    Overlays the segmentation mask on the original image and saves the result.

    Args:
        image_path (str): Path to the original RGB image.
        mask (np.ndarray): Predicted mask array with shape (height, width, 1).
        output_path (str): Path to save the overlay image.
    """
    # Load the original image
    image = Image.open(image_path).convert("RGB")
    
    # Ensure mask is 2D
    mask = mask.squeeze()
    
    # Resize image to match mask if necessary
    if image.size != (mask.shape[1], mask.shape[0]):
        image = image.resize((mask.shape[1], mask.shape[0]))
    
    # Create a red mask
    mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
    red_mask = Image.new("RGB", mask_image.size, color=(255, 0, 0))
    
    # Composite the red mask onto the original image using the mask
    overlay = Image.composite(red_mask, image, mask_image)
    
    # Blend the original image and the overlay
    blended = Image.blend(image, overlay, alpha=0.5)
    
    # Save the blended image
    blended.save(output_path)

def visualize_false_color_composite(stacked_hsi: np.ndarray, bands: List[int] = [29, 19, 9], figsize=(10, 10), save_path: str = None):
    """
    Creates a false-color composite from specified HSI bands.

    Args:
        stacked_hsi (np.ndarray): Stacked HSI data with shape (height, width, bands).
        bands (List[int]): List of three band indices to use for RGB channels.
        figsize (tuple): Size of the figure.
        save_path (str, optional): Path to save the composite image.

    Returns:
        None
    """
    try:
        if len(bands) != 3:
            logging.error("Three band indices must be provided for RGB channels.")
            return
        
        # Extract the specified bands
        rgb = stacked_hsi[:, :, bands]
        
        # Normalize the bands for display
        rgb_normalized = rgb / np.max(rgb, axis=(0, 1), keepdims=True)
        
        plt.figure(figsize=figsize)
        plt.imshow(rgb_normalized)
        plt.title(f'False-Color Composite (Bands {bands})')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path)
            logging.info(f"False-Color Composite saved to: {save_path}")
        
        plt.show()
    
    except Exception as e:
        logging.error(f"Error in False-Color Composite visualization: {e}")


def visualize_pca_composite(stacked_hsi: np.ndarray, n_components: int = 3, figsize=(10, 10), save_path: str = None):
    """
    Reduces HSI data to specified principal components and visualizes as a single RGB composite image.

    Args:
        stacked_hsi (np.ndarray): Stacked HSI data with shape (height, width, bands).
        n_components (int): Number of principal components to retain (default is 3 for RGB).
        figsize (tuple): Size of the figure.
        save_path (str, optional): Path to save the PCA composite image.

    Returns:
        None
    """
    try:
        height, width, bands = stacked_hsi.shape
        # Reshape the data for PCA
        reshaped_hsi = stacked_hsi.reshape(-1, bands)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(reshaped_hsi)
        
        # Reshape back to image format
        pca_image = principal_components.reshape(height, width, n_components)
        
        # Normalize each component to [0, 1]
        pca_normalized = pca_image - np.min(pca_image, axis=(0, 1), keepdims=True)
        pca_normalized /= np.max(pca_normalized, axis=(0, 1), keepdims=True)
        
        if n_components == 3:
            # Assign each principal component to RGB channels
            rgb_composite = pca_normalized[:, :, :3]
            
            plt.figure(figsize=figsize)
            plt.imshow(rgb_composite)
            plt.title(f'PCA Composite (Top {n_components} Components)')
            plt.axis('off')
            
            if save_path:
                plt.savefig(save_path)
                logging.info(f"PCA Composite saved to: {save_path}")
            
            plt.show()
        else:
            logging.error("n_components must be 3 for RGB visualization.")
    
    except Exception as e:
        logging.error(f"Error in PCA Composite visualization: {e}")

def visualize_pca_3d(stacked_hsi: np.ndarray, n_components: int = 3):
    """
    Creates an interactive 3D scatter plot of the first three PCA components.

    Args:
        stacked_hsi (np.ndarray): Stacked HSI data with shape (height, width, bands).
        n_components (int): Number of principal components to use for 3D visualization.

    Returns:
        None
    """
    try:
        height, width, bands = stacked_hsi.shape
        reshaped_hsi = stacked_hsi.reshape(-1, bands)
        
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(reshaped_hsi)
        
        fig = go.Figure(data=[go.Scatter3d(
            x=principal_components[:, 0],
            y=principal_components[:, 1],
            z=principal_components[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=principal_components[:, 0],  # Color by first principal component
                colorscale='Viridis',
                opacity=0.8
            )
        )])
        
        fig.update_layout(
            title='3D PCA Scatter Plot',
            scene=dict(
                xaxis_title='PC 1',
                yaxis_title='PC 2',
                zaxis_title='PC 3'
            ),
            width=800,
            height=800
        )
        
        fig.show()
    
    except Exception as e:
        logging.error(f"Error in 3D PCA visualization: {e}")

def compile_model(model, learning_rate=1e-4):
    model.compile(optimizer=Adam(learning_rate),
                  loss='binary_crossentropy',  # Change to 'categorical_crossentropy' for multi-class
                  metrics=['accuracy'])
    return model

def load_hsi_images(hsi_dir: str, target_size=(256, 256)) -> np.ndarray:
    """
    Loads all HSI band images from the specified directory and stacks them into a single 3D NumPy array.

    Args:
        hsi_dir (str): Directory containing HSI band images (e.g., .tiff files).
        target_size (tuple): Desired image size as (width, height).

    Returns:
        np.ndarray: Stacked HSI data with shape (height, width, bands).
    """
    try:
        # List and sort all HSI band files
        band_files = sorted([
            f for f in os.listdir(hsi_dir)
            if f.lower().endswith(('.tiff', '.tif'))
        ])
        
        if not band_files:
            logging.error(f"No HSI band files found in directory: {hsi_dir}")
            return np.array([])
        
        bands = []
        for band_file in band_files:
            band_path = os.path.join(hsi_dir, band_file)
            band_image = tiff.imread(band_path)
            
            # Resize if necessary
            if band_image.shape != target_size:
                band_image = np.array(Image.fromarray(band_image).resize(target_size, Image.BILINEAR))
            
            bands.append(band_image)
            logging.debug(f"Loaded band {band_file} with shape {band_image.shape}")
        
        # Stack bands along the third axis to form (height, width, bands)
        stacked_hsi = np.stack(bands, axis=-1)
        logging.info(f"HSI data stacked with shape: {stacked_hsi.shape}")
        return stacked_hsi
    
    except Exception as e:
        logging.error(f"Error loading and stacking HSI images: {e}")
        return np.array([])


def visualize_hsi_stacked(stacked_hsi: np.ndarray, save_path: str = None):
    """
    Visualizes the stacked HSI data as an image with shape (height, width, bands).

    Args:
        stacked_hsi (np.ndarray): Stacked HSI data with shape (height, width, bands).
        save_path (str, optional): Path to save the visualization image. If None, the image is not saved.

    Returns:
        None
    """
    try:
        if stacked_hsi.ndim != 3:
            logging.error(f"HSI data must be a 3D array, got {stacked_hsi.ndim}D array instead.")
            return
        
        height, width, bands = stacked_hsi.shape
        logging.info(f"Visualizing stacked HSI with shape: {stacked_hsi.shape}")
        
        # Optionally, normalize the data for better visualization
        stacked_hsi_normalized = stacked_hsi.astype(float)
        for i in range(bands):
            band_min = np.min(stacked_hsi[:, :, i])
            band_max = np.max(stacked_hsi[:, :, i])
            if band_max - band_min > 0:
                stacked_hsi_normalized[:, :, i] = (stacked_hsi[:, :, i] - band_min) / (band_max - band_min)
            else:
                stacked_hsi_normalized[:, :, i] = 0
            logging.debug(f"Band {i} normalized: min={band_min}, max={band_max}")
        
        # Display the stacked HSI as a multi-channel image
        plt.figure(figsize=(12, 12))
        plt.imshow(stacked_hsi_normalized)
        plt.title('Stacked HSI Image (Height x Width x Bands)')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            logging.info(f"Stacked HSI visualization saved to: {save_path}")
        
        plt.show()
    
    except Exception as e:
        logging.error(f"Error visualizing stacked HSI: {e}")


def visualize_all_hsi_bands(filepath: str, bands: List[int] = None, figsize=(20, 15)) -> np.ndarray:
    """
    Visualizes all HSI bands in a grid and returns the HSI data as a NumPy array.

    Args:
        filepath (str): Path to the HSI TIFF file.
        bands (List[int], optional): Specific band indices to visualize. If None, visualize all bands.
        figsize (tuple): Size of the matplotlib figure.

    Returns:
        np.ndarray: The loaded HSI data with shape (height, width, bands).
    """
    try:
        hsi = tiff.imread(filepath)
        total_bands = hsi.shape[-1]
        
        if bands is None:
            bands = list(range(total_bands))
        else:
            # Validate band indices
            bands = [b for b in bands if 0 <= b < total_bands]
            if not bands:
                logging.error("No valid band indices provided.")
                return np.array([])
        
        num_bands = len(bands)
        cols = 5  # Number of columns in the grid
        rows = num_bands // cols + int(num_bands % cols != 0)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten()
        
        for idx, band in enumerate(bands):
            ax = axes[idx]
            band_data = hsi[:, :, band]
            ax.imshow(band_data, cmap='gray')
            ax.set_title(f'Band {band}')
            ax.axis('off')
        
        # Hide any remaining subplots if num_bands is not a multiple of cols
        for idx in range(num_bands, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        logging.info(f"Displayed {num_bands} bands from {filepath}")
        return hsi
    
    except Exception as e:
        logging.error(f"Error visualizing all HSI bands: {e}")
        return np.array([])


def create_hsi_grid_image(stacked_hsi: np.ndarray, cols: int = 5, figsize=(25, 20), save_path: str = None):
    """
    Creates a single image grid displaying all HSI bands.

    Args:
        stacked_hsi (np.ndarray): Stacked HSI data with shape (height, width, bands).
        cols (int): Number of columns in the grid.
        figsize (tuple): Size of the figure.
        save_path (str, optional): Path to save the grid image. If None, the image is not saved.

    Returns:
        None
    """
    try:
        total_bands = stacked_hsi.shape[-1]
        rows = total_bands // cols + int(total_bands % cols != 0)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten()
        
        for idx in range(total_bands):
            ax = axes[idx]
            band_data = stacked_hsi[:, :, idx]
            ax.imshow(band_data, cmap='gray')
            ax.set_title(f'Band {idx}')
            ax.axis('off')
        
        # Hide remaining axes
        for idx in range(total_bands, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logging.info(f"HSI grid image saved to: {save_path}")
        
        plt.show()
    
    except Exception as e:
        logging.error(f"Error creating HSI grid image: {e}")


def visualize_stacked_hsi(stacked_hsi: np.ndarray, save_path: str = None):
    """
    Visualizes all 31 HSI bands as separate greyscale layers in a single image.
    Each band is displayed as a semi-transparent layer to preserve spatial and spectral data.
    
    Args:
        stacked_hsi (np.ndarray): Stacked HSI data with shape (height, width, bands).
        save_path (str, optional): Path to save the visualization image. If None, the image is not saved.
    
    Returns:
        None
    """
    try:
        if stacked_hsi.ndim != 3:
            logging.error(f"HSI data must be a 3D array, got {stacked_hsi.ndim}D array instead.")
            return
        
        height, width, bands = stacked_hsi.shape
        logging.info(f"Visualizing stacked HSI with shape: {stacked_hsi.shape}")
        
        # Create an empty canvas
        canvas = np.zeros((height, width), dtype=float)
        
        # Normalize each band and add to the canvas with opacity
        opacity = 1.0 / bands  # Distribute opacity across bands
        
        for i in range(bands):
            band = stacked_hsi[:, :, i]
            band_min = band.min()
            band_max = band.max()
            if band_max - band_min > 0:
                band_normalized = (band - band_min) / (band_max - band_min)
            else:
                band_normalized = np.zeros_like(band)
                logging.warning(f"Band {i+1} has constant value. Displaying as black.")
            
            # Accumulate the normalized bands
            canvas += band_normalized * opacity
        
        # Clip the canvas to [0,1]
        canvas = np.clip(canvas, 0, 1)
        
        # Plot the accumulated canvas
        plt.figure(figsize=(8, 8))
        plt.imshow(canvas, cmap='gray')
        plt.title('Stacked HSI Image (All 31 Bands Combined)')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            logging.info(f"Stacked HSI visualization saved to: {save_path}")
        
        plt.show()
    
    except Exception as e:
        logging.error(f"Error visualizing stacked HSI: {e}")

def load_tiff_images(tiff_dir):
    """
    Loads TIFF images from a specified directory.
    This function traverses through the given directory, identifies subdirectories,
    and loads all TIFF images found within those subdirectories into a dictionary.
    The keys of the dictionary are the filenames of the TIFF images, and the values
    are the corresponding image data as NumPy arrays.
    Args:
        tiff_dir (str): The path to the directory containing TIFF images.
    Returns:
        dict: A dictionary where the keys are TIFF filenames and the values are 
              the image data as NumPy arrays.
    """
    tiff_images = {}
    for folder in os.listdir(tiff_dir):
        folder_path = os.path.join(tiff_dir, folder)
        if os.path.isdir(folder_path):
            tiff_files = [f for f in os.listdir(
                folder_path) if f.endswith('.tiff')]
            for tiff_file in tiff_files:
                img = Image.open(os.path.join(folder_path, tiff_file))
                tiff_images[tiff_file] = np.array(img)

    return tiff_images


def map_hsi_to_rgb(hsi_images, band_wavelengths, rgb_images):
    """
    Maps hyperspectral images (HSI) to RGB images based on specified band wavelengths.
    Args:
        hsi_images (dict): A dictionary where keys are filenames and values are hyperspectral image cubes (3D numpy arrays).
        band_wavelengths (dict): A dictionary where keys are filenames and values are 1D numpy arrays of wavelengths corresponding to the bands in the hyperspectral image cubes.
        rgb_images (dict): A dictionary where keys are filenames and values are RGB images (3D numpy arrays) to be used for resizing the mapped HSI images if necessary.
    Returns:
        dict: A dictionary where keys are filenames and values are the mapped RGB images (3D numpy arrays).
    """
    mapped_rgb_images = {}
    rgb_wavelengths = {
        "red": 650,
        "green": 550,
        "blue": 450
    }

    for mat_file, hsi_cube in hsi_images.items():
        wavelengths = band_wavelengths[mat_file]
        red_band_index = np.argmin(
            np.abs(wavelengths - rgb_wavelengths["red"]))
        green_band_index = np.argmin(
            np.abs(wavelengths - rgb_wavelengths["green"]))
        blue_band_index = np.argmin(
            np.abs(wavelengths - rgb_wavelengths["blue"]))

        # Print the indices and corresponding wavelengths for debugging
        print(f"Processing {mat_file}:")
        print(
            f"Red band index: {red_band_index}, Wavelength: {wavelengths[red_band_index]}")
        print(
            f"Green band index: {green_band_index}, Wavelength: {wavelengths[green_band_index]}")
        print(
            f"Blue band index: {blue_band_index}, Wavelength: {wavelengths[blue_band_index]}")

        # Extract RGB bands from the hyperspectral cube
        rgb_from_hsi = hsi_cube[:, :, [
            red_band_index, green_band_index, blue_band_index]]
        print(
            f"RGB from HSI min: {rgb_from_hsi.min()}, max: {rgb_from_hsi.max()}")
        rgb_scaled = np.clip(rgb_from_hsi * 255, 0, 255).astype(np.uint8)

        # Convert to RGB image
        hsi_rgb_image = Image.fromarray(rgb_scaled)
        base_name = mat_file.split('.')[0]
        rgb_file_name = f"{base_name}_clean.png"

        if rgb_file_name in rgb_images:
            rgb_image = Image.fromarray(rgb_images[rgb_file_name])
            # Resize if dimensions do not match
            if hsi_rgb_image.size != rgb_image.size:
                hsi_rgb_image = hsi_rgb_image.resize(
                    rgb_image.size, Image.ANTIALIAS)

            mapped_rgb_images[rgb_file_name] = np.array(hsi_rgb_image)

    return mapped_rgb_images


def check_normalization(hsi_images: dict):
    """
    Checks the normalization of hyperspectral images by printing the minimum and maximum values for each image.

    Args:
        hsi_images (dict): A dictionary where the keys are filenames (str) and the values are hyperspectral images (numpy arrays).

    Returns:
        None
    """
    for fname, img in hsi_images.items():
        min_val = img.min()
        max_val = img.max()
        print(f"{fname}: Min = {min_val}, Max = {max_val}")


import os
import tensorflow as tf
import logging

def load_rgb_images(rgb_path, target_size=(256, 256)):
    """
    Load RGB images from the specified directory.
    
    Args:
        rgb_path (str): Path to RGB images directory.
        target_size (tuple): (width, height) for resizing.
    
    Returns:
        Tuple of (image array, filenames)
    """
    # Get sorted filenames to ensure consistent order
    filenames = sorted([
        f for f in os.listdir(rgb_path)
        if f.endswith('_clean.png') or (f[:-4].isdigit() and f.endswith('.png'))
    ])

    if not filenames:
        raise ValueError(f"No RGB images found in {rgb_path}")

    images = []
    valid_filenames = []

    for filename in filenames:
        try:
            img_path = os.path.join(rgb_path, filename)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
            img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            if img.shape[:2] != target_size:
                logging.warning(f"Image {filename} has incorrect size {img.shape[:2]}, expected {target_size}. Skipping.")
                continue
            images.append(img)
            valid_filenames.append(filename)
        except Exception as e:
            logging.warning(f"Failed to load RGB image {filename}: {str(e)}")
            continue

    if not images:
        raise ValueError(f"No valid RGB images found in {rgb_path} after processing.")

    return tf.convert_to_tensor(images, dtype=tf.float32), valid_filenames

def load_paired_images(rgb_path: str, hsi_path: str):
    """
    Load both RGB and HSI images ensuring perfect correspondence.

    Args:
        rgb_path: Path to RGB images directory
        hsi_path: Path to HSI base directory
        target_size: (width, height) for resizing

    Returns:
        Tuple of (rgb_images, hsi_images) with guaranteed matching order
    """
    # First load RGB images and get filenames
    rgb_images, rgb_filenames = load_rgb_images(rgb_path)

    # Then load HSI images using RGB filenames to maintain order
    hsi_images = load_hsi_images_from_all_folders(hsi_path, rgb_filenames)

    # Verify we have matching counts
    if len(rgb_images) != len(hsi_images):
        raise ValueError(
            f"Mismatch in number of images: {len(rgb_images)} RGB vs {len(hsi_images)} HSI")

    return rgb_images, hsi_images


def load_hsi_images_from_all_folders(base_folder: str, rgb_filenames: List[str],
                                     target_size=(IMG_WIDTH, IMG_HEIGHT)):
    """
    Load HSI images maintaining exact correspondence with RGB images.

    Args:
        base_folder: Path to HSI base directory containing subdirectories
        rgb_filenames: List of RGB filenames to match against
        target_size: (width, height) for resizing

    Returns:
        numpy array of HSI images in same order as RGB files
    """
    all_hsi_images = []
    expected_bands = 31

    # Process each RGB filename to find corresponding HSI folder
    for rgb_filename in rgb_filenames:
        # Extract the base name (e.g., 'ARAD_HS_0001' from 'ARAD_HS_0001_clean.png')
        folder_name = rgb_filename.replace('_clean.png', '')
        folder_path = os.path.join(base_folder, folder_name)

        if not os.path.isdir(folder_path):
            raise ValueError(
                f"HSI folder not found for RGB image {rgb_filename}")

        # Load all TIFF files from the folder
        tiff_files = sorted([f for f in os.listdir(folder_path)
                             if f.lower().endswith(('.tiff', '.tif'))])

        if len(tiff_files) != expected_bands:
            raise ValueError(
                f"Expected {expected_bands} bands but found {len(tiff_files)} "
                f"in folder {folder_name}")

        # Load and process each band
        bands = []
        for tiff_file in tiff_files:
            img_path = os.path.join(folder_path, tiff_file)
            try:
                img = Image.open(img_path)
                img = img.resize(target_size)
                img_array = np.array(img)
                bands.append(img_array)
            except Exception as e:
                raise ValueError(
                    f"Failed to load HSI band {tiff_file}: {str(e)}")

        # Stack bands and reshape to HxWxC
        stacked_image = np.stack(bands)
        stacked_image = stacked_image.transpose(1, 2, 0)  # Reshape to HxWxC
        all_hsi_images.append(stacked_image)

    return np.array(all_hsi_images)


def save_mapped_images(mapped_rgb_images, OUT_DIR_PATHectory):
    """
    Save mapped RGB images to the specified output directory.

    Args:
        mapped_rgb_images (dict): A dictionary where the keys are file names (str) 
                                  and the values are RGB images (numpy arrays).
        OUT_DIR_PATHectory (str): The path to the output directory where images will be saved.

    Returns:
        None
    """
    for rgb_file_name, rgb_image in mapped_rgb_images.items():
        output_path = os.path.join(OUT_DIR_PATHectory, rgb_file_name)
        # Convert to PIL image and save
        pil_image = Image.fromarray(rgb_image)
        pil_image.save(output_path)
        print(f"Saved: {output_path}")


def aggregate_weights():
    """
    Aggregates the weights of multiple checkpoints for a generator and discriminator model.
    This function loads the weights from a list of checkpoint paths, averages them, 
    and then sets the averaged weights back to the generator and discriminator models. 
    Finally, it saves the aggregated weights to a new checkpoint.
    Args:
        None
    Returns:
        None
    """
    ckpt_list: list = ['./checkpoints/abdul_ckpt-2', './checkpoints/akshath_ckpt-2',
                       './checkpoints/harshitha_ckpt-5', './checkpoints/keshihan_ckpt-2',
                       './checkpoints/nihaal_ckpt-2', './checkpoints/hrithik_ckpt-2']

    gen = Generator()
    disc = Discriminator()

    num_ckpt = len(ckpt_list)
    checkpoint = tf.train.Checkpoint(generator=gen, discriminator=disc)

    gen_weights = np.zeros_like(gen.get_weights())
    disc_weights = np.zeros_like(disc.get_weights())

    for path in ckpt_list:
        checkpoint.restore(path).expect_partial()
        gen_weights = np.add(gen_weights, gen.get_weights())
        disc_weights = np.add(disc_weights, disc.get_weights())

    gen_weights = np.divide(gen_weights, num_ckpt)
    disc_weights = np.divide(disc_weights, num_ckpt)

    gen.set_weights(gen_weights)
    disc.set_weights(disc_weights)
    checkpoint.save('checkpoints/global_1ckpt')


def visualize_generated_images(rgb_batch, generated_hsi, hsi_batch, epoch, batch):
    """
    Visualizes and saves a comparison of RGB input images, generated hyperspectral images (HSI), 
    and original HSI as composite images.
    Parameters:
    rgb_batch (tensor): Batch of RGB input images with shape [batch_size, height, width, channels].
    generated_hsi (tensor): Batch of generated HSI images with shape [batch_size, height, width, channels].
    hsi_batch (tensor): Batch of original HSI images with shape [batch_size, height, width, channels].
    epoch (int): Current epoch number.
    batch (int): Current batch number.
    Returns:
    None
    """
    # Assuming generated_hsi and hsi_batch are tensors in [batch_size, height, width, channels]
    fig, axes = plt.subplots(3, len(rgb_batch), figsize=(15, 5))

    for i in range(len(rgb_batch)):
        # Convert each tensor to a numpy array
        rgb_img = rgb_batch[i].numpy()
        gen_img = generated_hsi[i].numpy()
        real_img = hsi_batch[i].numpy()

        # Display RGB input
        axes[0, i].imshow(rgb_img)
        axes[0, i].axis('off')
        axes[0, i].set_title('RGB Input')

        # Display Original HSI as a composite image
        # Average intensity across all channels
        real_img_composite = real_img.mean(axis=-1)
        axes[1, i].imshow(real_img_composite, cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title('Original HSI')

        # Display Generated HSI as a composite image
        # Average intensity across all channels
        gen_img_composite = gen_img.mean(axis=-1)
        axes[2, i].imshow(gen_img_composite, cmap='gray')
        axes[2, i].axis('off')
        axes[2, i].set_title('Generated HSI')

    plt.suptitle(f'Epoch {epoch}, Batch {batch}')

    # Save the plot to a file
    if not os.path.exists(OUT_DIR_PATH):
        print(f"Creating directory: {OUT_DIR_PATH}")
        os.makedirs(OUT_DIR_PATH, exist_ok=True)
    else:
        print(f"Directory already exists: {OUT_DIR_PATH}")

    file_path = os.path.join(OUT_DIR_PATH, f'epoch_{epoch}_batch_{batch}.png')
    print(f"Saving plot to: {file_path}")
    plt.savefig(file_path)
    plt.close()
    print(f"Plot saved successfully.")


def apply_paired_augmentation(rgb_batch, hsi_batch):
    """
    Apply paired data augmentation to batches of RGB and HSI images.
    This function takes batches of RGB and HSI images, applies the same random
    transformations to both images in each pair to ensure that the augmentations
    are synchronized, and returns the augmented batches.
    Parameters:
    rgb_batch (tf.Tensor): A batch of RGB images with shape (batch_size, height, width, channels).
    hsi_batch (tf.Tensor): A batch of HSI images with shape (batch_size, height, width, channels).
    Returns:
    tuple: A tuple containing two tf.Tensors:
        - augmented_rgb_batch: The augmented batch of RGB images.
        - augmented_hsi_batch: The augmented batch of HSI images.
    """
    data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    # Ensure inputs are numpy arrays with correct shape
    rgb_np = rgb_batch.numpy()
    hsi_np = hsi_batch.numpy()

    # Get batch size and shapes
    batch_size = rgb_np.shape[0]
    augmented_rgb_list = []
    augmented_hsi_list = []

    # Process each image pair in the batch separately
    for i in range(batch_size):
        # Get single images
        rgb_img = rgb_np[i]  # Should be (height, width, channels)
        hsi_img = hsi_np[i]  # Should be (height, width, channels)

        # Combine for synchronized augmentation
        combined = np.concatenate([rgb_img, hsi_img], axis=-1)

        # Get and apply random transformation
        transform_parameters = data_gen.get_random_transform(rgb_img.shape)
        augmented_combined = data_gen.apply_transform(
            combined, transform_parameters)

        # Split back
        rgb_channels = rgb_img.shape[-1]
        augmented_rgb = augmented_combined[..., :rgb_channels]
        augmented_hsi = augmented_combined[..., rgb_channels:]

        augmented_rgb_list.append(augmented_rgb)
        augmented_hsi_list.append(augmented_hsi)

    # Stack back into batches
    augmented_rgb_batch = tf.stack(augmented_rgb_list)
    augmented_hsi_batch = tf.stack(augmented_hsi_list)

    # Uncomment the return statement to return the augmented batches
    return augmented_rgb_batch, augmented_hsi_batch


def extract_bands(input_dir: str, output_dir: str):
    """
    Extracts bands from .mat files in the input directory and saves them as TIFF files in the output directory.
    This function processes each .mat file in the specified input directory, extracts the 'cube' data, and saves each band
    of the 'cube' as a separate TIFF file in a corresponding folder within the output directory.
    Args:
        input_dir (str): The directory containing the .mat files to be processed.
        output_dir (str): The directory where the extracted TIFF files will be saved.
    Raises:
        FileNotFoundError: If the input directory does not exist.
        KeyError: If the 'cube' key is not found in a .mat file.
    Example:
        extract_bands('/path/to/input_dir', '/path/to/output_dir')
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    mat_files = [f for f in os.listdir(input_dir) if f.endswith('.mat')]

    for mat_file in mat_files:
        # Full path of the .mat file
        mat_file_path = os.path.join(input_dir, mat_file)

        # Load the .mat file
        mat_data = loadmat(mat_file_path)

        # Extract the 'cube' data
        if 'cube' in mat_data:
            cube_data = mat_data['cube']

            # Create a folder for the current .mat file's bands
            mat_file_name = os.path.splitext(mat_file)[0]
            mat_output_folder = os.path.join(output_dir, mat_file_name)
            os.makedirs(mat_output_folder, exist_ok=True)

            # Loop through each band and save as a TIFF file
            for i in range(cube_data.shape[2]):
                band_data = cube_data[:, :, i]
                tiff_file_path = os.path.join(
                    mat_output_folder, f'{mat_file_name}_band_{i + 1}.tiff')
                tiff.imwrite(tiff_file_path, band_data.astype(
                    np.float32))  # Save the TIFF file
            print(
                f"Extracted {cube_data.shape[2]} bands from {mat_file} and saved as TIFF in {mat_output_folder}.")
        else:
            print(f"'cube' key not found in {mat_file}.")


def save_hsi_image(hsi_image: np.ndarray, filename: str, save_dir: str, expected_channels: int = 31):
    """
    Saves a single HSI image as a multi-band TIFF file.

    Args:
        hsi_image (np.ndarray): HSI image array.
        filename (str): Base filename without extension.
        save_dir (str): Directory to save the TIFF file.
        expected_channels (int): Expected number of spectral channels.
    """
    try:
        # Verify and correct the shape
        print(f"Original HSI Image Shape: {hsi_image.shape}")
        if hsi_image.shape[-1] != expected_channels:
            if hsi_image.shape[0] == expected_channels:
                hsi_image = np.transpose(hsi_image, (1, 2, 0))
                print(f"Transposed HSI Image Shape: {hsi_image.shape}")
            else:
                raise ValueError(f"HSI image has {hsi_image.shape[-1]} channels, expected {expected_channels}.")

        # Define the output file path
        output_path = os.path.join(save_dir, f"{filename}_hsi.tiff")
        
        # Save using tifffile as a multi-band TIFF
        tiff.imwrite(output_path, hsi_image, photometric='minisblack')
        logging.info(f"Saved HSI image: {output_path}")
        print(f"HSI image saved successfully at: {output_path}")
    except Exception as e:
        logging.error(f"Failed to save HSI image for {filename}: {str(e)}")
        print(f"Error: {str(e)}")
        

def load_data(image_dir, mask_dir, img_height=IMG_HEIGHT, img_width=IMG_WIDTH):
    images = []
    masks = []

    image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith(('.tiff', '.tif'))])
    mask_filenames = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.tiff', '.tif'))])

    for img_name, mask_name in zip(image_filenames, mask_filenames):
        # Load RGB Image
        img_path = os.path.join(image_dir, img_name)
        rgb_image = tiff.imread(img_path)  # Shape: (H, W, 3)
        
        # Normalize RGB Image
        rgb_image = rgb_image.astype(np.float32) / 255.0

        # Load HSI Image
        hsi_path = img_path.replace('rgb_micro', 'hsi_micro')  # Adjust if HSI images are in a different directory
        hsi_image = tiff.imread(hsi_path)  # Shape: (H, W, 31)
        
        # Normalize HSI Image
        hsi_image = hsi_image.astype(np.float32) / np.max(hsi_image)

        # Load Mask
        mask_path = os.path.join(mask_dir, mask_name)
        mask = tiff.imread(mask_path)  # Shape: (H, W)
        
        # If mask has multiple channels, convert to single channel
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        # Resize Images and Masks if necessary
        rgb_image = tf.image.resize(rgb_image, [img_height, img_width]).numpy()
        hsi_image = tf.image.resize(hsi_image, [img_height, img_width]).numpy()
        mask = tf.image.resize(mask[..., np.newaxis], [img_height, img_width], method='nearest').numpy()

        images.append(hsi_image)
        masks.append(mask)

    images = np.array(images)
    masks = np.array(masks)

    return images, masks

def generate_and_save_masks(model, input_dir, output_dir, img_height=IMG_HEIGHT, img_width=IMG_WIDTH):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_filenames = sorted([f for f in os.listdir(input_dir) if f.endswith(('.tiff', '.tif'))])
    
    for img_name in image_filenames:
        img_path = os.path.join(input_dir, img_name)
        hsi_image = tiff.imread(img_path)
        hsi_image = hsi_image.astype(np.float32) / np.max(hsi_image)
        hsi_image = tf.image.resize(hsi_image, [img_height, img_width]).numpy()
        hsi_image = np.expand_dims(hsi_image, axis=0)  # Add batch dimension
        
        # Predict mask
        pred_mask = model.predict(hsi_image)[0]
        pred_mask = (pred_mask > 0.5).astype(np.uint8)  # Binarize if using sigmoid
        
        # Save mask
        mask_save_path = os.path.join(output_dir, img_name.replace('.tiff', '_mask.tiff'))
        tiff.imwrite(mask_save_path, pred_mask.squeeze())
        print(f"Saved mask to {mask_save_path}")


def clear_session():
    """
    Clears the TensorFlow session to free up memory.
    """
    tf.keras.backend.clear_session()
