## Implementing Single Image Segmentation Prediction in `unet_main.py`

#To perform segmentation overlay on a single RGB image using your trained U-Net model, follow the steps below. This involves modifying `unet_main.py` to accept a single image input, perform prediction, and overlay the segmentation mask on the original image.

### **Step 1: Update `load_model_and_predict` Function in `main.py`**

#Ensure that the `load_model_and_predict` function can handle both single images and batches. Modify it to accept an optional parameter for single image paths.

import tensorflow as tf
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from config import CHECKPOINT_DIR
import tifffile as tiff
from PIL import Image
from unet_model import build_unet  # Ensure this imports the U-Net model correctly
from utils import load_rgb_images, save_mask_overlay
from typing import List, Tuple
import cv2
from sklearn.decomposition import PCA
import matplotlib.cm as cm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data(img_dir: str, mask_dir: str, img_height: int = 256, img_width: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads and preprocesses images and their corresponding masks.

    Args:
        img_dir (str): Directory containing HSI images.
        mask_dir (str): Directory containing mask images.
        img_height (int): Desired image height after resizing.
        img_width (int): Desired image width after resizing.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of NumPy arrays containing images and masks.
    """
    try:
        # Supported image and mask extensions
        img_extensions = ('.tiff', '.tif', '.png', '.jpg', '.jpeg')
        mask_extensions = ('.tiff', '.tif', '.png', '.jpg', '.jpeg')

        # List all image and mask files
        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(img_extensions)]
        mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(mask_extensions)]

        logging.info(f"Found {len(img_files)} HSI images and {len(mask_files)} mask images.")

        # Create a mapping from image base name to mask file
        mask_dict = {}
        for mask_file in mask_files:
            mask_base = os.path.splitext(mask_file)[0].lower()
            mask_dict[mask_base] = mask_file

        X = []
        Y = []
        matched = 0
        skipped_no_mask = 0
        skipped_load_error = 0

        for img_file in img_files:
            img_base = os.path.splitext(img_file)[0].lower()  # e.g., '011_hsi'
            
            # Extract the numerical part from the image filename
            # Assuming image filenames are in the format '###_hsi.tiff'
            if '_hsi' in img_base:
                num_part = img_base.split('_hsi')[0]  # e.g., '011'
            else:
                logging.warning(f"Image filename '{img_file}' does not contain '_hsi'. Skipping.")
                skipped_no_mask += 1
                continue

            # Construct the expected mask filename based on your naming convention
            # Masks are saved as '011-1.tiff', '012-1.tiff', etc.
            expected_mask_base = f"{num_part}-1"
            mask_file = mask_dict.get(expected_mask_base)

            if not mask_file:
                logging.warning(f"No mask found for image '{img_file}'. Skipping.")
                skipped_no_mask += 1
                continue

            img_path = os.path.join(img_dir, img_file)
            mask_path = os.path.join(mask_dir, mask_file)

            try:
                # Load and preprocess the HSI image (31 channels)
                img = tiff.imread(img_path)  # Shape: (Bands, Height, Width)
                img = img.transpose(1, 0, 2)  # Convert to (Height, Width, Bands)
                img = tf.image.resize(img, [img_height, img_width]).numpy()
                img = img / np.max(img)  # Normalize based on max value
                X.append(img)

                # Load and preprocess the mask
                mask = Image.open(mask_path).convert("L")  # Grayscale
                mask = mask.resize((img_width, img_height))
                mask_array = np.array(mask) / 255.0  # Normalize to [0, 1]
                mask_array = np.expand_dims(mask_array, axis=-1)  # Add channel dimension
                Y.append(mask_array)

                matched += 1

            except Exception as e:
                logging.error(f"Error loading image or mask pair '{img_file}' and '{mask_file}': {e}")
                skipped_load_error += 1
                continue

        logging.info(f"Matched and loaded {matched} image-mask pairs.")
        if skipped_no_mask > 0:
            logging.info(f"Skipped {skipped_no_mask} images due to missing masks.")
        if skipped_load_error > 0:
            logging.info(f"Skipped {skipped_load_error} image-mask pairs due to loading errors.")

        X = np.array(X)
        Y = np.array(Y)

        return X, Y

    except Exception as e:
        logging.error(f"An error occurred while loading data: {e}")
        return np.array([]), np.array([])
    
def load_model(checkpoint_path: str) -> tf.keras.Model:
    """
    Load the trained U-Net model from the .keras checkpoint file.

    Args:
        checkpoint_path (str): Full path to the 'best_model.keras' file.

    Returns:
        tf.keras.Model: Loaded U-Net model.
    """
    try:
        model = tf.keras.models.load_model(checkpoint_path)
        logging.info(f"Model loaded from: {checkpoint_path}")
        return model
    except FileNotFoundError:
        logging.error(f"Checkpoint file not found at: {checkpoint_path}")
        raise
    except tf.errors.NotFoundError:
        logging.error(f"Model architecture not found in checkpoint: {checkpoint_path}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error loading model from {checkpoint_path}: {e}")
        raise

def preprocess_image(image_path: str, target_size=(256, 256), expected_bands: int = 31) -> np.ndarray:
    """
    Load and preprocess the HSI image.

    Args:
        image_path (str): Path to the HSI image.
        target_size (tuple): Desired image size.
        expected_bands (int): Number of expected spectral bands.

    Returns:
        np.ndarray: Preprocessed image array.
    """
    try:
        # Read the multi-band TIFF image
        img = tiff.imread(image_path)  # Shape: (Bands, Height, Width) or (Height, Width, Bands)
        logging.info(f"Original image shape: {img.shape}")
        
        # Check the shape and transpose if necessary
        if img.ndim == 3:
            if img.shape[0] == expected_bands:
                # Assume shape is (Bands, Height, Width), transpose to (Height, Width, Bands)
                img = img.transpose(1, 2, 0)
                logging.info(f"Image transposed to shape: {img.shape}")
            elif img.shape[2] == expected_bands:
                # Already in (Height, Width, Bands)
                logging.info(f"Image already in (Height, Width, Bands) with shape: {img.shape}")
            else:
                logging.warning(
                    f"Image has {img.shape[2] if img.shape[2] >= img.shape[0] else img.shape[0]} bands, "
                    f"expected {expected_bands}. Truncating or padding as necessary."
                )
                if img.shape[2] < expected_bands:
                    # Pad with zeros
                    padding = expected_bands - img.shape[2]
                    pad_width = ((0, 0), (0, 0), (0, padding))
                    img = np.pad(img, pad_width, mode='constant', constant_values=0)
                    logging.info(f"Image padded to shape: {img.shape}")
                else:
                    # Truncate bands
                    img = img[:, :, :expected_bands]
                    logging.info(f"Image truncated to shape: {img.shape}")
        else:
            logging.error(f"Unsupported image dimensions: {img.ndim}D. Expected 3D array.")
            raise ValueError("Image must be a 3D array.")
        
        # Convert to float32 if not already
        if img.dtype != np.float32:
            img = img.astype(np.float32)
            logging.info("Converted image to float32.")
        
        # Resize the image to the target size
        img = tf.image.resize(img, target_size).numpy()
        logging.info(f"Image resized to: {img.shape[:2]}")
        
        # Normalize the image based on the max value in the image
        max_val = np.max(img)
        if max_val > 0:
            img = img / max_val
            logging.info("Image normalized by max value.")
        else:
            logging.warning("Max value of image is 0. Skipping normalization.")
        
        # Add a batch dimension
        img = np.expand_dims(img, axis=0)  # Shape: (1, Height, Width, Bands)
        logging.info(f"Image shape after adding batch dimension: {img.shape}")
        
        return img
    
    except Exception as e:
        logging.error(f"Error preprocessing image '{image_path}': {e}")
        raise e
    

def predict_mask(model: tf.keras.Model, image: np.ndarray, output_dir: str, apply_sigmoid: bool = False) -> np.ndarray:
    """
    Predict the segmentation mask using the U-Net model.

    Args:
        model (tf.keras.Model): Loaded U-Net model.
        image (np.ndarray): Preprocessed image array.
        output_dir (str): Directory to save the mask images.
        apply_sigmoid (bool): Whether to apply sigmoid activation to the predictions.

    Returns:
        np.ndarray: Predicted binary mask.
    """
    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        prediction = model.predict(image)
        logging.info(f"Prediction shape: {prediction.shape}")

        # Handle activation
        if apply_sigmoid:
            prediction = tf.keras.activations.sigmoid(prediction).numpy()
            logging.info("Applied sigmoid activation to predictions.")
        elif prediction.max() > 1.0:
            logging.warning("Prediction values exceed 1.0 but sigmoid activation was not applied.")

        # Assuming binary segmentation; handle single channel
        if prediction.shape[-1] == 1:
            raw_pred = prediction[0, :, :, 0]  # Shape: (Height, Width)
        else:
            # If multiple classes, consider using argmax or other strategies
            raw_pred = np.argmax(prediction, axis=-1)[0]
            logging.info("Applied argmax to multi-channel predictions.")

        # Save raw prediction
        plt.figure(figsize=(6, 6))
        plt.imshow(raw_pred, cmap='viridis')
        plt.colorbar()
        plt.title("Raw Model Prediction")
        raw_pred_path = os.path.join(output_dir, "raw_prediction.png")
        plt.savefig(raw_pred_path)
        plt.close()
        logging.info(f"Raw prediction saved to {raw_pred_path}")

        # Binarize the mask using a threshold of 0.5 (only relevant if sigmoid was applied)
        if apply_sigmoid or prediction.max() > 1.0:
            mask = (prediction > 0.5).astype(np.uint8)
            logging.info("Binarized the mask with threshold 0.5.")
        else:
            # If already binary, ensure it's in uint8
            mask = prediction.astype(np.uint8)
            logging.info("Converted prediction to uint8 mask.")

        # Remove unnecessary dimensions
        mask = np.squeeze(mask)
        logging.info(f"Mask shape after squeezing: {mask.shape}")

        # Log mask statistics
        unique_values = np.unique(mask)
        logging.info(f"Mask Statistics - Min: {mask.min()}, Max: {mask.max()}, Unique Values: {unique_values}")

        # Save the mask as an image for inspection
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask_uint8, mode='L')
        mask_path = os.path.join(output_dir, "predicted_mask.png")
        mask_image.save(mask_path)
        logging.info(f"Saved predicted mask to {mask_path}")

        return mask
    except Exception as e:
        logging.error(f"Error during mask prediction: {e}")
        raise e


def overlay_mask(image_path: str, mask: np.ndarray, output_path: str, alpha: float = 0.5):
    """
    Overlay the predicted mask on the original HSI image converted to grayscale.

    Args:
        image_path (str): Path to the original HSI image.
        mask (np.ndarray): Predicted binary mask.
        output_path (str): Path to save the overlaid image.
        alpha (float): Transparency factor for the mask overlay.

    Returns:
        None
    """
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert HSI to Grayscale for visualization
        img = Image.open(image_path)
        img = img.convert("L")  # Convert to grayscale
        img = img.resize((mask.shape[1], mask.shape[0]))
        logging.info(f"Grayscale image resized to: {img.size}")
        
        # Convert mask to RGB with colormap for better visibility
        cmap = cm.get_cmap('jet')  # Choose a colormap
        mask_colored = cmap(mask)[:, :, :3]  # RGB channels
        mask_colored = (mask_colored * 255).astype(np.uint8)
        mask_rgb = Image.fromarray(mask_colored, mode='RGB')
        
        # Convert grayscale image to RGB for blending
        img_rgb = img.convert("RGB")
        
        # Blend the grayscale RGB image with the colored mask
        overlay = Image.blend(img_rgb, mask_rgb, alpha=alpha)
        
        overlay.save(output_path)
        logging.info(f"Overlay image saved to: {output_path}")
    
    except Exception as e:
        logging.error(f"Error overlaying mask: {e}")
        raise e

def visualize_segmentation(image_path: str, mask: np.ndarray, output_path: str):
    """
    Visualize the original HSI image, predicted mask, and their overlay.

    Args:
        image_path (str): Path to the original HSI TIFF file.
        mask (np.ndarray): Predicted binary mask with shape (Height, Width).
        output_path (str): Path to save the overlay visualization.

    Returns:
        None
    """
    try:
        # Load the HSI image using tifffile
        hsi = tiff.imread(image_path)  # Shape: (Bands, Height, Width) or (Height, Width, Bands)
        logging.info(f"Loaded HSI image shape: {hsi.shape}, dtype: {hsi.dtype}")

        # Transpose if necessary to get (Height, Width, Bands)
        if hsi.ndim == 3 and hsi.shape[0] == 31:
            hsi = hsi.transpose(1, 2, 0)
            logging.info(f"HSI image transposed to shape: {hsi.shape}")
        elif hsi.ndim == 3 and hsi.shape[2] != 31:
            logging.warning(f"Expected 31 bands, but found {hsi.shape[2]} bands.")
            hsi = hsi[:, :, :31]
            logging.info(f"HSI image truncated to shape: {hsi.shape}")
        else:
            logging.info(f"HSI image already in (Height, Width, Bands) format with shape: {hsi.shape}")

        # Apply PCA to convert HSI to RGB
        rgb_image = hsi_to_rgb_pca(hsi)  # Shape: (Height, Width, 3)
        logging.info(f"PCA-based RGB image shape: {rgb_image.shape}")

        # Normalize RGB image for display
        rgb_min = rgb_image.min()
        rgb_max = rgb_image.max()
        rgb_image_normalized = (rgb_image - rgb_min) / (rgb_max - rgb_min) if rgb_max > rgb_min else rgb_image
        logging.info(f"RGB image normalized to range [0, 1].")

        # Create a figure with three subplots
        plt.figure(figsize=(18, 6))

        # Original HSI RGB Image (PCA-based)
        plt.subplot(1, 3, 1)
        plt.imshow(rgb_image_normalized)
        plt.title('Original HSI Image (PCA RGB)')
        plt.axis('off')

        # Predicted Mask
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')

        # Overlay Mask on HSI Image
        plt.subplot(1, 3, 3)
        plt.imshow(rgb_image_normalized)
        
        # Create a colored mask for better visibility (e.g., red)
        colored_mask = np.zeros_like(rgb_image_normalized)
        colored_mask[..., 0] = mask  # Assign mask to the Red channel
        
        # Normalize mask for visualization to prevent division by zero
        mask_max = mask.max()
        mask_normalized = mask / mask_max if mask_max > 0 else mask
        
        plt.imshow(colored_mask, cmap='Reds', alpha=0.5)
        plt.title('Overlay')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Segmentation visualization saved to {output_path}")

    except Exception as e:
        logging.error(f"Error in visualize_segmentation: {e}")
        raise e

def visualize_overlay(hsi_path: str, mask: np.ndarray, output_path: str, alpha: float = 0.5):
    """
    Visualize and save the overlay of the predicted mask on the HSI image in grayscale with red highlights.

    Args:
        hsi_path (str): Path to the original HSI image.
        mask (np.ndarray): Predicted binary mask with shape (Height, Width).
        output_path (str): Path to save the overlaid image.
        alpha (float): Transparency factor for the mask overlay.

    Returns:
        None
    """
    try:
        # Load the HSI image using tifffile
        hsi = tiff.imread(hsi_path)  # Shape: (Bands, Height, Width) or (Height, Width, Bands)
        logging.info(f"Loaded HSI image shape: {hsi.shape}, dtype: {hsi.dtype}")

        # Transpose if necessary to get (Height, Width, Bands)
        if hsi.ndim == 3 and hsi.shape[0] == 31:
            hsi = hsi.transpose(1, 2, 0)
            logging.info(f"HSI image transposed to shape: {hsi.shape}")
        elif hsi.ndim == 3 and hsi.shape[2] != 31:
            logging.warning(f"Expected 31 bands, but found {hsi.shape[2]} bands.")
            hsi = hsi[:, :, :31]
            logging.info(f"HSI image truncated to shape: {hsi.shape}")
        else:
            logging.info(f"HSI image already in (Height, Width, Bands) format with shape: {hsi.shape}")

        # Convert HSI to Grayscale using PCA (could be replaced with simple averaging)
        rgb_image = hsi_to_rgb_pca(hsi)  # Shape: (Height, Width, 3)
        hsi_grayscale = rgb_image.mean(axis=2)
        hsi_grayscale = (hsi_grayscale - hsi_grayscale.min()) / (
            hsi_grayscale.max() - hsi_grayscale.min()) if hsi_grayscale.max() > hsi_grayscale.min() else hsi_grayscale
        hsi_grayscale = (hsi_grayscale * 255).astype(np.uint8)
        hsi_image = Image.fromarray(hsi_grayscale, mode='L')
        hsi_image = hsi_image.resize((mask.shape[1], mask.shape[0]))
        logging.info(f"Grayscale HSI image resized to: {hsi_image.size}")

        # Create a red mask
        mask_red = Image.new("RGB", hsi_image.size, (0, 0, 0))
        mask_pixels = mask_red.load()
        for i in range(mask_red.size[0]):
            for j in range(mask_red.size[1]):
                if mask[j, i]:  # Assuming mask is in (Height, Width) format
                    mask_pixels[i, j] = (255, 0, 0)  # Red
                
        # Blend the grayscale image with the red mask
        overlay = Image.blend(hsi_image.convert("RGB"), mask_red, alpha=alpha)
        overlay.save(output_path)
        logging.info(f"Overlay visualization saved to: {output_path}")

    except Exception as e:
        logging.error(f"Error in visualize_overlay: {e}")
        raise e
    
    
def load_model_and_predict(image_path: str, checkpoint_path: str, output_dir: str, overlay_filename: str = "overlay.png") -> np.ndarray:
    """
    Load the model, predict the mask for a single image, visualize, and save the overlay.

    Args:
        image_path (str): Path to the input HSI image.
        checkpoint_path (str): Path to the saved model.
        output_dir (str): Directory to save the prediction results.
        overlay_filename (str): Filename for the overlay image.

    Returns:
        np.ndarray: Predicted binary mask.
    """
    try:
        # Load the model
        model = load_model(checkpoint_path)
        logging.info("Model loaded successfully.")

        # Preprocess the image
        preprocessed_image = preprocess_image(image_path)
        logging.info("Image preprocessed successfully.")

        # Predict the mask
        mask = predict_mask(
            model=model,
            image=preprocessed_image,
            output_dir=output_dir,
            apply_sigmoid=True  # Assuming binary segmentation with sigmoid activation
        )
        logging.info("Mask predicted successfully.")

        # Define overlay path
        overlay_path = os.path.join(output_dir, overlay_filename)

        # Overlay the mask on the original image
        overlay_mask(
            image_path=image_path,
            mask=mask,
            output_path=overlay_path,
            alpha=0.5  # Semi-transparent
        )
        logging.info(f"Overlay created and saved to {overlay_path}.")

        return mask
    except Exception as e:
        logging.error(f"Failed to load model and predict: {e}")
        raise e


def hsi_to_rgb_pca(hsi_image: np.ndarray, n_components: int = 3) -> np.ndarray:
    """
    Convert HSI image to RGB using PCA.

    Args:
        hsi_image (np.ndarray): HSI image array with shape (Height, Width, Bands).
        n_components (int): Number of principal components to retain.

    Returns:
        np.ndarray: RGB image array with shape (Height, Width, 3).
    """
    try:
        if hsi_image.ndim != 3:
            logging.error(f"HSI image must be 3D. Received shape: {hsi_image.shape}")
            raise ValueError("HSI image must be a 3D array.")

        # Ensure data is float32
        if hsi_image.dtype != np.float32:
            hsi_image = hsi_image.astype(np.float32)
            logging.info("Converted HSI image to float32 for PCA.")

        # Reshape HSI image to 2D array for PCA
        height, width, bands = hsi_image.shape
        reshaped_hsi = hsi_image.reshape(-1, bands)
        logging.info(f"HSI image reshaped for PCA: {reshaped_hsi.shape}")

        # Normalize the data (mean=0, variance=1)
        reshaped_hsi -= reshaped_hsi.mean(axis=0)
        reshaped_hsi /= reshaped_hsi.std(axis=0) + 1e-8  # Avoid division by zero
        logging.info("HSI data normalized for PCA.")

        # Apply PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(reshaped_hsi)
        logging.info(f"PCA result shape: {pca_result.shape}")

        # Reshape back to image dimensions
        rgb_image = pca_result.reshape(height, width, n_components)
        logging.info(f"RGB image shape after PCA reshape: {rgb_image.shape}")

        # Normalize RGB image to [0,1]
        rgb_min = rgb_image.min()
        rgb_max = rgb_image.max()
        rgb_image_normalized = (rgb_image - rgb_min) / (rgb_max - rgb_min) if rgb_max > rgb_min else rgb_image
        logging.info("RGB image normalized to [0, 1].")

        return rgb_image_normalized

    except Exception as e:
        logging.error(f"Error converting HSI to RGB using PCA: {e}")
        raise e
    
def display_mask(mask: np.ndarray):
    """
    Display the predicted mask.

    Args:
        mask (np.ndarray): Predicted binary mask.

    Returns:
        None
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis('off')
    plt.show()


def create_synthetic_mask(height: int, width: int, top_left: tuple = (50, 50), bottom_right: tuple = (200, 200), value: int = 1) -> np.ndarray:
    """
    Create a synthetic binary mask with a filled rectangle.

    Args:
        height (int): Height of the mask.
        width (int): Width of the mask.
        top_left (tuple): Top-left corner of the rectangle.
        bottom_right (tuple): Bottom-right corner of the rectangle.
        value (int): Fill value for the rectangle.

    Returns:
        np.ndarray: Synthetic binary mask.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, top_left, bottom_right, value, -1)  # Filled rectangle
    logging.info(f"Synthetic mask created with rectangle from {top_left} to {bottom_right}.")
    return mask


def visualize_synthetic_overlay(image_path: str, output_path: str, mask_alpha: float = 0.3):
    """
    Create and overlay a synthetic mask on the HSI image.

    Args:
        image_path (str): Path to the input HSI image.
        output_path (str): Path to save the overlaid image.
        mask_alpha (float): Transparency factor for the synthetic mask.

    Returns:
        None
    """
    try:
        # Create synthetic mask
        synthetic_mask = create_synthetic_mask(height=256, width=256, top_left=(50, 50), bottom_right=(200, 200), value=1)
        logging.info(f"Synthetic mask created with shape: {synthetic_mask.shape}")

        # Overlay the synthetic mask on the image
        overlay_mask(
            image_path=image_path,
            mask=synthetic_mask,
            output_path=output_path,
            alpha=mask_alpha
        )
        logging.info(f"Synthetic mask overlay saved to: {output_path}")

    except Exception as e:
        logging.error(f"Failed to create synthetic mask overlay: {e}")
        raise e
