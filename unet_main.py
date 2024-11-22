import os
import shutil
import sys
import zipfile
import logging
from PIL import Image
import numpy as np
import tifffile as tiff
from config import CHECKPOINT_DIR
from matplotlib import pyplot as plt
from main import load_model_and_predict
from unet_utils import load_data, visualize_overlay, display_mask, preprocess_image, create_synthetic_mask, visualize_overlay, visualize_synthetic_overlay
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore
from unet_model import build_unet, HSIGenerator  # Import from the new module
from utils import load_rgb_images
from utils import (
    visualize_all_hsi_bands, 
    create_hsi_grid_image,
    visualize_stacked_hsi,
    visualize_false_color_composite,
    visualize_pca_composite,
    visualize_pca_3d
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses INFO and WARNING messages

# Configure logging

import io

# Reconfigure stdout and stderr to use UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log", encoding='utf-8'),  # Log to file with UTF-8
        logging.StreamHandler(sys.stdout),                        # Log to stdout
        logging.StreamHandler(sys.stderr)                         # Log to stderr
    ]
)


# Define paths using raw strings
extract_dir = r"C:\Harshi\ECS-II\Dataset\extracted"
rgb_dir = r"C:\Harshi\ECS-II\Dataset\temp-rgb-micro"
mask_dir = r"C:\Harshi\ECS-II\Dataset\mask_micro"
zip_file_path = r"C:\Harshi\ECS-II\Dataset\dataverse_files full"


"""
# Extract the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
    logging.info(f"Extracted {zip_file_path} to {extract_dir}")

# Create the directories if they don't exist
os.makedirs(rgb_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)
logging.info(f"Ensured directories {rgb_dir} and {mask_dir} exist.")

# Iterate through the files in the extracted folder and move them into the correct directories
for root, _, files in os.walk(extract_dir):
    for filename in files:
        if filename.endswith(".png"):
            # Check if it's a mask image (ends with '-1.png')
            if filename.endswith("-1.png"):
                shutil.move(os.path.join(root, filename), os.path.join(mask_dir, filename))
                logging.info(f"Moved mask image {filename} to {mask_dir}")
            else:
                shutil.move(os.path.join(root, filename), os.path.join(rgb_dir, filename))
                logging.info(f"Moved RGB image {filename} to {rgb_dir}")

logging.info("Images sorted into 'rgb_micro' and 'mask_micro'.")

# List the contents of the directories
logging.info("Contents of 'rgb_micro':")
logging.info(os.listdir(rgb_dir))

logging.info("Contents of 'mask_micro':")
logging.info(os.listdir(mask_dir))
"""

# Call load_model_and_predict with the sorted images
try:
    load_model_and_predict(
        rgb_path=rgb_dir,
        checkpoint_path=CHECKPOINT_DIR
    )
except Exception as e:
    logging.error(f"An error occurred during prediction: {str(e)}")

# Load the saved multi-band TIFF
loaded_hsi = tiff.imread(r'c:\Harshi\ECS-II\Dataset\gen_hsi\190_hsi.tiff')

print(f"Loaded HSI Shape: {loaded_hsi.shape}")  # Should be (height, width, 31)

# Visualize the first spectral band
plt.imshow(loaded_hsi[:, :, 0], cmap='gray')
plt.title('HSI Channel 1')
plt.axis('off')
plt.show()

# Visualization of all HSI bands and stacking
try:
    # Visualize all bands
    hsi_data = visualize_all_hsi_bands(
        filepath=r"C:\Harshi\ECS-II\Dataset\gen_hsi\027_hsi.tiff",  # Use the specific TIFF file
        bands=None,              # Set to None to visualize all bands
        figsize=(25, 20)         # Adjust figsize as needed
    )
    
    if hsi_data.size != 0:
        # Stack bands into a single 3D NumPy array
        stacked_hsi = hsi_data  # hsi_data is already a 3D array (height, width, bands)
        logging.info(f"HSI data loaded with shape: {stacked_hsi.shape}")
        
        # Save the stacked HSI data as a multi-band TIFF
        output_stacked_path = r"C:\Harshi\ECS-II\Dataset\gen_hsi\stacked_hsi.tiff"
        try:
            tiff.imwrite(output_stacked_path, stacked_hsi)
            logging.info(f"Stacked HSI saved to: {output_stacked_path}")
        except Exception as e:
            logging.error(f"Failed to save stacked HSI: {e}")
        
        # Create and save a grid image of all bands
        grid_save_path = r"C:\Harshi\ECS-II\Dataset\gen_hsi\HSI_Bands_Grid.png"
        try:
            create_hsi_grid_image(
                stacked_hsi=stacked_hsi,
                cols=5,
                figsize=(25, 20),
                save_path=grid_save_path
            )
        except Exception as e:
            logging.error(f"An error occurred during grid image creation: {e}")
        
        """# False-Color Composite Visualization
        try:
            false_color_save_path = r"C:\Harshi\ECS-II\Dataset\gen_hsi\False_Color_Composite.png"
            visualize_false_color_composite(
                stacked_hsi=stacked_hsi,
                bands=[19, 29, 9],  # Example band indices for RGB channels
                figsize=(10, 10),
                save_path=false_color_save_path
            )
        except Exception as e:
            logging.error(f"An error occurred during False-Color Composite visualization: {e}")
        """
        # PCA Composite Visualization
        try:
            pca_save_path = r"C:\Harshi\ECS-II\Dataset\gen_hsi\PCA_Composite.png"
            visualize_pca_composite(
                stacked_hsi=stacked_hsi,
                n_components=3,
                figsize=(10, 10),
                save_path=pca_save_path
            )
        except Exception as e:
            logging.error(f"An error occurred during PCA Composite visualization: {e}")
        
        """# Interactive 3D PCA Visualization (Optional)
        try:
            visualize_pca_3d(
                stacked_hsi=stacked_hsi,
                n_components=3
            )
        except Exception as e:
            logging.error(f"An error occurred during 3D PCA visualization: {e}")
        
        # Stacked IMG visualization
        try:
            visualize_stacked_hsi(stacked_hsi, save_path=r"C:\Harshi\ECS-II\Dataset\gen_hsi\stacked_hsi.tiff")
        except Exception as e:
            logging.error(f"An error occurred during stacked HSI visualization: {str(e)}")"""
    
    else:
        logging.error("HSI data is empty. Stacking and visualization skipped.")

except Exception as e:
    logging.error(f"An error occurred during HSI bands visualization and stacking: {str(e)}")


"""def test_generator(generator):
    X_test, Y_test = generator.__getitem__(0)
    print(f"Test Batch - X shape: {X_test.shape}, Y shape: {Y_test.shape}")
    
    # Optional: Visualize one sample
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    
    # Display the first band of the first image in the batch
    plt.subplot(1, 3, 1)
    plt.imshow(X_test[0][:,:,0], cmap='gray')
    plt.title('Sample HSI Band 1')
    
    # Display the corresponding mask
    plt.subplot(1, 3, 2)
    plt.imshow(Y_test[0].squeeze(), cmap='gray')
    plt.title('Sample Mask')
    
    # Display the last band of the first image in the batch
    plt.subplot(1, 3, 3)
    plt.imshow(X_test[0][:,:, -1], cmap='gray')
    plt.title('Sample HSI Band 31')
    
    plt.show()

if __name__ == "__main__":
    IMG_HEIGHT, IMG_WIDTH = 256, 256
    N_CLASSES = 1  # Binary segmentation
    BATCH_SIZE = 16
    EPOCHS = 5
    IMG_PATH = r"C:\Harshi\ECS-II\Dataset\temp-gen-hsi"  # Path to your HSI images
    MASK_PATH = r"C:\Harshi\ECS-II\Dataset\temp-mask"  # Path to your masks
    MODEL_PATH = r"C:\Harshi\ecs-venv\rgb-to-hyper\rgb-to-hyper-main\rgb-to-hyper"

    # Build and summarize the model
    model = build_unet(input_shape=(256, 256, 31), num_classes=1)  # Adjust input channels if necessary

    # Redirect model.summary() to a file
    with open("model_summary.txt", "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    logging.info("Model summary saved to 'model_summary.txt'")

    # Load data
    try:
        X, Y = load_data(IMG_PATH, MASK_PATH)
    except ValueError as ve:
        print(f"Data loading error: {ve}")
        exit(1)

    print(f"Loaded {X.shape[0]} samples.")
    print(f"Image shape: {X.shape[1:]}")  # Expected: (256, 256, 31)
    print(f"Mask shape: {Y.shape[1:]}")    # Expected: (256, 256, 1) 

    if X.shape[0] == 0:
        print("No samples loaded. Please check your data directories and file formats.")
        exit(1)

    # Split the data
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

    # Initialize data generator
    train_generator = HSIGenerator(
        img_dir=IMG_PATH,
        mask_dir=MASK_PATH,
        batch_size=BATCH_SIZE,
        img_height=256,
        img_width=256,
        desired_channels=31,
        shuffle=True
    )

    # Test the generator
    test_generator(train_generator)

    # Validation generator
    val_generator = HSIGenerator(
        img_dir=IMG_PATH,
        mask_dir=MASK_PATH,
        batch_size=BATCH_SIZE,
        img_height=256,
        img_width=256,
        desired_channels=31,
        shuffle=False
    )

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)]
    )

    # Callbacks
    checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model using generators
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stop]
    )

    print("Training complete!")"""

"""from unet_utils import load_model_and_predict

if __name__ == "__main__":
    IMAGE_PATH = r"C:\Harshi\ECS-II\Dataset\val_set_micro_hsi\035_hsi.tiff"
    CHECKPOINT_PATH = r"C:\Harshi\ecs-venv\rgb-to-hyper\rgb-to-hyper-main\rgb-to-hyper\best_model.keras"
    OUTPUT_DIR = r"C:\Harshi\ECS-II\Dataset\gen_overlay"
    OUTPUT_FILENAME = "003_overlay.png"

    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    
    try:
        # Predict the mask and visualize the overlay
        mask = load_model_and_predict(IMAGE_PATH, CHECKPOINT_PATH, OUTPUT_PATH)
        if mask is not None:
            logging.info("Mask obtained and overlay visualization created successfully.")
        else:
            logging.error("Mask prediction failed. Overlay was not created.")
    except Exception as e:
        logging.error(f"Failed to load model and predict: {e}")"""

"""
    try:
        # Create and visualize synthetic mask overlay
        visualize_synthetic_overlay(IMAGE_PATH, OUTPUT_PATH)
        logging.info("Synthetic mask overlay created successfully.")
    except Exception as e:
        logging.error(f"Failed to create synthetic mask overlay: {e}")
"""
