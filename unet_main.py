import os
import shutil
import zipfile
import logging
from main import load_model_and_predict
from config import CHECKPOINT_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("unet_main.log"),
        logging.StreamHandler()
    ]
)

# Define paths using raw strings
extract_dir = r"C:\Harshi\ECS-II\Dataset\extracted"
rgb_dir = r"/workspaces/rgb-to-hyper/rgb_micro"
mask_dir = r"C:\Harshi\ECS-II\Dataset\mask_micro"
zip_file_path = r"C:\Harshi\ECS-II\Dataset\images.zip"

"""# Extract the zip file
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
logging.info(os.listdir(mask_dir))"""

# Call load_model_and_predict with the sorted images
try:
    load_model_and_predict(
        rgb_path=rgb_dir,
        checkpoint_path=r"/workspaces/rgb-to-hyper/checkpoints"
    )
except Exception as e:
    logging.error(f"An error occurred during prediction: {str(e)}")
