# ECSGAN
This is the end-to-end implementation of detecting microplastics in water using Hyperspectral Imaging and Generative Adverserial Networks (GANs).

## Workflow

This project employs a GAN and U-Net in a novel approach to microplastic detection. The GAN acts as a translator, generating hyperspectral images from RGB inputs, overcoming the limitations of RGB data. Subsequently, the U-Net, a powerful segmentation model, identifies and delineates microplastic particles within the generated hyperspectral images. This two-stage process provides a robust solution for automated microplastic analysis.

## Project Tree

## Setup and Installation
### Step 0: Clone the repository through HTTPS/SSH
```bash
$ git clone https://github.com/akshathmangudi/rgb-to-hyper.git
```

### Step 1: Create a virtual environemnt 
```bash
$ python -m venv <env_name>
```

### Step 2: Install all the dependencies 
```bash
$ pip install -r requirements.txt
```

### Step 3: Training Script

The repository consists of tow stages:
1. One for reconstructing HSI out of RGB images. 
2. One for running detection on microplastic data. 

#### To train the GAN: 
**Step 1**: cURL or download the NTIRE2020 dataset. 

You can either download it manually or use a downloader like cURL to get 


Summary of the Training and Prediction Flow

A. Training Flow (mode = "global"):

1. Model and Optimizer Initialization:
   - Instantiate Generator and Discriminator.
   - Initialize their respective Adam optimizers.

2. Logging Setup:
   - Create a summary_writer for TensorBoard to log training metrics.

3. Checkpoint Path Determination:
     - Set checkpoint_path to 'global_ckpt' for global training.

4. Training Execution:
   - Call train_gan with all necessary parameters.
   - Inside train_gan:
     - Checkpoint Restoration: Load existing checkpoints if available.
     - Data Loading and Preparation: Load and preprocess paired RGB and HSI images.
     - Epoch Loop: For each epoch, shuffle data and iterate over batches.
       - Batch Processing: Perform augmentation, train discriminator, train generator, compute metrics, and log progress.
       - Checkpoint Saving: Save model states at the end of each epoch.
     - Post-Training: Save final metrics and optionally generate sample outputs.


B. Prediction Flow (mode = "predict")

1. Checkpoint Restoration:
  - Use load_model_and_predict to load the Generator model from the 'global_ckpt' directory.

2. Data Loading:
   - Load RGB images designated for prediction.

3. HSI Generation:
  - Use the restored Generator to create HSI images from the RGB inputs.

4. Output Saving:
   - Save the generated HSI images as TIFF files in the specified directory.
