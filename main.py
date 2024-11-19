# Generates 31 .tiffs for each HSI (each .tiff corresponds to each channel in HSI)
import os
import numpy as np
import config
import tensorflow as tf
import imageio
import logging
import tifffile
from config import IMG_HEIGHT, IMG_WIDTH, RGB_IMAGE_PATH,HSI_IMAGE_PATH, RGB_MICRO_PATH
from model import Generator, Discriminator
from loss import peak_signal_to_noise_ratio, spectral_angle_mapper, generator_loss, mean_squared_error, discriminator_loss
from utils import load_paired_images, visualize_generated_images, apply_paired_augmentation, load_rgb_images, save_hsi_image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train_gan(rgb_path: str, hsi_path: str, generator: Generator,
              discriminator: Discriminator, 
              generator_optimizer, discriminator_optimizer, checkpoint_path: str,
              target_size=(IMG_WIDTH, IMG_HEIGHT),
              mode: str = "global"):    
    """
    Trains the GAN model with the provided parameters.

    Args:
        rgb_path (str): Path to RGB images.
        hsi_path (str): Path to HSI images.
        generator (tf.keras.Model): The generator model.
        discriminator (tf.keras.Model): The discriminator model.
        generator_optimizer (tf.keras.optimizers.Optimizer): Optimizer for the generator.
        discriminator_optimizer (tf.keras.optimizers.Optimizer): Optimizer for the discriminator.
        mode (str): Training mode ('global' or 'local').
        checkpoint_path (str): Path to save/load checkpoints.
    """
    # Set up checkpointing
    checkpoint = tf.train.Checkpoint(
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer
    )
    
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=checkpoint_path,
        max_to_keep=5
    )
    
    # Restore from the latest checkpoint if it exists
    if checkpoint_manager.latest_checkpoint:
        status = checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
        try:
            status.assert_existing_objects_matched()
            status.assert_consumed()
            logging.info(f"Successfully restored from checkpoint: {checkpoint_manager.latest_checkpoint}")
        except Exception as e:
            logging.warning(f"Checkpoint restoration incomplete: {e}")
    else:
        logging.info("No checkpoint found. Initializing from scratch.")

    # Clear the TensorFlow session
    #clear_session()
    """
    Train GAN with properly paired RGB and HSI images, using synchronized augmentation.
    """
    # Create metrics directory if it doesn't exist
    metrics_dir = './metrics'
    os.makedirs(metrics_dir, exist_ok=True)

    # Create output directory for generated images if it doesn't exist
    generated_hsi_dir = r'C:\Harshi\ECS-II\Dataset\gen_hsi'
    os.makedirs(generated_hsi_dir, exist_ok=True)

    # Load paired images
    print("Loading and pairing images...")
    try:
        rgb_images, hsi_images = load_paired_images(rgb_path, hsi_path)
        print(f"Successfully loaded {len(rgb_images)} paired images")
    except Exception as e:
        print(f"Error loading images: {str(e)}")
        return

    # Convert to tensors
    rgb_images = tf.convert_to_tensor(rgb_images, dtype=tf.float32)
    hsi_images = tf.convert_to_tensor(hsi_images, dtype=tf.float32)

    # Dictionary to store final epoch metrics
    final_metrics = {
        'discriminator_loss': [],
        'generator_loss': [],
        'mse': [],
        'psnr': [],
        'sam': []
    }

    # Training loop
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")

        for i in range(0, len(rgb_images), config.BATCH_SIZE):
            # Get batches
            rgb_batch = rgb_images[i:i + config.BATCH_SIZE]
            hsi_batch = hsi_images[i:i + config.BATCH_SIZE]

            # Apply synchronized augmentation
            try:
                augmented_rgb_batch, augmented_hsi_batch = apply_paired_augmentation(
                    rgb_batch, hsi_batch)
            except Exception as e:
                print(f"Error in data augmentation: {str(e)}")
                continue

            # Generator forward pass
            generated_hsi = generator(augmented_rgb_batch)

            # Visualize current results
            try:
                visualize_generated_images(
                    augmented_rgb_batch, generated_hsi, augmented_hsi_batch,
                    epoch, i // config.BATCH_SIZE)
            except Exception as e:
                logging.error(f"Error during visualization at Epoch {epoch}, Batch {i//config.BATCH_SIZE}: {str(e)}")

            # Save generated HSI images
            for j in range(generated_hsi.shape[0]):
                # Convert tensor to numpy array and clip values if necessary
                generated_hsi_np = tf.clip_by_value(generated_hsi[j], 0, 1).numpy()
                
                # Normalize and ensure the data is in the correct format
                # Depending on your specific needs, you may want to apply different transformations.
                generated_hsi_np = np.moveaxis(generated_hsi_np, -1, 0)  # Move channels to first dimension
                
                # Ensure that the image is in a valid shape (C, H, W) for saving
                if generated_hsi_np.ndim == 3 and generated_hsi_np.shape[0] > 3:
                    # Save as a multi-page TIFF file if there are more than 3 channels
                    image_path = os.path.join(generated_hsi_dir, f'generated_hsi_epoch{epoch+1}_batch{i//config.BATCH_SIZE}_img{j}.tiff')
                    imageio.mimwrite(image_path, generated_hsi_np.astype(np.float32), format='tiff')
                else:
                    # Otherwise, save as normal RGB or grayscale
                    image_path = os.path.join(generated_hsi_dir, f'generated_hsi_epoch{epoch+1}_batch{i//config.BATCH_SIZE}_img{j}.png')
                    imageio.imwrite(image_path, (generated_hsi_np * 255).astype(np.uint8))  # Scale to 0-255 if saving as PNG
            
            # Prepare discriminator inputs by concatenating RGB first, then HSI
            combined_real = tf.concat([augmented_rgb_batch, augmented_hsi_batch], axis=-1)
            combined_fake = tf.concat([augmented_rgb_batch, generated_hsi], axis=-1)
            
            # Train discriminator
            with tf.GradientTape() as disc_tape:
                disc_real = discriminator(combined_real, training=True)
                disc_fake = discriminator(combined_fake, training=True)
                # Pass concatenated inputs instead of discriminator outputs
                disc_loss = discriminator_loss(combined_real, combined_fake, discriminator)
            
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            gradients_of_discriminator = [tf.clip_by_norm(g, 1.0) for g in gradients_of_discriminator]
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
            
            # Train generator
            with tf.GradientTape() as gen_tape:
                generated_hsi = generator(augmented_rgb_batch, training=True)
                combined_fake = tf.concat([augmented_rgb_batch, generated_hsi], axis=-1)
                fake_output = discriminator(combined_fake, training=True)
                gen_loss = generator_loss(fake_output, generated_hsi, augmented_hsi_batch, lambda_pixel=20, lambda_perceptual=0.5)
            
            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            
            # Ensure no None gradients before clipping
            gradients_of_generator = [
                tf.clip_by_norm(g, 1.0) if g is not None else tf.zeros_like(v)
                for g, v in zip(gradients_of_generator, generator.trainable_variables)
            ]
            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            
            # Log weight norms for Generator
            for var in generator.trainable_variables:
                var_norm = tf.norm(var).numpy()
                logging.info(f"Generator {var.name} norm: {var_norm:.4f}")
            
            # Log weight norms for Discriminator
            for var in discriminator.trainable_variables:
                var_norm = tf.norm(var).numpy()
                logging.info(f"Discriminator {var.name} norm: {var_norm:.4f}")

            # Calculate metrics
            mse = mean_squared_error(augmented_hsi_batch, generated_hsi)
            psnr = peak_signal_to_noise_ratio(augmented_hsi_batch, generated_hsi)
            sam = spectral_angle_mapper(augmented_hsi_batch, generated_hsi)

            # Store metrics for the final epoch
            if epoch == config.EPOCHS - 1:
                final_metrics['discriminator_loss'].append(disc_loss.numpy())
                final_metrics['generator_loss'].append(gen_loss.numpy())
                final_metrics['mse'].append(mse.numpy())
                final_metrics['psnr'].append(psnr.numpy())
                final_metrics['sam'].append(sam.numpy())

            # Print progress
            batch_idx = i // config.BATCH_SIZE
            print(f'Epoch: {epoch}, Batch: {batch_idx}, '
                  f'Disc Loss: {disc_loss.numpy():.4f}, '
                  f'Gen Loss: {gen_loss.numpy():.4f}, '
                  f'MSE: {mse.numpy():.4f}, '
                  f'PSNR: {psnr.numpy():.4f}, '
                  f'SAM: {sam.numpy():.4f}')

        # Save checkpoint at end of epoch
        checkpoint.save(file_prefix=checkpoint_path)

        # Save metrics after the final epoch
        if epoch == config.EPOCHS - 1:
            metrics_file = os.path.join(metrics_dir, f'final_metrics_{mode}.txt')
            with open(metrics_file, 'w') as f:
                f.write(f"Training Mode: {mode}\n")
                f.write(f"Number of Batches: {len(final_metrics['mse'])}\n\n")
                f.write("Final Epoch Metrics (Average across all batches):\n")
                f.write("-" * 50 + "\n")
                f.write(
                    f"Discriminator Loss: {sum(final_metrics['discriminator_loss']) / len(final_metrics['discriminator_loss']):.4f}\n")
                f.write(
                    f"Generator Loss: {sum(final_metrics['generator_loss']) / len(final_metrics['generator_loss']):.4f}\n")
                f.write(
                    f"Mean Squared Error: {sum(final_metrics['mse']) / len(final_metrics['mse']):.4f}\n")
                f.write(
                    f"Peak Signal-to-Noise Ratio: {sum(final_metrics['psnr']) / len(final_metrics['psnr']):.4f}\n")
                f.write(
                    f"Spectral Angle Mapper: {sum(final_metrics['sam']) / len(final_metrics['sam']):.4f}\n")


def load_model_and_predict(rgb_path: str, checkpoint_path: str):
    """
    Load the pre-trained model and use it to make predictions on new data.
    
    Args:
        rgb_path (str): Path to RGB images directory.
        checkpoint_path (str): Path to the checkpoint directory containing the saved model weights.
    
    Returns:
        None
    """
    # Load RGB images and filenames
    rgb_images, filenames = load_rgb_images(rgb_path)
    
    # Check if any images were loaded by verifying the filenames list
    if not filenames:
        logging.error("No valid RGB images to process.")
        return
    
    # Convert RGB images to tensor
    rgb_tensor = tf.convert_to_tensor(rgb_images, dtype=tf.float32)
    print("RGB images converted to tensor")
    
    # Load the generator model
    generator = Generator()
    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=checkpoint_path,
        max_to_keep=5
    )
    latest_ckpt = checkpoint_manager.latest_checkpoint
    
    if latest_ckpt:
        status = checkpoint.restore(latest_ckpt)
        try:
            status.assert_existing_objects_matched()
            status.assert_consumed()
            logging.info(f"Successfully restored from checkpoint: {latest_ckpt}")
        except Exception as e:
            logging.warning(f"Checkpoint restoration incomplete: {e}")
    else:
        logging.error("No checkpoint found. Please check the checkpoint path.")
        return
    
    # Make predictions
    generated_hsi = generator(rgb_tensor, training=False)
    
    # Verify tensor shape
    print(f"Generated HSI shape: {generated_hsi.shape}")
    
    # Ensure expected channels
    expected_channels = 31
    actual_channels = generated_hsi.shape[-1]
    if actual_channels != expected_channels:
        logging.warning(f"Generated HSI has {actual_channels} channels instead of {expected_channels}.")
        # Attempt to transpose if channel dimension is incorrect
        if generated_hsi.shape[1] == expected_channels:
            generated_hsi = tf.transpose(generated_hsi, perm=[0, 2, 3, 1])
            logging.info(f"Transposed Generated HSI shape: {generated_hsi.shape}")
        else:
            logging.error("Unexpected HSI tensor shape. Cannot proceed with saving.")
            return
    
    # Convert to numpy for saving
    generated_hsi = generated_hsi.numpy()
    
    # Define the directory to save generated HSI TIFF files
    generated_hsi_dir = r'C:\Harshi\ECS-II\Dataset\gen_hsi'
    os.makedirs(generated_hsi_dir, exist_ok=True)
    
    # Iterate through each generated HSI image and save as TIFF
    for j in range(generated_hsi.shape[0]):
        try:
            # Extract individual image
            hsi_image = generated_hsi[j]
            
            # Normalize to [0, 1] if necessary
            hsi_image = (hsi_image + 1.0) / 2.0  # Assuming tanh activation
            hsi_image = np.clip(hsi_image, 0.0, 1.0)
            
            # Scale to [0, 255] and convert to uint8
            hsi_image = (hsi_image * 255).astype(np.uint8)
            
            # Ensure the shape is (height, width, channels)
            if hsi_image.shape[-1] != expected_channels:
                logging.error(f"Image {j} has incorrect channel dimension: {hsi_image.shape[-1]}")
                continue
            
            # Save the HSI image using the utility function
            save_hsi_image(hsi_image, os.path.splitext(filenames[j])[0], generated_hsi_dir)
        
        except Exception as e:
            logging.error(f"Failed to save HSI image for {filenames[j]}: {str(e)}")
            continue
    
    logging.info("HSI generation and saving completed.")

# Train the GAN
if __name__ == "__main__":
    mode = "global"
    if mode == "predict":
        load_model_and_predict(rgb_path=RGB_MICRO_PATH, checkpoint_path=config.CHECKPOINT_DIR)
    else:
        generator = Generator()
        discriminator = Discriminator()

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=config.LEARNING_RATE,
            decay_steps=100000,
            decay_rate=0.96,
            staircase=True
        )
        
        generator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            beta_1=config.BETA_1
        )
        
        discriminator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            beta_1=config.BETA_1
        )
        
        # To save model checkpoints
        if mode == "global":
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'global_ckpt')
        else:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'local_ckpt')

        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_path, exist_ok=True)
        logging.info(f"Checkpoint directory set to: {checkpoint_path}")

        # Logging and Checkpointing
        log_dir = config.LOG_DIR
        summary_writer = tf.summary.create_file_writer(log_dir)

        # Initialize Checkpoint and CheckpointManager
        checkpoint = tf.train.Checkpoint(
            generator=generator,
            discriminator=discriminator,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer
        )
           
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint,
            directory=checkpoint_path,
            max_to_keep=5
        )

        # Restore from the latest checkpoint if it exists
        if checkpoint_manager.latest_checkpoint:
            checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
            logging.info(f"Successfully restored from checkpoint: {checkpoint_manager.latest_checkpoint}")
        else:
            logging.info("No checkpoint found. Initializing from scratch.")
        
        train_gan(rgb_path=RGB_IMAGE_PATH, hsi_path=HSI_IMAGE_PATH,
                  generator=generator, discriminator=discriminator,
                  generator_optimizer=generator_optimizer, 
                  discriminator_optimizer=discriminator_optimizer,
                  mode=mode,
                  checkpoint_path=checkpoint_path)
        print("Training complete!")
