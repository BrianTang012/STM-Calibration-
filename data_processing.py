import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Function 1: Resize Images
def resize_images(input_directory, output_directory, final_resolution=(64, 64)):
    """
    Resizes images by averaging pixel values within blocks and saves them in the specified output directory.
    """
    def resize_image(original_image, final_resolution):
        block_size = original_image.shape[0] // final_resolution[0]
        resized_image = np.zeros(final_resolution)
        for i in range(final_resolution[0]):
            for j in range(final_resolution[1]):
                start_row = i * block_size
                end_row = (i + 1) * block_size
                start_col = j * block_size
                end_col = (j + 1) * block_size
                block = original_image[start_row:end_row, start_col:end_col]
                block_mean = np.nanmean(block)
                resized_image[i, j] = block_mean
        return resized_image

    os.makedirs(output_directory, exist_ok=True)
    npy_files = [f for f in os.listdir(input_directory) if f.endswith('.npy')]
    for npy_file in npy_files:
        original_image_path = os.path.join(input_directory, npy_file)
        original_image = np.load(original_image_path)
        resized_image = resize_image(original_image, final_resolution)
        output_file_path = os.path.join(output_directory, f"{os.path.splitext(npy_file)[0]}_resized.npy")
        np.save(output_file_path, resized_image)
        print(f"Resized image saved: {output_file_path}")

# Function 2: Augment Images
def augment_images(resized_directory, augmented_directory):
    """
    Augments images in the specified directory and saves them in the 'augmented' subdirectory.
    """
    def visualize(original, augmented, title):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original, cmap='viridis')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        axes[1].imshow(augmented, cmap='viridis')
        axes[1].set_title(title)
        axes[1].axis('off')
        plt.show()

    os.makedirs(augmented_directory, exist_ok=True)

    # Get a list of all .npy files in the resized directory
    npy_files = [f for f in os.listdir(resized_directory) if f.endswith('.npy')]

    # Limit to visualize only 10 images
    image_count = 0

    # Loop through each .npy file in the directory
    for npy_file in npy_files:
        # Load the resized image data
        resized_image_path = os.path.join(resized_directory, npy_file)
        resized_image = np.load(resized_image_path)

        # Save the original image to the augmented directory
        original_image_path = os.path.join(augmented_directory, f"{os.path.splitext(npy_file)[0]}_original.npy")
        np.save(original_image_path, resized_image, allow_pickle=False, fix_imports=True)
        print(f"Original image saved: {original_image_path}")

        # Convert resized image to TensorFlow tensor and add batch and channel dimensions
        resized_image_tf = tf.convert_to_tensor(resized_image, dtype=tf.float64)
        resized_image_tf = tf.expand_dims(resized_image_tf, axis=-1)  # Add channel dimension
        resized_image_tf = tf.expand_dims(resized_image_tf, axis=0)  # Add batch dimension

        # Perform data augmentation
        augmentations = {}

        # Flip the image horizontally
        flipped = tf.image.flip_left_right(resized_image_tf)
        augmentations["flipped"] = tf.squeeze(flipped).numpy()

        # Brighten the image by adding 0.5 to pixel values (50% brighter)
        bright = tf.image.adjust_brightness(resized_image_tf, 0.5)
        augmentations["bright"] = tf.squeeze(bright).numpy()

        # Rotate the image 90 degrees counterclockwise
        rotated = tf.image.rot90(resized_image_tf)
        augmentations["rotated"] = tf.squeeze(rotated).numpy()

        # Save augmented images to files
        for aug_name, aug_image in augmentations.items():
            output_file_path = os.path.join(augmented_directory, f"{os.path.splitext(npy_file)[0]}_{aug_name}.npy")
            np.save(output_file_path, aug_image, allow_pickle=False, fix_imports=True)
            print(f"{aug_name.capitalize()} image saved: {output_file_path}")

        # Visualize one of the augmentations
        if image_count < 10:
            visualize(resized_image, augmentations["flipped"], "Flipped Image")
            image_count += 1

# Example usage
if __name__ == "__main__":
    # Set your NPY output directory here
    npy_output_folder = r'C:\Users\Razer\DeepSPM\classifier\npy_output'
    resized_output_directory = os.path.join(npy_output_folder, 'resized')
    augmented_output_directory = os.path.join(npy_output_folder, 'augmented')

    # Run the resizing function
    resize_images(
        input_directory=npy_output_folder,
        output_directory=resized_output_directory,
        final_resolution=(64, 64)
    )

    # Run the augmentation function
    augment_images(
        resized_directory=resized_output_directory,
        augmented_directory=augmented_output_directory
    )
