import os
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms

class ImagePredictor:
    def __init__(self, root_dir, filtered_save_dir=None, model_path=None, device=None):
        # Set up device
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the model if a model path is provided
        if model_path:
            self.model = self.load_model(model_path)
        else:
            self.model = None

        # On-the-fly transformations for prediction
        self.transform = self.get_transform()

        # Data tracking
        self.root_dir = root_dir
        self.filtered_save_dir = filtered_save_dir
        self.valid_data = []  # Store only valid images
        self.nan_count = 0  # To count images with NaN values
        self.same_min_max_count = 0  # To count images with same min and max values

        # Create filtered save directory if it doesn't exist
        if self.filtered_save_dir and not os.path.exists(self.filtered_save_dir):
            os.makedirs(self.filtered_save_dir)

        # Load all .npy files from the directory
        self.load_images_from_directory()

        print(f"Total images removed due to NaN: {self.nan_count}")
        print(f"Total images removed due to min == max: {self.same_min_max_count}")

    def load_model(self, model_path):
        """Loads the trained model from the specified path."""
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.BatchNorm1d(num_ftrs),
            torch.nn.Linear(num_ftrs, 1)  # Output a single score for binary classification
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()  # Set the model to evaluation mode
        return model

    def initial_transform(self):
        """Defines the initial transformations to save into the .transformed.npy files."""
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ])

    def get_transform(self):
        """Defines the on-the-fly transformations for images used during prediction."""
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
            transforms.Resize((224, 224)),  # Resize for ResNet input
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_images_from_directory(self):
        """Load valid images from the directory."""
        for filename in os.listdir(self.root_dir):
            if filename.endswith('.npy'):
                file_path = os.path.join(self.root_dir, filename)
                img = np.load(file_path)

                # Check for NaN values
                if np.isnan(img).any():
                    print(f"Image {filename} contains NaN values and is excluded.")
                    self.nan_count += 1
                elif img.min() == img.max():
                    print(f"Image {filename} has no variation (min == max) and is excluded.")
                    self.same_min_max_count += 1
                else:
                    self.valid_data.append(file_path)

                    # Save valid images to the filtered_save_dir
                    if self.filtered_save_dir:
                        filtered_save_file = os.path.join(self.filtered_save_dir, filename)
                        np.save(filtered_save_file, img)

    def preprocess_image(self, image_path):
        """Preprocesses the image stored in a .npy file with initial transformations."""
        img = np.load(image_path)

        # Check for NaN values and handle them
        if np.isnan(img).any():
            self.nan_count += 1
            print(f"Image at {image_path} contains NaN values. Skipping.")
            return None

        # Normalize the image
        min_value = img.min()
        max_value = img.max()

        # Handle images with zero-range (min == max)
        if min_value == max_value:
            self.same_min_max_count += 1
            print(f"Image at {image_path} has same min and max values. Skipping.")
            return None

        img = (img - min_value) / (max_value - min_value)
        img = (img * 255).astype('uint8')

        img = Image.fromarray(img, 'L')  # Convert to grayscale image

        # Apply initial transformations (flip, color jitter, etc.)
        transform = self.initial_transform()
        img = transform(img)

        # Resize to 64x64 and save as numpy array
        img_resized = img.resize((64, 64), Image.Resampling.LANCZOS)
        img_resized = np.array(img_resized).astype('uint8')  # Convert to numpy array

        return img_resized

    def save_images(self, output_dir):
        """Save the preprocessed and transformed images."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for idx, img_path in enumerate(self.valid_data):
            transformed_img = self.preprocess_image(img_path)

            if transformed_img is not None:
                base_filename = os.path.splitext(os.path.basename(img_path))[0]
                output_file_transformed = os.path.join(output_dir, base_filename + '_transformed.npy')
                np.save(output_file_transformed, transformed_img)
                print(f"Saved: {output_file_transformed}")

    def predict(self, image_path):
        """Makes a prediction for a single .npy file (requires model)."""
        if not self.model:
            print("Model is not loaded. Cannot perform prediction.")
            return None, None

        # Load the pre-saved transformed image
        img = np.load(image_path)
        img = Image.fromarray(img, 'L')

        # Apply the final on-the-fly transformations
        img = self.transform(img)

        img = img.unsqueeze(0).to(self.device)  # Add batch dimension
        with torch.no_grad():
            output = self.model(img)
            probability = torch.sigmoid(output).item()
            predicted_label = 'good' if probability > 0.5 else 'bad'
            return predicted_label, probability

    def process_directory(self):
        """Process and save valid images from the root directory."""
        self.save_images(self.filtered_save_dir)

if __name__ == "__main__":
    # Paths for root directory and output directory
    root_dir = r'C:\Users\Razer\DeepSPM\classifier\npy_output\augmented'  # Directory containing .npy files
    filtered_save_dir = r'C:\Users\Razer\DeepSPM\classifier\npy_output\filtered'  # Directory to save valid images
    model_path = r'C:\Users\Razer\DeepSPM\classifier\model\checkpoint.pt'  # Path to the trained model

    # Instantiate the ImagePredictor class
    predictor = ImagePredictor(root_dir=root_dir, filtered_save_dir=filtered_save_dir, model_path=model_path)

    # Process the images and save the preprocessed transformed images
    predictor.process_directory()

    # Optionally, predict on a sample image
    sample_image_path = r'C:\Users\Razer\DeepSPM\classifier\npy_output\filtered\some_image_transformed.npy'
    label, probability = predictor.predict(sample_image_path)
    print(f"Predicted label: {label}, Probability: {probability:.4f}")

