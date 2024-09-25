import os
import logging
import time
import json
from nanonis_basic import nanonis_basic
import data_processing
from model_testing import ImagePredictor
from grid_manager import GridManager

# Configure logging for recording all activities
logging.basicConfig(filename='nanonis_scan_log_test.txt', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration settings from a JSON file, fallback to defaults if no file is found
def load_config(json_path='config.json'):
    if os.path.exists(json_path):
        with open(json_path, 'r') as file:
            config = json.load(file)
            logging.info(f"Configuration successfully loaded from {json_path}")
            return config
    else:
        logging.warning(f"Configuration file not found at {json_path}, using default settings.")
        return None

# Establish connection to Nanonis device, retrying if needed
def connect_nanonis(config=None):
    IP = config.get('nanonis_ip', '127.0.0.1') if config else '127.0.0.1'
    PORT = config.get('nanonis_port', 6501) if config else 6501
    retry_interval = config.get('retry_interval', 5) if config else 5
    
    nanonis = nanonis_basic(IP, PORT)

    # Attempt to connect to the Nanonis device
    while True:
        try:
            logging.info(f"Attempting to connect to Nanonis at IP: {IP}, PORT: {PORT}")
            nanonis.connect()
            logging.info(f"Successfully connected to Nanonis at {IP}:{PORT}")
            return nanonis
        except (ConnectionRefusedError, OSError):
            logging.warning(f"Unable to connect to Nanonis. Retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)

# Function to clear contents of a directory without removing the directory itself
def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except Exception as e:
            logging.error(f'Failed to delete {file_path}. Reason: {e}')

# Perform a scan and process images
def perform_scan_and_process_images(nanonis, npy_output_folder, resized_output_folder, augmented_output_folder, filtered_save_dir, model_path, grid_manager):
    try:
        # Ensure the output folders exist
        os.makedirs(npy_output_folder, exist_ok=True)
        os.makedirs(resized_output_folder, exist_ok=True)  # Ensure resized directory exists
        os.makedirs(augmented_output_folder, exist_ok=True)  # Ensure augmented directory exists
        os.makedirs(filtered_save_dir, exist_ok=True)

        recalibration_attempts = 0
        max_recalibration_attempts = 5
        initial_z_position_1V = None  # To store initial Z position at 1V after first scan

        # Get the next grid position for the tip
        while True:
            next_position = grid_manager.get_next_position()
            if next_position is None:
                logging.info("Reached the end of the grid scanning area.")
                break

            center_x, center_y = next_position
            logging.info(f"Moving STM tip to grid position with center coordinates (X={center_x}, Y={center_y})")

            # Set the bias to 1V for scanning (normal condition)
            nanonis.BiasSet(1.0)
            logging.info("Bias voltage set to 1V for scanning.")

            # Set the tunneling current to 100 pA
            nanonis.SetpointSet(100e-12)
            logging.info("Tunneling current set to 100 pA.")

            # Ensure the feedback loop is on (constant current mode)
            feedback_status = nanonis.FeedbackOnOffGet()
            if feedback_status == 'Off':
                logging.info("Feedback loop is off. Activating constant current mode...")
                nanonis.FeedbackOnOffSet('On')
                logging.info("Constant current mode activated (feedback loop on).")
            else:
                logging.info("Feedback loop is already on (constant current mode enabled).")

            # Set the scan frame size and log the position
            scan_size_x = 100e-9  # 100nm in meters
            scan_size_y = 100e-9  # 100nm in meters
            nanonis.ScanFrameSet(center_x=center_x, center_y=center_y, size_x=scan_size_x, size_y=scan_size_y)
            logging.info(f"STM scan frame set to 100nm x 100nm with center at coordinates (X={center_x}, Y={center_y}).")

            # Start the scan
            logging.info(f"Initiating scan at grid center (X={center_x}, Y={center_y})...")
            nanonis.ScanAction('Start', 'Down')

            # Wait for the scan to complete and convert the result to NPY format
            nanonis.ScanWaitEndOfScan(npy_output_folder=npy_output_folder)
            logging.info(f"Scan completed and saved as NPY files in: {npy_output_folder}")

            # Move the tip to the last scanned point (lower-right corner)
            last_scanned_x = center_x + (scan_size_x / 2)
            last_scanned_y = center_y - (scan_size_y / 2)
            nanonis.TipXYSet(last_scanned_x, last_scanned_y)
            logging.info(f"Moved tip to last scanned point (X={last_scanned_x}, Y={last_scanned_y}).")

            # Log the initial Z position at 1V after the first scan
            if recalibration_attempts == 0 and initial_z_position_1V is None:
                initial_z_position_1V = nanonis.TipZGet()
                logging.info(f"Initial Z position at 1V recorded at last scan point: {initial_z_position_1V} meters")

            # Create a unique folder for the latest scan inside filtered_save_dir
            timestamp = int(time.time())
            temp_scan_folder = os.path.join(filtered_save_dir, f"scan_{timestamp}")
            os.makedirs(temp_scan_folder, exist_ok=True)

            # Move the latest scan files (only new images) to the temp_scan_folder
            for file_name in os.listdir(npy_output_folder):
                file_path = os.path.join(npy_output_folder, file_name)
                if os.path.isfile(file_path) and file_name.endswith('.npy'):
                    os.rename(file_path, os.path.join(temp_scan_folder, file_name))

            # **Clear the resized and augmented directories before processing the latest scan**
            clear_directory(resized_output_folder)
            clear_directory(augmented_output_folder)

            # Process the images (resize and augment) only for the latest scan
            process_images(temp_scan_folder, resized_output_folder, augmented_output_folder)

            # Preprocess images before prediction and save as .transformed.npy in temp_scan_folder
            preprocess_images_before_prediction(augmented_output_folder, temp_scan_folder)

            # Perform majority voting on the latest scan only
            prediction_successful = predict_filtered_images_with_majority_vote(temp_scan_folder, model_path)

            # Log the result of majority voting
            logging.info(f"Majority vote prediction outcome: {'Successful (Good)' if prediction_successful else 'Unsuccessful (Bad)'}")

            # If no 'good' label, attempt recalibration up to a set number of tries
            if not prediction_successful:
                recalibration_attempts += 1
                logging.info(f"Scan labeled as 'bad'. Performing controlled tip dip for recalibration (attempt {recalibration_attempts})")

                # Turn off feedback loop before bias adjustment
                logging.info("Turning off feedback loop for safe bias adjustment.")
                nanonis.FeedbackOnOffSet('Off')

                # Reduce the bias to 0.1V before performing the dip
                nanonis.BiasSet(0.1)
                logging.info("Bias voltage reduced to 0.1V for safe dipping.")
                time.sleep(0.5)  # Delay for stability

                # Turn the feedback loop back on
                nanonis.FeedbackOnOffSet('On')
                logging.info("Feedback loop turned back on after bias adjustment.")

                # Record the Z position after adjusting the bias to 0.1V, at the last scan point
                initial_z_position_0_1V = nanonis.TipZGet()
                logging.info(f"Z position recorded after bias adjustment to 0.1V at last scan point: {initial_z_position_0_1V} meters")

                # Perform the controlled dip by adjusting the tip's Z position using the initial Z position at 1V as reference
                nanonis.TipZSet(initial_z_position_1V - 1e-9)  # Lower the tip by 1nm
                logging.info(f"Controlled dip performed, lowering tip by 1nm to Z={initial_z_position_1V - 1e-9} meters.")

                # Perform controlled retraction before moving to the next scan region
                controlled_retraction(nanonis, initial_z_position_1V, retract_distance=100e-9, retract_steps=10, original_bias=1.0)

                # If recalibration attempts exceed the limit, move to the next grid position
                if recalibration_attempts >= max_recalibration_attempts:
                    logging.info("Maximum recalibration attempts reached. Proceeding to the next grid position.")
                    recalibration_attempts = 0
            else:
                logging.info("Tip successfully calibrated at this grid position.")
                break  # If prediction is successful, end the process

        logging.info("Scanning process completed for the entire grid.")

    except Exception as e:
        logging.error(f"An error occurred during the scan or image processing: {e}")

# Function to resize and augment images
def process_images(npy_input_folder, resized_output_directory, augmented_output_directory):
    try:
        logging.info(f"Resizing images from {npy_input_folder} into {resized_output_directory}...")
        data_processing.resize_images(input_directory=npy_input_folder,
                                      output_directory=resized_output_directory,
                                      final_resolution=(64, 64))

        logging.info(f"Augmenting resized images from {resized_output_directory} and saving them into {augmented_output_directory}...")
        data_processing.augment_images(resized_directory=resized_output_directory,
                                       augmented_directory=augmented_output_directory)

        logging.info(f"Image resizing and augmentation completed. Augmented images saved in: {augmented_output_directory}")

    except Exception as e:
        logging.error(f"An error occurred during image resizing or augmentation: {e}")

# Function to preprocess images and save only .transformed.npy files before prediction
def preprocess_images_before_prediction(augmented_output_directory, filtered_save_dir):
    try:
        logging.info(f"Preprocessing augmented images from {augmented_output_directory}...")

        # Instantiate the ImagePredictor
        predictor = ImagePredictor(root_dir=augmented_output_directory, filtered_save_dir=filtered_save_dir)

        # Process and save only transformed images ending with .transformed.npy to the filtered_save_dir
        predictor.process_directory()

        logging.info(f"Image preprocessing completed. Transformed images saved in: {filtered_save_dir}")

    except Exception as e:
        logging.error(f"An error occurred during image preprocessing: {e}")

# Function to predict the filtered images (only .transformed.npy) with majority voting
def predict_filtered_images_with_majority_vote(filtered_save_dir, model_path):
    try:
        logging.info(f"Initiating prediction on the latest scan images in {filtered_save_dir}...")

        # Instantiate the ImagePredictor with the model loaded
        predictor = ImagePredictor(root_dir=filtered_save_dir, model_path=model_path)

        good_count = 0
        bad_count = 0

        # Loop through the images in the filtered_save_dir and make predictions only on .transformed.npy files
        for filename in os.listdir(filtered_save_dir):
            if filename.endswith('_transformed.npy'):
                image_path = os.path.join(filtered_save_dir, filename)
                label, probability = predictor.predict(image_path)
                logging.info(f"Image: {filename}, Predicted label: {label}, Probability: {probability:.4f}")

                # Tally the results
                if label == 'good':
                    good_count += 1
                else:
                    bad_count += 1

        logging.info(f"Prediction results - Good: {good_count}, Bad: {bad_count}")

        # Return True if the majority label is 'good'
        return good_count > bad_count

    except Exception as e:
        logging.error(f"An error occurred during image prediction: {e}")
        return False

# Perform a controlled retraction of the STM tip
def controlled_retraction(nanonis, initial_z_position, retract_distance=100e-9, retract_steps=10, original_bias=1.0):
    """
    Perform a controlled retraction of the STM tip by retracting in steps.
    The tip retracts by 'retract_distance' in steps, but always returns to the initial Z position recorded after the first scan.
    
    Args:
        nanonis: Instance of Nanonis interface class.
        initial_z_position: The Z position recorded after the first scan.
        retract_distance: The total distance to retract the tip (e.g., 100e-9 meters).
        retract_steps: Number of steps to divide the retraction into.
        original_bias: The bias voltage to restore after retraction (e.g., 1.0V for scanning).
    """
    
    # Step 1: Get the current Z position and calculate the retraction step size
    current_z_position = nanonis.TipZGet()
    logging.info(f"Current Z position: {current_z_position}")
    
    # Calculate the total retraction required to return to the initial Z position
    total_retraction_distance = initial_z_position - current_z_position
    retract_step_size = total_retraction_distance / retract_steps
    logging.info(f"Retracting the tip in {retract_steps} steps, each step: {retract_step_size} meters.")

    # Step 2: Retract the tip in steps
    for step in range(retract_steps):
        current_z_position += retract_step_size  # Move the tip up by the step size
        nanonis.TipZSet(current_z_position)
        time.sleep(0.1)  # Add a small delay between steps for stability
        logging.info(f"Step {step + 1}/{retract_steps}: Tip Z position now {current_z_position} meters")
    
    # Step 3: After retraction, return to the initial Z position recorded after the first scan
    logging.info(f"Returning tip to initial Z position: {initial_z_position} meters")
    nanonis.TipZSet(initial_z_position)
    time.sleep(0.5)

    # Step 4: Restore original bias voltage, setpoint current, and feedback loop
    logging.info(f"Restoring bias voltage to {original_bias}V.")
    nanonis.BiasSet(original_bias)
    time.sleep(0.5)

    nanonis.SetpointSet(100e-12)  # Restore setpoint current to 100 pA
    logging.info("Setpoint current restored to 100 pA.")
    
    logging.info("Turning feedback loop back on for continued scanning.")
    nanonis.FeedbackOnOffSet('On')

    logging.info("Tip retraction complete. Ready for next scan.")

# Main execution entry point
if __name__ == "__main__":
    config = load_config()  # Load configuration from JSON

    # Folders for NPY files, resized, augmented, and filtered images
    npy_output_folder = r'C:\Users\Razer\Downloads\FYP\npy_output'
    resized_output_folder = r'C:\Users\Razer\Downloads\FYP\npy_output\resized'
    augmented_output_folder = r'C:\Users\Razer\Downloads\FYP\npy_output\augmented'
    filtered_save_dir = r'C:\Users\Razer\Downloads\FYP\npy_output\filtered'
    model_path = r'C:\Users\Razer\Documents\pytorch\model\ResNet18_classimb_BCEwithlogitsloss_20240902-013641.pt'

    # Initialize GridManager for grid movement (set grid boundaries and step size)
    grid_manager = GridManager(start_x=0, start_y=0, end_x=1000e-9, end_y=1000e-9, step_size=100e-9)

    nanonis = connect_nanonis(config)
    perform_scan_and_process_images(nanonis, npy_output_folder, resized_output_folder, augmented_output_folder, filtered_save_dir, model_path, grid_manager)

    logging.info("Process completed successfully.")
