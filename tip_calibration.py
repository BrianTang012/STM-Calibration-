import time
import logging
from nanonis_basic import nanonis_basic  # Assuming this is the Nanonis interface

class TipCalibration:
    def __init__(self, nanonis):
        self.nanonis = nanonis

    def perform_controlled_dip(self, initial_z_position, dip_amount=1e-9):
        """
        Perform a controlled dip by lowering the tip by a specified amount.
        
        Args:
            initial_z_position: The Z position at the start (reference).
            dip_amount: The amount to lower the tip by (default 1nm).
        """
        new_z_position = initial_z_position - dip_amount
        self.nanonis.TipZSet(new_z_position)
        logging.info(f"Controlled dip: lowered tip by {dip_amount} to Z={new_z_position} meters.")

    def perform_controlled_retraction(self, initial_z_position, retract_distance=100e-9, retract_steps=10, original_bias=1.0):
        """
        Perform a controlled retraction of the STM tip in steps.
        
        Args:
            initial_z_position: The Z position to return to after retraction.
            retract_distance: Total distance to retract the tip (default 100nm).
            retract_steps: Number of steps to divide the retraction into.
            original_bias: Bias voltage to restore after retraction.
        """
        current_z_position = self.nanonis.TipZGet()
        total_retraction_distance = initial_z_position - current_z_position
        retract_step_size = total_retraction_distance / retract_steps
        
        logging.info(f"Retracting the tip in {retract_steps} steps, each step: {retract_step_size} meters.")

        # Retract the tip in steps
        for step in range(retract_steps):
            current_z_position += retract_step_size
            self.nanonis.TipZSet(current_z_position)
            time.sleep(0.1)
            logging.info(f"Step {step + 1}/{retract_steps}: Tip Z position now {current_z_position} meters")

        # Return to the initial Z position and restore settings
        self.nanonis.TipZSet(initial_z_position)
        logging.info(f"Returned tip to initial Z position: {initial_z_position} meters")
        
        # Restore original bias voltage, setpoint current, and feedback loop
        self.nanonis.BiasSet(original_bias)
        logging.info(f"Restored bias voltage to {original_bias}V.")
        self.nanonis.SetpointSet(100e-12)
        logging.info("Setpoint current restored to 100 pA.")
        self.nanonis.FeedbackOnOffSet('On')
        logging.info("Feedback loop turned on. Retraction complete.")

# Initialize main execution if the script is run directly
if __name__ == "__main__":
    # Configure logging to console for testing purposes
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Replace these with your real IP and PORT for Nanonis
    IP = '127.0.0.1'
    PORT = 6501

    # Create an instance of the Nanonis connection
    nanonis = nanonis_basic(IP, PORT)
    
    # Attempt to connect to the Nanonis device
    try:
        nanonis.connect()
        logging.info("Connected to Nanonis device successfully.")
    except Exception as e:
        logging.error(f"Failed to connect to Nanonis device: {e}")
        exit(1)

    # Create an instance of the TipCalibration class
    tip_calibration = TipCalibration(nanonis)

    # Example initial Z position (replace with actual Z position)
    initial_z_position = nanonis.TipZGet()

    # Perform a controlled dip (e.g., lower by 1nm)
    tip_calibration.perform_controlled_dip(initial_z_position, dip_amount=1e-9)

    # Perform a controlled retraction (e.g., retract by 100nm in 10 steps)
    tip_calibration.perform_controlled_retraction(initial_z_position, retract_distance=100e-9, retract_steps=10, original_bias=1.0)
