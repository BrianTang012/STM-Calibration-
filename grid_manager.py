import logging

class GridManager:
    def __init__(self, start_x, start_y, end_x, end_y, step_size):
        """
        Initializes the GridManager with the starting and ending coordinates of the grid, 
        along with the step size for moving within the grid.
        All parameters are expected to be in meters.
        """
        self.start_x = start_x  # in meters
        self.start_y = start_y  # in meters
        self.end_x = end_x      # in meters
        self.end_y = end_y      # in meters
        self.step_size = step_size  # in meters

        self.current_x = start_x
        self.current_y = start_y

    def get_next_position(self):
        """
        Calculates and returns the next position within the grid based on the current position.
        Moves in a line-by-line fashion, similar to raster scanning.
        """
        # If the current position is outside the grid, return None to indicate the end of the grid
        if self.current_y > self.end_y:
            return None

        # Store the current position to return
        next_position = (self.current_x, self.current_y)

        # Move to the next step along the x-axis
        self.current_x += self.step_size

        # If we've reached the end of the line, move to the next line in the grid
        if self.current_x > self.end_x:
            self.current_x = self.start_x  # Reset x to the start of the next line
            self.current_y += self.step_size  # Move down one step in the y direction

        return next_position

    def reset_position(self):
        """
        Resets the current position back to the starting point of the grid.
        This can be used if you want to start the scanning process from the beginning of the grid.
        """
        self.current_x = self.start_x
        self.current_y = self.start_y
        logging.info("Position reset to the starting point of the grid.")


