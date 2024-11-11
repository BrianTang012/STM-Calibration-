import socket
import atexit
import struct
import threading
import os
import numpy as np
import nanonispy as nap  # NanonisPy library from GitHub
import sys

# Define the module as nanonis_basic

datatype_dict = {
    'int': '>i',
    'uint16': '>H',
    'uint32': '>I',
    'float32': '>f',
    'float64': '>d',
    'string': None  # No fixed size for strings, handled separately, added on 2 Sept
}

datasize_dict = {
    'int': 4,
    'uint16': 2,
    'uint32': 4,
    'float32': 4,
    'float64': 8,
    'string': None  # Variable size for strings, handled separately, added on 2 Sept
}

datatype_py_dict = {
    'int': int,
    'uint16': int,
    'uint32': int,
    'float32': float,
    'float64': float,
}

si_prefix = {
    '': 1.0,
    'a': 1e-18,
    'f': 1e-15,
    'p': 1e-12,
    'n': 1e-9,
    'u': 1e-6,
    'm': 1e-3,
}

python_major_version = sys.version_info.major

class nanonisException(Exception):
    def __init__(self, message):
        super(nanonisException, self).__init__(message)

if python_major_version == 2:
    import thread
elif python_major_version == 3:
    import _thread as thread
else:
    raise nanonisException('Unknown Python version')

def decode_hex_from_string(input_string):
    if python_major_version == 2:
        return input_string.decode('hex')
    elif python_major_version == 3:
        return bytes.fromhex(input_string)
    else:
        raise nanonisException('Unknown Python version')

class nanonis_basic:
    
    def __init__(self, IP='127.0.0.1', PORT=6501):
        self.IP = IP
        self.PORT = PORT
        self.socket = None
        self.lock = threading.Lock()
        self.regex = None

        # Parameter limits in SI units (without prefixes, usually an SI base unit)
        self.BiasLimit = 10
        self.XScannerLimit = 1e-6
        self.YScannerLimit = 1e-6
        self.ZScannerLimit = 1e-7
        self.LowerSetpointLimit = 0
        self.UpperSetpointLimit = 1e-3

        # Executed at normal termination of Python interpreter
        @atexit.register
        def exit_handler():
            self.disconnect()

    def connect(self):
        """Establishes a TCP/IP connection to Nanonis."""
        if self.socket is None:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.IP, self.PORT))
            print(f"Connected to Nanonis at {self.IP}:{self.PORT}")
        else:
            print("Already connected to Nanonis.")

    def disconnect(self):
        """Closes the TCP/IP connection to Nanonis."""
        if self.socket:
            self.socket.close()
            self.socket = None
            print("Disconnected from Nanonis.")
        else:
            print("No active connection to disconnect.")

    def to_binary(self, datatype, input_data):
        """Convert data to binary format."""
        if datatype == 'string':
            return bytes(input_data, 'utf-8')
        try:
            return struct.pack(datatype_dict[datatype], input_data)
        except KeyError:
            raise nanonisException('Unknown Data Type: ' + str(datatype))

    def from_binary(self, datatype, input_data):
        """Convert binary data to the appropriate Python type."""
        if datatype == 'string':
            return input_data.decode('utf-8')
        try:
            return struct.unpack(datatype_dict[datatype], input_data)[0]
        except KeyError:
            raise nanonisException('Unknown Data Type ' + str(datatype))

    def construct_header(self, command_name, body_size, send_response_back=True):
        """Construct the command header."""
        cmd_name_bytes = self.to_binary('string', command_name)
        len_cmd_name_bytes = len(cmd_name_bytes)
        cmd_name_bytes += b'\0' * (32 - len_cmd_name_bytes)  # Pad command name with 0x00 to 32 bytes
        if send_response_back:
            response_flag = b'\x00\x01'  # Tell Nanonis to send a response to client
        else:
            response_flag = b'\0\0'  # Tell Nanonis to not send a response to client
        header = cmd_name_bytes + \
                 self.to_binary('int', body_size) + \
                 response_flag + b'\0\0'
        return header

    def construct_command(self, command_name, *vargs):
        """Construct the full command to be sent."""
        if len(vargs) % 2 != 0:
            raise nanonisException('Unbalanced number of arguments')
        body_size = 0
        body = b''
        datatype = ''
        for idx, arg in enumerate(vargs):
            if idx % 2 == 0:
                #Check to see if the input type is a 1D array
                if arg.split("_")[0] == "1DArr":
                    arrayDims = 1
                    datatype = arg.split("_")[1] #Set the datatype to the second part of arg
                    if type(vargs[idx-1]) == int: #Look for the argument that specifies array length (should be argument before for first array)
                        arrLen = vargs[idx-1] #Set the array length if arg two before current is int (if not will use previously set value)
                        if arrLen != len(vargs[idx+1]):
                            raise nanonisException('Array length ' + str(len(arg)) + ' but input array length is ' + str(arrLen))
                    if datatype == 'string': #Special case to deal with an array of string
                        for string in vargs[idx+1]:
                            body_size += len(string)+4 #Adds on the number of bytes equal to str length + 4 for the integer containing string length
                    else:
                        body_size += datasize_dict[datatype]*arrLen
                
                else:
                    arrayDims = 0
                    datatype = arg
                    if datatype == 'string':
                        if vargs[idx-1] == len(vargs[idx+1]):
                            body_size += vargs[idx-1]
                        else:
                            raise nanonisException('String size is ' + str(len(arg)) + ' but input string length is ' + str(vargs[idx-1]))
                    else:
                        body_size += datasize_dict[datatype]
            else:
                if arrayDims == 0 : #For data that is not in an array
                    body += self.to_binary(datatype, arg)
                if arrayDims == 1:
                    if datatype == 'string': #Special case for string
                        for string in arg:    
                            body += self.to_binary('int', len(string)) #Add an integer with the string length
                            body += self.to_binary(datatype, string) #Add the string to the body
                    else:
                        for value in arg:
                            body += self.to_binary(datatype, value) #Add the value to the command body

        header = self.construct_header(command_name, body_size)
        return header + body

    def send(self, command_name, *vargs):
        """Send the command to Nanonis and return the response."""
        try:
            self.lock.acquire()
            response = self.transmit(self.construct_command(command_name, *vargs))
            returned_command = self.from_binary('string', response[:32])
            body_size = self.from_binary('int', response[32:36])
            body = response[40:]
            # Check to make sure the body size actually matches the body size specified in the response header.
            if body_size != len(body):
                errorMessage = 'Response body size error: ' + \
                               returned_command + ', ' + \
                               str(body_size) + ', ' + \
                               self.from_binary('string', body)
                raise nanonisException(errorMessage)
        except:
            raise
        finally:
            self.lock.release() # Release lock
        return {'command_name': returned_command,
                'body_size': body_size,
                'body': body
                }

    def transmit(self, message):
        """Transmit the command to Nanonis."""
        self.socket.sendall(message)
        return self.socket.recv(4096)

    def parse_response(self, response, *vargs):
        """Parse the response from Nanonis."""
        bytecursor = 0
        parsed = {}
        for idx, arg in enumerate(vargs):
            if arg.split("_")[0] == "1DArr": #Check to see if data type is a 1D array
                #Search for the length of the array from previously parsed arguments (it will be the most recent integer argument)
                for _i in range(len(parsed)):
                    if type(parsed[str(idx-1-_i)]) == int:
                        arrLen = parsed[str(idx-1-_i)]
                        break
                        if idx-1-_i == 0:
                            raise nanonisException('No array length found for 1D array')    
                #array = np.zeros(arrLen) #Create a 1D array to put the values into
                if arg.split("_")[1] == "string": #Special case if the data type is string, need to find the byte size of each element in the array                   
                    array = np.zeros(arrLen, dtype=object)
                    for _j in range(arrLen):
                        bytesize = self.from_binary('int', response['body'][bytecursor:bytecursor + 4]) #Read in the integer that specifies the string length
                        bytecursor += 4
                        array[_j] = self.from_binary('string', response['body'][bytecursor:bytecursor + bytesize]) #Read the string for the current array element
                        bytecursor += bytesize

                else: #For non string data types
                    datatype = arg.split("_")[1]
                    bytesize = datasize_dict[datatype]
                    array = np.zeros(arrLen, dtype=datatype_py_dict[datatype]) #Initialise an array with the correct datatype
                    for _j in range(arrLen):
                        array[_j] = self.from_binary(datatype, response['body'][bytecursor:bytecursor + bytesize])
                        bytecursor += bytesize
                parsed[str(idx)] = array
                
            elif arg.split("_")[0] == "2DArr": #Check to see if data type is 2D array
                #Search for the size of the array from previously parsed arguments (it will be the most recent two integer arguments)
                for _i in range(len(parsed)):
                    if type(parsed[str(idx-1-_i)]) == int:
                        nCols = parsed[str(idx-1-_i)]
                        if type(parsed[str(idx-2-_i)]) == int:
                            nRows = parsed[str(idx-2-_i)]
                        else:
                            raise nanonisException('No number of row information found for 2D array')
                        break
                        if idx-1-_i == 0: #Raise exception if ints are not found
                            raise nanonisException('No array size found for 2D array')    
                if arg.split("_")[1] == "string": #Special case if the data type is string, need to find the byte size of each element in the array                   
                    array = np.zeros([nRows, nCols], dtype=object) #Initialise an array
                    for _i in range(nRows):
                        for _j in range(nCols):
                            bytesize = self.from_binary('int', response['body'][bytecursor:bytecursor + 4]) #Read in the integer that specifies the string length
                            bytecursor += 4
                            array[_i, _j] = self.from_binary('string', response['body'][bytecursor:bytecursor + bytesize]) #Read the string for the current array element
                            bytecursor += bytesize
                else: #For non string data types
                    datatype = arg.split("_")[1]
                    bytesize = datasize_dict[datatype]
                    array = np.zeros([nRows, nCols], dtype=datatype_py_dict[datatype]) #Initialise an array with the correct datatype
                    for _i in range(nRows):
                        for _j in range(nCols):
                            array[_i, _j] = self.from_binary(datatype, response['body'][bytecursor:bytecursor + bytesize])
                            bytecursor += bytesize
                parsed[str(idx)] = array
            #For data not in an array
            else: 
                if arg == 'string':
                    bytesize = parsed[str(idx-1)]
                    parsed[str(idx)] = self.from_binary(arg, response['body'][bytecursor:bytecursor + bytesize])
                    bytecursor += bytesize
                else:
                    bytesize = datasize_dict[arg]
                    parsed[str(idx)] = self.from_binary(arg, response['body'][bytecursor:bytecursor + bytesize])
                    bytecursor += bytesize
        parsed['Error status'] = self.from_binary('uint32', response['body'][bytecursor:bytecursor + 4])
        bytecursor += 4
        parsed['Error size'] = self.from_binary('int', response['body'][bytecursor:bytecursor + 4])
        bytecursor += 4
        if parsed['Error size'] != 0:
            parsed['Error description'] = self.from_binary('string', response['body'][bytecursor:bytecursor + parsed['Error size']])
            bytecursor += parsed['Error size']
        
        # If the total number of bytes requested by the user does not match body_size minus the error size, raise an exception.
        if bytecursor != response['body_size']:
            raise nanonisException('Response parse error: body_size = ' + str(response['body_size']) + ' number of bytes expected is ' + str(bytecursor))
        
        return parsed

    def convert(self, input_data):
        """Convert a number with SI prefix into its numeric value."""
        if self.regex is None:
            import re
            self.regex = re.compile(r'^(-)?([0-9.]+)\s*([A-Za-z]*)$')
        match = self.regex.match(input_data)
        if match is None:
            raise nanonisException('Malformed number: Not a correctly formatted number')
        groups = match.groups()
        if groups[0] is None:
            sign = 1
        elif groups[0] == '-':
            sign = -1
        else:
            pass
        try:
            return sign * float(groups[1]) * si_prefix[groups[2]]
        except KeyError:
            raise nanonisException('Malformed number: SI prefix not recognized')

    # Setting and getting the XY coordinates of the tip
    def TipXYSet(self, X, Y, wait=1):
        """Set the X, Y tip coordinates (m)."""
        if type(X) is str:
            X_val = self.convert(X)
        else:
            X_val = float(X)
        if type(Y) is str:
            Y_val = self.convert(Y)
        else:
            Y_val = float(Y)
        if not (-self.XScannerLimit <= X_val <= self.XScannerLimit):
            raise nanonisException('X out of bounds')
        if not (-self.YScannerLimit <= Y_val <= self.YScannerLimit):
            raise nanonisException('Y out of bounds')
        self.send('FolMe.XYPosSet', 'float64', X_val, 'float64', Y_val, 'uint32', wait)

    def TipXYGet(self, wait=1):
        """Get the X, Y tip coordinates (m)."""
        parsedResponse = self.parse_response(self.send('FolMe.XYPosGet', 'uint32', wait), 'float64', 'float64')
        return {'X': parsedResponse['0'], 'Y': parsedResponse['1']}

    # Setting and getting the Z position (height) of the tip
    def TipZSet(self, Z):
        """Set the Z tip height (m)."""
        if type(Z) is str:
            Z_val = self.convert(Z)
        else:
            Z_val = float(Z)
        if not (-self.ZScannerLimit <= Z_val <= self.ZScannerLimit):
            raise nanonisException('Z out of bounds')
        self.send('ZCtrl.ZPosSet', 'float32', Z_val)

    def TipZGet(self):
        """Get the Z tip height (m)."""
        parsedResponse = self.parse_response(self.send('ZCtrl.ZPosGet'), 'float32')['0']
        return parsedResponse

    # Control the feedback loop for the Z-controller
    def FeedbackOnOffSet(self, feedbackStatus):
        """Turn on/off the Z-controller feedback."""
        if type(feedbackStatus) is str:
            if feedbackStatus.lower() == 'on':
                ZCtrlStatus = 1
            elif feedbackStatus.lower() == 'off':
                ZCtrlStatus = 0
            else:
                raise nanonisException('Feedback On or Off?')
        elif type(feedbackStatus) is int:
            if feedbackStatus == 1:
                ZCtrlStatus = 1
            elif feedbackStatus == 0:
                ZCtrlStatus = 0
            else:
                raise nanonisException('Feedback On or Off?')
        else:
            raise nanonisException('Feedback On or Off?')
        self.send('ZCtrl.OnOffSet', 'uint32', ZCtrlStatus)

    def FeedbackOnOffGet(self):
        """Get the Z-controller feedback status ('On' or 'Off')."""
        parsedResponse = self.parse_response(self.send('ZCtrl.OnOffGet'), 'uint32')['0']
        if parsedResponse == 1:
            return 'On'
        elif parsedResponse == 0:
            return 'Off'
        else:
            raise nanonisException('Unknown Feedback State')

    def Withdraw(self, wait=1, timeout=-1):
        """Turn off the feedback and fully withdraw the tip."""
        self.send('ZCtrl.Withdraw', 'uint32', wait, 'int', timeout)

    def Home(self):
        """Turn off feedback and move the tip to the Home position."""
        self.send('ZCtrl.Home')

    # Setpoint control for the Z-controller
    def SetpointSet(self, setpoint):
        """Set the setpoint value (A)."""
        if type(setpoint) is str:
            setpoint_val = self.convert(setpoint)
        else:
            setpoint_val = float(setpoint)
        if not (self.LowerSetpointLimit <= setpoint_val <= self.UpperSetpointLimit):
            raise nanonisException('Setpoint out of bounds')
        self.send('ZCtrl.SetpntSet', 'float32', setpoint_val)

    def SetpointGet(self):
        """Get the setpoint value (A)."""
        parsedResponse = self.parse_response(self.send('ZCtrl.SetpntGet'), 'float32')['0']
        return parsedResponse

    def CurrentGet(self):
        """Get the value of the current (A)."""
        parsedResponse = self.parse_response(self.send('Current.Get'), 'float32')['0']
        return parsedResponse

    # Bias control for the Z-controller
    def BiasSet(self, bias):
        """Set the bias (V)."""
        if type(bias) is str:
            bias_val = self.convert(bias)
        else:
            bias_val = float(bias)
        if -self.BiasLimit <= bias_val <= self.BiasLimit:
            self.send('Bias.Set', 'float32', bias_val)
        else:
            raise nanonisException('Bias out of bounds')

    def BiasGet(self):
        """Get the bias (V)."""
        return self.parse_response(self.send('Bias.Get'), 'float32')['0']

    # Getting scan frame parameters
    def ScanFrameGet(self):
        """Get the parameters for the scan frame."""
        parsedResponse = self.parse_response(self.send('Scan.FrameGet'), 'float32', 'float32', 'float32', 'float32', 'float32')
        return {'centre': [parsedResponse['0'], parsedResponse['1']], 'size': [parsedResponse['2'], parsedResponse['3']], 'angle': parsedResponse['4']}

    # Setting scan frame parameters
    def ScanFrameSet(self, center_x, center_y, size_x, size_y, angle=0):
        """Set the parameters for the scan frame.

        Args:
            center_x (float): X coordinate of the center of the scan frame (in meters).
            center_y (float): Y coordinate of the center of the scan frame (in meters).
            size_x (float): Width of the scan frame (in meters).
            size_y (float): Height of the scan frame (in meters).
            angle (float): Rotation angle of the scan frame (in degrees, optional).
        """
        # Ensure the connection is established
        self.ensure_connection()

        # Convert size and center to meters if given in other units
        if isinstance(center_x, str):
            center_x = self.convert(center_x)
        if isinstance(center_y, str):
            center_y = self.convert(center_y)
        if isinstance(size_x, str):
            size_x = self.convert(size_x)
        if isinstance(size_y, str):
            size_y = self.convert(size_y)

        # Send the command to set the scan frame
        self.send('Scan.FrameSet', 'float32', center_x, 'float32', center_y, 'float32', size_x, 'float32', size_y, 'float32', angle)
        print(f"Scan frame set to center: ({center_x}, {center_y}), size: ({size_x} x {size_y}), angle: {angle}°")

    # Starting, stopping, and controlling the direction of a scan
    def ScanAction(self, action, direction):
        """Sets a scan action and associated direction."""
        if isinstance(action, str):
            if action.lower() == 'start':
                action = 0
            elif action.lower() == 'stop':
                action = 1
            elif action.lower() == 'pause':
                action = 2
            elif action.lower() == 'resume':
                action = 3
            else:
                raise nanonisException('Invalid argument for action')

        if isinstance(direction, str):
            if direction.lower() == 'down':
                direction = 0
            elif direction.lower() == 'up':
                direction = 1
            else:
                raise nanonisException('Invalid argument for direction')

        self.send('Scan.Action', 'uint16', action, 'uint32', direction)

    np.float = np.float64 #added on 8 Sept
    np.int = int #added on 8 Sept

    # Convert SXM files to NPY format (NanonisPy)
    def convert_sxm_to_npy(self, sxm_file_path, output_folder):
        """Convert a single SXM file to a NumPy array and save it in the specified output folder."""
        try:
            # Read the SXM file
            sxm_data = nap.read.Scan(sxm_file_path)
            
            # Print the available channel names
            available_channels = sxm_data.signals.keys()
            print(f"Available channel names in {os.path.basename(sxm_file_path)}:", available_channels)
            
            # Choose the first available channel
            if available_channels:
                channel_name = next(iter(available_channels))
                # Access the data for the chosen channel
                channel_data = sxm_data.signals[channel_name]['forward']
                print(f"Accessing data from channel: {channel_name}")
                
                # Construct the output file path for the NumPy array
                numpy_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(sxm_file_path))[0]}_{channel_name}_data.npy")
                # Save the data as a NumPy array
                np.save(numpy_file_path, channel_data)
                print(f"Data saved as {numpy_file_path}")
            else:
                print(f"No channels found in {os.path.basename(sxm_file_path)}.")
                
        except Exception as e:
            print(f"An unexpected error occurred during conversion: {e}")
    
    # Wait for the current scan to complete and convert the result to NPY format
    def ScanWaitEndOfScan(self, timeout=-1, npy_output_folder=None):
        """Waits for the current scan to finish before returning and converts SXM to NPY format."""
        parsedResponse = self.parse_response(self.send('Scan.WaitEndOfScan', 'int', timeout), 'uint32', 'uint32', 'string')
        filePath = parsedResponse.get('2')  # Adjust the index based on your actual parsing sequence
    
        if filePath and npy_output_folder:
            print(f"Scan completed. File saved at: {filePath}")
            # Ensure the output folder exists
            os.makedirs(npy_output_folder, exist_ok=True)
            # Convert the SXM file to NPY using the specified output folder
            self.convert_sxm_to_npy(filePath, npy_output_folder)
        else:
            print("Scan did not complete successfully or no file path was provided.")
    
        return parsedResponse

    def ensure_connection(self):
        """Ensures that a connection to Nanonis is established."""
        if self.socket is None:
            self.connect()

    # Additional Methods from nanonis_programming_interface

    def AtomTrackCtrlSet(self, control, status):
        """Turns the selected Atom Tracking control (modulation, controller or drift measurement) on or off."""
        # Convert control input to the necessary format
        if isinstance(control, str):
            if control.lower() == 'modulation':
                control = 0
            elif control.lower() == 'controller':
                control = 1
            elif control.lower() == 'drift':
                control = 2
            else:
                raise nanonisException('Invalid atom tracking control')
        
        # Convert from string to int if necessary
        if isinstance(status, str):
            if status.lower() == 'on':
                on = 1
            elif status.lower() == 'off':
                on = 0
            else:
                raise nanonisException('Feedback On or Off?')
        # Send the command
        if on:
            self.send('AtomTrack.CtrlSet', 'uint16', control, 'uint16', 1)
        else:
            self.send('AtomTrack.CtrlSet', 'uint16', control, 'uint16', 0)

    def AtomTrackStatusGet(self, control):
        """Get the status of the atom tracking control module."""
        if isinstance(control, str):
            if control.lower() == 'modulation':
                control = 0
            elif control.lower() == 'controller':
                control = 1
            elif control.lower() == 'drift':
                control = 2
            else:
                raise nanonisException('Invalid atom tracking control')
                
        parsedResponse = self.parse_response(self.send('AtomTrack.StatusGet', 'uint16', control), 'uint16')['0']
        return parsedResponse
    
    def AtomTrackPropsGet(self):
        """Get the atom tracking parameters."""
        parsedResponse = self.parse_response(self.send('AtomTrack.PropsGet'), 'float32', 'float32', 'float32', 'float32', 'float32')
        # Check to see if error has been returned
        if parsedResponse['Error status']:
            raise nanonisException('Error executing AtomTrackPropsGet')
        else:
            return {'iGain': parsedResponse['0'], 'freq': parsedResponse['1'], 'amplitude': parsedResponse['2'], 'phase': parsedResponse['3'], 'soDelay': parsedResponse['4']}
    
    def FolMePSOnOffSet(self, psStatus):
        """Set the point and shoot option in follow me to on or off."""
        if isinstance(psStatus, str):
            if psStatus.lower() == 'on':
                psStatus = 1
            elif psStatus.lower() == 'off':
                psStatus = 0
            else:
                raise nanonisException('Point and shoot On or Off?')
        elif isinstance(psStatus, int):
            if psStatus != 1 and psStatus != 0:
                raise nanonisException('Invalid point and shoot status value, use 0 for off and 1 for on')
        else:
            raise nanonisException('Invalid point and shoot status argument, expected int or string')
            
        self.send('FolMe.PSOnOffSet', 'uint32', psStatus)
        
    def ZCtrlTipLiftSet(self, tipLift):
        """Set the value of the Z controller 'tipLift' (the amount the tip moves in Z when Z controller is turned off)."""
        if isinstance(tipLift, str):
            tipLiftVal = self.convert(tipLift)
        else:
            tipLiftVal = float(tipLift)
        if not (-self.ZScannerLimit <= tipLiftVal <= self.ZScannerLimit):
            raise nanonisException('Z out of bounds')
        self.send('ZCtrl.TipLiftSet', 'float32', tipLiftVal)
        
    def PiezoDriftCompGet(self):
        """Get the drift compensation parameters applied to the piezos."""
        parsedResponse = self.parse_response(self.send('Piezo.DriftCompGet'), 'uint32', 'float32', 'float32', 'float32', 'uint32', 'uint32', 'uint32', 'float32')
        if parsedResponse['Error status']:
            raise nanonisException('Error executing PiezoDriftCompGet')
        else:
            return {'Status': parsedResponse['0'], 'Vx': parsedResponse['1'], 'Vy': parsedResponse['2'], 'Vz': parsedResponse['3'], 'Xsat': bool(parsedResponse['4']), 'Ysat': bool(parsedResponse['5']), 'Zsat': bool(parsedResponse['6']), 'SatLim': bool(parsedResponse['7'])}
        
    def PiezoDriftCompSet(self, on, Vxyz, satLim=10):
        """Set the drift compensation parameters applied to the piezos."""
        # Convert Vxyz values if input as strings
        for Vn, i in enumerate(Vxyz):
            if isinstance(Vn, str):
                Vxyz[i] = self.convert(Vn)
        
        self.send('Piezo.DriftCompSet', 'int', on, 'float32', Vxyz[0], 'float32', Vxyz[1], 'float32', Vxyz[2], 'float32', satLim)
        
    def ZSpectrPropsGet(self):
        """Get the Z Spectroscopy parameters."""
        parsedResponse = self.parse_response(self.send('ZSpectr.PropsGet'), 'uint16', 'int', 'string', 'string', 'string', 'uint16', 'uint16')
        
        # Check if an error has been returned
        if parsedResponse['Error status']:
            raise nanonisException('Error executing ZSpectrPropsGet')
        else:
            # Parse the channels, parameters, and fixed parameters arrays
            channels = parsedResponse['2'].split('\n')
            parameters = parsedResponse['3'].split('\n')
            fixedParameters = parsedResponse['4'].split('\n')
            
            return {
                'backwardSweep': parsedResponse['0'],
                'numPoints': parsedResponse['1'],
                'channels': channels,
                'parameters': parameters,
                'fixedParameters': fixedParameters,
                'numSweeps': parsedResponse['5'],
                'saveAll': parsedResponse['6']
            }
        
    def SignalsNamesGet(self):
        """Get the list of channel names as a numpy array. Numpy array index corresponds to channel index in nanonis."""
        parsedResponse = self.parse_response(self.send('Signals.NamesGet'), 'int', 'int', '1DArr_string')
        return parsedResponse['2']
        
    def SignalsValGet(self, signal, waitForNewest=True):
        """Get the value of a specified signal."""
        # Convert waitForNewest into an integer
        if waitForNewest:
            wait = 1
        else:
            wait = 0
        parsedResponse = self.parse_response(self.send('Signals.ValGet', 'int', signal, 'uint32', wait), 'float32')
        
        return parsedResponse['0']