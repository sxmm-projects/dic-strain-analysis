import base64
import io
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from threading import Event, Thread

import cv2
import gxipy as gx
import numpy as np
import pandas as pd
import webview
from flask import Flask, Response, jsonify, render_template, request
from flask_socketio import SocketIO, emit
from lxml import etree
from PIL import Image

from dem_concentricCircleDetection import findKeypoints

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

font = cv2.FONT_HERSHEY_SIMPLEX
color = (255, 0, 0)
SIZE_IMAGE = (800, 600)

capturing = Event()
capturing.clear()
pause = Event()
pause.clear()
switch = False
cam1 = None
cam2 = None

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

templates_path = resource_path("templates")
static_path = resource_path("static")

app = Flask(__name__, template_folder=templates_path, static_folder=static_path)
socketio = SocketIO(app)
window = webview.create_window('SSCs-gui', app)

# Load the configuration file
with open('config.json', 'r') as f:
    config = json.load(f)

DICe_exe_path = config.get('DICe_exe_path')

class Camera:
    def __init__(self, name, path, index):
        self.name = name
        self.index = index
        self.cam = None
        self.frame_count = 0
        self.start_time = time.time()
        self.path = path
        self.__set_path()
        self._connect()

    def __set_path(self):
        Path(self.path).mkdir(parents=True, exist_ok=True)

    def _connect(self):
        self.device_manager = gx.DeviceManager()
        self.dev_num, self.dev_info_list = self.device_manager.update_device_list()
        if self.dev_num <= self.index:
            raise Exception(f"No device found at index {self.index}")
        self.cam = self.device_manager.open_device_by_index(self.index)

    def stream_on(self):
        self.cam.stream_on()
        self.cam.ExposureTime.set(12000)

    def stream_off(self):
        self.cam.stream_off()

    def get_img(self):
        raw_image = self.cam.data_stream[0].get_image()
        if raw_image is None:
            return None
        numpy_image = raw_image.get_numpy_array()
        return numpy_image

    def save_image(self, count, img):
        output_name = Path(self.path) / f'img{count}.jpg'
        cv2.imwrite(str(output_name), img)

def capture_images(frequency, min, sec, burstmode, shotmode):
    global capturing, pause
    if frequency == '':
        frequency = 12
    if min == '' and sec == '':
        min = 1
        sec = 30
    elif min == '':
        min = 0
    elif sec == '':
        sec = 0

    total_time = int(min) * 60 + int(sec)
    start_time = time.time()

    if burstmode:
        count_1 = 0
        count_2 = 0
        while capturing.is_set() and (time.time() - start_time < total_time):
            new_frame_time = time.time()

            img1 = cam1.get_img()
            img2 = cam2.get_img()

            while pause.is_set():
                pass

            if img1 is not None:
                img1_resize = cv2.resize(img1, SIZE_IMAGE, interpolation=cv2.INTER_LINEAR)
                cam1.save_image(count_1, img1_resize)
                count_1 += 1

            if img2 is not None:
                img2_resize = cv2.resize(img2, SIZE_IMAGE, interpolation=cv2.INTER_LINEAR)
                cam2.save_image(count_2, img2_resize)
                count_2 += 1

            time_error = time.time() - new_frame_time
            if time_error < (1 / int(frequency)):
                time.sleep((1 / int(frequency)) - time_error)
    elif shotmode:
        img1 = cam1.get_img()
        img2 = cam2.get_img()

        if img1 is not None:
            cam1.save_image(0, img1)

        if img2 is not None:
            cam2.save_image(0, img2)

    capturing.clear()
    
@app.route('/start_realtime_keypoints', methods=['POST'])
def start_realtime_keypoints():
    try:
        data = request.get_json()
        image_folder = data.get('image_folder', 'uploads')

        if not os.path.exists(image_folder) or not os.listdir(image_folder):
            return jsonify({'status': 'error', 'message': 'No images found in the specified folder'}), 400

        thread = Thread(target=process_images_realtime, args=(image_folder,))
        thread.start()

        return jsonify({'status': 'success', 'message': 'Real-time keypoint detection started'})
    except Exception as e:
        logging.error(f"Error in start_realtime_keypoints: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

def process_images_realtime(image_folder):
    """Process images in real time and emit results to the frontend."""
    image_files = sorted(os.listdir(image_folder))  
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                logging.warning(f"Unable to read image: {image_path}")
                continue
            
            frame_with_keypoints = detect_keypoints(frame)
            
            _, buffer = cv2.imencode('.jpg', frame_with_keypoints)
            img_str = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('realtime_keypoint_frame', {'image': img_str})
            
            time.sleep(0.1)
        except Exception as e:
            logging.error(f"Error processing {image_path}: {str(e)}")


def detect_keypoints(frame):
    """Detect keypoints in the frame and return the processed frame."""
    try:
        orb = cv2.ORB_create()
        keypoints = orb.detect(frame, None)
        frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0))

        logging.info(f"Detected {len(keypoints)} keypoints.")
        return frame_with_keypoints
    except Exception as e:
        logging.error(f"Error detecting keypoints: {str(e)}")
        return frame 

def detect_keypoints_in_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create()
    keypoints = orb.detect(img, None)
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))
    processed_image_path = Path(image_path).stem + "_keypoints.jpg"
    cv2.imwrite(processed_image_path, img_with_keypoints)
    return processed_image_path

def generate_frames(image_folder):
    """Generate frames from a folder of images and detect keypoints."""
    try:
        image_files = sorted(os.listdir(image_folder))  
        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            frame = cv2.imread(image_path)
            if frame is not None:
                frame_with_keypoints = detect_keypoints(frame)
                _, buffer = cv2.imencode('.jpg', frame_with_keypoints)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
                _, buffer = cv2.imencode('.jpg', frame_with_keypoints)
                img_str = base64.b64encode(buffer).decode('utf-8')
                socketio.emit('realtime_keypoint_frame', {'image': img_str})
                time.sleep(0.1)
    except Exception as e:
        logging.error(f"Error in generate_frames: {str(e)}")
        
def log_message(message):
    print(message)
    socketio.emit('log_update', {'message': message})

def process_images_realtime(image_folder):
    """Process images in real time and emit results to the frontend."""
    image_files = sorted(os.listdir(image_folder))
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder, image_file)
        log_message(f"Processing image {idx + 1}/{len(image_files)}: {image_file}")
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                log_message(f"Error: Unable to read image: {image_file}")
                continue
            frame_with_keypoints = detect_keypoints(frame)
            _, buffer = cv2.imencode('.jpg', frame_with_keypoints)
            img_str = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('realtime_keypoint_frame', {'image': img_str})
            time.sleep(0.1)
        except Exception as e:
            log_message(f"Error processing {image_file}: {str(e)}")



@app.route('/')
def index():
    return render_template("index.html")

@app.route('/calibration')
def calibration():
    return render_template("calibration.html")

@app.route('/strain_cal', methods=['GET'])
def strain_cal():
    try:
        logging.debug(f"Attempting to run executable: {DICe_exe_path}")

        if os.path.exists(DICe_exe_path):
            subprocess.Popen([DICe_exe_path], shell=True)
            logging.debug("Executable started successfully.")
            return jsonify({"status": "success"})
        else:
            logging.error("Executable not found.")
            return jsonify({"status": "error", "message": "Executable not found."})
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/start_capture', methods=['POST'])
def start_capture():
    global capturing
    if not capturing.is_set() and switch:
        data = request.get_json()
        frequency = data.get('frequency')
        min = data.get('min')
        sec = data.get('sec')
        burstmode = data.get('burstmode')
        shotmode = data.get('shotmode')
        capturing.set()
        capture_thread = Thread(target=capture_images, args=(frequency, min, sec, burstmode, shotmode,))
        capture_thread.start()
        return jsonify({'status': 'Capture started'})
    else:
        return jsonify({'status': 'Capture already in progress'})

@app.route('/pause_capture')
def pause_capture():
    global pause
    if pause.is_set():
        pause.clear()
        return jsonify({'status': 'Capture paused'})
    else:
        pause.set()
        return jsonify({'status': 'Capture resumed'})

@app.route('/stop_capture')
def stop_capture():
    global capturing
    if capturing.is_set():
        capturing.clear()
        return jsonify({'status': 'Capture stopped'})
    else:
        return jsonify({'status': 'No capture in progress'})

@app.route('/camera')
def camera():
    global switch
    if not switch:
        switch = True
        print("Turn on")
        cam1.stream_on()
        cam2.stream_on()
    else:
        switch = False
        print("Turn off")
        cam1.stream_off()
        cam2.stream_off()

    return jsonify({'status': 'Camera status toggled', 'switch': switch})

@app.route('/cam1_route')
def cam1_route():
    if cam1 is not None:
        return Response(generate_frames(cam1), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/cam2_route')
def cam2_route():
    if cam2 is not None:
        return Response(generate_frames(cam2), mimetype='multipart/x-mixed-replace; boundary=frame')

# File upload route
@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        parameter_file = request.files.get('cal_input_file')  
        keypoints_left_file = request.files.get('keypoints_left_file')  
        keypoints_right_file = request.files.get('keypoints_right_file')  
        
        save_path = r'C:\Sxmm\4th years\Project\exe_srtaincal\Senior work1'  

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if parameter_file:
            parameter_file.save(os.path.join(save_path, 'cal_input.xml'))
        if keypoints_left_file:
            keypoints_left_file.save(os.path.join(save_path, 'keypoints_left.xml'))
        if keypoints_right_file:
            keypoints_right_file.save(os.path.join(save_path, 'keypoints_right.xml'))

        return jsonify({'status': 'success', 'message': 'Files uploaded successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# Parsing input or read parameters from XML 
def parse_input_parameters(xml_file):
    tree = etree.parse(xml_file)
    root = tree.getroot()

#store the parameter
    params = {}
    for param in root.findall(".//Parameter"):
        name = param.attrib['name']
        value = param.attrib['value']
        type_ = param.attrib['type']

        # Convert types accordingly
        if type_ == 'int':
            value = int(value)
        elif type_ == 'double':
            value = float(value)
        elif type_ == 'bool':
            value = value.lower() == 'true'
        params[name] = value

    return params

# Parsing keypoints from XML
def parse_keypoints_xml(keypoints_file):
    tree = etree.parse(keypoints_file)
    root = tree.getroot()

#store keypoint data
    keypoints = {}
    for img in root.findall(".//ParameterList[@name]"):
        img_name = img.attrib['name']
        all_corners = []
        for row in img.findall(".//ParameterList[@name]"):
            x_values = row.find("Parameter[@name='X']")
            y_values = row.find("Parameter[@name='Y']")

            if x_values is not None and y_values is not None:
                x_values = list(map(float, x_values.attrib['value'].strip("{}").split(",")))
                y_values = list(map(float, y_values.attrib['value'].strip("{}").split(",")))

#matching the corner
                if len(x_values) == len(y_values):
                    all_corners.extend(list(zip(x_values, y_values)))
                else:
                    print(f"Skipping {row.attrib['name']} in {img_name}: Mismatched X and Y lengths.")
            else:
                print(f"Skipping {row.attrib['name']} in {img_name}: Missing X or Y data.")

        if all_corners:
            keypoints[img_name] = all_corners

    if not keypoints:
        print(f"Warning: No keypoints found in the XML file: {keypoints_file}")

    return keypoints

# Match left and right images based on extracted index
def extract_index(filename):
    return filename.replace('left', '').replace('right', '').replace('.jpg', '')

# Calibration functions
def calibrate_single_camera(objpoints, imgpoints, img_size, camera_matrix, dist_coeffs):
    camera_matrix = cv2.initCameraMatrix2D(objpoints, imgpoints, img_size)
    dist_coeffs = np.zeros((5, 1))  # Initial assumption of zero distortion

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, camera_matrix, dist_coeffs,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_PRINCIPAL_POINT
    )
    return camera_matrix, dist_coeffs, rvecs, tvecs

def calibrate_stereo_camera(objpoints, imgpoints_left, imgpoints_right, camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, img_size):
    R = np.eye(3, dtype=np.float64)  # Identity matrix for rotation
    T = np.zeros((3, 1), dtype=np.float64)  # Zero vector for translation
    
    ret, camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        camera_matrix_left, dist_coeffs_left,
        camera_matrix_right, dist_coeffs_right,
        img_size, R, T,
        flags=cv2.CALIB_FIX_INTRINSIC + cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_ZERO_TANGENT_DIST
    )
    return R, T, E, F

def compute_epipolar_error(imgpoints_left, imgpoints_right, camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, F):
    total_error = 0
    num_points = 0

    for i in range(len(imgpoints_left)):
        # Undistort points
        undistorted_left = cv2.undistortPoints(imgpoints_left[i], camera_matrix_left, dist_coeffs_left, P=camera_matrix_left)
        undistorted_right = cv2.undistortPoints(imgpoints_right[i], camera_matrix_right, dist_coeffs_right, P=camera_matrix_right)

        # Compute epipolar lines
        lines1 = cv2.computeCorrespondEpilines(undistorted_right.reshape(-1, 1, 2), 2, F)
        lines1 = lines1.reshape(-1, 3)

        lines2 = cv2.computeCorrespondEpilines(undistorted_left.reshape(-1, 1, 2), 1, F)
        lines2 = lines2.reshape(-1, 3)

        for j in range(len(undistorted_left)):
            error1 = abs(undistorted_left[j][0][0] * lines1[j][0] + undistorted_left[j][0][1] * lines1[j][1] + lines1[j][2])
            error2 = abs(undistorted_right[j][0][0] * lines2[j][0] + undistorted_right[j][0][1] * lines2[j][1] + lines2[j][2])
            total_error += (error1 + error2)
            num_points += 1

    mean_error = total_error / num_points
    return mean_error

# Stereo rectification
def rectify_stereo(camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, R, T, img_size):
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, img_size, R, T)
    return R1, R2, P1, P2

# Save calibration data to XML, including Skew Parameter (Fs)
def save_calibration_to_xml(filename, mtx_left, dist_left, mtx_right, dist_right, R, T, F, img_size, epipolar_error):
    root = etree.Element("ParameterList")
    etree.SubElement(root, "Parameter", name="xml_file_format", type="string", value="DICe_xml_camera_system_file")
    etree.SubElement(root, "Parameter", name="system_type_3D", type="string", value="OPENCV")
    etree.SubElement(root, "Parameter", name="extrinsics_relative_camera_to_camera", type="bool", value="true")
    etree.SubElement(root, "Parameter", name="avg_epipolar_error", type="double", value=f"{epipolar_error:.12e}")

    # Save left camera parameters (CAMERA 0)
    camera_left = etree.SubElement(root, "ParameterList", name="CAMERA 0")
    etree.SubElement(camera_left, "Parameter", name="CX", type="double", value=f"{mtx_left[0,2]:.12e}")
    etree.SubElement(camera_left, "Parameter", name="CY", type="double", value=f"{mtx_left[1,2]:.12e}")
    etree.SubElement(camera_left, "Parameter", name="FX", type="double", value=f"{mtx_left[0,0]:.12e}")
    etree.SubElement(camera_left, "Parameter", name="FY", type="double", value=f"{mtx_left[1,1]:.12e}")
    etree.SubElement(camera_left, "Parameter", name="Fs", type="double", value=f"{mtx_left[0,1]:.12e}")  # Skew Parameter (Fs)

    k1 = dist_left[0, 0] if dist_left.size > 0 else 0.0
    k2 = dist_left[1, 0] if dist_left.size > 1 else 0.0
    k3 = dist_left[4, 0] if dist_left.size > 4 else 0.0

    etree.SubElement(camera_left, "Parameter", name="K1", type="double", value=f"{k1:.12e}")
    etree.SubElement(camera_left, "Parameter", name="K2", type="double", value=f"{k2:.12e}")
    etree.SubElement(camera_left, "Parameter", name="K3", type="double", value=f"{k3:.12e}")
    etree.SubElement(camera_left, "Parameter", name="LENS_DISTORTION_MODEL", type="string", value="OPENCV_LENS_DISTORTION")

    # Default TX, TY, TZ and Identity Rotation Matrix for CAMERA 0
    etree.SubElement(camera_left, "Parameter", name="TX", type="double", value="0.000000000000e+00")
    etree.SubElement(camera_left, "Parameter", name="TY", type="double", value="0.000000000000e+00")
    etree.SubElement(camera_left, "Parameter", name="TZ", type="double", value="0.000000000000e+00")
    rotation_matrix_left = etree.SubElement(camera_left, "ParameterList", name="rotation_3x3_matrix")
    etree.SubElement(rotation_matrix_left, "Parameter", name="ROW 0", type="string", value="{ 1.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00 }")
    etree.SubElement(rotation_matrix_left, "Parameter", name="ROW 1", type="string", value="{ 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00 }")
    etree.SubElement(rotation_matrix_left, "Parameter", name="ROW 2", type="string", value="{ 0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00 }")

    etree.SubElement(camera_left, "Parameter", name="IMAGE_HEIGHT_WIDTH", type="string", value=f"{{ {img_size[0]}, {img_size[1]} }}")
    etree.SubElement(camera_left, "Parameter", name="PIXEL_DEPTH", type="int", value="-1")

    # Save right camera parameters (CAMERA 1)
    camera_right = etree.SubElement(root, "ParameterList", name="CAMERA 1")
    etree.SubElement(camera_right, "Parameter", name="CX", type="double", value=f"{mtx_right[0,2]:.12e}")
    etree.SubElement(camera_right, "Parameter", name="CY", type="double", value=f"{mtx_right[1,2]:.12e}")
    etree.SubElement(camera_right, "Parameter", name="FX", type="double", value=f"{mtx_right[0,0]:.12e}")
    etree.SubElement(camera_right, "Parameter", name="FY", type="double", value=f"{mtx_right[1,1]:.12e}")
    etree.SubElement(camera_right, "Parameter", name="Fs", type="double", value=f"{mtx_right[0,1]:.12e}")  # Skew Parameter (Fs)
    k1_right = dist_right[0, 0] if dist_right.size > 0 else 0.0
    k2_right = dist_right[1, 0] if dist_right.size > 1 else 0.0
    k3_right = dist_right[4, 0] if dist_right.size > 4 else 0.0

    etree.SubElement(camera_right, "Parameter", name="K1", type="double", value=f"{k1_right:.12e}")
    etree.SubElement(camera_right, "Parameter", name="K2", type="double", value=f"{k2_right:.12e}")
    etree.SubElement(camera_right, "Parameter", name="K3", type="double", value=f"{k3_right:.12e}")
    etree.SubElement(camera_right, "Parameter", name="LENS_DISTORTION_MODEL", type="string", value="OPENCV_LENS_DISTORTION")

    # TX, TY, TZ and Rotation Matrix from calibration for CAMERA 1
    etree.SubElement(camera_right, "Parameter", name="TX", type="double", value=f"{T[0,0]:.12e}")   
    etree.SubElement(camera_right, "Parameter", name="TY", type="double", value=f"{T[1,0]:.12e}")
    etree.SubElement(camera_right, "Parameter", name="TZ", type="double", value=f"{T[2,0]:.12e}")

    rotation_matrix_right = etree.SubElement(camera_right, "ParameterList", name="rotation_3x3_matrix")
    for i in range(3):
        etree.SubElement(rotation_matrix_right, "Parameter", name=f"ROW {i}", type="string", value=f"{{ {R[i,0]:.12e}, {R[i,1]:.12e}, {R[i,2]:.12e} }}")
    etree.SubElement(camera_right, "Parameter", name="IMAGE_HEIGHT_WIDTH", type="string", value=f"{{ {img_size[0]}, {img_size[1]} }}")
    etree.SubElement(camera_right, "Parameter", name="PIXEL_DEPTH", type="int", value="-1")

    # Save the XML tree to file
    tree = etree.ElementTree(root)
    with open(filename, "wb") as fh:
        tree.write(fh, pretty_print=True, xml_declaration=True, encoding="UTF-8")

    print(f"Calibration data saved to {filename}")

# Function for emitting console output
def emit_console_output(message):
    """Emit console output to the GUI via WebSocket."""
    socketio.emit('console_output', message)
    
def findKeypoints(img, coordRatio, innerRatio, xNum, yNum, loc, sd, biFac):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create()
        keypoints = orb.detect(gray, None)
        logging.debug(f"Detected {len(keypoints)} keypoints")

        if len(keypoints) < xNum * yNum:
            raise ValueError(f"Invalid keypoints detected: {len(keypoints)} < {xNum * yNum}")

        img_with_keypoints = cv2.drawKeypoints(gray, keypoints, None, color=(0, 255, 0))
        
        _, buffer = cv2.imencode('.jpg', img_with_keypoints)
        img_str = base64.b64encode(buffer).decode('utf-8')
        socketio.emit('realtime_keypoint_frame', {'image': img_str})
        
        for i, kp in enumerate(keypoints[:xNum * yNum]):
            loc[i // yNum, i % yNum] = kp.pt

        return sd, biFac  # Returning the original values unchanged
    except Exception as e:
        socketio.emit('console_output', f"Error in findKeypoints: {str(e)}")
        return sd, biFac


def process_images_in_folder(folder_path, xNum, yNum, coordRatio, innerRatio):
    images = sorted(os.listdir(folder_path))  
    loc = np.zeros((len(images), xNum, yNum, 2), dtype=float)
    sd, biFac = 2.0, 1.5 

    for idx, image_name in enumerate(images):
        image_path = os.path.join(folder_path, image_name)
        img = cv2.imread(image_path)

        if img is None:
            socketio.emit('console_output', f"Skipping {image_name}: Unable to read the image.")
            continue

        sd, biFac = findKeypoints(img, coordRatio, innerRatio, xNum, yNum, loc[idx], sd, biFac)

        time.sleep(0.5)

@app.route('/start-keypoint-detection', methods=['POST'])
def start_keypoint_detection():
    try:
        data = request.json
        app.logger.debug(f"Received data for keypoint detection: {data}")

        xNum = int(data['xNum'])
        yNum = int(data['yNum'])
        coordOuter = float(data['coordOuter'])
        coordInner = float(data['coordInner'])
        fieldOuter = float(data['fieldOuter'])
        fieldInner = float(data['fieldInner'])
        patternSpacing = float(data['patternSpacing'])
        offsetX = float(data['offsetX'])
        offsetY = float(data['offsetY'])

        imgNumberStart = int(data['imgNumberStart'])
        imgNumberEnd = int(data['imgNumberEnd'])
        imgDirectories = data['imgDirectories']
        datDirectories = data['datDirectories']

        app.logger.info(f"Keypoint detection parameters: {xNum}, {yNum}, {coordOuter}, {coordInner}, {patternSpacing}, {offsetX}, {offsetY}")

        coordRatio = coordOuter / coordInner
        innerRatio = fieldOuter / fieldInner
        imgNamePrefix = ["left", "right"]
        viewNum = imgNumberEnd - imgNumberStart + 1
        loc = np.zeros((viewNum, xNum, yNum, 2), dtype=float)
        shape = loc[0].shape
        sd, biFac = 2.0, 1.5 

        for camIdx, (imgDirectory, datDirectory) in enumerate(zip(imgDirectories, datDirectories)):
            outdir = Path(f"{datDirectory}/{imgNamePrefix[camIdx]}")
            os.makedirs(outdir, exist_ok=True)
            socketio.emit('console_output', f"Processing Camera {camIdx + 1}...")

            for viewIdx in range(viewNum):
                imgNumber = imgNumberStart + viewIdx
                savedFile = Path(f"{outdir}/{imgNamePrefix[camIdx]}{imgNumber}.csv")
                
                if savedFile.exists():
                    loc[viewIdx] = pd.read_csv(savedFile, header=None).to_numpy().reshape(shape[0], shape[1], shape[2])
                    socketio.emit('console_output', f"Keypoints for View {viewIdx + 1} loaded from {savedFile}.")
                    continue

                img_path = Path(imgDirectory[viewIdx])
                if not img_path.exists():
                    socketio.emit('console_output', f"Processing {img_path}...")
                    socketio.emit('console_output', f"Error: File does not exist: {img_path}")
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    socketio.emit('console_output', f"Error: Unable to read image: {img_path}")
                    continue

                try:
                    sd, biFac = findKeypoints(img, coordRatio, innerRatio, xNum, yNum, loc[viewIdx], sd, biFac)
                    pd.DataFrame(loc[viewIdx].reshape(shape[0], -1)).to_csv(savedFile, header=False, index=False)
                    socketio.emit('console_output', f"Keypoints for {img_path} saved to {savedFile}")
                except Exception as e:
                    socketio.emit('console_output', f"Error detecting keypoints in {img_path}: {str(e)}")

        return jsonify({"success": True, "message": "Keypoint detection completed."})

    except Exception as e:
        logging.error(f"Error in start_keypoint_detection: {str(e)}")
        socketio.emit('console_output', f"Error: {str(e)}")
        return jsonify({"success": False, "message": str(e)})


if __name__ == '__main__':
    socketio.run(app, debug=True)

    
@app.route('/calibrate', methods=['POST'])
def calibrate():
    try:
        # Load keypoints XML files
        keypoints_left_path = os.path.join(r'C:\Sxmm\4th years\Project\exe_srtaincal\Senior work1', 'keypoints_left.xml')
        keypoints_right_path = os.path.join(r'C:\Sxmm\4th years\Project\exe_srtaincal\Senior work1', 'keypoints_right.xml')
        cal_input_path = os.path.join(r'C:\Sxmm\4th years\Project\exe_srtaincal\Senior work1', 'cal_input.xml')

        keypoints_left = parse_keypoints_xml(keypoints_left_path)
        keypoints_right = parse_keypoints_xml(keypoints_right_path)

        input_params = parse_input_parameters(cal_input_path)
        num_fiducials_x = input_params['num_cal_fiducials_x']
        num_fiducials_y = input_params['num_cal_fiducials_y']
        target_spacing = input_params['cal_target_spacing_size']
        img_size = (2448, 2048)  

        objp = np.zeros((num_fiducials_x * num_fiducials_y, 3), np.float32)
        objp[:, :2] = np.mgrid[0:num_fiducials_y, 0:num_fiducials_x].T.reshape(-1, 2) * target_spacing

        objpoints = []  # 3D points for calibration
        imgpoints_left = []  # Image points for the left camera
        imgpoints_right = []  # Image points for the right camera

        matched_keys = []
        for left_key in keypoints_left.keys():
            left_index = extract_index(left_key)
            for right_key in keypoints_right.keys():
                right_index = extract_index(right_key)
                if left_index == right_index:
                    matched_keys.append((left_key, right_key))

        for left_key, right_key in matched_keys:
            if len(keypoints_left[left_key]) == 60 and len(keypoints_right[right_key]) == 60:
                objpoints.append(objp)
                imgpoints_left.append(np.array(keypoints_left[left_key], dtype=np.float32))
                imgpoints_right.append(np.array(keypoints_right[right_key], dtype=np.float32))

        # Calibration process
        if len(objpoints) > 0:
            camera_matrix_left = cv2.initCameraMatrix2D(objpoints, imgpoints_left, img_size)
            camera_matrix_right = cv2.initCameraMatrix2D(objpoints, imgpoints_right, img_size)
            dist_coeffs_left = np.zeros((5, 1))  # Assuming zero distortion initially
            dist_coeffs_right = np.zeros((5, 1))

            # Calibrate both cameras
            mtx_left, dist_left, rvecs_left, tvecs_left = calibrate_single_camera(
                objpoints, imgpoints_left, img_size, camera_matrix=camera_matrix_left, dist_coeffs=dist_coeffs_left
            )
            mtx_right, dist_right, rvecs_right, tvecs_right = calibrate_single_camera(
                objpoints, imgpoints_right, img_size, camera_matrix=camera_matrix_right, dist_coeffs=dist_coeffs_right
            )

            # Stereo calibration
            R, T, E, F = calibrate_stereo_camera(
                objpoints, imgpoints_left, imgpoints_right, mtx_left, dist_left, mtx_right, dist_right, img_size
            )

            # Compute epipolar error
            epipolar_error = compute_epipolar_error(imgpoints_left, imgpoints_right, mtx_left, dist_left, mtx_right, dist_right, F)

            # Save calibration results
            save_calibration_to_xml("outputlatest2.xml", mtx_left, dist_left, mtx_right, dist_right, R, T, F, img_size, epipolar_error)

            # Return parameters to the GUI
            return jsonify({
                'status': 'Calibration successful',
                'mtx_left': mtx_left.tolist(),
                'dist_left': dist_left.tolist(),
                'mtx_right': mtx_right.tolist(),
                'dist_right': dist_right.tolist(),
                'R': R.tolist(),
                'T': T.tolist(),
                'F': F.tolist(),
                'epipolar_error': epipolar_error
            })
        else:
            return jsonify({'status': 'error', 'message': 'Not enough valid data points for calibration'}), 400

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    

if __name__ == '__main__':
    try:
        cam1 = Camera("Camera_1", "cam1", 1)
        cam2 = Camera("Camera_2", "cam2", 2)
        print("Cameras connected")
    except Exception as e:
        print(f"No camera: {str(e)}")
    socketio.run(app, debug=True)
    #webview.start()