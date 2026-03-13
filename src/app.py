import ast
import io
import json
import logging
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from threading import Event, Thread

import cv2
import easygui
import gxipy as gx
import numpy as np
import pandas as pd
import webview
from flask import (Flask, Response, abort, flash, jsonify, redirect,
                   render_template, request, send_file, url_for)
from lxml import etree
from werkzeug.utils import secure_filename

import dem_concentricCircleDetection
import dem_keypointDetection
from test111111111 import (calibrate_single_camera, calibrate_stereo_camera,
                           compute_epipolar_error, parse_input_parameters,
                           parse_keypoints_xml, rectify_stereo,
                           save_calibration_to_xml)

img_Keydetect = dem_concentricCircleDetection.Imgs
img_Keydetect1 = dem_concentricCircleDetection.Imgs1
img_Keydetect2 = dem_concentricCircleDetection.Imgs2

font = cv2.FONT_HERSHEY_SIMPLEX
color = (255, 0, 0)
SIZE_IMAGE = (800, 600)

image_cam1 = []
image_cam2 = []
directories = {}

capturing = Event()
capturing.clear()
pause = Event()
pause.clear()
real_fps = None
switch = False
key_process = False

xNum = None
yNum = None
coordOuter = None
coordInner = None
fieldOuter = None
fieldInner = None
innerX = None
innerY = None
patternSpacing = None

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

templates_path = resource_path("templates")
static_path = resource_path("static")

app = Flask(__name__, template_folder=templates_path, static_folder=static_path)
window = webview.create_window('SSCs-gui', app)

with open('config.json', 'r') as f:
    config = json.load(f)

DICe_exe_path = os.path.abspath(config.get('DICe_exe_path'))
print(f"Resolved DICe_exe_path: {DICe_exe_path}")


UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
ALLOWED_EXTENSIONS = {'xml'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


class Camera:
    def __init__(self, name, path, shot_path, index):
        self.name = name
        self.index = index
        self.cam = None
        self.frame_count = 0
        self.start_time = time.time()
        self.path = path
        self.shot_path = shot_path
        self.__set_path()
        self._connect()

    def __set_path(self):
        Path(self.path).mkdir(parents=True, exist_ok=True)

    def _connect(self):
        self.device_manager = gx.DeviceManager()
        self.dev_num, self.dev_info_list = self.device_manager.update_device_list()
        self.cam = self.device_manager.open_device_by_index(self.index)

    def stream_on(self):
        self.cam.stream_on()
        self.cam.ExposureTime.set(12000)
        
    def stream_off(self):
        self.cam.stream_off()
        #self.cam.close_device()

    def get_img(self):
        raw_image = self.cam.data_stream[0].get_image()
        if raw_image is None:
            return None
        numpy_image = raw_image.get_numpy_array()
        return numpy_image

    def save_image_burst(self, count, img):
        if prefix is None or prefix == '':
            output_name = Path(self.path) / f'img_{count:05d}.jpg'
        elif prefix is not None:
            output_name = Path(self.path) / f'{prefix}_{count:05d}.jpg'
        cv2.imwrite(str(output_name), img)

    def save_image_shot(self, img):
        output_name = Path(self.shot_path) / f'shot_img.jpg'
        cv2.imwrite(str(output_name), img)

cam1 = None
cam2 = None

def capture_images(frequency, min, sec, burstmode, shotmode):
    global capturing, pause, real_fps, image_cam1, image_cam2
    if frequency == '':
        frequency = 12
    if min == '' and sec =='':
        min = 1
        sec = 30
    elif min == '':
        min = 0
    elif sec == '':
        sec = 0
    count_1 = 0
    count_2 = 0

    if image_cam1 and image_cam2:
        image_cam1.clear()
        image_cam2.clear()

    frame_inteval = 1 / int(frequency)

    start_time = time.time()
    stop_time = int(min) * 60 + int(sec)
    MAX_CAPTURE = stop_time * int(frequency)
    if burstmode:
        while capturing.is_set() and (count_1 < MAX_CAPTURE or count_2 < MAX_CAPTURE):
            new_frame_time = time.time()

            img_cam1 = cam1.get_img()
            img_cam2 = cam2.get_img()

            while pause.is_set():
                pass

            if img_cam1 is None:
                print(f'{cam1.name}: Error image')
            elif count_1 < MAX_CAPTURE:
                img_cam1_resize = cv2.resize(img_cam1, SIZE_IMAGE, interpolation=cv2.INTER_LINEAR)
                count_1 += 1
                cam1.frame_count += 1

                cv2.putText(img_cam1_resize, f'count = {count_1} of {MAX_CAPTURE}', (50, 50), font, 0.6, color, thickness=2)
                cv2.imshow(f'{cam1.name}_realtime', img_cam1_resize)
                image_cam1.append(img_cam1_resize)

            if img_cam2 is None:
                print(f'{cam2.name}: Error image')
            elif count_2 < MAX_CAPTURE:
                img_cam2_resize = cv2.resize(img_cam2, SIZE_IMAGE, interpolation=cv2.INTER_LINEAR)
                count_2 += 1
                cam2.frame_count += 1

                cv2.putText(img_cam2_resize, f'count = {count_2} of {MAX_CAPTURE}', (50, 50), font, 0.6, color, thickness=2)
                cv2.imshow(f'{cam2.name}_realtime', img_cam2_resize)
                image_cam2.append(img_cam2_resize)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time_error = time.time() - new_frame_time
            if time_error < frame_inteval:
                time.sleep(frame_inteval - time_error)
        times = time.time() - start_time
        print("times : ", times)

        #for i, img in enumerate(image_cam1):
        #    cam1.save_image_burst(i, img)
        #for i, img in enumerate(image_cam2):
        #    cam2.save_image_burst(i, img)

        capturing.clear()
        #cam1.stream_off()
        #cam2.stream_off()
        real_fps = round(MAX_CAPTURE / times, 2)
        cv2.destroyAllWindows()
    elif shotmode:
        img_cam1 = cam1.get_img()
        img_cam2 = cam2.get_img()

        if img_cam1 is None:
            print(f'{cam1.name}: Error image')
        else:
            cam1.save_image_shot(img_cam1)

        if img_cam2 is None:
            print(f'{cam2.name}: Error image')
        else:
            cam2.save_image_shot(img_cam2)

    capturing.clear()

def generate_frames(cam):
    while True:
        frame = cam.get_img()
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/start_capture', methods=['POST'])
def start_capture():
    global capturing, real_fps, prefix
    if not capturing.is_set() and switch:
        data = request.get_json()
        frequency = data.get('frequency')
        min = data.get('min')
        sec = data.get('sec')
        burstmode = data.get('burstmode')
        shotmode = data.get('shotmode')
        prefix = data.get('prefix')
        capturing.set()
        capture_thread = Thread(target=capture_images, args=(frequency, min, sec, burstmode, shotmode,))
        capture_thread.start()
        capture_thread.join()
        return jsonify({'status': 'Capture finished', 'real_fps': real_fps, 
                        'count1': len(image_cam1), 'count2': len(image_cam2)})
    else:
        if pause.is_set():
            pause.clear()
            return jsonify({'status': 'Unpause'})
        else:
            return jsonify({'status': 'Capture already in progress'})

@app.route('/pause_capture')
def pause_capture():
    global pause
    if pause.is_set():
        return jsonify({'status': 'Capture pause'})
    else:
        pause.set()
        return jsonify({'status': 'No capture in progress'})

@app.route('/stop_capture')
def stop_capture():
    global capturing
    if capturing.is_set():
        pause.clear()
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

    return jsonify({'status': 'Capture stopped', 'switch': switch})

@app.route('/reconnect')
def cam_reconnect():
    global cam1, cam2
    cam1 = Camera("Camera_1", "cam1", "shot_cam1", 1)
    cam2 = Camera("Camera_2", "cam2", "shot_cam2", 2)
    return jsonify({'status': 'Cameras reconnected and streaming started'})

@app.route('/cam1_route')
def cam1_route():
    global cam1, cam1_thread
    if cam1 is not None:
        cam1_thread = Thread(target=generate_frames, args=(cam1,))
        cam1_thread.start()
        return Response(generate_frames(cam1), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/cam2_route')
def cam2_route():
    global cam2, cam2_thread
    if cam2 is not None:
        cam2_thread = Thread(target=generate_frames, args=(cam2,))
        cam2_thread.start()
        return Response(generate_frames(cam2), mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/image1/<int:index>')
def get_image1(index):
    if index < 0 or index >= len(image_cam1):
        abort(404)

    img = image_cam1[index]
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = img_encoded.tobytes()

    return send_file(io.BytesIO(img_bytes), mimetype='image/jpeg')

@app.route('/image2/<int:index>')
def get_image2(index):
    if index < 0 or index >= len(image_cam2):
        abort(404)

    img = image_cam2[index]
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = img_encoded.tobytes()

    return send_file(io.BytesIO(img_bytes), mimetype='image/jpeg')

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
    
@app.route('/calibration')
def calibration():
    return render_template("calibration.html")

def extract_index(filename):
    """Extract index from the filename by removing specific substrings."""
    return filename.replace('left', '').replace('right', '').replace('.jpg', '')


def perform_calibration(parameter_path, keypoints_left_path, keypoints_right_path):
    try:
        # Load keypoints XML files
        keypoints_left = parse_keypoints_xml(keypoints_left_path)
        keypoints_right = parse_keypoints_xml(keypoints_right_path)

        # Load and extract parameters
        input_params = parse_input_parameters(parameter_path)
        num_fiducials_x = input_params['num_cal_fiducials_x']
        num_fiducials_y = input_params['num_cal_fiducials_y']
        target_spacing = input_params['cal_target_spacing_size']

        img_size = (2448, 2048)
        objp = np.zeros((num_fiducials_x * num_fiducials_y, 3), np.float32)
        objp[:, :2] = np.mgrid[0:num_fiducials_y, 0:num_fiducials_x].T.reshape(-1, 2) * target_spacing

        objpoints = []  # 3D points for calibration
        imgpoints_left = []  # Image points for the left camera
        imgpoints_right = []  # Image points for the right camera

        # Match left and right images based on extracted index
        def extract_index(filename):
            return filename.replace('left', '').replace('right', '').replace('.jpg', '')

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
            else:
                print(f"Skipping pair {left_key}, {right_key}: Expected 60 points, but found {len(keypoints_left[left_key])} (left) and {len(keypoints_right[right_key])} (right)")

        # If sufficient points for calibration
        if len(objpoints) > 0:
            camera_matrix_left = cv2.initCameraMatrix2D(objpoints, imgpoints_left, img_size)
            camera_matrix_right = cv2.initCameraMatrix2D(objpoints, imgpoints_right, img_size)
            dist_coeffs_left = np.zeros((5, 1))
            dist_coeffs_right = np.zeros((5, 1))

            mtx_left, dist_left, _, _ = calibrate_single_camera(
                objpoints, imgpoints_left, img_size, camera_matrix=camera_matrix_left, dist_coeffs=dist_coeffs_left
            )
            mtx_right, dist_right, _, _ = calibrate_single_camera(
                objpoints, imgpoints_right, img_size, camera_matrix=camera_matrix_right, dist_coeffs=dist_coeffs_right
            )

            R, T, _, F = calibrate_stereo_camera(
                objpoints, imgpoints_left, imgpoints_right, mtx_left, dist_left, mtx_right, dist_right, img_size
            )

            epipolar_error = compute_epipolar_error(
                imgpoints_left, imgpoints_right, mtx_left, dist_left, mtx_right, dist_right, F
            )

            save_path = "outputlatest2.xml"  # Path where the file is saved
            save_calibration_to_xml(save_path, mtx_left, dist_left, mtx_right, dist_right, R, T, F, img_size, epipolar_error)
            
            return {
            "mtx_left": mtx_left.tolist(),
            "dist_left": dist_left.tolist(),
            "mtx_right": mtx_right.tolist(),
            "dist_right": dist_right.tolist(),
            "epipolar_error": epipolar_error,
            "warnings": [],
            "save_path": save_path
            }

        else:
            # Insufficient data for calibration
            return {
                "warnings": ["Insufficient keypoints or data. Check input files."],
                "mtx_left": None,
                "dist_left": None,
                "mtx_right": None,
                "dist_right": None,
                "epipolar_error": None
            }
    except Exception as e:
        print(f"Error in perform_calibration: {e}")
        return None



@app.route('/upload_files', methods=['POST'])
def upload_files():
    """Handle file uploads."""
    try:
        # Debugging ALLOWED_EXTENSIONS
        print("DEBUG: ALLOWED_EXTENSIONS:", app.config.get('ALLOWED_EXTENSIONS'))

        # Get uploaded files
        parameter_file = request.files.get('parameter_file')
        keypoints_left_file = request.files.get('keypoints_left_file')
        keypoints_right_file = request.files.get('keypoints_right_file')

        if not parameter_file or not keypoints_left_file or not keypoints_right_file:
            return jsonify({"status": "error", "message": "All files are required."}), 400

        # Validate file extensions
        for file in [parameter_file, keypoints_left_file, keypoints_right_file]:
            if not allowed_file(file.filename):
                return jsonify({"status": "error", "message": f"Invalid file type: {file.filename}"}), 400

        # Save files securely
        parameter_file.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(parameter_file.filename)))
        keypoints_left_file.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(keypoints_left_file.filename)))
        keypoints_right_file.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(keypoints_right_file.filename)))

        return jsonify({"status": "success", "message": "Files uploaded successfully."}), 200
    except Exception as e:
        print("Error in /upload_files:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/start_calibration', methods=['POST'])
def start_calibration():
    try:
        # Retrieve uploaded file paths
        parameter_file = os.path.join(app.config['UPLOAD_FOLDER'], 'cal_input.xml')
        keypoints_left_file = os.path.join(app.config['UPLOAD_FOLDER'], 'keypoints_left.xml')
        keypoints_right_file = os.path.join(app.config['UPLOAD_FOLDER'], 'keypoints_right.xml')

        # Check if files exist
        for file_path in [parameter_file, keypoints_left_file, keypoints_right_file]:
            if not os.path.exists(file_path):
                return jsonify({"status": "error", "message": f"Missing file: {os.path.basename(file_path)}"}), 400

        # Perform calibration
        result = perform_calibration(parameter_file, keypoints_left_file, keypoints_right_file)

        if result:
            print(f"\nCalibration Result:")
            print(f"Left Camera Matrix: {result['mtx_left']}")
            print(f"Right Camera Matrix: {result['mtx_right']}")
            print(f"Left Camera Distortion Coefficients: {result['dist_left']}")
            print(f"Right Camera Distortion Coefficients: {result['dist_right']}")
            print(f"Epipolar Error: {result['epipolar_error']}")
            print(f"Saved Output File: {result['save_path']}")
            print(f"Warnings: {result.get('warnings', 'None')}")
            
            # Success or partial success
            response = {
                "status": "success" if not result["warnings"] else "partial_success",
                "message": "Calibration completed successfully." if not result["warnings"] else "Calibration completed with warnings.",
                "details": {
                    "mtx_left": result["mtx_left"],
                    "mtx_right": result["mtx_right"],
                    "dist_left": result["dist_left"],
                    "dist_right": result["dist_right"],
                    "epipolar_error": result["epipolar_error"],
                    "save_path": result["save_path"],
                },
                "warnings": result.get("warnings", [])
            }
            return jsonify(response), 200

        else:
            # General failure
            return jsonify({
                "status": "error",
                "message": "Calibration failed due to an unknown issue."
            }), 400
    except Exception as e:
        print(f"Error in /start_calibration: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/executeDetect', methods=['POST'])
def executeDetect():
    global directories, innerX, innerY, patternSpacing
    try:
        data = request.get_json()
        #xNum = int(data.get('total_x'))
        #yNum = int(data.get('total_y'))
        coordOuter = float(data.get('coord_outer'))
        coordInner = float(data.get('coord_inner'))
        fieldOuter = float(data.get('field_outer'))
        fieldInner = float(data.get('field_inner'))
        numstart = int(data.get('s'))
        numend = int(data.get('e'))
        ##################
        innerX = int(data.get('inner_x'))
        innerY = int(data.get('inner_y'))
        #patternSpacing = float(data.get('pSI'))

        print("xNum: ", xNum)
        print("Amount: ", len(directories))

        if len(directories) == 4:
            # Call your function (ensure it's error-free)
            generate_dice_xml("output_dice_calibration.xml", innerX, innerY, patternSpacing)
            dem_keypointDetection.dectectKeypoints(directories, innerX, innerY, coordOuter, coordInner, fieldOuter, fieldInner, numstart, numend)
            return jsonify({
                'status': 'Capture finished',
                'countKey1': len(img_Keydetect1),
                'countKey2': len(img_Keydetect2)
            })
        else:
            return jsonify({
                "error": "Required input is missing. Please provide all necessary fields."
            }), 400

    except Exception as e:
        # Log the exception and return a meaningful error
        print(f"Error in /executeDetect: {str(e)}")
        return jsonify({
            "error": "An error occurred during keypoint detection.",
            "details": str(e)
        }), 500
    
def generate_dice_xml(output_file, inx, iny, psi):
    # Root Element
    root = ET.Element("ParameterList")
    
    # Main Parameters
    ET.SubElement(root, "Parameter", name="xml_file_format", type="string", value="DICe_xml_calibration_file")
    ET.SubElement(root, "Parameter", name="image_file_extension", type="string", value=".jpg")
    ET.SubElement(root, "Parameter", name="num_cal_fiducials_x", type="int", value=str(inx))
    ET.SubElement(root, "Parameter", name="num_cal_fiducials_y", type="int", value=str(iny))
    ET.SubElement(root, "Parameter", name="cal_target_spacing_size", type="double", value=str(psi))
    ET.SubElement(root, "Parameter", name="draw_intersection_image", type="bool", value="true")
    ET.SubElement(root, "Parameter", name="cal_debug_folder", type="string", 
                  value="C:\\Users\\66879\\Desktop\\TEST")

    # Nested Parameter List
    cal_opencv_options = ET.SubElement(root, "ParameterList", name="cal_opencv_options")
    ET.SubElement(cal_opencv_options, "Parameter", name="CALIB_USE_INTRINSIC", type="bool", value="true")
    ET.SubElement(cal_opencv_options, "Parameter", name="CALIB_ZERO_TANGENT_DIST", type="bool", value="true")
    
    # Generate XML tree
    tree = ET.ElementTree(root)
    
    # Write to output file
    tree.write(output_file, encoding="UTF-8", xml_declaration=True)
    print(f"XML file generated at: {output_file}")


@app.route('/image_Keydetect1')
def get_image_Keydetect1():
    if len(img_Keydetect) >= 1:
        img = img_Keydetect[0]
        _, img_encoded = cv2.imencode('.jpg', img)
        img_bytes = img_encoded.tobytes()
        return send_file(io.BytesIO(img_bytes), mimetype='image/jpeg')
    else:
        return abort(404, description="Image not found")

@app.route('/image_Keydetect2')
def get_image_Keydetect2():
    if len(img_Keydetect) >= 2:
        img = img_Keydetect[1]
        _, img_encoded = cv2.imencode('.jpg', img)
        img_bytes = img_encoded.tobytes()
        return send_file(io.BytesIO(img_bytes), mimetype='image/jpeg')
    else:
        return abort(404, description="Image not found")
    
@app.route('/imageKey1/<int:index>')
def get_imageKey1(index):
    if index < 0 or index >= len(img_Keydetect1):
        abort(404)

    img = img_Keydetect1[index]
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = img_encoded.tobytes()

    return send_file(io.BytesIO(img_bytes), mimetype='image/jpeg')

@app.route('/imageKey2/<int:index>')
def get_imageKey2(index):
    if index < 0 or index >= len(img_Keydetect2):
        abort(404)

    img = img_Keydetect2[index]
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = img_encoded.tobytes()

    return send_file(io.BytesIO(img_bytes), mimetype='image/jpeg')
    
@app.route('/upload-directory', methods=['POST'])
def upload_directory():
    global directories

    try:
        data = request.get_json()
        button_id = data.get('buttonId')

        if button_id is None:
            return jsonify({"error": "Button ID not provided"}), 400

        # Ask the user to select a folder
        folder_path = easygui.diropenbox(title="Select a Folder")

        # Check if a folder was selected
        if folder_path:
            # Replace backslashes with forward slashes for consistency
            folder_path = folder_path.replace("\\", "/")
            directories[button_id] = folder_path

            print(f"Button {button_id} selected directory: {directories[button_id]}")
            print("All directories:", directories)

            # Return a success response
            return jsonify({"status": "success", "buttonId": button_id, "folderPath": folder_path})
        else:
            return jsonify({"error": "No folder selected"}), 400

    except Exception as e:
        # Catch and handle unexpected exceptions
        print(f"Error in /upload-directory: {str(e)}")
        return jsonify({"error": "An unexpected error occurred.", "details": str(e)}), 500


@app.route('/check_pattern_spacing', methods=['GET', 'POST'])
def check_pattern_spacing():
    global innerX, innerY, patternSpacing
    try:
        data = request.json
        innerX = int(data['inner_x'])
        innerY = int(data['inner_y'])
        xLength = float(data['x_length'])
        yLength = float(data['y_length'])
        patternSpacing = float(data['pattern_spacing_input'])

        patternSpacingX = xLength / (innerX - 1)
        patternSpacingY = yLength / (innerY - 1)

        avg_spacing = (patternSpacing + patternSpacingX + patternSpacingY) / 3

        error_input = abs(patternSpacing - avg_spacing) / avg_spacing * 100
        error_x = abs(patternSpacingX - avg_spacing) / avg_spacing * 100
        error_y = abs(patternSpacingY - avg_spacing) / avg_spacing * 100
        
        print(f"Pattern Spacing Input: {patternSpacing}")
        print(f"Pattern Spacing X: {patternSpacingX}")
        print(f"Pattern Spacing Y: {patternSpacingY}")
        print(f"Error Input: {error_input}")
        print(f"Error X: {error_x}")
        print(f"Error Y: {error_y}")

        if max(error_input, error_x, error_y) > 1:
            return jsonify({
                "status": "error",
                "message": "Pattern spacing values differ by more than 1%.",
                "details": {
                    "pattern_spacing_input": patternSpacing,
                    "pattern_spacing_x": patternSpacingX,
                    "pattern_spacing_y": patternSpacingY,
                    "error_input": error_input,
                    "error_x": error_x,
                    "error_y": error_y
                }
            }), 400

        return jsonify({
            "status": "success",
            "message": "Pattern spacing values are within 1% tolerance.",
            "details": {
                "pattern_spacing_input": patternSpacing,
                "pattern_spacing_x": patternSpacingX,
                "pattern_spacing_y": patternSpacingY,
                "error_input": error_input,
                "error_x": error_x,
                "error_y": error_y
            }
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500



if __name__ == '__main__':
    try:
        cam1 = Camera("Camera_1", "cam1", "shot_cam1", 1)
        cam2 = Camera("Camera_2", "cam2", "shot_cam2", 2)
        print("Camera connected")
    except:
        print("No camera")
    #app.run()
    webview.start(debug= True)