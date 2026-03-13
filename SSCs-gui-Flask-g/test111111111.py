import cv2
import numpy as np
from lxml import etree


# Parsing input parameters from XML
def parse_input_parameters(xml_file):
    tree = etree.parse(xml_file)
    root = tree.getroot()

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

    # Save the Fundamental Matrix F
    # fundamental_matrix = etree.SubElement(root, "ParameterList", name="Fundamental_Matrix")
    # for i in range(3):
    #     etree.SubElement(fundamental_matrix, "Parameter", name=f"ROW {i}", type="string", value=f"{{ {F[i,0]:.12e}, {F[i,1]:.12e}, {F[i,2]:.12e} }}")

    # Save the XML tree to file
    tree = etree.ElementTree(root)
    with open(filename, "wb") as fh:
        tree.write(fh, pretty_print=True, xml_declaration=True, encoding="UTF-8")

    print(f"Calibration data saved to {filename}")

# Load keypoints XML files
keypoints_left = parse_keypoints_xml('C:/Sxmm/4th years/Project/exe_srtaincal/Senior work1/keypoints_left.xml')
keypoints_right = parse_keypoints_xml('C:/Sxmm/4th years/Project/exe_srtaincal/Senior work1/keypoints_right.xml')

# Load and extract parameters
input_params = parse_input_parameters('C:/Sxmm/4th years/Project/exe_srtaincal/Senior work1/cal_input.xml')
num_fiducials_x = input_params['num_cal_fiducials_x']
num_fiducials_y = input_params['num_cal_fiducials_y']
target_spacing = input_params['cal_target_spacing_size']

img_size = (2448, 2048)

# Generate object points (based on grid size)
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

# Now process the matched pairs
for left_key, right_key in matched_keys:
    if len(keypoints_left[left_key]) == 60 and len(keypoints_right[right_key]) == 60:
        objpoints.append(objp)
        imgpoints_left.append(np.array(keypoints_left[left_key], dtype=np.float32))
        imgpoints_right.append(np.array(keypoints_right[right_key], dtype=np.float32))
    else:
        print(f"Skipping pair {left_key}, {right_key}: Expected 60 points, but found {len(keypoints_left[left_key])} (left) and {len(keypoints_right[right_key])} (right)")

print(f"Left image keys: {list(keypoints_left.keys())}")
print(f"Right image keys: {list(keypoints_right.keys())}")
print(f"Total object points: {len(objpoints)}")
print(f"Total image points for left camera: {len(imgpoints_left)}")
print(f"Total image points for right camera: {len(imgpoints_right)}")

if len(objpoints) > 0:
    # สร้างค่าเริ่มต้นสำหรับ camera_matrix และ dist_coeffs
    camera_matrix_left = cv2.initCameraMatrix2D(objpoints, imgpoints_left, img_size)
    camera_matrix_right = cv2.initCameraMatrix2D(objpoints, imgpoints_right, img_size)
    dist_coeffs_left = np.zeros((5, 1))  # สมมติให้ค่าการบิดเบือนเป็นศูนย์สำหรับการเริ่มต้น
    dist_coeffs_right = np.zeros((5, 1))

    # Calibrate the left and right cameras
    mtx_left, dist_left, rvecs_left, tvecs_left = calibrate_single_camera(
        objpoints, imgpoints_left, img_size, camera_matrix=camera_matrix_left, dist_coeffs=dist_coeffs_left
    )
    mtx_right, dist_right, rvecs_right, tvecs_right = calibrate_single_camera(
        objpoints, imgpoints_right, img_size, camera_matrix=camera_matrix_right, dist_coeffs=dist_coeffs_right
    )

    # Calibrate stereo camera
    R, T, E, F = calibrate_stereo_camera(objpoints, imgpoints_left, imgpoints_right, mtx_left, dist_left, mtx_right, dist_right, img_size)
    print("Left camera matrix (mtx_left):")
    print(mtx_left)
    print("Left camera distortion coefficients (dist_left):")
    print(dist_left)
    print("Right camera matrix (mtx_right):")
    print(mtx_right)
    print("Right camera distortion coefficients (dist_right):")
    print(dist_right)

    # Calculate the epipolar error
    epipolar_error = compute_epipolar_error(imgpoints_left, imgpoints_right, mtx_left, dist_left, mtx_right, dist_right, F)
    print(f"Epipolar error: {epipolar_error:.12f}")

    # Rectify the images
    R1, R2, P1, P2 = rectify_stereo(mtx_left, dist_left, mtx_right, dist_right, R, T, img_size)

    # Save the calibration results, including epipolar error
    save_calibration_to_xml("outputlatest2.xml", mtx_left, dist_left, mtx_right, dist_right, R, T, F, img_size, epipolar_error)
else:
    print("Error: Not enough valid data points for calibration.")
