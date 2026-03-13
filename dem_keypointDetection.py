import os
import io
import cv2
import numpy as np
import pandas as pd
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from dem_concentricCircleDetection import findKeypoints

def dectectKeypoints(D, ix, iy, co, ci, fo, fi, ns, ne):
    all_csv_left = []
    all_csv_right = []
    left_filenames = []
    right_filenames = []

    xNum, yNum = ix, iy
    coordOuter, coordInner = co, ci
    fieldOuter, fieldInner = fo, fi
    coordRatio, innerRatio = coordOuter / coordInner, fieldOuter / fieldInner
    camNum = 2
    imgNamePrefix = ["left", "right"]
    imgDirectory = (f"{D[1]}", f"{D[2]}")
    datDirectory = (f"{D[3]}", f"{D[4]}")
    imgNumber = [ns, ne]
    sd, biFac = 2.0, 1.5  # standard deviation and binarization threshold 
    (first, last) = imgNumber
    viewNum = (last - first) + 1
    loc = np.zeros((viewNum, xNum, yNum, 2), dtype=float)
    shape = loc[0].shape

    for i in range(viewNum):
        for j in range(camNum):
            # Check and create output directory
            outdir = Path(f"{datDirectory[j]}/{imgNamePrefix[j]}")
            outdir.mkdir(parents=True, exist_ok=True)

            print(f"\n################ Camera {j + 1} ################")
            print(f"\n######## View {j + 1}.{i + 1} ########")

            savedFile = Path(f"{outdir}/{imgNamePrefix[j]}{i + first}.csv")
            if savedFile.exists():
                loc[i] = pd.read_csv(savedFile, header=None).to_numpy().reshape(shape[0], shape[1], shape[2])
                print(f"All keypoints from View {j + 1}.{i + 1} successfully loaded")
                continue

            img_path = f"{imgDirectory[j]}/{imgNamePrefix[j]}{i + first}.jpg"
            print(f"Attempting to load image: {img_path}")

            # Check if the image exists
            if not Path(img_path).exists():
                print(f"Error: The file {img_path} does not exist.")
                continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"Error: Unable to load image {img_path}.")
                continue

            # Detect keypoints
            sd, biFac = findKeypoints(img, coordRatio, innerRatio, xNum, yNum, loc[i], j, sd, biFac)

            # Save keypoints to CSV
            pd.DataFrame(loc[i].reshape(shape[0], -1)).to_csv(savedFile, header=False, index=False)

            # Capture CSV content
            csv_buffer = io.StringIO()
            pd.DataFrame(loc[i].reshape(shape[0], -1)).to_csv(csv_buffer, header=False, index=False)
            csv_content = csv_buffer.getvalue()

            filename = f"{imgNamePrefix[j]}{i + first}.jpg"  # Dynamically set filename
            if j == 0:  # Left camera
                all_csv_left.append(csv_content)
                left_filenames.append(filename)
            else:  # Right camera
                all_csv_right.append(csv_content)
                right_filenames.append(filename)

    # Generate separate XML files for left and right
    generate_combined_xml(all_csv_left, "left.xml", "left", left_filenames)
    generate_combined_xml(all_csv_right, "right.xml", "right", right_filenames)
    print("Generated left.xml and right.xml")

def generate_combined_xml(csv_files, output_file, camera_name, filenames):
    root = ET.Element("ParameterLists")  # Main root element

    for i, csv_content in enumerate(csv_files):
        if csv_content:  # Ensure content is not empty
            csv_data = parse_csv_from_string(csv_content, filenames[i])  # Pass filename

            if csv_data:
                for filename, rows in csv_data.items():
                    file_node = ET.SubElement(root, "ParameterList", name=filename)

                    for row_index, (x_values, y_values) in enumerate(rows):
                        row_node = ET.SubElement(file_node, "ParameterList", name=f"ROW_{row_index}")

                        ET.SubElement(
                            row_node, "Parameter", name="X", type="string",
                            value=f"{{{', '.join(map(str, x_values))}}}"
                        )
                        ET.SubElement(
                            row_node, "Parameter", name="Y", type="string",
                            value=f"{{{', '.join(map(str, y_values))}}}"
                        )

    # Save to XML file
    xml_string = ET.tostring(root, encoding='unicode', method='xml')
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"<?xml version='1.0' encoding='UTF-8'?>\n{xml_string}")

def parse_csv_from_string(csv_content, filename):
    data = {}
    csv_reader = csv.reader(io.StringIO(csv_content))
    
    for row in csv_reader:
        if not row:
            continue

        try:
            # Extract all X-values (even indexes) and Y-values (odd indexes)
            x_values = [float(row[i]) for i in range(0, len(row), 2)]
            y_values = [float(row[i]) for i in range(1, len(row), 2)]

            if filename not in data:
                data[filename] = []
            data[filename].append((x_values, y_values))
        except ValueError:
            print(f"Skipping invalid row: {row}")

    return data
