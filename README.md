# DIC-Based Strain Analysis Application

## Overview

This project presents a **Digital Image Correlation (DIC) based strain analysis application** designed for experimental mechanics and material deformation analysis.
The system processes stereo image pairs captured from cameras to estimate displacement fields and compute strain distributions using computer vision techniques.

The application integrates **Python-based DIC algorithms with a Flask web interface**, allowing users to upload images, perform correlation analysis, and visualize deformation results.

---

## Key Features

* Stereo image processing for deformation analysis
* keypoint detection
* Image correlation for displacement tracking
* Strain field calculation
* Web-based visualization using Flask
* Camera calibration support

---

## System Architecture

The system consists of three major components:

1. **Image Acquisition**

   * Stereo camera images of the specimen surface

2. **DIC Processing Engine**

   * Keypoint detection
   * Displacement tracking
   * Numerical strain computation

3. **Web Interface**

   * Image upload
   * Parameter configuration
   * Visualization of results

---

## Project Structure

```
dic-strain-analysis
│
├ src
│   ├ app
│   ├ dic_engine
│   └ dem_keypointDetection
│
├ static
├ templates
├ data_collection
│
└ README.md
```

---

## Technologies Used

* Python
* Flask
* OpenCV
* Numerical Methods
* Digital Image Correlation (DIC)

---

## Applications

This software can be applied in:

* Material deformation analysis
* Structural testing
* Experimental mechanics research
* Mechanical engineering laboratories

---

## Future Improvements

* GPU acceleration for correlation computation
* Real-time strain visualization
* Integration with industrial cameras
* Automated feature tracking optimization

---

## Author

Rungphailin Siamphupong <br>
Poomipat Thongthae <br>
Developed as part of a Robotics Engineering and Automation Systems project focusing on computer vision applications in engineering analysis.
