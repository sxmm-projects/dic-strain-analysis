let leftfiles,
    rightfiles,
    Dimension_i = [],
    Dimension_j = [],
    leftimages = [],
    rightimages = [],
    loadedCount = 0,
    selectedDirectories = {},
    patternSpacing = true;

let debounceTimer;

    const uploadForm = document.getElementById('upload-form');
    const consoleOutput = document.getElementById('console-output');

document.addEventListener('DOMContentLoaded', () => {
    const leftinput = document.getElementById('left-file-input');
    const rightinput = document.getElementById('right-file-input');
    
    leftinput.addEventListener('change', (e) => {
        leftfiles = e.target.files;
        if (leftfiles.length === 0) {
            return;
        }
        if (leftfiles.length === 1) {
            document.getElementById('leftcheck').textContent = leftfiles[0].name + ": ";
        } else {
            document.getElementById('leftcheck').textContent = `${leftfiles.length} Files selected for the left folder`;
        }
        filesMatch();
    });
    
    rightinput.addEventListener('change', (e) => {
        rightfiles = e.target.files;
        if (rightfiles.length === 0) {
            return;
        }
        if (rightfiles.length === 1) {
            document.getElementById('rightcheck').textContent = rightfiles[0].name + ": ";
        } else {
            document.getElementById('rightcheck').textContent = `${rightfiles.length} Files selected for the right folder`;
        }
        filesMatch();
    });

    function filesMatch(){
        if((leftfiles.length > 0 && rightfiles.length > 0) && (leftfiles.length === rightfiles.length)){
            check_folder_dimensions()
        }
        else{
            document.getElementById('console-output').value = "Please select the same number of files for both sides.";
            return;
        }
    }
    
    function check_folder_dimensions() {
        Dimension_i = []
        Dimension_j = []
        leftimages = []
        rightimages = []
        loadedCount = 0
        // Loop through the files and handle left and right images
        for (let i = 0; i < leftfiles.length; i++) {
            const filei = leftfiles[i];
            const filej = rightfiles[i];
            const imgi = new Image();
            const imgj = new Image();
    
            // Create object URLs for the files
            const fileiURL = URL.createObjectURL(filei);
            const filejURL = URL.createObjectURL(filej);
    
            // Left image loading handler
            imgi.onload = function() {
                Dimension_i[i] = [imgi.naturalWidth, imgi.naturalHeight];
                URL.revokeObjectURL(fileiURL);  // Clean up the object URL after loading
                loadedCount++; // Increment counter when imgi loads
                checkIfAllImagesLoaded();
            };
    
            // Right image loading handler
            imgj.onload = function() {
                Dimension_j[i] = [imgj.naturalWidth, imgj.naturalHeight];
                URL.revokeObjectURL(filejURL);  // Clean up the object URL after loading
                loadedCount++; // Increment counter when imgj loads
                checkIfAllImagesLoaded();
            };
    
            // Set the image sources to the object URLs
            imgi.src = fileiURL;
            imgj.src = filejURL;
        }
    }
    
    function checkIfAllImagesLoaded() {
        if (loadedCount === leftfiles.length * 2) {
            checkDimension();
        }
    }
    
    // Function to log dimensions
    function checkDimension(){
        for (let i = 0; i < Dimension_i.length; i++) {
            if ((Dimension_i[i][0] === Dimension_j[i][0]) && (Dimension_i[i][1] === Dimension_j[i][1])) {
    
            } else {
                leftimages.push(leftfiles[i])
                rightimages.push(rightfiles[i])
            }
        }
        if(leftimages.length > 0 && rightimages.length > 0){
            let mismatch = "These image pairs are mismatched.\n"
            for(let i = 0; i < leftimages.length; i++){
               mismatch += `${leftimages[i].name}, ${rightimages[i].name}\n`;
            }
            document.getElementById('console-output').value = mismatch;
            return;
        }
        document.getElementById('console-output').value = "Process success: All image are matched.";
    }
    const imgElement1 = document.getElementById('imageKeydetect1');
    const imgElement2 = document.getElementById('imageKeydetect2');
    setInterval(() => {
        const url1 = `/image_Keydetect1?t=${new Date().getTime()}`;
        const url2 = `/image_Keydetect2?t=${new Date().getTime()}`;
        
        imgElement1.src = url1;
        imgElement2.src = url2;
    }, 1000);

    document.getElementById('leftImgDirectory').addEventListener('click', () => sendDirectory(1, '/upload-directory'));
    document.getElementById('rightImgDirectory').addEventListener('click', () => sendDirectory(2, '/upload-directory'));
    document.getElementById('leftDatDirectory').addEventListener('click', () => sendDirectory(3, '/upload-directory'));
    document.getElementById('rightDatDirectory').addEventListener('click', () => sendDirectory(4, '/upload-directory'));
});

const buttonCompletionStatus = {
    1: false,
    2: false,
    3: false,
    4: false,
};

// Modified sendDirectory function
async function sendDirectory(buttonId, endpoint) {
    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ buttonId: buttonId }),
        });
        if (!response.ok) {
            const errorText = await response.text();
            console.error(`Failed to send directory from Button ${buttonId}. Status: ${response.status}. Response: ${errorText}`);
        } else {
            const data = await response.json();
            console.log(`Directory from Button ${buttonId} sent to server successfully:`, data);

            if (data.folderPath) {
                selectedDirectories[buttonId] = data.folderPath;
                console.log(`Selected directory for Button ${buttonId}: ${selectedDirectories[buttonId]}`);
            } else {
                console.error(`No folderPath received for Button ${buttonId}.`);
            }
            // Mark the button as completed
            buttonCompletionStatus[buttonId] = true;
        }
    } catch (error) {
        console.error(`Error in sendDirectory for Button ${buttonId}:`, error.message);
    }
}


function strain_cal() {
    fetch('/strain_cal')
    .then(response => response.json())
    .then(data => {
        console.log(data);
    })
    .catch(error => {
        console.log(error.message);
    });
}

function validatePatternSpacing() {
    clearTimeout(debounceTimer);

    debounceTimer = setTimeout(() => {
        const patternSpacingInput = parseFloat(document.getElementById('pattern-spacing').value);
        const xLength = parseFloat(document.getElementById('x-length').value);
        const yLength = parseFloat(document.getElementById('y-length').value);
        const innerX = document.getElementById("inner-pattern-width").value;
        const innerY = document.getElementById("inner-pattern-height").value;

        console.log('Input values:', { patternSpacingInput, xLength, yLength, innerX, innerY });

        if (
            isNaN(patternSpacingInput) ||
            isNaN(xLength) ||
            isNaN(yLength) ||
            isNaN(innerX) ||
            isNaN(innerY)
        ) {
            document.getElementById('validation-result').textContent =
                'Please fill in all fields with valid numbers.';
            return;
        }

        const payload = {
            pattern_spacing_input: patternSpacingInput,
            x_length: xLength,
            y_length: yLength,
            inner_x: innerX,
            inner_y: innerY,
        };

        console.log('Payload sent to backend:', payload);

        fetch('/check_pattern_spacing', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        })
        .then((response) => {
            if (!response.ok) {
                return response.text().then((text) => {
                    throw new Error(`HTTP error! status: ${response.status}, message: ${text}`);
                });
            }
            return response.json();
        })
        
            .then((data) => {
                console.log('Backend response:', data);
                if (data.status === 'error') {
                    document.getElementById('validation-result').textContent = data.message;
                } else {
                    patternSpacing = true
                    document.getElementById('validation-result').textContent = 'Pattern spacing is valid.';
                }
            })
            .catch((error) => {
                console.error('Error:', error);

                if (error.message.includes('Unexpected token')) {
                    document.getElementById('validation-result').textContent =
                        'Server error: Invalid response. Please check your server logs.';
                } else {
                    patternSpacing = false
                    document.getElementById('validation-result').textContent =
                        'An error occurred during pattern spacing validation.';
                }
            });
    }, 500);
}

function uploadFiles() {
    const parameterFile = document.getElementById('parameter-file').files[0];
    const keypointsLeftFile = document.getElementById('keypoints-left-file').files[0];
    const keypointsRightFile = document.getElementById('keypoints-right-file').files[0];

    if (!parameterFile || !keypointsLeftFile || !keypointsRightFile) {
        alert("Please select all required files.");
        return;
    }

    const formData = new FormData();
    formData.append('parameter_file', parameterFile);
    formData.append('keypoints_left_file', keypointsLeftFile);
    formData.append('keypoints_right_file', keypointsRightFile);

    fetch('/upload_files', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            console.log('Upload successful:', data.message);
            document.getElementById('console-output').value = "Files uploaded successfully.";
        } else {
            console.error('Upload failed:', data.message);
            document.getElementById('console-output').value = `Upload failed: ${data.message}`;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('console-output').value = `Error: ${error.message}`;
    });
}

async function startKeypointDetection() {
    const consoleOutput = document.getElementById('console-output');
    const xNum = document.getElementById("total-pattern-width").value;
    const yNum = document.getElementById("total-pattern-height").value;
    const coordOuter = document.getElementById("outer-diameter").value;
    const coordInner = document.getElementById("inner-diameter").value;
    const fieldOuter = document.getElementById("field-outer").value;
    const fieldInner = document.getElementById("field-inner").value;
    //const offsetX = document.getElementById("original-offsetX").value;
    //const offsetY = document.getElementById("original-offsetY").value;
    const innerX = document.getElementById("inner-pattern-width").value;
    const innerY = document.getElementById("inner-pattern-height").value;
    const patternSpacingInput = document.getElementById('pattern-spacing').value;
    //const xLength = parseFloat(document.getElementById('x-length').value);
    //const yLength = parseFloat(document.getElementById('y-length').value);
    const start = document.getElementById('imgNumberStart').value;
    const end = document.getElementById('imgNumberEnd').value;
    const UserInput = 
    `Total X Keypoint Number: ${xNum}\n` +
    `Total Y Keypoint Number: ${yNum}\n` +
    `Inner X Keypoint Number: ${innerX}\n` +
    `Inner Y Keypoint Number: ${innerY}\n` +
    `Corner outer diameter: ${coordOuter}\n` +
    `Corner inner diameter: ${coordInner}\n` +
    `Keypoint Outer Diameter: ${fieldOuter}\n` +
    `Keypoint Inner Diameter: ${fieldInner}\n` +
    `Pattern Spacing (mm): ${patternSpacingInput}\n\n` +
    `Image Number Start: ${start}\n` +
    `Image Number End: ${end}\n\n` ;
    if (Object.values(buttonCompletionStatus).every((status) => status === true)) {
        if (patternSpacing) {
            document.getElementById('console-output').value = UserInput;
            consoleOutput.value += "Keypoint detection Processing"
            try {
                const response = await fetch('/executeDetect', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        //total_x: xNum,
                        //total_y: yNum,
                        coord_outer: coordOuter,
                        coord_inner: coordInner,
                        field_outer: fieldOuter,
                        field_inner: fieldInner,
                        s: start,
                        e: end,
                        //offset_x: offsetX,
                        //offset_y: offsetY,
                        inner_x: innerX,
                        inner_y: innerY
                        //pSI: patternSpacingInput
                    })
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }

                const data = await response.json();

                if (data.status === 'Capture finished') {
                    countImageKey1 = data.countKey1;
                    countImageKey2 = data.countKey2;
                    document.getElementById('console-output').value = UserInput;
                    consoleOutput.value += 
                    "Successfully at path:\n" +
                    `${selectedDirectories[3]}\n` +
                    `${selectedDirectories[4]}\n\n` +
                    "Please click 'View Images' to check the images in full size" ;
                }
                console.log(data.status);
            } catch (error) {
                console.error('Error during keypoint detection:', error.message);
            }
        }
        else{
            document.getElementById('console-output').value = "Pattern spacing values differ by more than 1%.";
        }
    }
    else{
        alert('Please select pics and data path');
    }
}

function startCalibration() {
    fetch('/start_calibration', {
        method: 'POST',
    })
    .then(response => response.json())
    .then(data => {
        let output = "";
        if (data.status === 'success') {
            output = `
Calibration completed successfully:
Left Camera Matrix: ${JSON.stringify(data.details.mtx_left, null, 2)}
Right Camera Matrix: ${JSON.stringify(data.details.mtx_right, null, 2)}
Epipolar Error: ${data.details.epipolar_error}
Saved Output File: ${data.details.save_path}
Warnings: ${data.warnings.length > 0 ? data.warnings.join("\n") : "None"}
            `;
        } else if (data.status === 'partial_success') {
            output = `
Calibration completed with warnings:
Warnings: ${data.warnings.join("\n")}
Left Camera Matrix: ${JSON.stringify(data.details.mtx_left, null, 2)}
Right Camera Matrix: ${JSON.stringify(data.details.mtx_right, null, 2)}
Epipolar Error: ${data.details.epipolar_error || "N/A"}
Saved Output File: ${data.details.save_path || "N/A"}
            `;
        } else {
            output = `Calibration failed: ${data.message}`;
        }
        document.getElementById('console-output').value = output;
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('console-output').value = `Calibration failed: ${error.message}`;
    });
}

function goHome() {
    history.back();
}

function viewKeypoint(){
    const container1 = document.getElementById("container1");
    const img1 = document.getElementById("modalImage1");
    const container2 = document.getElementById("container2");
    const img2 = document.getElementById("modalImage2");

    let viewphoto = document.getElementById("viewphoto");
    let value1 = 1,
        value2 = 1;
    //if (typeof countImageKey1 === "undefined") {
    //    alert('Please wait until finish');
    //    return
    //}
    currentPhotoIndex = 0;
    loadImageKey1(currentPhotoIndex);
    loadImageKey2(currentPhotoIndex);
    viewphoto.style.display = 'block';

    container1.addEventListener('wheel', (event) => {
        event.preventDefault();

        if (event.deltaY < 0) {
            value1 += 0.1;
        } 
        else if (event.deltaY > 0 && value1 > 1){
            value1 -= 0.1;
        }
        img1.style.transform = `scale(${value1})`;
        console.log(value1);
    }, { passive: false });

    container1.addEventListener("mousemove", (e) => {
        const x = e.clientX - e.target.offsetLeft;
        const y = e.clientY - e.target.offsetTop;

        console.log(x, y);
        img1.style.transformOrigin = `${x}px ${y}px`;
    });

    container2.addEventListener('wheel', (event) => {
        event.preventDefault();

        if (event.deltaY < 0) {
            value2 += 0.1;
        } 
        else if (event.deltaY > 0 && value2 > 1){
            value2 -= 0.1;
        }
        img2.style.transform = `scale(${value2})`;
        console.log(value2);
    }, { passive: false });

    container2.addEventListener("mousemove", (e) => {
        const x = e.clientX - e.target.offsetLeft;
        const y = e.clientY - e.target.offsetTop;

        console.log(x, y);
        img2.style.transformOrigin = `${x}px ${y}px`;
    });
}
function closebutton(){
    viewphoto.style.display = 'none';
}

function loadImageKey1(index) {
    const modalImage1 = document.getElementById('modalImage1');
    const url = `/imageKey1/${index}?t=${new Date().getTime()}`;
    modalImage1.src = url;
}
function loadImageKey2(index) {
    const modalImage2 = document.getElementById('modalImage2');
    const url = `/imageKey2/${index}?t=${new Date().getTime()}`;
    modalImage2.src = url;
}

function changePhoto(direction) {
    currentPhotoIndex += direction;
    const maxPhotos = Math.max(countImageKey1, countImageKey2);
    if (currentPhotoIndex >= maxPhotos) {
        currentPhotoIndex = 0;
    } else if (currentPhotoIndex < 0) {
        currentPhotoIndex = maxPhotos - 1;
    }
    const displayNumber = currentPhotoIndex + 1;
    const counterElement = document.getElementById('image-counter');
    if (counterElement) {
        counterElement.textContent = ` ${displayNumber} / ${maxPhotos}`;
    }
    if (currentPhotoIndex < countImageKey1) {
        loadImageKey1(currentPhotoIndex);
    }
    if (currentPhotoIndex < countImageKey2) {
        loadImageKey2(currentPhotoIndex);
    }
}