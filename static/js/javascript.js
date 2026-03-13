let currentPhotoIndex = 0,
countImage1 = 0,
countImage2 = 0,
photos1 = [],
photos2 = [],
directory,
folderHandle1,
folderHandle2;
async function startCapture() {
    let frequency = document.getElementById("frequency").value,
    min = document.getElementById("minute").value,
    sec = document.getElementById("second").value,
    burstmode = document.getElementById("burst").checked,
    shotmode = document.getElementById("shot").checked,
    status_start = document.getElementById("status1"),
    status_pause = document.getElementById("status2"),
    status_stop = document.getElementById("status3"),
    viewphoto = document.getElementById("viewphoto"),
    prefix = document.getElementById("prefix").value;
    if(directory == null){
        directory = await window.showDirectoryPicker();
        folderHandle1 = await directory.getDirectoryHandle("cam1", { create: true });
        folderHandle2 = await directory.getDirectoryHandle("cam2", { create: true });
    }
    fetch('/start_capture', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
             frequency: frequency,
             min: min,
             sec: sec,
             burstmode: burstmode,
             shotmode: shotmode,
             prefix: prefix
        })
    })
    .then(response => response.json())
    .then(data => {
        if(data.status === 'Capture finished'){
            countImage1 = data.count1;
            countImage2 = data.count2;
            document.getElementById('real-fps').textContent = data.real_fps;
            takePhoto()
        }
        console.log(data.status);
    })
    .catch(error => {
        console.log(error.message);
    });
    status_start.classList.add('display-status');
    status_pause.classList.remove('display-status');
    status_stop.classList.remove('display-status');
    c = 1
}
async function takePhoto(){
    if (directory) {
        photos1.length = 0;
        photos2.length = 0;
        const prefix = document.getElementById("prefix").value;
        try {
            for (let i = 0; i < countImage1; i++) {
                const imgResponse = await fetch(`/image1/${i}`);
                const blob = await imgResponse.blob();
                const photoURL1 = URL.createObjectURL(blob);
                if (prefix === ""){
                    photos1.push({url: photoURL1, name: `img_${String(i).padStart(5, '0')}.jpg`});
                }
                else{
                    photos1.push({url: photoURL1, name: `${prefix}_${String(i).padStart(5, '0')}.jpg`});
                }
            }
            for (let i = 0; i < countImage2; i++) {
                const imgResponse = await fetch(`/image2/${i}`);
                const blob = await imgResponse.blob();
                const photoURL2 = URL.createObjectURL(blob);
                if (prefix === ""){
                    photos2.push({url: photoURL2, name: `img_${String(i).padStart(5, '0')}.jpg`});
                }
                else{
                    photos2.push({url: photoURL2, name: `${prefix}_${String(i).padStart(5, '0')}.jpg`});
                }
            }
            for (const photo of photos1) {
                const fileHandle1 = await folderHandle1.getFileHandle(photo.name, { create: true });
                const writable1 = await fileHandle1.createWritable();
                const response1 = await fetch(photo.url);
                await writable1.write(await response1.blob());
                await writable1.close();
            }
            for (const photo of photos2) {
                const fileHandle2 = await folderHandle2.getFileHandle(photo.name, { create: true });
                const writable2 = await fileHandle2.createWritable();
                const response2 = await fetch(photo.url);
                await writable2.write(await response2.blob());
                await writable2.close();
            }
        } catch (error) {
            console.error('Error fetching images or processing them:', error);
        }
        console.log('Using directory:', directory);
    } else {
        console.log('No directory selected yet.');
    }
}
async function saveDirectory(){
    directory = await window.showDirectoryPicker();
    folderHandle1 = await directory.getDirectoryHandle("cam1", { create: true });
    folderHandle2 = await directory.getDirectoryHandle("cam2", { create: true });
}
function pauseCapture() {
    const status_start = document.getElementById("status1"),
    status_pause = document.getElementById("status2"),
    status_stop = document.getElementById("status3");
    fetch('/pause_capture')
    .then(response => response.json())
    .then(data => {
        console.log(data);
    })
    .catch(error => {
        console.log(error.massage);
    });
    if(c === 1){
        status_start.classList.remove('display-status');
        status_pause.classList.add('display-status');
        status_stop.classList.remove('display-status');
    }
}
function stopCapture() {
    const status_start = document.getElementById("status1"),
    status_pause = document.getElementById("status2"),
    status_stop = document.getElementById("status3");
    fetch('/stop_capture')
    .then(response => response.json())
    .then(data => {
        console.log(data);
    })
    .catch(error => {
        console.log(error.massage);
    });
    if(c === 1){
        status_start.classList.remove('display-status');
        status_pause.classList.remove('display-status');
        status_stop.classList.add('display-status');
        c = ""
    }
}
function viewCapture(){
    currentPhotoIndex = 0;
    loadImage1(currentPhotoIndex);
    loadImage2(currentPhotoIndex);
    viewphoto.style.display = 'block';
}
function closebutton(){
    viewphoto.style.display = 'none';
}
function loadImage1(index) {
    const modalImage1 = document.getElementById('modalImage1');
    const url = `/image1/${index}?t=${new Date().getTime()}`;
    modalImage1.src = url;
}
function loadImage2(index) {
    const modalImage2 = document.getElementById('modalImage2');
    const url = `/image2/${index}?t=${new Date().getTime()}`;
    modalImage2.src = url;
}
function changePhoto(direction) {
    currentPhotoIndex += direction;
    if ((currentPhotoIndex >= countImage1) && (currentPhotoIndex >= countImage2)) {
        currentPhotoIndex = 0;
    } else if (currentPhotoIndex < 0) {
        currentPhotoIndex = countImage1 - 1;
    }
    loadImage1(currentPhotoIndex);
    loadImage2(currentPhotoIndex);
}
function reload(){
    const img1 = document.getElementById('realtime1');
    const img2 = document.getElementById('realtime2');
    fetch('/reconnect')
    .then(response => response.json())
    .then(data => {
        if(data.status === 'Cameras reconnected and streaming started'){
            img1.src = img1.src.split('?')[0] + '?' + new Date().getTime();
            img2.src = img2.src.split('?')[0] + '?' + new Date().getTime();
        }else{
            alert(data.status);
        }
    })
    .catch(error => {
        console.log(error.message);
    });
}
function shot_mode(){
    const shotmode = document.getElementById("shot"),
    burstmode = document.getElementById("burst");
    if(shotmode.checked){
        shotmode.checked = true;
        burstmode.checked = false;
    }
    else{
        shotmode.checked = true;
    }
}
function burst_mode(){
    const shotmode = document.getElementById("shot"),
    burstmode = document.getElementById("burst");
    if(burstmode.checked){
        shotmode.checked = false;
        burstmode.checked = true;
    }
    else{
        burstmode.checked = true;
    }
}

document.addEventListener('DOMContentLoaded', function () {
    document.getElementById('frequency').addEventListener('input', function (e) {
        this.value = this.value.replace(/\D/g, '');
        // if (parseInt(this.value) > 24) {
        //     this.value = 24;
        // }
    });
    document.querySelectorAll('.time').forEach(function(input) {
        input.addEventListener('input', function(e) {
            this.value = this.value.replace(/\D/g, '');
            if (parseInt(this.value) > 59) {
                this.value = 59;
            }
        });
    });
});