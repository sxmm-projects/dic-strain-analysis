let c = "";  // This variable should be defined globally if used across multiple functions.

function startCapture() {
    let frequency = document.getElementById("frequency").value,
    min = document.getElementById("minute").value,
    sec = document.getElementById("second").value,
    burstmode = document.getElementById("burst").checked,
    shotmode = document.getElementById("shot").checked,
    status_start = document.getElementById("status1"),
    status_pause = document.getElementById("status2"),
    status_stop = document.getElementById("status3");
    
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
             shotmode: shotmode
        })
    })
    .then(response => response.json())
    .then(data => {
        if(data.status === 'Capture started'){
            document.getElementById('real-fps').textContent = data.real_fps;
        } else {
            console.log(data);
            alert(data.status);
        }
    })
    .catch(error => {
        console.log(error.message);
    });

    status_start.classList.add('display-status');
    status_pause.classList.remove('display-status');
    status_stop.classList.remove('display-status');
    c = 1;
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
        console.log(error.message);
    });

    if(c % 2 === 1){
        status_start.classList.remove('display-status');
        status_pause.classList.add('display-status');
        status_stop.classList.remove('display-status');
        c++;
    } else if(c % 2 === 0 && c !== ""){
        status_start.classList.add('display-status');
        status_pause.classList.remove('display-status');
        status_stop.classList.remove('display-status');
        c++;
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
        console.log(error.message);
    });

    status_start.classList.remove('display-status');
    status_pause.classList.remove('display-status');
    status_stop.classList.add('display-status');
    c = "";
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

function reload(){
    const img1 = document.getElementById('realtime1');
    const img2 = document.getElementById('realtime2');

    fetch('/reconnect')
    .then(response => response.json())
    .then(data => {
        if(data.status === 'Cameras reconnected and streaming started'){
            img1.src = img1.src.split('?')[0] + '?' + new Date().getTime();
            img2.src = img2.src.split('?')[0] + '?' + new Date().getTime();
        } else {
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
    } else {
        shotmode.checked = true;
    }
}

function burst_mode(){
    const shotmode = document.getElementById("shot"),
    burstmode = document.getElementById("burst");
    if(burstmode.checked){
        shotmode.checked = false;
        burstmode.checked = true;
    } else {
        burstmode.checked = true;
    }
}

document.addEventListener('DOMContentLoaded', function () {
    document.getElementById('frequency').addEventListener('input', function (e) {
        this.value = this.value.replace(/\D/g, '');
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
