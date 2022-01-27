function jump(){
	window.location.href="../jump";
}
function showImg(input) {
        var file = input.files[0];
        var reader = new FileReader()
        // 图片读取成功回调函数
        reader.onload = function(e) {
            document.getElementById('upload').src=e.target.result
        }
        reader.readAsDataURL(file)
}
window.onload = function () {
    //访问用户媒体设备的兼容方法
    function getUserMedia(constraints, success, error) {
        if (navigator.mediaDevices.getUserMedia) {
            //最新的标准API
            navigator.mediaDevices.getUserMedia(constraints).then(success).catch(error);
        } else if (navigator.webkitGetUserMedia) {
            //webkit核心浏览器
			navigator.webkitGetUserMedia(constraints, success, error)
		} else if (navigator.mozGetUserMedia) {
			//firfox浏览器
		    navigator.mozGetUserMedia(constraints, success, error);
        } else if (navigator.getUserMedia) {
			//旧版API
			navigator.getUserMedia(constraints, success, error);
		}
	}
    
    let video = document.getElementById('video');
	let canvas = document.getElementById('canvas');
	let context = canvas.getContext('2d');

	function success(stream) {
		//兼容webkit核心浏览器
		let CompatibleURL = window.URL || window.webkitURL;
		//将视频流设置为video元素的源
        video.srcObject = stream;
		video.play();
	}
    function error(error) {
		console.log(`访问用户媒体设备失败${error.name}, ${error.message}`);
	}

	if (navigator.mediaDevices.getUserMedia || navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia) {
		if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
			console.log("enumerateDevices() not supported.");
			return;
		}
		// 列出摄像头和麦克风
		var exArray = [];
		navigator.mediaDevices.enumerateDevices()
			.then(function (devices) {
				devices.forEach(function (device) {
					if (device.kind == "videoinput") {
						exArray.push(device.deviceId);
					}
				});
				var mediaOpts = { video: { width: 420, height: 120 } };
				var mediaOpts =
				{
					video:
					{
						deviceId: { exact: exArray[1] }
					}
				};
				//调用用户媒体设备, 访问摄像头
				getUserMedia(mediaOpts, success, error);
			})
			.catch(function (err) {
				console.log(err.name + ": " + err.message);
			});

	} else {
		alert('不支持访问用户媒体');
	}
	
	var img = new Image();
	
	document.getElementById('capture').addEventListener('click', function () {
		context.drawImage(video, 0, 0, 425, 320);
    })

    function send(){
		var imgData = canvas.toDataURL("image/jpg",0.95);
		var formdata = new FormData();
    	formdata.append("face", imgData);  
		var xhr = new XMLHttpRequest();
		xhr.open('POST', '/camera/', true);
		//xhr.setRequestHeader("Content-type", "application/json");
		xhr.send(formdata);
	}
	document.getElementById('face').addEventListener('click', function () {
		send();
    })
}