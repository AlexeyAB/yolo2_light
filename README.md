# yolo2_light
Light version of convolutional neural network Yolo v3 & v2 for objects detection with a minimum of dependencies (INT8-inference, BIT1-XNOR-inference)

This repository supports:

* both Windows and Linux
* both OpenCV <= 3.3.0 and OpenCV 2.4.13
* both cuDNN >= 7.1.1
* CUDA >= 8.0

How to compile:
* To compile for CPU just do `make` on Linux or build `yolo_cpu.sln` on Windows
* To compile for GPU set flag `GPU=1` in the `Makefile` on Linux or build `yolo_gpu.sln` on Windows
    
    Required both [CUDA >= 8.0](https://developer.nvidia.com/cuda-toolkit-archive) and [cuDNN >= 7.1.1](https://developer.nvidia.com/rdp/cudnn-archive)

How to start:
* Download [`yolov3.weights`](https://pjreddie.com/media/files/yolov3.weights) to the `bin` directory and run `./yolo.sh` on Linux (or `yolo_cpu.cmd` / `yolo_gpu.cmd` on Windows)
* Download [`yolov3-tiny.cfg`](https://pjreddie.com/media/files/yolov3-tiny.weights) to the `bin` directory and run `./tiny-yolo.sh`

How to use **INT8**-inference:
* Use flag `-quantized` at the end of command, for example, [`tiny-yolo-int8.sh`](https://github.com/AlexeyAB/yolo2_light/blob/master/bin/tiny-yolo-int8.sh) or [`yolo_cpu_int8.cmd`](https://github.com/AlexeyAB/yolo2_light/blob/master/bin/yolo_cpu_int8.cmd)
* For the custom dataset, you should use `input_calibration=` parameter in your cfg-file, from the correspon cfg-file: [`yolov3-tiny.cfg`](https://github.com/AlexeyAB/yolo2_light/blob/29905072f194ee86fdeed6ff2d12fed818712411/bin/yolov3-tiny.cfg#L25) or [`yolov3.cfg`](https://github.com/AlexeyAB/yolo2_light/blob/29905072f194ee86fdeed6ff2d12fed818712411/bin/yolov3.cfg#L25), ...

How to use **BIT1-XNOR**-inference - only for custom models (you should train it by yourself):
* You should base your cfg-file on [`tiny-yolo-obj_xnor.cfg`](https://github.com/AlexeyAB/yolo2_light/blob/master/bin/tiny-yolo-obj_xnor.cfg) and train it by using this repository as usual https://github.com/AlexeyAB/darknet
* Then use it for Detection-test or for getting Accuracy (mAP):
    * `./darknet detector test data/obj.names tiny-yolo-obj_xnor.cfg data/tiny-yolo-obj_xnor_5000.weights -thresh 0.15 dog.jpg`
	* `./darknet detector map data/obj.data tiny-yolo-obj_xnor.cfg data/tiny-yolo-obj_xnor_5000.weights -thresh 0.15`

Other models by the link: https://pjreddie.com/darknet/yolo/
