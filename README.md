# yolo2_light
Light version of convolutional neural network Yolo v2 for objects detection with a minimum of dependencies

This repository supports:

* both Windows and Linux
* both OpenCV 3.x and OpenCV 2.4.13
* both cuDNN 5 and cuDNN 6
* CUDA >= 7.5

How to compile:
* To compile for CPU just do `make` on Linux or build `yolo_cpu.sln` on Windows
* To compile for GPU set flag `GPU=1` in the `Makefile` on Linux or build `yolo_gpu.sln` on Windows
    
    Required both [CUDA >= 7.5](https://developer.nvidia.com/cuda-toolkit-archive) and [cuDNN >= 5.1](https://developer.nvidia.com/rdp/cudnn-archive)

How to start:
* Download [`yolo-voc.weights`](https://drive.google.com/open?id=0BwRgzHpNbsWBSzB6eldFTHJLRTA) to the `bin` directory and run `./yolo.sh` on Linux (or `yolo_cpu.cmd` / `yolo_gpu.cmd` on Windows)
* Download [`tiny-yolo-voc.weights`](https://drive.google.com/open?id=0BwRgzHpNbsWBeHBaM3ltVUZCZW8) to the `bin` directory and run `./tiny-yolo.sh`


Other models by the link: https://pjreddie.com/darknet/yolo/
