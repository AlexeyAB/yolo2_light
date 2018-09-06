
yolo_cpu.exe detector test coco.names yolov3-tiny.cfg yolov3-tiny.weights -thresh 0.2 dog.jpg


yolo_cpu.exe detector demo coco.names yolov3-tiny.cfg yolov3-tiny.weights -thresh 0.2 test.mp4


rem yolo_cpu.exe detector demo coco.names yolov3-tiny.cfg yolov3-tiny.weights -thresh 0.2 test.mp4 -quantized



pause