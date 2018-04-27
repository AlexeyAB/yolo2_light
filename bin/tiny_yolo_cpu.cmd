
rem yolo_cpu.exe detector test voc.names yolo-voc.cfg yolo-voc.weights -thresh 0.03 dog.jpg -quantized


yolo_cpu.exe detector demo voc.names tiny-yolo-voc.cfg tiny-yolo-voc.weights -thresh 0.2 -quantized test.mp4


rem yolo_cpu.exe detector test voc.names tiny-yolo-voc.cfg tiny-yolo-voc.weights -thresh 0.2 -quantized


yolo_cpu.exe detector test voc.names tiny-yolo-voc.cfg tiny-yolo-voc.weights -thresh 0.2 dog.jpg -quantized


yolo_cpu.exe detector test coco.names tiny-yolo.cfg tiny-yolo.weights -thresh 0.2 dog.jpg -quantized

pause