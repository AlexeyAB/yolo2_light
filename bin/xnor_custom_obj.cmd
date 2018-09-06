REM You should train your own model based on tiny-yolo-obj_xnor.cfg by yourself (train as usual)


yolo_cpu.exe detector test data/obj.names tiny-yolo-obj_xnor.cfg data/tiny-yolo-obj_xnor_5000.weights -thresh 0.15 dog.jpg

yolo_cpu.exe detector demo data/obj.names tiny-yolo-obj_xnor.cfg data/tiny-yolo-obj_xnor_5000.weights -thresh 0.15 test.mp4


pause