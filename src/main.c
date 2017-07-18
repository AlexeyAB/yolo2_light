#include <stdio.h>
#include <stdlib.h>


#include "option_list.h"
#include "parser.h"
#include "image.h"
#include "pthread.h"

#ifdef OPENCV
#pragma comment(lib, "opencv_core249.lib")
#pragma comment(lib, "opencv_imgproc249.lib")
#pragma comment(lib, "opencv_highgui249.lib")

#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#endif


// get prediction boxes: yolov2_forward_network.c
void get_region_boxes_cpu(layer l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness, int *map);


void draw_detections_cpu(image im, int num, float thresh, box *boxes, float **probs, char **names, image **alphabet, int classes)
{
	int i;
	// number of bounded boxes = (width_last_layer * height_last_layer * anchors)
	for (i = 0; i < num; ++i) {
		int class = max_index(probs[i], classes);
		float prob = probs[i][class];
		if (prob > thresh) {	// if (probability > threshold) the draw bonded box

			int width = im.h * .012;

			printf("%s: %.0f%%\n", names[class], prob * 100);
			int offset = class * 123457 % classes;
			float red = get_color(2, offset, classes);
			float green = get_color(1, offset, classes);
			float blue = get_color(0, offset, classes);
			float rgb[3];

			//width = prob*20+2;

			rgb[0] = red;
			rgb[1] = green;
			rgb[2] = blue;
			box b = boxes[i];

			int left = (b.x - b.w / 2.)*im.w;
			int right = (b.x + b.w / 2.)*im.w;
			int top = (b.y - b.h / 2.)*im.h;
			int bot = (b.y + b.h / 2.)*im.h;

			if (left < 0) left = 0;
			if (right > im.w - 1) right = im.w - 1;
			if (top < 0) top = 0;
			if (bot > im.h - 1) bot = im.h - 1;

			draw_box_width(im, left, top, right, bot, width, red, green, blue);
			if (alphabet) {
				image label = get_label(alphabet, names[class], (im.h*.03) / 10);
				draw_label(im, top + width, left, label, rgb);
			}
		}
	}
}


// --------------- Detect on the Image ---------------

// detect on CPU: yolov2_forward_network.c
float *network_predict_cpu(network net, float *input);

// detect on GPU: yolov2_forward_network_gpu.cu
float *network_predict_gpu_cudnn(network net, float *input);

// Detect on Image: this function uses other functions not from this file
void test_detector_cpu(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh)
{
	list *options = read_data_cfg(datacfg);		// option_list.c
	char *name_list = option_find_str(options, "names", "data/names.list");	// option_list.c
	char **names = get_labels(name_list);		// data.c

	image **alphabet = load_alphabet();			// image.c
	network net = parse_network_cfg(cfgfile);	// parser.c
	if (weightfile) {
		load_weights(&net, weightfile);			// parser.c
	}
	set_batch_network(&net, 1);					// network.c
	srand(2222222);
	clock_t time;
	char buff[256];
	char *input = buff;
	int j;
	float nms = .4;
	while (1) {
		if (filename) {
			strncpy(input, filename, 256);
		}
		else {
			printf("Enter Image Path: ");
			fflush(stdout);
			input = fgets(input, 256, stdin);
			if (!input) return;
			strtok(input, "\n");
		}
		image im = load_image_color(input, 0, 0);		// image.c
		image sized = resize_image(im, net.w, net.h);	// image.c
		layer l = net.layers[net.n - 1];

		box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
		float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
		for (j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));

		float *X = sized.data;
		time = clock();
		//network_predict(net, X);
#ifdef GPU
		network_predict_gpu_cudnn(net, X);
#else
		network_predict_cpu(net, X);
#endif
		printf("%s: Predicted in %f seconds.\n", input, (float)(clock() - time) / CLOCKS_PER_SEC); //sec(clock() - time));
		get_region_boxes_cpu(l, 1, 1, thresh, probs, boxes, 0, 0);			// get_region_boxes(): region_layer.c

																			//  nms (non maximum suppression) - if (IoU(box[i], box[j]) > nms) then remove one of two boxes with lower probability
		if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);	// box.c
		draw_detections_cpu(im, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes);	// draw_detections(): image.c
		save_image(im, "predictions");	// image.c
		show_image(im, "predictions");	// image.c

		free_image(im);					// image.c
		free_image(sized);				// image.c
		free(boxes);
		free_ptrs((void **)probs, l.w*l.h*l.n);	// utils.c
#ifdef OPENCV
		cvWaitKey(0);
		cvDestroyAllWindows();
#endif
		if (filename) break;
	}
}


// --------------- Detect on the Video ---------------

#ifdef OPENCV
static char **demo_names;
static int demo_classes;

static float **probs;
static box *boxes;
static network net;
static image in;
static image in_s;
static image det;
static image det_s;
static image disp = { 0 };
static CvCapture * cap;
static float fps = 0;
static float demo_thresh = 0;

image get_image_from_stream_resize(CvCapture *cap, int w, int h, IplImage** in_img);	// image.c
IplImage* in_img;
IplImage* det_img;
IplImage* show_img;

// draw bounded boxes of found objects on the image, from: image.c
void draw_detections_cv_cpu(IplImage* show_img, int num, float thresh, box *boxes, float **probs, char **names, int classes)
{
	int i;

	for (i = 0; i < num; ++i) {
		int class = max_index(probs[i], classes);
		float prob = probs[i][class];
		if (prob > thresh) {

			int width = show_img->height * .012;

			if (0) {
				width = pow(prob, 1. / 2.) * 10 + 1;
			}

			printf("%s: %.0f%%\n", names[class], prob * 100);
			int offset = class * 123457 % classes;
			float red = get_color(2, offset, classes);
			float green = get_color(1, offset, classes);
			float blue = get_color(0, offset, classes);
			float rgb[3];

			//width = prob*20+2;

			rgb[0] = red;
			rgb[1] = green;
			rgb[2] = blue;
			box b = boxes[i];

			int left = (b.x - b.w / 2.)*show_img->width;
			int right = (b.x + b.w / 2.)*show_img->width;
			int top = (b.y - b.h / 2.)*show_img->height;
			int bot = (b.y + b.h / 2.)*show_img->height;

			if (left < 0) left = 0;
			if (right > show_img->width - 1) right = show_img->width - 1;
			if (top < 0) top = 0;
			if (bot > show_img->height - 1) bot = show_img->height - 1;

			float const font_size = show_img->height / 1000.F;
			CvPoint pt1, pt2, pt_text, pt_text_bg1, pt_text_bg2;
			pt1.x = left;
			pt1.y = top;
			pt2.x = right;
			pt2.y = bot;
			pt_text.x = left;
			pt_text.y = top - 12;
			pt_text_bg1.x = left;
			pt_text_bg1.y = top - (10 + 25 * font_size);
			pt_text_bg2.x = right;
			pt_text_bg2.y = top;
			CvScalar color;
			color.val[0] = red * 256;
			color.val[1] = green * 256;
			color.val[2] = blue * 256;

			cvRectangle(show_img, pt1, pt2, color, width, 8, 0);

			cvRectangle(show_img, pt_text_bg1, pt_text_bg2, color, width, 8, 0);
			cvRectangle(show_img, pt_text_bg1, pt_text_bg2, color, CV_FILLED, 8, 0);	// filled
			CvScalar black_color;
			black_color.val[0] = 0;
			CvFont font;
			cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX, font_size, font_size, 0, font_size * 3, 8);
			cvPutText(show_img, names[class], pt_text, &font, black_color);
		}
	}
}

static void *fetch_in_thread(void *ptr)
{
	in = get_image_from_stream_resize(cap, net.w, net.h, &in_img);	// image.c
	if (!in.data) {
		error("Stream closed.");
	}
	in_s = make_image(in.w, in.h, in.c);	// image.c
	memcpy(in_s.data, in.data, in.h*in.w*in.c * sizeof(float));

	return 0;
}

static void *detect_in_thread(void *ptr)
{
	float nms = .4;
	layer l = net.layers[net.n - 1];
	float *X = det_s.data;

	//float *prediction = network_predict(net, X);
#ifdef GPU
	network_predict_gpu_cudnn(net, X);
#else
	network_predict_cpu(net, X);
#endif

	free_image(det_s);
	get_region_boxes_cpu(l, 1, 1, demo_thresh, probs, boxes, 0, 0);		// get_region_boxes(): region_layer.c
	if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);	// box.c
	printf("\033[2J");
	printf("\033[1;1H");
	printf("\nFPS:%.1f\n", fps);
	printf("Objects:\n\n");
	draw_detections_cv_cpu(det_img, l.w*l.h*l.n, demo_thresh, boxes, probs, demo_names, demo_classes);	// draw_detections(): image.c

	return 0;
}

static double get_wall_time()
{
	struct timeval time;
	if (gettimeofday(&time, NULL)) {
		return 0;
	}
	return (double)time.tv_sec + (double)time.tv_usec * .000001;
}



// Detect on Video: this function uses other functions not from this file
void demo_cpu(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix)
{
	int delay = frame_skip;
	demo_names = names;
	demo_classes = classes;
	demo_thresh = thresh;
	printf("Demo\n");
	net = parse_network_cfg(cfgfile);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	set_batch_network(&net, 1);

	srand(2222222);

	if (filename) {
		printf("video file: %s\n", filename);
		cap = cvCaptureFromFile(filename);
	}
	else {
		cap = cvCaptureFromCAM(cam_index);
	}

	if (!cap) error("Couldn't connect to webcam.\n");

	layer l = net.layers[net.n - 1];
	int j;

	boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
	probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
	for (j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));

	pthread_t fetch_thread;
	pthread_t detect_thread;

	fetch_in_thread(0);
	det_img = in_img;
	det = in;
	det_s = in_s;

	fetch_in_thread(0);
	detect_in_thread(0);
	disp = det;
	show_img = det_img;
	det_img = in_img;
	det = in;
	det_s = in_s;

	int count = 0;
	if (!prefix) {
		cvNamedWindow("Demo", CV_WINDOW_NORMAL);
		cvMoveWindow("Demo", 0, 0);
		cvResizeWindow("Demo", 1352, 1013);
	}

	double before = get_wall_time();

	while (1) {
		++count;
		if (pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
		if (pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");

		if (!prefix) {
			//show_image(disp, "Demo");
			show_image_cv_ipl(show_img, "Demo");
			int c = cvWaitKey(1);
		}
		else {
			char buff[256];
			sprintf(buff, "%s_%08d", prefix, count);
			save_image(disp, buff);
		}

		pthread_join(fetch_thread, 0);
		pthread_join(detect_thread, 0);

		if (delay == 0) {
			free_image(disp);
			disp = det;
			show_img = det_img;
		}
		det_img = in_img;
		det = in;
		det_s = in_s;

		--delay;
		if (delay < 0) {
			delay = frame_skip;

			double after = get_wall_time();
			float curr = 1. / (after - before);
			fps = curr;
			before = after;
		}
	}
}
#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix)
{
	fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif





void run_detector(int argc, char **argv)
{
	char *prefix = find_char_arg(argc, argv, "-prefix", 0);
	float thresh = find_float_arg(argc, argv, "-thresh", .24);
	int cam_index = find_int_arg(argc, argv, "-c", 0);
	int frame_skip = find_int_arg(argc, argv, "-s", 0);
	if (argc < 4) {
		fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
		return;
	}

	int clear = find_arg(argc, argv, "-clear");

	char *datacfg = argv[3];
	char *cfg = argv[4];
	char *weights = (argc > 5) ? argv[5] : 0;
	char *filename = (argc > 6) ? argv[6] : 0;
	if (0 == strcmp(argv[2], "test")) test_detector_cpu(datacfg, cfg, weights, filename, thresh);
	//else if (0 == strcmp(argv[2], "train")) train_detector(datacfg, cfg, weights, gpus, ngpus, clear);
	//else if (0 == strcmp(argv[2], "valid")) validate_detector(datacfg, cfg, weights);
	//else if (0 == strcmp(argv[2], "recall")) validate_detector_recall(datacfg, cfg, weights);
	else if (0 == strcmp(argv[2], "demo")) {
		list *options = read_data_cfg(datacfg);
		int classes = option_find_int(options, "classes", 20);
		char *name_list = option_find_str(options, "names", "data/names.list");
		char **names = get_labels(name_list);
		demo_cpu(cfg, weights, thresh, cam_index, filename, names, classes, frame_skip, prefix);
	}

}


int main(int argc, char **argv)
{
	if (argc < 2) {
		fprintf(stderr, "usage: %s <function>\n", argv[0]);
		return 0;
	}
	gpu_index = find_int_arg(argc, argv, "-i", 0);
	if (find_arg(argc, argv, "-nogpu")) {
		gpu_index = -1;
	}

#ifndef GPU
	gpu_index = -1;
#else
	if (gpu_index >= 0) {
		cuda_set_device(gpu_index);
	}
#endif
	run_detector(argc, argv);
	return 0;
}
