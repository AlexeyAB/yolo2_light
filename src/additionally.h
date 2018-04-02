#pragma once
#ifndef ADDITIONALLY_H
#define ADDITIONALLY_H


#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <limits.h>
#include <stdint.h>

#ifdef CUDNN
#include "cudnn.h"
#endif

#ifdef GPU
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#endif

#ifdef OPENCL
#include "CL/cl.h"
#endif

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/types_c.h"
#include "opencv2/core/version.hpp"

// for OpenCV 3.x only
#ifndef CV_VERSION_EPOCH
#include "opencv2/imgcodecs/imgcodecs_c.h"
#include "opencv2/videoio/videoio_c.h"
#endif

#endif

#ifdef __cplusplus
extern "C" {
#endif

extern int gpu_index;

// -------------- im2col.h --------------

// im2col.c
float im2col_get_pixel(float *im, int height, int width, int channels,
	int row, int col, int channel, int pad);

// im2col.c
//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(float* data_im,
	int channels, int height, int width,
	int ksize, int stride, int pad, float* data_col);

// -------------- blas.h --------------

// blas.c
void gemm_nn(int M, int N, int K, float ALPHA,
	float *A, int lda, float *B, int ldb, float *C, int ldc);

// blas.c
void fill_cpu(int N, float ALPHA, float *X, int INCX);


// -------------- list.h --------------


typedef struct node {
	void *val;
	struct node *next;
	struct node *prev;
} node;

typedef struct list {
	int size;
	node *front;
	node *back;
} list;


// list.c
list *get_paths(char *filename);

// list.c
void **list_to_array(list *l);

// list.c
void free_node(node *n);

// list.c
void free_list(list *l);

// list.c
char **get_labels(char *filename);


// -------------- utils.h --------------

#define TWO_PI 6.2831853071795864769252866

// utils.c
void error(const char *s);

// utils.c
void malloc_error();

// utils.c
void file_error(char *s);

// utils.c
char *fgetl(FILE *fp);

// utils.c
int *read_map(char *filename);

// utils.c
void del_arg(int argc, char **argv, int index);

// utils.c
int find_arg(int argc, char* argv[], char *arg);

// utils.c
int find_int_arg(int argc, char **argv, char *arg, int def);

// utils.c
float find_float_arg(int argc, char **argv, char *arg, float def);

// utils.c
char *find_char_arg(int argc, char **argv, char *arg, char *def);

// utils.c
void strip(char *s);

// utils.c
void list_insert(list *l, void *val);

// utils.c
float rand_uniform(float min, float max);

// utils.c
float rand_scale(float s);

// utils.c
int rand_int(int min, int max);

// utils.c
int constrain_int(int a, int min, int max);

// utils.c
float dist_array(float *a, float *b, int n, int sub);

// utils.c
float mag_array(float *a, int n);

// utils.c
int max_index(float *a, int n);

// utils.c
// From http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
float rand_normal();

// utils.c
void free_ptrs(void **ptrs, int n);


// --------------  activations.h --------------

typedef enum {
	LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN
}ACTIVATION;

static inline float linear_activate(float x) { return x; }
static inline float logistic_activate(float x) { return 1.0F / (1.0F + expf(-x)); }
static inline float leaky_activate(float x) { return (x>0) ? x : 0.1F * x; }

static inline ACTIVATION get_activation(char *s)
{
	if (strcmp(s, "logistic") == 0) return LOGISTIC;
	if (strcmp(s, "loggy") == 0) return LOGGY;
	if (strcmp(s, "relu") == 0) return RELU;
	if (strcmp(s, "elu") == 0) return ELU;
	if (strcmp(s, "relie") == 0) return RELIE;
	if (strcmp(s, "plse") == 0) return PLSE;
	if (strcmp(s, "hardtan") == 0) return HARDTAN;
	if (strcmp(s, "lhtan") == 0) return LHTAN;
	if (strcmp(s, "linear") == 0) return LINEAR;
	if (strcmp(s, "ramp") == 0) return RAMP;
	if (strcmp(s, "leaky") == 0) return LEAKY;
	if (strcmp(s, "tanh") == 0) return TANH;
	if (strcmp(s, "stair") == 0) return STAIR;
	fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
	return RELU;
}

// --------------  tree.h --------------

typedef struct {
	int *leaf;
	int n;
	int *parent;
	int *group;
	char **name;

	int groups;
	int *group_size;
	int *group_offset;
} tree;

// tree.c
void hierarchy_predictions(float *predictions, int n, tree *hier, int only_leaves);

// -------------- layer.h --------------

struct network_state;

struct layer;
typedef struct layer layer;

typedef enum {
	CONVOLUTIONAL,
	DECONVOLUTIONAL,
	CONNECTED,
	MAXPOOL,
	SOFTMAX,
	DETECTION,
	DROPOUT,
	CROP,
	ROUTE,
	COST,
	NORMALIZATION,
	AVGPOOL,
	LOCAL,
	SHORTCUT,
	ACTIVE,
	RNN,
	GRU,
	CRNN,
	BATCHNORM,
	NETWORK,
	XNOR,
	REGION,
	REORG,
	BLANK
} LAYER_TYPE;

typedef enum {
	SSE, MASKED, SMOOTH
} COST_TYPE;

struct layer {
	LAYER_TYPE type;
	ACTIVATION activation;
	COST_TYPE cost_type;
	void(*forward)   (struct layer, struct network_state);
	void(*backward)  (struct layer, struct network_state);
	void(*update)    (struct layer, int, float, float, float);
	void(*forward_gpu)   (struct layer, struct network_state);
	void(*backward_gpu)  (struct layer, struct network_state);
	void(*update_gpu)    (struct layer, int, float, float, float);
	int batch_normalize;
	int shortcut;
	int batch;
	int forced;
	int flipped;
	int inputs;
	int outputs;
	int truths;
	int h, w, c;
	int out_h, out_w, out_c;
	int n;
	int max_boxes;
	int groups;
	int size;
	int side;
	int stride;
	int reverse;
	int pad;
	int sqrt;
	int flip;
	int index;
	int binary;
	int xnor;
	int steps;
	int hidden;
	float dot;
	float angle;
	float jitter;
	float saturation;
	float exposure;
	float shift;
	float ratio;
	int softmax;
	int classes;
	int coords;
	int background;
	int rescore;
	int objectness;
	int does_cost;
	int joint;
	int noadjust;
	int reorg;
	int log;

	int adam;
	float B1;
	float B2;
	float eps;
	float *m_gpu;
	float *v_gpu;
	int t;
	float *m;
	float *v;

	tree *softmax_tree;
	int  *map;

	float alpha;
	float beta;
	float kappa;

	float coord_scale;
	float object_scale;
	float noobject_scale;
	float class_scale;
	int bias_match;
	int random;
	float thresh;
	int classfix;
	int absolute;

	int dontload;
	int dontloadscales;

	float temperature;
	float probability;
	float scale;

	int *indexes;
	float *rand;
	float *cost;
	char  *cweights;
	float *state;
	float *prev_state;
	float *forgot_state;
	float *forgot_delta;
	float *state_delta;

	float *concat;
	float *concat_delta;

	float *binary_weights;

	float *biases;
	float *bias_updates;

	float *scales;
	float *scale_updates;

	float *weights;
	float *weight_updates;

	float *col_image;
	int   * input_layers;
	int   * input_sizes;
	float * delta;
	float * output;
	float * squared;
	float * norms;

	float * spatial_mean;
	float * mean;
	float * variance;

	float * mean_delta;
	float * variance_delta;

	float * rolling_mean;
	float * rolling_variance;

	float * x;
	float * x_norm;

	struct layer *input_layer;
	struct layer *self_layer;
	struct layer *output_layer;

	struct layer *input_gate_layer;
	struct layer *state_gate_layer;
	struct layer *input_save_layer;
	struct layer *state_save_layer;
	struct layer *input_state_layer;
	struct layer *state_state_layer;

	struct layer *input_z_layer;
	struct layer *state_z_layer;

	struct layer *input_r_layer;
	struct layer *state_r_layer;

	struct layer *input_h_layer;
	struct layer *state_h_layer;

	float *z_cpu;
	float *r_cpu;
	float *h_cpu;

	float *binary_input;

	size_t workspace_size;

#ifdef GPU
	float *z_gpu;
	float *r_gpu;
	float *h_gpu;

	int *indexes_gpu;
	float * prev_state_gpu;
	float * forgot_state_gpu;
	float * forgot_delta_gpu;
	float * state_gpu;
	float * state_delta_gpu;
	float * gate_gpu;
	float * gate_delta_gpu;
	float * save_gpu;
	float * save_delta_gpu;
	float * concat_gpu;
	float * concat_delta_gpu;

	float *binary_input_gpu;
	float *binary_weights_gpu;

	float * mean_gpu;
	float * variance_gpu;

	float * rolling_mean_gpu;
	float * rolling_variance_gpu;

	float * variance_delta_gpu;
	float * mean_delta_gpu;

	float * col_image_gpu;

	float * x_gpu;
	float * x_norm_gpu;
	float * weights_gpu;
	float * weight_updates_gpu;

	float * biases_gpu;
	float * bias_updates_gpu;

	float * scales_gpu;
	float * scale_updates_gpu;

	float * output_gpu;
	float * delta_gpu;
	float * rand_gpu;
	float * squared_gpu;
	float * norms_gpu;
#ifdef CUDNN
	cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
	cudnnTensorDescriptor_t dsrcTensorDesc, ddstTensorDesc;
	cudnnFilterDescriptor_t weightDesc;
	cudnnFilterDescriptor_t dweightDesc;
	cudnnConvolutionDescriptor_t convDesc;
	cudnnConvolutionFwdAlgo_t fw_algo;
	cudnnConvolutionBwdDataAlgo_t bd_algo;
	cudnnConvolutionBwdFilterAlgo_t bf_algo;
#endif
#endif

#ifdef OPENCL
	cl_mem weights_ocl;
	cl_mem biases_ocl;
	cl_mem scales_ocl;
	cl_mem rolling_mean_ocl;
	cl_mem rolling_variance_ocl;

	cl_mem output_ocl;
	cl_mem indexes_ocl;
	cl_mem x_ocl;
#endif
};

typedef layer local_layer;
typedef layer convolutional_layer;
typedef layer softmax_layer;
typedef layer region_layer;
typedef layer reorg_layer;
typedef layer maxpool_layer;
typedef layer route_layer;

void free_layer(layer);


// -------------- network.h --------------

typedef enum {
	CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
} learning_rate_policy;

typedef struct network {
	float *workspace;
	int n;
	int batch;
	uint64_t *seen;
	float epoch;
	int subdivisions;
	float momentum;
	float decay;
	layer *layers;
	int outputs;
	float *output;
	learning_rate_policy policy;

	float learning_rate;
	float gamma;
	float scale;
	float power;
	int time_steps;
	int step;
	int max_batches;
	float *scales;
	int   *steps;
	int num_steps;
	int burn_in;

	int adam;
	float B1;
	float B2;
	float eps;

	int inputs;
	int h, w, c;
	int max_crop;
	int min_crop;
	float angle;
	float aspect;
	float exposure;
	float saturation;
	float hue;

	int gpu_index;
	tree *hierarchy;

#ifdef GPU
	float **input_gpu;
	float **truth_gpu;
#endif

#ifdef OPENCL
	cl_mem workspace_ocl;
#endif
} network;

typedef struct network_state {
	float *truth;
	float *input;
	float *delta;
	float *workspace;
	int train;
	int index;
	network net;
#ifdef OPENCL
	cl_mem input_ocl;
	cl_mem workspace_ocl;
#endif
} network_state;


// network.c
network make_network(int n);


// network.c
#ifdef GPU
#ifdef CUDNN
void cudnn_convolutional_setup(layer *l);
void cuda_set_device(int n);
#endif
#endif

#ifdef OPENCL
bool ocl_initialize();
#endif

// network.c
void set_batch_network(network *net, int b);


// -------------- softmax_layer.h --------------

// softmax_layer.c
softmax_layer make_softmax_layer(int batch, int inputs, int groups);

// -------------- reorg_layer.h --------------

// reorg_layer.c
layer make_reorg_layer(int batch, int w, int h, int c, int stride, int reverse);

// -------------- route_layer.h --------------

// route_layer.c
route_layer make_route_layer(int batch, int n, int *input_layers, int *input_sizes);

// -------------- region_layer.h --------------

//  region_layer.c
region_layer make_region_layer(int batch, int w, int h, int n, int classes, int coords);


// -------------- maxpool_layer.h --------------

// maxpool_layer.c
maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding);

// -------------- convolutional_layer.h --------------

// convolutional_layer.c
convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam);



// -------------- image.c --------------

// image.c
typedef struct {
	int h;
	int w;
	int c;
	float *data;
} image;

// image.c
void rgbgr_image(image im);

// image.c
image make_empty_image(int w, int h, int c);

// image.c
void free_image(image m);

// image.c
void draw_box(image a, int x1, int y1, int x2, int y2, float r, float g, float b);

// image.c
void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b);

// image.c
image make_image(int w, int h, int c);

// image.c
float get_pixel(image m, int x, int y, int c);

// image.c
void set_pixel(image m, int x, int y, int c, float val);

// image.c
image resize_image(image im, int w, int h);

// image.c
image load_image(char *filename, int w, int h, int c);

// image.c
image load_image_stb(char *filename, int channels);

#ifdef OPENCV
// image.c
image ipl_to_image(IplImage* src);

// image.c
void show_image_cv_ipl(IplImage *disp, const char *name);
#endif

// image.c
image load_image_cv(char *filename, int channels);

// image.c
float get_color(int c, int x, int max);

// image.c
void save_image_png(image im, const char *name);

// image.c
void show_image(image p, const char *name);


// -------------- parser.c --------------------

// parser.c
network parse_network_cfg(char *filename);

// parser.c
void load_weights_upto_cpu(network *net, char *filename, int cutoff);


// -------------- yolov2_forward_network.c --------------------


// detect on CPU: yolov2_forward_network.c
float *network_predict_cpu(network net, float *input);

// quantizized
float *network_predict_quantized(network net, float *input);

// -------------- yolov2_forward_network_gpu.c --------------------

#ifdef GPU
// detect on GPU: yolov2_forward_network_gpu.cu
float *network_predict_gpu_cudnn(network net, float *input);
#endif

// -------------- yolov2_forward_network_ocl.c --------------------

#ifdef OPENCL
// detect using OpenCL: yolov2_forward_network_gpu.cpp
float *network_predict_opencl(network net, float *input);
#endif


// -------------- gettimeofday for Windows--------------------

#if defined(_MSC_VER) 
#include <time.h>
#include <windows.h> //I've ommited this line.
#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#include <sys/time.h>
#endif

struct timezone
{
	int  tz_minuteswest; /* minutes W of Greenwich */
	int  tz_dsttime;     /* type of dst correction */
};

int gettimeofday(struct timeval *tv, struct timezone *tz);
#endif

#ifdef __cplusplus
}
#endif

#endif	// ADDITIONALLY_H