#pragma once
#ifndef GPU_H
#define GPU_H

#ifdef __cplusplus
extern "C" {
#endif


#ifdef GPU

struct layer;
typedef struct layer layer;
typedef struct layer local_layer;

void pull_batchnorm_layer(layer l); // not required now
void push_batchnorm_layer(layer l); // not required now
void pull_local_layer(local_layer l); // not required now
void push_local_layer(local_layer l); // not required now
void pull_connected_layer(local_layer l); // not required now
void push_connected_layer(local_layer l); // not required now


void check_error(cudaError_t status);
void cuda_set_device(int n);
int cuda_get_device();

#ifdef CUDNN
cudnnHandle_t cudnn_handle();
#endif

float *cuda_make_array(float *x, size_t n);
int *cuda_make_int_array(size_t n);
void cuda_free(float *x_gpu);
void cuda_push_array(float *x_gpu, float *x, size_t n);
void cuda_pull_array(float *x_gpu, float *x, size_t n);
float *get_network_output_layer_gpu(network net, int i);
float *get_network_output_gpu(network net);
dim3 cuda_gridsize(size_t n);
void pull_convolutional_layer(convolutional_layer layer);
void push_convolutional_layer(convolutional_layer layer);

// -------------------- CUDA functions -------------------

// add BIAS
void add_bias_gpu(float *output, float *biases, int batch, int n, int size);

// normalization
void normalize_gpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);

// fill array
void fill_ongpu(int N, float ALPHA, float * X, int INCX);

// scale BIAS
void scale_bias_gpu(float *output, float *biases, int batch, int n, int size);

// max-pool layer
void forward_maxpool_layer_gpu(maxpool_layer layer, network_state state);

// flatten
void flatten_ongpu(float *x, int spatial, int layers, int batch, int forward, float *out);


// activations
void activate_array_ongpu(float *x, int n, ACTIVATION a);

// softmax layer
void softmax_gpu(float *input, int n, int offset, int groups, float temp, float *output);

// reorg layer
void reorg_ongpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out);

#endif // GPU

#ifdef __cplusplus
}
#endif

#endif	// GPU_H