#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "additionally.h"
#include "gpu.h"

extern int gpu_index;

#define BLOCK 512


void pull_batchnorm_layer(layer l) {} // not required now
void push_batchnorm_layer(layer l) {} // not required now
void pull_local_layer(local_layer l) {} // not required now
void push_local_layer(local_layer l) {} // not required now
void pull_connected_layer(local_layer l) {} // not required now
void push_connected_layer(local_layer l) {} // not required now


void check_error(cudaError_t status)
{
	//cudaDeviceSynchronize();
	cudaError_t status2 = cudaGetLastError();
	if (status != cudaSuccess)
	{
		const char *s = cudaGetErrorString(status);
		char buffer[256];
		printf("CUDA Error: %s\n", s);
		assert(0);
		snprintf(buffer, 256, "CUDA Error: %s", s);
		error(buffer);
	}
	if (status2 != cudaSuccess)
	{
		const char *s = cudaGetErrorString(status);
		char buffer[256];
		printf("CUDA Error Prev: %s\n", s);
		assert(0);
		snprintf(buffer, 256, "CUDA Error Prev: %s", s);
		error(buffer);
	}
}

void cuda_set_device(int n)
{
	gpu_index = n;
	cudaError_t status = cudaSetDevice(n);
	check_error(status);
}

int cuda_get_device()
{
	int n = 0;
	cudaError_t status = cudaGetDevice(&n);
	check_error(status);
	return n;
}

#ifdef CUDNN
cudnnHandle_t cudnn_handle()
{
	static int init[16] = { 0 };
	static cudnnHandle_t handle[16];
	int i = cuda_get_device();
	if (!init[i]) {
		cudnnCreate(&handle[i]);
		init[i] = 1;
	}
	return handle[i];
}
#endif

float *cuda_make_array(float *x, size_t n)
{
	float *x_gpu;
	size_t size = sizeof(float)*n;
	cudaError_t status = cudaMalloc((void **)&x_gpu, size);
	check_error(status);
	if (x) {
		status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
		check_error(status);
	}
	if (!x_gpu) error("Cuda malloc failed\n");
	return x_gpu;
}


int *cuda_make_int_array(size_t n)
{
	int *x_gpu;
	size_t size = sizeof(int)*n;
	cudaError_t status = cudaMalloc((void **)&x_gpu, size);
	check_error(status);
	return x_gpu;
}

void cuda_free(float *x_gpu)
{
	cudaError_t status = cudaFree(x_gpu);
	check_error(status);
}

void cuda_push_array(float *x_gpu, float *x, size_t n)
{
	size_t size = sizeof(float)*n;
	cudaError_t status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
	check_error(status);
}

void cuda_pull_array(float *x_gpu, float *x, size_t n)
{
	size_t size = sizeof(float)*n;
	cudaError_t status = cudaMemcpy(x, x_gpu, size, cudaMemcpyDeviceToHost);
	check_error(status);
}

float *get_network_output_layer_gpu(network net, int i)
{
	layer l = net.layers[i];
	if (l.type != REGION) cuda_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
	return l.output;
}

float *get_network_output_gpu(network net)
{
	int i;
	for (i = net.n - 1; i > 0; --i) if (net.layers[i].type != COST) break;
	return get_network_output_layer_gpu(net, i);
}

dim3 cuda_gridsize(size_t n) {
	size_t k = (n - 1) / BLOCK + 1;
	size_t x = k;
	size_t y = 1;
	if (x > 65535) {
		x = ceil(sqrtf(k));
		y = (n - 1) / (x*BLOCK) + 1;
	}
	dim3 d;
	d.x = x;
	d.y = y;
	d.z = 1;
	//printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
	return d;
}


void pull_convolutional_layer(convolutional_layer layer)
{
	cuda_pull_array(layer.weights_gpu, layer.weights, layer.c*layer.n*layer.size*layer.size);
	cuda_pull_array(layer.biases_gpu, layer.biases, layer.n);
	cuda_pull_array(layer.weight_updates_gpu, layer.weight_updates, layer.c*layer.n*layer.size*layer.size);
	cuda_pull_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
	if (layer.batch_normalize) {
		cuda_pull_array(layer.scales_gpu, layer.scales, layer.n);
		cuda_pull_array(layer.rolling_mean_gpu, layer.rolling_mean, layer.n);
		cuda_pull_array(layer.rolling_variance_gpu, layer.rolling_variance, layer.n);
	}
	if (layer.adam) {
		cuda_pull_array(layer.m_gpu, layer.m, layer.c*layer.n*layer.size*layer.size);
		cuda_pull_array(layer.v_gpu, layer.v, layer.c*layer.n*layer.size*layer.size);
	}
}

void push_convolutional_layer(convolutional_layer layer)
{
	cuda_push_array(layer.weights_gpu, layer.weights, layer.c*layer.n*layer.size*layer.size);
	cuda_push_array(layer.biases_gpu, layer.biases, layer.n);
	cuda_push_array(layer.weight_updates_gpu, layer.weight_updates, layer.c*layer.n*layer.size*layer.size);
	cuda_push_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
	if (layer.batch_normalize) {
		cuda_push_array(layer.scales_gpu, layer.scales, layer.n);
		cuda_push_array(layer.rolling_mean_gpu, layer.rolling_mean, layer.n);
		cuda_push_array(layer.rolling_variance_gpu, layer.rolling_variance, layer.n);
	}
	if (layer.adam) {
		cuda_push_array(layer.m_gpu, layer.m, layer.c*layer.n*layer.size*layer.size);
		cuda_push_array(layer.v_gpu, layer.v, layer.c*layer.n*layer.size*layer.size);
	}
}

// -------------------- CUDA functions -------------------

// add BIAS
__global__ void add_bias_kernel(float *output, float *biases, int n, int size)
{
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	int filter = blockIdx.y;
	int batch = blockIdx.z;

	if (offset < size) output[(batch*n + filter)*size + offset] += biases[filter];
}

void add_bias_gpu(float *output, float *biases, int batch, int n, int size)
{
	dim3 dimGrid((size - 1) / BLOCK + 1, n, batch);
	dim3 dimBlock(BLOCK, 1, 1);

	add_bias_kernel << <dimGrid, dimBlock >> >(output, biases, n, size);
	check_error(cudaPeekAtLastError());
}

// normalization
__global__ void normalize_kernel(int N, float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
	int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (index >= N) return;
	int f = (index / spatial) % filters;

	x[index] = (x[index] - mean[f]) / (sqrtf(variance[f]) + .000001f);
}

void normalize_gpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
	size_t N = batch*filters*spatial;
	normalize_kernel << <cuda_gridsize(N), BLOCK >> >(N, x, mean, variance, batch, filters, spatial);
	check_error(cudaPeekAtLastError());
}

// fill array
__global__ void fill_kernel(int N, float ALPHA, float *X, int INCX)
{
	int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i < N) X[i*INCX] = ALPHA;
}

void fill_ongpu(int N, float ALPHA, float * X, int INCX)
{
	fill_kernel << <cuda_gridsize(N), BLOCK >> >(N, ALPHA, X, INCX);
	check_error(cudaPeekAtLastError());
}

// scale BIAS
__global__ void scale_bias_kernel(float *output, float *biases, int n, int size)
{
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	int filter = blockIdx.y;
	int batch = blockIdx.z;

	if (offset < size) output[(batch*n + filter)*size + offset] *= biases[filter];
}

void scale_bias_gpu(float *output, float *biases, int batch, int n, int size)
{
	dim3 dimGrid((size - 1) / BLOCK + 1, n, batch);
	dim3 dimBlock(BLOCK, 1, 1);

	scale_bias_kernel << <dimGrid, dimBlock >> >(output, biases, n, size);
	check_error(cudaPeekAtLastError());
}

// max-pool layer
__global__ void forward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride, int size, int pad, float *input, float *output, int *indexes)
{
	int h = (in_h + 2 * pad) / stride;
	int w = (in_w + 2 * pad) / stride;
	int c = in_c;

	int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (id >= n) return;

	int j = id % w;
	id /= w;
	int i = id % h;
	id /= h;
	int k = id % c;
	id /= c;
	int b = id;

	int w_offset = -pad;
	int h_offset = -pad;

	int out_index = j + w*(i + h*(k + c*b));
	float max = -INFINITY;
	int max_i = -1;
	int l, m;
	for (l = 0; l < size; ++l) {
		for (m = 0; m < size; ++m) {
			int cur_h = h_offset + i*stride + l;
			int cur_w = w_offset + j*stride + m;
			int index = cur_w + in_w*(cur_h + in_h*(k + b*in_c));
			int valid = (cur_h >= 0 && cur_h < in_h &&
				cur_w >= 0 && cur_w < in_w);
			float val = (valid != 0) ? input[index] : -INFINITY;
			max_i = (val > max) ? index : max_i;
			max = (val > max) ? val : max;
		}
	}
	output[out_index] = max;
	indexes[out_index] = max_i;
}

void forward_maxpool_layer_gpu(maxpool_layer layer, network_state state)
{
	int h = layer.out_h;
	int w = layer.out_w;
	int c = layer.c;

	size_t n = h*w*c*layer.batch;

	forward_maxpool_layer_kernel << <cuda_gridsize(n), BLOCK >> >(n, layer.h, layer.w, layer.c, layer.stride, layer.size, layer.pad, state.input, layer.output_gpu, layer.indexes_gpu);
	check_error(cudaPeekAtLastError());
}

// flatten
__global__ void flatten_kernel(int N, float *x, int spatial, int layers, int batch, int forward, float *out)
{
	int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i >= N) return;
	int in_s = i%spatial;
	i = i / spatial;
	int in_c = i%layers;
	i = i / layers;
	int b = i;

	int i1 = b*layers*spatial + in_c*spatial + in_s;
	int i2 = b*layers*spatial + in_s*layers + in_c;

	if (forward) out[i2] = x[i1];
	else out[i1] = x[i2];
}

void flatten_ongpu(float *x, int spatial, int layers, int batch, int forward, float *out)
{
	int size = spatial*batch*layers;
	flatten_kernel << <cuda_gridsize(size), BLOCK >> >(size, x, spatial, layers, batch, forward, out);
	check_error(cudaPeekAtLastError());
}


// activations
__device__ float linear_activate_kernel(float x) { return x; }
__device__ float leaky_activate_kernel(float x) { return (x>0) ? x : .1f*x; }

__device__ float activate_kernel(float x, ACTIVATION a)
{
	switch (a) {
	case LINEAR:
		return linear_activate_kernel(x);
	case LEAKY:
		return leaky_activate_kernel(x);
	}
	return 0;
}

__global__ void activate_array_kernel(float *x, int n, ACTIVATION a)
{
	int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i < n) x[i] = activate_kernel(x[i], a);
}

void activate_array_ongpu(float *x, int n, ACTIVATION a)
{
	activate_array_kernel << <cuda_gridsize(n), BLOCK >> >(x, n, a);
	check_error(cudaPeekAtLastError());
}

// softmax layer
__device__ void softmax_device(int n, float *input, float temp, float *output)
{
	int i;
	float sum = 0;
	float largest = -INFINITY;
	for (i = 0; i < n; ++i) {
		int val = input[i];
		largest = (val>largest) ? val : largest;
	}
	for (i = 0; i < n; ++i) {
		float e = expf(input[i] / temp - largest / temp);
		sum += e;
		output[i] = e;
	}
	for (i = 0; i < n; ++i) {
		output[i] /= sum;
	}
}

__global__ void softmax_kernel(int n, int offset, int batch, float *input, float temp, float *output)
{
	int b = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (b >= batch) return;
	softmax_device(n, input + b*offset, temp, output + b*offset);
}

void softmax_gpu(float *input, int n, int offset, int groups, float temp, float *output)
{
	int inputs = n;
	int batch = groups;
	softmax_kernel << <cuda_gridsize(batch), BLOCK >> >(inputs, offset, batch, input, temp, output);
	check_error(cudaPeekAtLastError());
}

// reorg layer
__global__ void reorg_kernel(int N, float *x, int w, int h, int c, int batch, int stride, int forward, float *out)
{
	int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i >= N) return;
	int in_index = i;
	int in_w = i%w;
	i = i / w;
	int in_h = i%h;
	i = i / h;
	int in_c = i%c;
	i = i / c;
	int b = i%batch;

	int out_c = c / (stride*stride);

	int c2 = in_c % out_c;
	int offset = in_c / out_c;
	int w2 = in_w*stride + offset % stride;
	int h2 = in_h*stride + offset / stride;

	int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));

	out[in_index] = x[out_index];
}

void reorg_ongpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out)
{
	int size = w*h*c*batch;
	reorg_kernel << <cuda_gridsize(size), BLOCK >> >(size, x, w, h, c, batch, stride, forward, out);
	check_error(cudaPeekAtLastError());
}
