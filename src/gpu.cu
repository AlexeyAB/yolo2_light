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
    if (layer.batch_normalize) {
        cuda_pull_array(layer.scales_gpu, layer.scales, layer.n);
        cuda_pull_array(layer.rolling_mean_gpu, layer.rolling_mean, layer.n);
        cuda_pull_array(layer.rolling_variance_gpu, layer.rolling_variance, layer.n);
    }
}

void push_convolutional_layer(convolutional_layer layer)
{
    cuda_push_array(layer.weights_gpu, layer.weights, layer.c*layer.n*layer.size*layer.size);
    cuda_push_array(layer.biases_gpu, layer.biases, layer.n);
    if (layer.batch_normalize) {
        cuda_push_array(layer.scales_gpu, layer.scales, layer.n);
        cuda_push_array(layer.rolling_mean_gpu, layer.rolling_mean, layer.n);
        cuda_push_array(layer.rolling_variance_gpu, layer.rolling_variance, layer.n);
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
__device__ float lhtan_activate_kernel(float x)
{
    if (x < 0) return .001*x;
    if (x > 1) return .001*(x - 1) + 1;
    return x;
}
__device__ float lhtan_gradient_kernel(float x)
{
    if (x > 0 && x < 1) return 1;
    return .001;
}

__device__ float hardtan_activate_kernel(float x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}
__device__ float linear_activate_kernel(float x) { return x; }
__device__ float logistic_activate_kernel(float x) { return 1. / (1. + exp(-x)); }
__device__ float loggy_activate_kernel(float x) { return 2. / (1. + exp(-x)) - 1; }
__device__ float relu_activate_kernel(float x) { return x*(x>0); }
__device__ float elu_activate_kernel(float x) { return (x >= 0)*x + (x < 0)*(exp(x) - 1); }
__device__ float relie_activate_kernel(float x) { return (x>0) ? x : .01*x; }
__device__ float ramp_activate_kernel(float x) { return x*(x>0) + .1*x; }
__device__ float leaky_activate_kernel(float x) { return (x>0) ? x : .1*x; }
__device__ float tanh_activate_kernel(float x) { return (2 / (1 + exp(-2 * x)) - 1); }
__device__ float plse_activate_kernel(float x)
{
    if (x < -4) return .01 * (x + 4);
    if (x > 4)  return .01 * (x - 4) + 1;
    return .125*x + .5;
}
__device__ float stair_activate_kernel(float x)
{
    int n = floor(x);
    if (n % 2 == 0) return floor(x / 2.);
    else return (x - n) + floor(x / 2.);
}


__device__ float activate_kernel(float x, ACTIVATION a)
{
    switch (a) {
    case LINEAR:
        return linear_activate_kernel(x);
    case LOGISTIC:
        return logistic_activate_kernel(x);
    case LOGGY:
        return loggy_activate_kernel(x);
    case RELU:
        return relu_activate_kernel(x);
    case ELU:
        return elu_activate_kernel(x);
    case RELIE:
        return relie_activate_kernel(x);
    case RAMP:
        return ramp_activate_kernel(x);
    case LEAKY:
        return leaky_activate_kernel(x);
    case TANH:
        return tanh_activate_kernel(x);
    case PLSE:
        return plse_activate_kernel(x);
    case STAIR:
        return stair_activate_kernel(x);
    case HARDTAN:
        return hardtan_activate_kernel(x);
    case LHTAN:
        return lhtan_activate_kernel(x);
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



// upsample layer
__global__ void upsample_kernel(size_t N, float *x, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{
    size_t i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int out_index = i;
    int out_w = i % (w*stride);
    i = i / (w*stride);
    int out_h = i % (h*stride);
    i = i / (h*stride);
    int out_c = i%c;
    i = i / c;
    int b = i%batch;

    int in_w = out_w / stride;
    int in_h = out_h / stride;
    int in_c = out_c;

    int in_index = b*w*h*c + in_c*w*h + in_h*w + in_w;


    if (forward) out[out_index] += scale * x[in_index];
    else atomicAdd(x + in_index, scale * out[out_index]);
}

extern "C" void upsample_gpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{
    size_t size = w*h*c*batch*stride*stride;
    upsample_kernel << <cuda_gridsize(size), BLOCK >> >(size, in, w, h, c, batch, stride, forward, scale, out);
    check_error(cudaPeekAtLastError());
}


__global__ void copy_kernel(int N, float *X, int OFFX, int INCX, float *Y, int OFFY, int INCY)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < N) Y[i*INCY + OFFY] = X[i*INCX + OFFX];
}

extern "C" void copy_ongpu_offset(int N, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY)
{
    copy_kernel << <cuda_gridsize(N), BLOCK>> >(N, X, OFFX, INCX, Y, OFFY, INCY);
    check_error(cudaPeekAtLastError());
}

extern "C" void copy_ongpu(int N, float * X, int INCX, float * Y, int INCY)
{
    copy_ongpu_offset(N, X, 0, INCX, Y, 0, INCY);
}


// shortcut layer
__global__ void shortcut_kernel(int size, int minw, int minh, int minc, int stride, int sample, int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float *out)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;
    int i = id % minw;
    id /= minw;
    int j = id % minh;
    id /= minh;
    int k = id % minc;
    id /= minc;
    int b = id % batch;

    int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
    int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
    out[out_index] += add[add_index];
}

extern "C" void shortcut_gpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float *out)
{
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int stride = w1 / w2;
    int sample = w2 / w1;
    assert(stride == h1 / h2);
    assert(sample == h2 / h1);
    if (stride < 1) stride = 1;
    if (sample < 1) sample = 1;

    int size = batch * minw * minh * minc;
    shortcut_kernel << <cuda_gridsize(size), BLOCK>> >(size, minw, minh, minc, stride, sample, batch, w1, h1, c1, add, w2, h2, c2, out);
    check_error(cudaPeekAtLastError());
}

// ----------- Quantinization --------------

__host__ __device__ int max_abs(int src, int max_val) {
    if (abs(src) > abs(max_val)) src = (src > 0) ? max_val : -max_val;
    return src;
}

__global__ void cuda_f32_to_int8(float* input_f32, size_t size, int8_t *output_int8, float multipler, int max_val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output_int8[idx] = max_abs(input_f32[idx] * multipler, max_val); // 7-bit (1-bit sign) 

}

void cuda_convert_f32_to_int8(float* input_f32, size_t size, int8_t *output_int8, float multipler, int max_val) {
    cuda_f32_to_int8 << < size / BLOCK + 1, BLOCK >> >(input_f32, size, output_int8, multipler, max_val);
}



__global__ void cuda_f32_to_int8_nomax(float* input_f32, size_t size, int8_t *output_int8, float multipler)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output_int8[idx] = input_f32[idx] * multipler; // 7-bit (1-bit sign) 

}

void cuda_convert_f32_to_int8_nomax(float* input_f32, size_t size, int8_t *output_int8, float multipler) {
    cuda_f32_to_int8_nomax << < size / BLOCK + 1, BLOCK >> >(input_f32, size, output_int8, multipler);
}



__global__ void cuda_int8_to_f32(int8_t* input_int8, size_t size, float *output_f32, float multipler)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output_f32[idx] = input_int8[idx] * multipler; // 7-bit (1-bit sign) 

}

void cuda_convert_int8_to_f32(int8_t* input_int8, size_t size, float *output_f32, float multipler) {
    cuda_int8_to_f32 << < size / BLOCK + 1, BLOCK >> >(input_int8, size, output_f32, multipler);
}



__global__ void cuda_multiply_f32(float *input_output, size_t size, float multipler)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) input_output[idx] = input_output[idx] * multipler; // 7-bit (1-bit sign) 

}

void cuda_do_multiply_f32(float *input_output, size_t size, float multipler) {
    cuda_multiply_f32 << < size / BLOCK + 1, BLOCK >> >(input_output, size, multipler);
}