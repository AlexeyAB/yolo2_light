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
    int h = (in_h + pad - size) / stride + 1;
    int w = (in_w + pad - size) / stride + 1;
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

    int w_offset = -pad / 2;
    int h_offset = -pad / 2;

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
    if (layer.stride == layer.size) {
    //if(1) {
        cudnnStatus_t maxpool_status;

        float alpha = 1, beta = 0;
        maxpool_status = cudnnPoolingForward(
            cudnn_handle(),
            layer.poolingDesc,
            &alpha,
            layer.srcTensorDesc,
            state.input,
            &beta,
            layer.dstTensorDesc,
            layer.output_gpu);

        //maxpool_status = cudnnDestroyPoolingDescriptor(poolingDesc);
        //cudnnDestroyTensorDescriptor(layer.srcTensorDesc);
        //cudnnDestroyTensorDescriptor(layer.dstTensorDesc);
    }
    else {
        int h = layer.out_h;
        int w = layer.out_w;
        int c = layer.c;

        size_t n = h*w*c*layer.batch;

        forward_maxpool_layer_kernel << <cuda_gridsize(n), BLOCK >> > (n, layer.h, layer.w, layer.c, layer.stride, layer.size, layer.pad, state.input, layer.output_gpu, layer.indexes_gpu);
        check_error(cudaPeekAtLastError());
    }
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

__global__ void activate_array_leaky_kernel(float *x, int n)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < n) {
        float val = x[index];
        x[index] = (val > 0) ? val : val / 10;
    }
}

extern "C" void activate_array_ongpu(float *x, int n, ACTIVATION a)
{
    if (a == LEAKY) activate_array_leaky_kernel << <(n / BLOCK + 1), BLOCK, 0, 0 >> >(x, n);
    else activate_array_kernel << <cuda_gridsize(n), BLOCK, 0, 0 >> >(x, n, a);
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

// --------------------------------
// ------------- XNOR -------------
// --------------------------------

__global__ void binarize_weights_kernel(float *weights, int n, int size, float *binary)
{
    int f = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (f >= n) return;
    int i = 0;
    float mean = 0;
    for (i = 0; i < size; ++i) {
        mean += fabs(weights[f*size + i]);
    }
    mean = mean / size;
    for (i = 0; i < size; ++i) {
        binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        //binary[f*size + i] = weights[f*size + i];
    }
}

void binarize_weights_gpu(float *weights, int n, int size, float *binary)
{
    binarize_weights_kernel << <cuda_gridsize(n), BLOCK >> >(weights, n, size, binary);
    check_error(cudaPeekAtLastError());
}
// --------------------------------

__global__ void binarize_kernel(float *x, int n, float *binary)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    binary[i] = (x[i] >= 0) ? 1 : -1;
}

void binarize_gpu(float *x, int n, float *binary)
{
    binarize_kernel << <cuda_gridsize(n), BLOCK >> >(x, n, binary);
    check_error(cudaPeekAtLastError());
}
// --------------------------------

void swap_binary(convolutional_layer *l)
{
    float *swap = l->weights;
    l->weights = l->binary_weights;
    l->binary_weights = swap;

#ifdef GPU
    swap = l->weights_gpu;
    l->weights_gpu = l->binary_weights_gpu;
    l->binary_weights_gpu = swap;
#endif
}
// --------------------------------


#define WARP_SIZE 32

__global__ void im2col_align_gpu_kernel(const int n, const float* data_im,
    const int height, const int width, const int ksize,
    const int pad,
    const int stride,
    const int height_col, const int width_col,
    float *data_col, const int bit_align)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    for (; index < n; index += blockDim.x*gridDim.x) {
        int w_out = index % width_col;
        int h_index = index / width_col;
        int h_out = h_index % height_col;
        int channel_in = h_index / height_col;
        int channel_out = channel_in * ksize * ksize;
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;
        float* data_col_ptr = data_col;
        //data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
        data_col_ptr += channel_out * bit_align + h_out * width_col + w_out;
        float* data_col_ptr_32 = data_col + (channel_out * bit_align + h_out * width_col + w_out) / 32;
        const float* data_im_ptr = data_im;
        data_im_ptr += (channel_in * height + h_in) * width + w_in;
        for (int i = 0; i < ksize; ++i) {
            for (int j = 0; j < ksize; ++j) {
                int h = h_in + i;
                int w = w_in + j;

                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                    data_im_ptr[i * width + j] : 0;

                //float src_val = (h >= 0 && w >= 0 && h < height && w < width) ? data_im_ptr[i * width + j] : 0;
                //unsigned int bit_mask = __ballot_sync(0xffffffff, src_val > 0);
                //if (threadIdx.x % WARP_SIZE == 0) *((unsigned int*)data_col_ptr_32) = bit_mask;
                //data_col_ptr_32 += bit_align / 32;

                //data_col_ptr += height_col * width_col;
                data_col_ptr += bit_align;
            }
        }
    }
}

void im2col_align_ongpu(float *im,
    int channels, int height, int width,
    int ksize, int stride, int pad, float *data_col, int bit_align)
{
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height_col * width_col;
    im2col_align_gpu_kernel << <(num_kernels + BLOCK - 1) / BLOCK,
        BLOCK, 0, 0>> >(
            num_kernels, im, height, width, ksize, pad,
            stride, height_col,
            width_col, data_col, bit_align);
}
// --------------------------------



// binary im2col - stride=1
__global__ void im2col_align_bin_gpu_kernel(const int n, const float* data_im,
    const int height, const int width, const int ksize, const int channels,
    const int pad,
    const int stride,
    const int height_col, const int width_col,
    float *data_col, const int bit_align)
{
    __shared__ float tmp_s[1];
    __shared__ ulonglong4 tmp256_s[1];


    //#define SHRED_VALS ((BLOCK / 169) * )
    //__shared__ float dst_s[1024];
    //__shared__ float dst_s[1024];
    //__shared__ uint32_t bit_s[32];
    //__shared__ uint8_t bit_s[128];

    int index = blockIdx.x*blockDim.x + threadIdx.x;
    //for (; index < n; index += blockDim.x*gridDim.x)
    {
        int c_index = index;
        int channel_in = c_index % channels;

        //int h_out = index % height_col;
        //int c_index = index / height_col;
        //int channel_in = c_index % channels;

        int channel_out = channel_in * ksize * ksize;

        int j_index = c_index / channels;
        int j = j_index % ksize;
        int i = j_index / ksize;

        int pre_out_index = (channel_out + i*ksize + j) * bit_align;
        int j_pad = (j - pad);
        int i_pad = (i - pad);

        for (int wh_index = 0; wh_index < (height_col*width_col); wh_index += 32)
            //for (int h_out = 0; h_out < height_col; ++h_out)
        {

            // the end of padding
            //if(0)
            //for (int w_out = 0; w_out < (width_col); w_out += 32)
            {
                const int w_out = wh_index % width_col;
                const int h_out = wh_index / width_col;

                const int w = w_out + j_pad;
                const int h = h_out + i_pad;

                int pre_in_index = channel_in * height * width;
                int pre_in_wh_index = h * width + w;

                int send_wh_index = wh_index;
                if (i >= ksize) send_wh_index = height_col*width_col;

                #pragma unroll
                for (int t = 0; t < WARP_SIZE; ++t)
                {
                    const int lane_id = threadIdx.x % WARP_SIZE;

                    const int cur_wh_index = __shfl(send_wh_index, t) + lane_id;

                    if (cur_wh_index < (width_col*height_col))// && (cur_i_pad+pad) < ksize)
                    {
                        const int cur_pre_out_index = __shfl(pre_out_index, t);

                        const int cur_pre_in_index = __shfl(pre_in_index, t);
                        const int cur_pre_in_wh_index = __shfl(pre_in_wh_index, t) + lane_id;

                        int w = cur_pre_in_wh_index % width;
                        int h = cur_pre_in_wh_index / width;
                        int in_index = cur_pre_in_index + cur_pre_in_wh_index;

                        int out_index = cur_pre_out_index + cur_wh_index;

                        float val = (w >= 0 && w < width && h >= 0 && h < height) ?
                            data_im[in_index] : float();

                        //data_col[out_index] = val;
                        //tmp_s[0] = val;

                        uint32_t bit_mask = __ballot(val > 0);
                        if (lane_id == 0) {
                            uint8_t *bit8_ptr = &(((uint8_t *)data_col)[out_index / 8]);
                            uint32_t *bit32_ptr = (uint32_t *)bit8_ptr;
                            *bit32_ptr = bit_mask;
                        }
                    }


                }

            }// w_out

        }
    }
}


void im2col_align_bin_ongpu(float *im,
    int channels, int height, int width,
    int ksize, int stride, int pad, float *data_col, int bit_align) {
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    //int num_kernels = channels * height_col * width_col * ksize * ksize;
    //int num_kernels = channels * ksize * ksize * height_col;
    int num_kernels = channels * ksize * ksize;
    int num_blocks = num_kernels / BLOCK + 1;

    //im2col_align_bin_gpu_kernel << <(num_kernels + BLOCK - 1) / BLOCK,
    im2col_align_bin_gpu_kernel << <num_blocks,
        BLOCK, 0, 0 >> >(
            num_kernels, im, height, width, ksize, channels, pad,
            stride, height_col,
            width_col, data_col, bit_align);
}
// --------------------------------

__global__ void float_to_bit_gpu_kernel(float *src, unsigned char *dst, size_t size)
{
    //const int size_aligned = size + (WARP_SIZE - size % WARP_SIZE);

    int index = blockIdx.x*blockDim.x + threadIdx.x;
    float src_val;

    //for (; index < size_aligned; index += blockDim.x*gridDim.x)
    {
        //src_val = src[index];
        if (index < size) src_val = src[index];
        else src_val = 0;
        //unsigned int bit_mask = __ballot_sync(0xffffffff, src_val > 0);
        unsigned int bit_mask = __ballot(src_val > 0);
        if (threadIdx.x % WARP_SIZE == 0) ((unsigned int*)dst)[index / 32] = bit_mask;
    }
}


void float_to_bit_gpu(float *src, unsigned char *dst, size_t size)
{
    const int num_blocks = size / BLOCK + 1;
    float_to_bit_gpu_kernel << <num_blocks, BLOCK, 0, 0 >> >(src, dst, size);
}
// --------------------------------


__device__ __host__ static inline void remove_bit(unsigned char *const dst, size_t index) {
    size_t dst_i = index / 8;
    int dst_shift = index % 8;
    dst[dst_i] &= ~(1 << dst_shift);
}

__device__ __host__ static inline void set_bit(unsigned char *const dst, size_t index) {
    size_t dst_i = index / 8;
    int dst_shift = index % 8;
    dst[dst_i] |= 1 << dst_shift;
    //dst[dst_i] |= 1 << (8 - dst_shift);
}

__device__ __host__ static inline unsigned char get_bit(unsigned char const*const src, size_t index) {
    size_t src_i = index / 8;
    int src_shift = index % 8;
    unsigned char val = (src[src_i] & (1 << src_shift)) > 0;
    //unsigned char val = (src[src_i] & (1 << (8 - src_shift))) > 0;
    return val;
}

// Intel CPUs and nVidia CUDA GPU are little endian
__device__ __host__ unsigned char reverse_byte(unsigned char a)
{
    return ((a & 0x1) << 7) | ((a & 0x2) << 5) |
        ((a & 0x4) << 3) | ((a & 0x8) << 1) |
        ((a & 0x10) >> 1) | ((a & 0x20) >> 3) |
        ((a & 0x40) >> 5) | ((a & 0x80) >> 7);
}

__device__ unsigned char reverse_byte_CUDA(unsigned char a)
{
    uint32_t tmp = __brev(a);
    return tmp >> 24;
}

__device__ __host__ unsigned char reverse_byte_2(unsigned char a)
{
    return ((a * 0x0802LU & 0x22110LU) | (a * 0x8020LU & 0x88440LU)) * 0x10101LU >> 16;
}

__device__ void transpose8rS32_reversed_diagonale(unsigned char* A, int m, int n, unsigned char* B)
{
    unsigned x, y, t;

    // Load the array and pack it into x and y.
    x = (A[0] << 24) | (A[m] << 16) | (A[2 * m] << 8) | A[3 * m];
    y = (A[4 * m] << 24) | (A[5 * m] << 16) | (A[6 * m] << 8) | A[7 * m];

    t = (x ^ (x >> 7)) & 0x00AA00AA;  x = x ^ t ^ (t << 7);
    t = (y ^ (y >> 7)) & 0x00AA00AA;  y = y ^ t ^ (t << 7);

    t = (x ^ (x >> 14)) & 0x0000CCCC;  x = x ^ t ^ (t << 14);
    t = (y ^ (y >> 14)) & 0x0000CCCC;  y = y ^ t ^ (t << 14);

    t = (x & 0xF0F0F0F0) | ((y >> 4) & 0x0F0F0F0F);
    y = ((x << 4) & 0xF0F0F0F0) | (y & 0x0F0F0F0F);
    x = t;

    B[7 * n] = reverse_byte_CUDA(x >> 24);  B[6 * n] = reverse_byte_CUDA(x >> 16);  B[5 * n] = reverse_byte_CUDA(x >> 8);  B[4 * n] = reverse_byte_CUDA(x);
    B[3 * n] = reverse_byte_CUDA(y >> 24);  B[2 * n] = reverse_byte_CUDA(y >> 16);  B[1 * n] = reverse_byte_CUDA(y >> 8);  B[0 * n] = reverse_byte_CUDA(y);
}


__global__ void transpose_bin_gpu_kernel(unsigned char *A, unsigned char *B, const int n, const int m,
    const int lda, const int ldb, const int block_size)
{
    int i;
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    //for (i = 0; i < n; i += 8)
    {
        i = (index * 8) % n;
        int j;
        //for (j = 0; j < m - 8; j += 8)
        {
            j = ((index * 8) / n) * 8;
            if (j < m - 8) {
                int a_index = i*lda + j;
                int b_index = j*ldb + i;
                transpose8rS32_reversed_diagonale(&A[a_index / 8], lda / 8, ldb / 8, &B[b_index / 8]);
            }
            else if (j < m) {
                for (; j < m; ++j) {
                    if (get_bit(A, i*lda + j)) set_bit(B, j*ldb + i);
                    else remove_bit(B, j*ldb + i);
                }
            }
        }
    }
}



__device__ __host__ uint8_t reverse_8_bit(uint8_t a) {
    return ((a * 0x0802LU & 0x22110LU) | (a * 0x8020LU & 0x88440LU)) * 0x10101LU >> 16;
}

__device__ uint32_t reverse_32_bit(uint32_t a)
{
    // __device__ ​ unsigned int __brev(unsigned int  x) // CUDA
    // unsigned int __rbit(unsigned int val) // for ARM    //__asm__("rbit %0, %1\n" : "=r"(output) : "r"(input));
    return __brev(a);
    //return (reverse_8_bit(a >> 24) << 0) |
    //    (reverse_8_bit(a >> 16) << 8) |
    //    (reverse_8_bit(a >> 8) << 16) |
    //    (reverse_8_bit(a >> 0) << 24);
}

#define swap(a0, a1, j, m) t = (a0 ^ (a1 >>j)) & m; a0 = a0 ^ t; a1 = a1 ^ (t << j);

__device__ void transpose32_optimized(uint32_t A[32]) {
    int j, k;
    unsigned m, t;

    //m = 0x0000FFFF;
    //for (j = 16; j != 0; j = j >> 1, m = m ^ (m << j)) {
    //    for (k = 0; k < 32; k = (k + j + 1) & ~j) {
    //        t = (A[k] ^ (A[k + j] >> j)) & m;
    //        A[k] = A[k] ^ t;
    //        A[k + j] = A[k + j] ^ (t << j);
    //    }
    //}

    j = 16;
    m = 0x0000FFFF;
    for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

    j = 8;
    m = 0x00ff00ff;
    for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

    j = 4;
    m = 0x0f0f0f0f;
    for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

    j = 2;
    m = 0x33333333;
    for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

    j = 1;
    m = 0x55555555;
    for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

    // reverse Y
    for (j = 0; j < 16; ++j) {
        uint32_t tmp = A[j];
        A[j] = reverse_32_bit(A[31 - j]);
        A[31 - j] = reverse_32_bit(tmp);
    }
}

#define BLOCK_TRANSPOSE32 256

__device__ void transpose_32x32_bits_reversed_diagonale(uint32_t *A, uint32_t *B, int m, int n)
{
    //unsigned A_tmp[32];
    //int i;
    //#pragma unroll
    //for (i = 0; i < 32; ++i) A_tmp[i] = A[i * m];
    //transpose32_optimized(A_tmp);
    //#pragma unroll
    //for (i = 0; i < 32; ++i) B[i*n] = A_tmp[i];

    __shared__ uint32_t A_shared[32 * BLOCK_TRANSPOSE32];
    uint32_t *A_tmp = &A_shared[32 * threadIdx.x];

    int i;
    #pragma unroll 32
    for (i = 0; i < 32; ++i) A_tmp[i] = A[i * m];
    transpose32_optimized(A_tmp);
    #pragma unroll 32
    for (i = 0; i < 32; ++i) B[i*n] = A_tmp[i];
}


// transpose 32x32 bit
__global__ void transpose_bin_gpu_kernel_32(uint32_t *A, uint32_t *B, const int n, const int m,
    const int lda, const int ldb, const int block_size)
{
    int i;
    int index = (blockIdx.x*blockDim.x + threadIdx.x) * 32;

    //for (i = 0; i < n; i += 8)
    {
        i = index % n;
        int j;
        //for (j = 0; j < m - 8; j += 8)
        {
            j = (index / n) * 32;
            if (j < m) {
                int a_index = i*lda + j;
                int b_index = j*ldb + i;
                transpose_32x32_bits_reversed_diagonale(&A[a_index / 32], &B[b_index / 32], lda / 32, ldb / 32);
            }
        }
    }
}

void transpose_bin_gpu(unsigned char *A, unsigned char *B, const int n, const int m,
    const int lda, const int ldb, const int block_size)
{
    size_t size = n*m / (8 * 8) + 1;
    size_t size32 = n*m / (32 * 32) + 1;
    const int num_blocks = size / BLOCK + 1;
    const int num_blocks32 = size32 / BLOCK_TRANSPOSE32 + 1;
    transpose_bin_gpu_kernel_32 << <num_blocks32, BLOCK_TRANSPOSE32, 0, 0 >> >((uint32_t *)A, (uint32_t *)B, n, m, lda, ldb, block_size);
    //transpose_bin_gpu_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> >(A, B, n, m, lda, ldb, block_size);
}
// --------------------------------


__global__ void fill_int8_gpu_kernel(unsigned char *src, unsigned char val, size_t size) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < size) src[index] = 0;
}

void fill_int8_gpu(unsigned char *src, unsigned char val, size_t size)
{
    const int num_blocks = size / BLOCK + 1;
    fill_int8_gpu_kernel << <num_blocks, BLOCK, 0, 0 >> >(src, val, size);
}
// --------------------------------

//typedef unsigned long long int uint64_t;
//typedef unsigned int uint32_t;
//typedef unsigned char uint8_t;
//typedef char int8_t;

__device__ __host__ static inline uint64_t broadcast_bit_1_to_64(uint8_t src) {
    return (src > 0) ? 0xFFFFFFFFFFFFFFFF : 0;
}

__device__ __host__ static inline uint8_t xnor_bit1(uint8_t a, uint8_t b) {
    return ~(a^b) & 0b1;
}

__device__ __host__ static inline uint32_t xnor_int32(uint32_t a, uint32_t b) {
    return ~(a^b);
}

__device__ __host__ static inline uint64_t xnor_int64(uint64_t a, uint64_t b) {
    return ~(a^b);
}

__device__ __host__ static inline uint4 xnor_int128(uint4 a, uint4 b) {
    uint4 res;
    res.w = ~(a.w^b.w);
    res.x = ~(a.x^b.x);
    res.y = ~(a.y^b.y);
    res.z = ~(a.z^b.z);
    return res;
}

__device__ __host__ static inline ulonglong4 xnor_int256(ulonglong4 a, ulonglong4 b) {
    ulonglong4 res;
    res.w = ~(a.w^b.w);
    res.x = ~(a.x^b.x);
    res.y = ~(a.y^b.y);
    res.z = ~(a.z^b.z);
    return res;
}

/*
// A (weights) in the shared_memory
__global__ void gemm_nn_custom_bin_mean_transposed_gpu_kernel(int M, int N, int K,
    unsigned char *A, int lda,
    unsigned char *B, int ldb,
    float *C, int ldc, float *mean_arr)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    __shared__ uint64_t A_s[6144];  // 48 KB // [lda x M`]
                                    //__shared__ uint8_t A_s[6144*8];  // 48 KB // [lda x M`]

    int start_i = blockIdx.x*blockDim.x / N;
    int end_i = (blockIdx.x*blockDim.x + blockDim.x) / N + 1;

    size_t shared_size = lda * (end_i - start_i);

    int i_cur = index / N;
    int local_i = i_cur - start_i;

    for (int k = threadIdx.x * 64; k < shared_size; k += blockDim.x * 64) {
        int x = start_i*lda + k;
        if (x < (M*lda)) *((uint64_t *)(A_s + k / 8)) = *((uint64_t *)(A + x / 8));
    }

    //if (i_cur < M && (index % N == 0 || threadIdx.x == 0)) {
    //for (int k = 0; k < K; k += 64) {   // l.size*l.size*l.c - one filter size [27 - 9216]
    //*((uint64_t *)(A_s + (local_i*lda + k) / 8)) = *((uint64_t *)(A + (i_cur*lda + k) / 8));    // weights
    //  }
    //}

    __syncthreads();

    int i, j, k, h;

    j = index % N;
    {    // out_h*out_w - one channel output size [169 - 173056]
        i = index / N;
        if (i < M)  // l.n - filters [16 - 55 - 1024]
        {
            float mean_val = mean_arr[i];
            int count = 0;

            for (k = 0; k < K; k += 64) {   // l.size*l.size*l.c - one filter size [27 - 9216]
                                            //uint64_t a_bit64 = *((uint64_t *)(A + (i*lda + k) / 8));    // weights
                uint64_t a_bit64 = *((uint64_t *)(A_s + (local_i*lda + k) / 8));    // weights
                uint64_t b_bit64 = *((uint64_t *)(B + (j*ldb + k) / 8));            // input
                uint64_t c_bit64 = xnor_int64(a_bit64, b_bit64);

                int tmp_count = __popcll(c_bit64);

                if (K - k < 64)  tmp_count = tmp_count - (64 - (K - k));    // remove extra bits
                count += tmp_count;
            }

            C[i*ldc + j] = (2 * count - K) * mean_val;
        }
    }
}

#include <cstdio>

void gemm_nn_custom_bin_mean_transposed_gpu(int M, int N, int K,
    unsigned char *A, int lda,
    unsigned char *B, int ldb,
    float *C, int ldc, float *mean_arr)
{
    size_t size = M*N;
    const int num_blocks = size / BLOCK + 1;

    gemm_nn_custom_bin_mean_transposed_gpu_kernel << <num_blocks, BLOCK, 0, 0 >> >(
        M, N, K,
        A, lda,
        B, ldb,
        C, ldc,
        mean_arr);
}
*/
// --------------------------------

__inline__ __device__
int warpAllReduceSum(int val) {
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2)
        val += __shfl_xor(val, mask);
    return val;
}


// Coalesced memory access
// A (weights) in the shared_memory - GOOD
__global__ void gemm_nn_custom_bin_mean_transposed_gpu_kernel(int M, int N, int K,
    unsigned char *A, int lda,
    unsigned char *B, int ldb,
    float *C, int ldc, float *mean_arr, float *bias_arr)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    __shared__ uint8_t A_s[6144 * 8 / 4];
    //__shared__ uint64_t A_s[6144];  // 48 KB // [lda x M`]
    //__shared__ uint8_t A_s[6144*8];  // 48 KB // [lda x M`]

    int start_i = blockIdx.x*blockDim.x / N;
    int end_i = (blockIdx.x*blockDim.x + blockDim.x) / N + 1;

    size_t shared_size = lda * (end_i - start_i);

    int i_cur = index / N;
    int local_i = i_cur - start_i;

    for (int k = threadIdx.x * 64; k < shared_size; k += blockDim.x * 64) {
        int x = start_i*lda + k;
        if (x < (M*lda)) *((uint64_t *)(A_s + k / 8)) = *((uint64_t *)(A + x / 8));
    }
    __syncthreads();

    int i, j, k, h;

    j = index % N;
    {    // out_h*out_w - one channel output size [169 - 173056]
        i = index / N;
        //if (i < M)  // l.n - filters [16 - 55 - 1024]
        {
            int count = 0;
            k = 0;

#ifdef NOT_USED
            // 32 thread X 256 bit = 8192 bit
            for (; k < (K - 8192); k += 8192) {   // l.size*l.size*l.c - one filter size [27 - 9216]
                ulonglong4 c_bit256;

                //int64_t A_cur_index = (i*lda + k) / 8;
                int64_t A_cur_index = (local_i*lda + k) / 8;
                int64_t B_cur_index = (j*ldb + k) / 8;
                if (i >= M) A_cur_index = 0;

                #pragma unroll
                for (int t = 0; t < WARP_SIZE; ++t) {
                    const int lane_id = threadIdx.x % WARP_SIZE;

                    const int64_t A_i = __shfl(A_cur_index, t) + 32 * lane_id;
                    const int64_t B_i = __shfl(B_cur_index, t) + 32 * lane_id;

                    {
                        //ulonglong4 a_bit256 = *((ulonglong4 *)(A + A_i));    // weights
                        ulonglong4 a_bit256 = *((ulonglong4 *)(A_s + A_i));    // weights
                        ulonglong4 b_bit256 = *((ulonglong4 *)(B + B_i));    // input
                        c_bit256 = xnor_int256(a_bit256, b_bit256);
                        int tmp_count = __popcll(c_bit256.w) + __popcll(c_bit256.x) +
                            __popcll(c_bit256.y) + __popcll(c_bit256.z);

                        int sum_count = warpAllReduceSum(tmp_count);
                        if (lane_id == t) count += sum_count;
                    }
                }
            }
#endif

            //#ifdef NOT_USED
            // 32 thread X 64 bit = 2048 bit
            for (; k < (K - 2048); k += 2048) {   // l.size*l.size*l.c - one filter size [27 - 9216]
                uint64_t c_bit64;

                //int64_t A_cur_index = (i*lda + k) / 8;
                int64_t A_cur_index = (local_i*lda + k) / 8;
                int64_t B_cur_index = (j*ldb + k) / 8;
                if (i >= M) A_cur_index = 0;

                #pragma unroll
                for (int t = 0; t < WARP_SIZE; ++t) {
                    const int lane_id = threadIdx.x % WARP_SIZE;

                    const int64_t A_i = __shfl(A_cur_index, t) + 8 * lane_id;
                    const int64_t B_i = __shfl(B_cur_index, t) + 8 * lane_id;

                    {
                        //uint64_t a_bit64 = *((uint64_t *)(A + A_i));    // weights
                        uint64_t a_bit64 = *((uint64_t *)(A_s + A_i));    // weights
                        uint64_t b_bit64 = *((uint64_t *)(B + B_i));    // input
                        c_bit64 = xnor_int64(a_bit64, b_bit64);
                        int tmp_count = __popcll(c_bit64);

                        int sum_count = warpAllReduceSum(tmp_count);
                        if (lane_id == t) count += sum_count;
                    }
                }
            }
            //#endif

            //#ifdef NOT_USED
            // 32 thread X 32 bit = 1024 bit
            for (; k < (K - 1024); k += 1024) {   // l.size*l.size*l.c - one filter size [27 - 9216]

                                                  //int64_t A_cur_index = (i*lda + k) / 8;
                int64_t A_cur_index = (local_i*lda + k) / 8;
                int64_t B_cur_index = (j*ldb + k) / 8;
                if (i >= M) A_cur_index = 0;

                #pragma unroll
                for (int t = 0; t < WARP_SIZE; ++t) {
                    const int lane_id = threadIdx.x % WARP_SIZE;

                    const int64_t A_i = __shfl(A_cur_index, t) + 4 * lane_id;
                    const int64_t B_i = __shfl(B_cur_index, t) + 4 * lane_id;

                    {
                        //uint64_t a_bit64 = *((uint64_t *)(A + A_i));    // weights
                        uint32_t a_bit32 = *((uint32_t *)(A_s + A_i));    // weights
                        uint32_t b_bit32 = *((uint32_t *)(B + B_i));    // input
                        uint32_t c_bit32 = xnor_int32(a_bit32, b_bit32);
                        int tmp_count = __popc(c_bit32);

                        int sum_count = warpAllReduceSum(tmp_count);
                        if (lane_id == t) count += sum_count;
                    }
                }
            }
            //#endif

            if (i < M)
            {
                float mean_val = mean_arr[i];
                float bias_val = bias_arr[i];

                //#ifdef NOT_USED
                for (; k < K; k += 256) {   // l.size*l.size*l.c - one filter size [27 - 144 - 9216]
                                            //ulonglong4 a_bit256 = *((ulonglong4 *)(A + (i*lda + k) / 8));    // weights
                    ulonglong4 a_bit256 = *((ulonglong4 *)(A_s + (local_i*lda + k) / 8));    // weights
                    ulonglong4 b_bit256 = *((ulonglong4 *)(B + (j*ldb + k) / 8));    // input
                    ulonglong4 c_bit256 = xnor_int256(a_bit256, b_bit256);

                    count += __popcll(c_bit256.w) + __popcll(c_bit256.x) +
                        __popcll(c_bit256.y) + __popcll(c_bit256.z);
                }
                //#endif

#ifdef NOT_USED
                for (; k < K; k += 64) {   // l.size*l.size*l.c - one filter size [27 - 9216]
                                           //uint64_t a_bit64 = *((uint64_t *)(A + (i*lda + k) / 8));    // weights
                    uint64_t a_bit64 = *((uint64_t *)(A_s + (local_i*lda + k) / 8));    // weights
                    uint64_t b_bit64 = *((uint64_t *)(B + (j*ldb + k) / 8));            // input
                    uint64_t c_bit64 = xnor_int64(a_bit64, b_bit64);

                    count += __popcll(c_bit64);
                }
#endif

                const int bit_step = 256;
                int f1 = (K % bit_step == 0) ? 0 : (bit_step - (K % bit_step));
                count = count - f1;    // remove extra bits (from empty space for align only)

                C[i*ldc + j] = (2 * count - K) *mean_val + bias_val;
            }
        }
    }
}


/*
// Coalescing
// B (input) in the shared_memory - GOOD
__global__ void gemm_nn_custom_bin_mean_transposed_gpu_kernel(int M, int N, int K,
unsigned char *A, int lda,
unsigned char *B, int ldb,
float *C, int ldc, float *mean_arr, float *bias_arr)
{
int index = blockIdx.x*blockDim.x + threadIdx.x;

__shared__ uint8_t B_s[4096*8];  // 32 KB // [ldb x N`] // max = 262 144 bits
//__shared__ uint64_t B_s[4096];  // 32 KB // [ldb x N`] // max = 262 144 bits

int start_j = blockIdx.x*blockDim.x / M;
int end_j = (blockIdx.x*blockDim.x + blockDim.x) / M + 1;

size_t shared_size = ldb * (end_j - start_j);

int j_cur = index / M;
int local_j = j_cur - start_j;

for (int k = threadIdx.x * 256; k < shared_size; k += blockDim.x * 256) {
int x = start_j*ldb + k;
if (x < (N*ldb)) *((ulonglong4 *)(B_s + k / 8)) = *((ulonglong4 *)(B + x / 8));
}
__syncthreads();

int i, j, k;

i = index % M;   // l.n - filters [16 - 55 - 1024]
{
j = index / M;  // out_h*out_w - one channel output size [169 - 173056]
if (j < N)
{
int count = 0;
k = 0;

//#ifdef NOT_USED
// 32 thread X 64 bit = 2048 bit
for (; k < (K - 2048); k += 2048) {   // l.size*l.size*l.c - one filter size [27 - 9216]
uint64_t c_bit64;

int64_t A_cur_index = (i*lda + k) / 8;
//int64_t B_cur_index = (j*ldb + k) / 8;
int64_t B_cur_index = (local_j*ldb + k) / 8;
if (i >= M) A_cur_index = 0;

#pragma unroll
for (int t = 0; t < WARP_SIZE; ++t) {
const int lane_id = threadIdx.x % WARP_SIZE;

const int64_t A_i = __shfl(A_cur_index, t) + 8 * lane_id;
const int64_t B_i = __shfl(B_cur_index, t) + 8 * lane_id;

{
uint64_t a_bit64 = *((uint64_t *)(A + A_i));    // weights
//uint64_t b_bit64 = *((uint64_t *)(B + B_i));    // input
uint64_t b_bit64 = *((uint64_t *)(B_s + B_i));    // input
c_bit64 = xnor_int64(a_bit64, b_bit64);
int tmp_count = __popcll(c_bit64);

int sum_count = warpAllReduceSum(tmp_count);
if (lane_id == t) count += sum_count;
}
}
}
//#endif

//#ifdef NOT_USED
// 32 thread X 32 bit = 1024 bit
for (; k < (K - 1024); k += 1024) {   // l.size*l.size*l.c - one filter size [27 - 9216]

int64_t A_cur_index = (i*lda + k) / 8;
//int64_t B_cur_index = (j*ldb + k) / 8;
int64_t B_cur_index = (local_j*ldb + k) / 8;
if (i >= M) A_cur_index = 0;

#pragma unroll
for (int t = 0; t < WARP_SIZE; ++t) {
const int lane_id = threadIdx.x % WARP_SIZE;

const int64_t A_i = __shfl(A_cur_index, t) + 4 * lane_id;
const int64_t B_i = __shfl(B_cur_index, t) + 4 * lane_id;

{
uint32_t a_bit32 = *((uint32_t *)(A + A_i));    // weights
//uint32_t b_bit32 = *((uint32_t *)(B + B_i));    // input
uint32_t b_bit32 = *((uint32_t *)(B_s + B_i));    // input
uint32_t c_bit32 = xnor_int32(a_bit32, b_bit32);
int tmp_count = __popc(c_bit32);

int sum_count = warpAllReduceSum(tmp_count);
if (lane_id == t) count += sum_count;
}
}
}
//#endif

if (i < M)
{
float mean_val = mean_arr[i];
float bias_val = bias_arr[i];

//#ifdef NOT_USED
for (; k < K; k += 256) {   // l.size*l.size*l.c - one filter size [27 - 144 - 9216]
ulonglong4 a_bit256 = *((ulonglong4 *)(A + (i*lda + k) / 8));    // weights
//ulonglong4 b_bit256 = *((ulonglong4 *)(B + (j*ldb + k) / 8));    // input
ulonglong4 b_bit256 = *((ulonglong4 *)(B_s + (local_j*ldb + k) / 8));    // input
ulonglong4 c_bit256 = xnor_int256(a_bit256, b_bit256);

count += __popcll(c_bit256.w) + __popcll(c_bit256.x) +
__popcll(c_bit256.y) + __popcll(c_bit256.z);
}
//#endif

#ifdef NOT_USED
for (; k < K; k += 64) {   // l.size*l.size*l.c - one filter size [27 - 9216]
uint64_t a_bit64 = *((uint64_t *)(A + (i*lda + k) / 8));    // weights
//uint64_t b_bit64 = *((uint64_t *)(B + (j*ldb + k) / 8));            // input
uint64_t b_bit64 = *((uint64_t *)(B_s + (local_j*ldb + k) / 8));            // input
uint64_t c_bit64 = xnor_int64(a_bit64, b_bit64);

count += __popcll(c_bit64);
}
#endif

const int bit_step = 256;
int f1 = (K % bit_step == 0) ? 0 : (bit_step - (K % bit_step));
count = count - f1;    // remove extra bits (from empty space for align only)

C[i*ldc + j] = (2 * count - K) * mean_val + bias_val;
}
}
}
}
*/

// Coalesced memory access - GOOD
void gemm_nn_custom_bin_mean_transposed_gpu(int M, int N, int K,
    unsigned char *A, int lda,
    unsigned char *B, int ldb,
    float *C, int ldc, float *mean_arr, float *bias)
{
    size_t size = M*N;
    const int num_blocks = size / BLOCK + 1;

    /*
    printf("\n gemm_bin size = %d, num_blocks = %d, M*K = %d KB, N*K = %d KB \n (w) M*K/num_blocks = %d KB, (i) N*K/num_blocks = %d KB \n",
    size, num_blocks, M*K / 1024, N*K / 1024, M*lda / num_blocks / 1024, N*ldb / num_blocks / 1024);
    printf(" M / 512 = %d, N / 512 = %d, M*lda / 512 = %d, N*ldb / 512 = %d \n", M / 512, N / 512, M*lda/512, N*ldb/512);
    */
    //printf(" shared_memory: (w) lda*BLOCK/N = %d, (i) ldb*BLOCK/M = %d, \t lda = %d \n\n", lda*BLOCK / N, ldb*BLOCK / M, lda);

    gemm_nn_custom_bin_mean_transposed_gpu_kernel << <num_blocks, BLOCK, 0, 0 >> >(
        M, N, K,
        A, lda,
        B, ldb,
        C, ldc,
        mean_arr, bias);
}
// --------------------------------
// --------------------------------

// --------------------------------
// sequentially - B (input) in the shared_memory - BAD
// --------------------------------
__global__ void gemm_nn_custom_bin_mean_transposed_sequentially_gpu_kernel(int M, int N, int K,
    unsigned char *A, int lda,
    unsigned char *B, int ldb,
    float *C, int ldc, float *mean_arr)
{
    //__shared__ float mean_shared[32];
    //__shared__ uint32_t B_s[8192];  // 32 KB // [ldb x N`] // max = 262 144 bits
    //__shared__ uint32_t B_s[4096];  // 16 KB // [ldb x N`] // max = 131 072 bits
    __shared__ uint8_t B_s[4096 * 4];  // 16 KB // [ldb x N`] // max = 131 072 bits


    const int K_items = WARP_SIZE;
    int start_j = blockIdx.x*blockDim.x / (K_items * M);

    {
        int end_j = (blockIdx.x*blockDim.x + blockDim.x) / (K_items * M) + 1;
        if (end_j > N) end_j = N;
        size_t shared_size = ldb * (end_j - start_j);

        if (shared_size != 0) {
            //if(threadIdx.x == 0) printf(" start_j = %d, end_j = %d, shared_size = %d \n", start_j, end_j, shared_size);

            int k;
            for (int k = threadIdx.x * 32; k < shared_size; k += blockDim.x * 32) {
                int x = start_j*ldb + k;
                if (x < (N*ldb)) *((uint32_t *)(B_s + k / 8)) = *((uint32_t *)(B + x / 8));
            }
        }
    }
    __syncthreads();

    int index = blockIdx.x*blockDim.x + threadIdx.x;

    {
        int i;  // l.n
        int j;  // out_h*out_w
        int k;  // l.size * l.size * l.c

        const int index2 = index / K_items;
        i = index2 % M; // max M
        j = index2 / M; // max N
        int local_j = j - start_j;

        //if (i <= 1 && j <= 1 ) printf(" k = %d, K = %d, K_items = %d, i = %d, j = %d, lda = %d, ldb = %d, ldc = %d \n",
        //    k, K, K_items, i, j, lda, ldb, ldc);
        {   // l.n - filters [16 - 55 - 1024]
            // further improvements: for (l.n == 1024) iterate several (j)


            if (j < N)
            { // out_h*out_w - one channel output size [169 - 173056]

                int count = 0;


                const int bit_step = 32;
                for (k = (threadIdx.x % WARP_SIZE) * bit_step; k < K; k += bit_step*WARP_SIZE)
                {   // l.size*l.size*l.c - one filter size [27 - 144 - 9216]
                    uint32_t a_bit32 = *((uint32_t *)(A + (i*lda + k) / 8));    // weights
                                                                                //uint32_t b_bit32 = *((uint32_t *)(B + (j*ldb + k) / 8));    // input
                    uint32_t b_bit32 = *((uint32_t *)(B_s + (local_j*ldb + k) / 8));    // input
                    uint32_t c_bit32 = xnor_int32(a_bit32, b_bit32);

                    count += __popc(c_bit32);
                }

                for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
                    count += __shfl_down(count, offset);


                if (threadIdx.x % WARP_SIZE == 0) {
                    int f1 = (K % bit_step == 0) ? 0 : (bit_step - (K % bit_step));
                    count = count - f1;
                    float mean_val = mean_arr[i];
                    C[i*ldc + j] = (2 * count - K) * mean_val;
                    //B_s[threadIdx.x / WARP_SIZE] = (2 * count - K) * mean_val;
                }
            }
        }
    }
}

// sequentially - BAD
void gemm_nn_custom_bin_mean_transposed_sequentially_gpu(int M, int N, int K,
    unsigned char *A, int lda,
    unsigned char *B, int ldb,
    float *C, int ldc, float *mean_arr)
{
    //size_t size = M*N;
    size_t size = M*N * 32;

    const int num_blocks = size / BLOCK + 1;

    //printf(" K = %d \n", K);

    /*
    printf("\n gemm_bin size = %d, num_blocks = %d, M*K = %d KB, N*K = %d KB \n (w) M*K/num_blocks = %d KB, (i) N*K/num_blocks = %d KB \n",
    size, num_blocks, M*K / 1024, N*K / 1024, M*lda / num_blocks / 1024, N*ldb / num_blocks / 1024);
    printf(" M / 512 = %d, N / 512 = %d, M*lda / 512 = %d, N*ldb / 512 = %d \n", M / 512, N / 512, M*lda/512, N*ldb/512);
    */
    //printf(" shared_memory: (w) lda*BLOCK/N = %d, (i) ldb*BLOCK/M = %d, \t lda = %d \n\n", lda*BLOCK / N, ldb*BLOCK / M, lda);

    gemm_nn_custom_bin_mean_transposed_sequentially_gpu_kernel << <num_blocks, BLOCK, 0, 0 >> >(
        M, N, K,
        A, lda,
        B, ldb,
        C, ldc,
        mean_arr);
}
// --------------------------------
