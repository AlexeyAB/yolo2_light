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

    // copy
    void copy_ongpu(int N, float * X, int INCX, float * Y, int INCY);

    // activations
    void activate_array_ongpu(float *x, int n, ACTIVATION a);

    // softmax layer
    void softmax_gpu(float *input, int n, int offset, int groups, float temp, float *output);

    // reorg layer
    void reorg_ongpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out);
    
    // upsample layer
    void upsample_gpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out);

    void shortcut_gpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float *out);
    // -------------------- Quantinization -------------------

    void cuda_convert_f32_to_int8(float* input_f32, size_t size, int8_t *output_int8, float multipler, int max_val);

    void cuda_convert_f32_to_int8_nomax(float* input_f32, size_t size, int8_t *output_int8, float multipler);

    void cuda_convert_int8_to_f32(int8_t* input_int8, size_t size, float *output_f32, float multipler);

    void cuda_do_multiply_f32(float *input_output, size_t size, float multipler);

#endif // GPU

#ifdef __cplusplus
}
#endif

#endif    // GPU_H