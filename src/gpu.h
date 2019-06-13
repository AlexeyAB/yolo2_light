#pragma once
#ifndef GPU_H
#define GPU_H

#ifndef __DATE__
#define __DATE__
#endif

#ifndef __TIME__
#define __TIME__
#endif

#ifndef __FUNCTION__
#define __FUNCTION__
#endif

#ifndef __LINE__
#define __LINE__ 0
#endif

#ifndef __FILE__
#define __FILE__
#endif


#ifdef __cplusplus
extern "C" {
#endif


#ifdef GPU

    void check_error(cudaError_t status);
    void check_error_extended(cudaError_t status, const char *file, int line, const char *date_time);
#define CHECK_CUDA(X) check_error_extended(X, __FILE__ " : " __FUNCTION__, __LINE__,  __DATE__ " - " __TIME__ );

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

    void cudnn_check_error_extended(cudnnStatus_t status, const char *file, int line, const char *date_time);
#define CHECK_CUDNN(X) cudnn_check_error_extended(X, __FILE__ " : " __FUNCTION__, __LINE__,  __DATE__ " - " __TIME__ );
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

    void input_shortcut_gpu(float *in, int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float *out);
    // -------------------- Quantinization -------------------

    void cuda_convert_f32_to_int8(float* input_f32, size_t size, int8_t *output_int8, float multipler, int max_val);

    void cuda_convert_f32_to_int8_nomax(float* input_f32, size_t size, int8_t *output_int8, float multipler);

    void cuda_convert_int8_to_f32(int8_t* input_int8, size_t size, float *output_f32, float multipler);

    void cuda_do_multiply_f32(float *input_output, size_t size, float multipler);

    // -------------------- XNOR -------------------

    void swap_binary(convolutional_layer *l);

    void binarize_weights_gpu(float *weights, int n, int size, float *binary);

    void binarize_gpu(float *x, int n, float *binary);

    void repack_input_gpu_bin(float *input, uint32_t *re_packed_input_bin, int w, int h, int c);

    void transpose_uint32_gpu(uint32_t *src, uint32_t *dst, int src_h, int src_w, int src_align, int dst_align);

    void im2col_ongpu(float *im,
        int channels, int height, int width,
        int ksize, int stride, int pad, float *data_col);

    void im2col_align_ongpu(float *im,
        int channels, int height, int width,
        int ksize, int stride, int pad, float *data_col, int bit_align);

    void im2col_align_bin_ongpu(float *im,
        int channels, int height, int width,
        int ksize, int stride, int pad, float *data_col, int bit_align);

    void float_to_bit_gpu(float *src, unsigned char *dst, size_t size);

    void transpose_bin_gpu(unsigned char *A, unsigned char *B, const int n, const int m,
        const int lda, const int ldb, const int block_size);

    void fill_int8_gpu(unsigned char *src, unsigned char val, size_t size);

    void gemm_nn_custom_bin_mean_transposed_gpu(int M, int N, int K,
        unsigned char *A, int lda,
        unsigned char *B, int ldb,
        float *C, int ldc, float *mean_arr, float *bias, int leaky_activation,
        float *shortcut_in_gpu, float *shortcut_out_gpu);

#endif // GPU

#ifdef __cplusplus
}
#endif

#endif    // GPU_H