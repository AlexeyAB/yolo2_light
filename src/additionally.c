#include "additionally.h"
#include "gpu.h"

#ifdef OPENCL
#include "ocl.h"
#endif

#ifdef CUDNN
#pragma comment(lib, "cudnn.lib")
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// global GPU index: cuda.c
int gpu_index = 0;

// im2col.c
float im2col_get_pixel(float *im, int height, int width, int channels,
    int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

// im2col.c
//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(float* data_im,
    int channels, int height, int width,
    int ksize, int stride, int pad, float* data_col)
{
    int c, h, w;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                    im_row, im_col, c_im, pad);
            }
        }
    }
}

// -------------- my own --------------

// fuse convolutional and batch_norm weights into one convolutional-layer
void yolov2_fuse_conv_batchnorm(network net)
{
    int j;
    for (j = 0; j < net.n; ++j) {
        layer *l = &net.layers[j];

        if (l->type == CONVOLUTIONAL) {
            printf(" Fuse Convolutional layer \t\t l->size = %d  \n", l->size);

            if (l->batch_normalize) {
                int f;
                for (f = 0; f < l->n; ++f)
                {
                    l->biases[f] = l->biases[f] - l->scales[f] * l->rolling_mean[f] / (sqrtf(l->rolling_variance[f]) + .000001f);

                    const size_t filter_size = l->size*l->size*l->c;
                    int i;
                    for (i = 0; i < filter_size; ++i) {
                        int w_index = f*filter_size + i;

                        l->weights[w_index] = l->weights[w_index] * l->scales[f] / (sqrtf(l->rolling_variance[f]) + .000001f);
                    }
                }

                l->batch_normalize = 0;
#ifdef GPU
                if (gpu_index >= 0) {
                    push_convolutional_layer(*l);
                }
#endif

#ifdef OPENCL
                //if (gpu_index >= 0) {
                ocl_push_convolutional_layer(*l);
                //}
#endif
            }
        }
        else {
            printf(" Skip layer: %d \n", l->type);
        }
    }
}

// -------------- XNOR-net ------------

void binarize_weights(float *weights, int n, int size, float *binary)
{
    int i, f;
    for (f = 0; f < n; ++f) {
        float mean = 0;
        for (i = 0; i < size; ++i) {
            mean += fabs(weights[f*size + i]);
        }
        mean = mean / size;
        for (i = 0; i < size; ++i) {
            binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        }
    }
}

void binarize_cpu(float *input, int n, float *binary)
{
    int i;
    for (i = 0; i < n; ++i) {
        binary[i] = (input[i] > 0) ? 1 : -1;
    }
}

static inline unsigned char xnor(unsigned char a, unsigned char b) {
    //return a == b;
    return !(a^b);
}

// INT-32
static inline uint32_t get_bit_int32(uint32_t const*const src, size_t index) {
    size_t src_i = index / 32;
    int src_shift = index % 32;
    unsigned char val = (src[src_i] & (1 << src_shift)) > 0;
    return val;
}

static inline uint32_t xnor_int32(uint32_t a, uint32_t b) {
    return ~(a^b);
}

static inline uint64_t xnor_int64(uint64_t a, uint64_t b) {
    return ~(a^b);
}


static inline uint32_t fill_bit_int32(char src) {
    if (src == 0) return 0x00000000;
    else return  0xFFFFFFFF;
}

static inline uint64_t fill_bit_int64(char src) {
    if (src == 0) return 0x0000000000000000;
    else return  0xFFFFFFFFFFFFFFFF;
}

void binary_int32_printf(uint32_t src) {
    int i;
    for (i = 0; i < 32; ++i) {
        if (src & 1) printf("1");
        else printf("0");
        src = src >> 1;
    }
    printf("\n");
}

void binary_int64_printf(uint64_t src) {
    int i;
    for (i = 0; i < 64; ++i) {
        if (src & 1) printf("1");
        else printf("0");
        src = src >> 1;
    }
    printf("\n");
}

void get_mean_array(float *src, size_t size, size_t filters, float *mean_arr) {
    size_t i, counter;
    counter = 0;
    for (i = 0; i < size; i += size / filters) {
        mean_arr[counter++] = fabs(src[i]);
    }
}

void binary_align_weights(convolutional_layer *l)
{
    int m = l->n;
    int k = l->size*l->size*l->c;
    size_t new_lda = k + (l->lda_align - k % l->lda_align); // (k / 8 + 1) * 8;
    l->new_lda = new_lda;

    binarize_weights(l->weights, m, k, l->binary_weights);

    size_t align_weights_size = new_lda * m;
    l->align_bit_weights_size = align_weights_size / 8;// +1;
    float *align_weights = calloc(align_weights_size, sizeof(float));
    l->align_bit_weights = calloc(l->align_bit_weights_size, sizeof(char));

    size_t i, j;
    // align A without transpose
    for (i = 0; i < m; ++i) {
        for (j = 0; j < k; ++j) {
            align_weights[i*new_lda + j] = l->binary_weights[i*k + j];
        }
    }
    float_to_bit(align_weights, l->align_bit_weights, align_weights_size);

    l->mean_arr = calloc(l->n, sizeof(float));
    get_mean_array(align_weights, align_weights_size, l->n, l->mean_arr);

#ifdef GPU
    cudaError_t status;
    l->align_workspace_size = l->bit_align * l->size * l->size * l->c;
    status = cudaMalloc((void **)&l->align_workspace_gpu, l->align_workspace_size * sizeof(float));
    status = cudaMalloc((void **)&l->transposed_align_workspace_gpu, l->align_workspace_size * sizeof(float));
    check_error(status);

    //l->align_bit_weights_gpu = cuda_make_array(l->align_bit_weights, l->align_bit_weights_size * sizeof(char)/sizeof(float));
    status = cudaMalloc((void **)&l->align_bit_weights_gpu, l->align_bit_weights_size);
    check_error(status);
    status = cudaMemcpy(l->align_bit_weights_gpu, l->align_bit_weights, l->align_bit_weights_size, cudaMemcpyHostToDevice);
    check_error(status);
    status = cudaMemcpy(l->binary_weights_gpu, l->binary_weights, m*k * sizeof(float), cudaMemcpyHostToDevice);
    check_error(status);

    l->mean_arr_gpu = cuda_make_array(l->mean_arr, l->n);
    cudaDeviceSynchronize();
#endif // GPU

    free(align_weights);
}

void calculate_binary_weights(network net)
{
    int j;
    for (j = 0; j < net.n; ++j) {
        layer *l = &net.layers[j];

        if (l->type == CONVOLUTIONAL) {
            //printf(" Merges Convolutional-%d and batch_norm \n", j);

            if (l->xnor) {
                //printf("\n %d \n", j);
                l->lda_align = 256; // 256bit for AVX2

                binary_align_weights(l);

                if (net.layers[j].use_bin_output) {
                    l->activation = LINEAR;
                }
            }
        }
    }
    //printf("\n calculate_binary_weights Done! \n");

}

static inline void set_bit(unsigned char *const dst, size_t index) {
    size_t dst_i = index / 8;
    int dst_shift = index % 8;
    dst[dst_i] |= 1 << dst_shift;
}

static inline unsigned char get_bit(unsigned char const*const src, size_t index) {
    size_t src_i = index / 8;
    int src_shift = index % 8;
    unsigned char val = (src[src_i] & (1 << src_shift)) > 0;
    return val;
}

/*
static inline unsigned char reverse_byte_1(char a)
{
    return ((a & 0x1) << 7) | ((a & 0x2) << 5) |
        ((a & 0x4) << 3) | ((a & 0x8) << 1) |
        ((a & 0x10) >> 1) | ((a & 0x20) >> 3) |
        ((a & 0x40) >> 5) | ((a & 0x80) >> 7);
}

static inline unsigned char reverse_byte(unsigned char a)
{
    return ((a * 0x0802LU & 0x22110LU) | (a * 0x8020LU & 0x88440LU)) * 0x10101LU >> 16;
}

static unsigned char lookup[16] = {
    0x0, 0x8, 0x4, 0xc, 0x2, 0xa, 0x6, 0xe,
    0x1, 0x9, 0x5, 0xd, 0x3, 0xb, 0x7, 0xf, };

static inline unsigned char reverse_byte_3(unsigned char n) {
    // Reverse the top and bottom nibble then swap them.
    return (lookup[n & 0b1111] << 4) | lookup[n >> 4];
}


static inline void transpose8rS32_reversed_diagonale(unsigned char* A, int m, int n, unsigned char* B)
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

    B[7 * n] = reverse_byte(x >> 24);  B[6 * n] = reverse_byte(x >> 16);  B[5 * n] = reverse_byte(x >> 8);  B[4 * n] = reverse_byte(x);
    B[3 * n] = reverse_byte(y >> 24);  B[2 * n] = reverse_byte(y >> 16);  B[1 * n] = reverse_byte(y >> 8);  B[0 * n] = reverse_byte(y);
}

void transpose_bin(char *A, char *B, const int n, const int m,
    const int lda, const int ldb, const int block_size)
{
    int i;
    #pragma omp parallel for
    for (i = 0; i < n; i += 8) {
        int j;
        for (j = 0; j < m - 8; j += 8) {
            int a_index = i*lda + j;
            int b_index = j*ldb + i;
            //transpose_8x8_bits_my(&A[a_index/8], &B[b_index/8], lda/8, ldb/8);
            transpose8rS32_reversed_diagonale(&A[a_index / 8], lda / 8, ldb / 8, &B[b_index / 8]);
        }
        for (; j < m; ++j) {
            if (get_bit(A, i*lda + j)) set_bit(B, j*ldb + i);
        }
    }
}
*/

uint8_t reverse_8_bit(uint8_t a) {
    return ((a * 0x0802LU & 0x22110LU) | (a * 0x8020LU & 0x88440LU)) * 0x10101LU >> 16;
}

uint32_t reverse_32_bit(uint32_t a)
{
    // unsigned int __rbit(unsigned int val) // for ARM    //__asm__("rbit %0, %1\n" : "=r"(output) : "r"(input));
    return (reverse_8_bit(a >> 24) << 0) |
        (reverse_8_bit(a >> 16) << 8) |
        (reverse_8_bit(a >> 8) << 16) |
        (reverse_8_bit(a >> 0) << 24);
}

#define swap(a0, a1, j, m) t = (a0 ^ (a1 >>j)) & m; a0 = a0 ^ t; a1 = a1 ^ (t << j);

void transpose32_optimized(uint32_t A[32]) {
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

void transpose_32x32_bits_reversed_diagonale(uint32_t *A, uint32_t *B, int m, int n)
{
    unsigned A_tmp[32];
    int i;
    #pragma unroll
    for (i = 0; i < 32; ++i) A_tmp[i] = A[i * m];
    transpose32_optimized(A_tmp);
    #pragma unroll
    for (i = 0; i < 32; ++i) B[i*n] = A_tmp[i];
}

// transpose by 32-bit
void transpose_bin(uint32_t *A, uint32_t *B, const int n, const int m,
    const int lda, const int ldb, const int block_size)
{
    int i;
    #pragma omp parallel for
    for (i = 0; i < n; i += 32) {
        int j;
        for (j = 0; j < m; j += 32) {
            int a_index = i*lda + j;
            int b_index = j*ldb + i;
            transpose_32x32_bits_reversed_diagonale(&A[a_index / 32], &B[b_index / 32], lda / 32, ldb / 32);
            //transpose_32x32_bits_my(&A[a_index/32], &B[b_index/32], lda/32, ldb/32);
        }
        for (; j < m; ++j) {
            if (get_bit(A, i*lda + j)) set_bit(B, j*ldb + i);
        }
    }
}

// -------------- blas.c --------------


#ifdef AVX

#ifdef _WIN64
// Windows
#include <intrin.h>
#else
// Linux
#include <x86intrin.h>
#endif

#include <ammintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <emmintrin.h>
// https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=broad&expand=561

// https://software.intel.com/sites/landingpage/IntrinsicsGuide
void gemm_nn(int M, int N, int K, float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float *C, int ldc)
{
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            float A_PART = ALPHA*A[i*lda + k];
            __m256 a256, b256, c256, result256;    // AVX
            a256 = _mm256_set1_ps(A_PART);
            for (j = 0; j < N - 8; j += 8) {
                b256 = _mm256_loadu_ps(&B[k*ldb + j]);
                c256 = _mm256_loadu_ps(&C[i*ldc + j]);
                // FMA - Intel Haswell (2013), AMD Piledriver (2012)
                //result256 = _mm256_fmadd_ps(a256, b256, c256);
                result256 = _mm256_mul_ps(a256, b256);
                result256 = _mm256_add_ps(result256, c256);
                _mm256_storeu_ps(&C[i*ldc + j], result256);
            }

            int prev_end = (N % 8 == 0) ? (N - 8) : (N / 8) * 8;
            for (j = prev_end; j < N; ++j)
                C[i*ldc + j] += A_PART*B[k*ldb + j];
        }
    }
}


#if defined(_MSC_VER) && _MSC_VER <= 1900
static inline __int32 _mm256_extract_epi64(__m256i a, const int index) {
    return a.m256i_i64[index];
}

static inline __int32 _mm256_extract_epi32(__m256i a, const int index) {
    return a.m256i_i32[index];
}
#endif

static inline float _castu32_f32(uint32_t a) {
    return *((float *)&a);
}

#if defined(_MSC_VER)
// Windows
static inline float _mm256_extract_float32(__m256 a, const int index) {
    return a.m256_f32[index];
}
#else
// Linux
static inline float _mm256_extract_float32(__m256 a, const int index) {
    return _castu32_f32(_mm256_extract_epi32(_mm256_castps_si256(a), index));
}
#endif

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu_custom(float* data_im,
    int channels, int height, int width,
    int ksize, int stride, int pad, float* data_col)
{

    int c;
    const int height_col = (height + 2 * pad - ksize) / stride + 1;
    const int width_col = (width + 2 * pad - ksize) / stride + 1;
    const int channels_col = channels * ksize * ksize;

    // optimized version
    if (height_col == height && width_col == width && stride == 1 && pad == 1)// && is_fma_avx())
    {
        #pragma omp parallel for
        for (c = 0; c < channels_col; ++c) {
            int h, w;
            int w_offset = c % ksize;
            int h_offset = (c / ksize) % ksize;
            int c_im = c / ksize / ksize;
            for (h = pad; h < height_col - pad; ++h) {
                for (w = pad; w < width_col - pad - 8; w += 8) {
                    int im_row = h_offset + h - pad;
                    int im_col = w_offset + w - pad;
                    int col_index = (c * height_col + h) * width_col + w;

                    //data_col[col_index] = data_im[im_col + width*(im_row + height*c_im)];
                    __m256 src256 = _mm256_loadu_ps((float *)(&data_im[im_col + width*(im_row + height*c_im)]));
                    _mm256_storeu_ps(&data_col[col_index], src256);
                }

                for (; w < width_col - pad; ++w) {
                    int im_row = h_offset + h - pad;
                    int im_col = w_offset + w - pad;
                    int col_index = (c * height_col + h) * width_col + w;

                    data_col[col_index] = data_im[im_col + width*(im_row + height*c_im)];
                }
            }

            {
                w = 0;
                for (h = 0; h < height_col; ++h) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    int col_index = (c * height_col + h) * width_col + w;
                    data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
                }
            }

            {
                w = width_col - 1;
                for (h = 0; h < height_col; ++h) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    int col_index = (c * height_col + h) * width_col + w;
                    data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
                }
            }

            {
                h = 0;
                for (w = 0; w < width_col; ++w) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    int col_index = (c * height_col + h) * width_col + w;
                    data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
                }
            }

            {
                h = height_col - 1;
                for (w = 0; w < width_col; ++w) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    int col_index = (c * height_col + h) * width_col + w;
                    data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
                }
            }
        }

    }
    else {
        //printf("\n Error: is no non-optimized version \n");
        im2col_cpu(data_im, channels, height, width, ksize, stride, pad, data_col);
    }
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu_custom_bin(float* data_im,
    int channels, int height, int width,
    int ksize, int stride, int pad, float* data_col, int bit_align)
{
    int c;
    const int height_col = (height + 2 * pad - ksize) / stride + 1;
    const int width_col = (width + 2 * pad - ksize) / stride + 1;
    const int channels_col = channels * ksize * ksize;

    // optimized version
    if (height_col == height && width_col == width && stride == 1 && pad == 1)
    {
        //__m256i all256_sing1 = _mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000);
        __m256 float_zero256 = _mm256_set1_ps(0.00);

        int new_ldb = bit_align;

        #pragma omp parallel for
        for (c = 0; c < channels_col; ++c) {
            int h, w;
            int w_offset = c % ksize;
            int h_offset = (c / ksize) % ksize;
            int c_im = c / ksize / ksize;
            for (h = pad; h < height_col - pad; ++h) {
                for (w = pad; w < width_col - pad - 8; w += 8) {
                    int im_row = h_offset + h - pad;
                    int im_col = w_offset + w - pad;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    //__m256i src256 = _mm256_loadu_si256((__m256i *)(&data_im[im_col + width*(im_row + height*c_im)]));
                    //__m256i result256 = _mm256_and_si256(src256, all256_sing1); // check sign in 8 x 32-bit floats
                    //uint16_t mask = _mm256_movemask_ps(_mm256_castsi256_ps(result256)); // (val >= 0) ? 0 : 1
                    //mask = ~mask;   // inverse mask,  (val >= 0) ? 1 : 0

                    __m256 src256 = _mm256_loadu_ps((float *)(&data_im[im_col + width*(im_row + height*c_im)]));
                    __m256 result256 = _mm256_cmp_ps(src256, float_zero256, _CMP_GT_OS);
                    uint16_t mask = _mm256_movemask_ps(result256); // (val > 0) ? 0 : 1

                    uint16_t *dst_ptr = &((unsigned char*)data_col)[col_index / 8];
                    *dst_ptr |= (mask << (col_index % 8));
                }

                for (; w < width_col - pad; ++w) {
                    int im_row = h_offset + h - pad;
                    int im_col = w_offset + w - pad;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    //data_col[col_index] = data_im[im_col + width*(im_row + height*c_im)];
                    float val = data_im[im_col + width*(im_row + height*c_im)];
                    if (val > 0) set_bit(data_col, col_index);
                }
            }

            {
                w = 0;
                for (h = 0; h < height_col; ++h) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    //data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    float val = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    if (val > 0) set_bit(data_col, col_index);
                }
            }

            {
                w = width_col - 1;
                for (h = 0; h < height_col; ++h) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    //data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    float val = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    if (val > 0) set_bit(data_col, col_index);
                }
            }

            {
                h = 0;
                for (w = 0; w < width_col; ++w) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    //data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    float val = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    if (val > 0) set_bit(data_col, col_index);
                }
            }

            {
                h = height_col - 1;
                for (w = 0; w < width_col; ++w) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    //data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    float val = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    if (val > 0) set_bit(data_col, col_index);
                }
            }
        }

    }
    else {
        printf("\n Error: is no non-optimized version \n");
        //im2col_cpu(data_im, channels, height, width, ksize, stride, pad, data_col); // must be aligned for transpose after float_to_bin
        // float_to_bit(b, t_input, src_size);
        // transpose_bin(t_input, *t_bit_input, k, n, bit_align, new_ldb, 8);
    }
}

void activate_array_cpu_custom(float *x, const int n, const ACTIVATION a)
{
    int i = 0;
    if (a == LINEAR)
    {
    }
    else if (a == LEAKY)
    {
        {
            __m256i all256_sing1 = _mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000);
            __m256 all256_01 = _mm256_set1_ps(0.1F);

            for (i = 0; i < n - 8; i += 8) {
                //x[i] = (x[i]>0) ? x[i] : .1*x[i];

                __m256 src256 = _mm256_loadu_ps(&x[i]);
                __m256 mult256 = _mm256_mul_ps((src256), all256_01); // mult * 0.1

                __m256i sign256 = _mm256_and_si256(_mm256_castps_si256(src256), all256_sing1); // check sign in 8 x 32-bit floats

                __m256 result256 = _mm256_blendv_ps(src256, mult256, _mm256_castsi256_ps(sign256)); // (sign>0) ? src : mult;
                _mm256_storeu_ps(&x[i], result256);
            }
        }

        for (; i < n; ++i) {
            x[i] = (x[i]>0) ? x[i] : .1*x[i];
        }
    }
    else {
        for (i = 0; i < n; ++i) {
            x[i] = activate(x[i], a);
        }
    }
}


void forward_maxpool_layer_avx(float *src, float *dst, int *indexes, int size, int w, int h, int out_w, int out_h, int c,
    int pad, int stride, int batch)
{

    const int w_offset = -pad / 2;
    const int h_offset = -pad / 2;
    int b, k;

    for (b = 0; b < batch; ++b) {
        #pragma omp parallel for
        for (k = 0; k < c; ++k) {
            int i, j, m, n;
            for (i = 0; i < out_h; ++i) {
                //for (j = 0; j < out_w; ++j) {
                j = 0;

                if (stride == 1) {
                    for (j = 0; j < out_w - 8 - (size - 1); j += 8) {
                        int out_index = j + out_w*(i + out_h*(k + c*b));
                        __m256 max256 = _mm256_set1_ps(-FLT_MAX);
                        for (n = 0; n < size; ++n) {
                            for (m = 0; m < size; ++m) {
                                int cur_h = h_offset + i*stride + n;
                                int cur_w = w_offset + j*stride + m;
                                int index = cur_w + w*(cur_h + h*(k + b*c));
                                int valid = (cur_h >= 0 && cur_h < h &&
                                    cur_w >= 0 && cur_w < w);
                                if (!valid) continue;

                                __m256 src256 = _mm256_loadu_ps(&src[index]);
                                max256 = _mm256_max_ps(src256, max256);
                            }
                        }
                        _mm256_storeu_ps(&dst[out_index], max256);

                    }
                }
                else if (size == 2 && stride == 2) {
                    for (j = 0; j < out_w - 4; j += 4) {
                        int out_index = j + out_w*(i + out_h*(k + c*b));
                        float max = -FLT_MAX;
                        int max_i = -1;
                        __m128 max128 = _mm_set1_ps(-FLT_MAX);

                        for (n = 0; n < size; ++n) {
                            //for (m = 0; m < size; ++m)
                            m = 0;
                            {
                                int cur_h = h_offset + i*stride + n;
                                int cur_w = w_offset + j*stride + m;
                                int index = cur_w + w*(cur_h + h*(k + b*c));
                                int valid = (cur_h >= 0 && cur_h < h &&
                                    cur_w >= 0 && cur_w < w);
                                if (!valid) continue;

                                __m256 src256 = _mm256_loadu_ps(&src[index]);
                                __m256 src256_2 = _mm256_permute_ps(src256, (1 << 0) | (3 << 4));
                                __m256 max256 = _mm256_max_ps(src256, src256_2);

                                __m128 src128_0 = _mm256_extractf128_ps(max256, 0);
                                __m128 src128_1 = _mm256_extractf128_ps(max256, 1);
                                __m128 src128 = _mm_shuffle_ps(src128_0, src128_1, (2 << 2) | (2 << 6));

                                max128 = _mm_max_ps(src128, max128);
                            }
                        }
                        _mm_storeu_ps(&dst[out_index], max128);
                    }
                }

                for (; j < out_w; ++j) {
                    int out_index = j + out_w*(i + out_h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for (n = 0; n < size; ++n) {
                        for (m = 0; m < size; ++m) {
                            int cur_h = h_offset + i*stride + n;
                            int cur_w = w_offset + j*stride + m;
                            int index = cur_w + w*(cur_h + h*(k + b*c));
                            int valid = (cur_h >= 0 && cur_h < h &&
                                cur_w >= 0 && cur_w < w);
                            float val = (valid != 0) ? src[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max = (val > max) ? val : max;
                        }
                    }
                    dst[out_index] = max;
                    indexes[out_index] = max_i;
                }
            }
        }
    }
}


// http://graphics.stanford.edu/~seander/bithacks.html
// https://stackoverflow.com/questions/17354971/fast-counting-the-number-of-set-bits-in-m128i-register
// https://arxiv.org/pdf/1611.07612.pdf

static inline int popcnt128(__m128i n) {
    const __m128i n_hi = _mm_unpackhi_epi64(n, n);
#ifdef _MSC_VER
    return __popcnt64(_mm_cvtsi128_si64(n)) + __popcnt64(_mm_cvtsi128_si64(n_hi));
#else
    return __popcntq(_mm_cvtsi128_si64(n)) + __popcntq(_mm_cvtsi128_si64(n_hi));
#endif
}

static inline int popcnt256(__m256i n) {
    return popcnt128(_mm256_extractf128_si256(n, 0)) + popcnt128(_mm256_extractf128_si256(n, 1));
}

static inline __m256i count256(__m256i v) {
    __m256i lookup =
        _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2,
            2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3,
            1, 2, 2, 3, 2, 3, 3, 4);

    __m256i low_mask = _mm256_set1_epi8(0x0f);

    __m256i lo = _mm256_and_si256(v, low_mask);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi32(v, 4), low_mask);
    __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo);
    __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi);
    __m256i total = _mm256_add_epi8(popcnt1, popcnt2);

    return _mm256_sad_epu8(total, _mm256_setzero_si256());
}

static inline int popcnt256_custom(__m256i n) {
    __m256i val = count256(n);

    //return val.m256i_i64[0] +
    //val.m256i_i64[1] +
    //val.m256i_i64[2] +
    //val.m256i_i64[3];
    return _mm256_extract_epi64(val, 0)
        + _mm256_extract_epi64(val, 1)
        + _mm256_extract_epi64(val, 2)
        + _mm256_extract_epi64(val, 3);
}

// 5x times faster than gemm()-float32
// further optimizations: do mean-mult only for the last layer
void gemm_nn_custom_bin_mean_transposed(int M, int N, int K, float ALPHA_UNUSED,
    unsigned char *A, int lda,
    unsigned char *B, int ldb,
    float *C, int ldc, float *mean_arr)
{

#if defined(_OPENMP)
    static int max_num_threads = 0;
    if (max_num_threads == 0) {
        max_num_threads = omp_get_max_threads();
        //omp_set_num_threads(max_num_threads / 2);
    }
#endif

    int i;
    #pragma omp parallel for
    for (i = 0; i < M; ++i)
    {   // l.n - filters [16 - 55 - 1024]
        float mean_val = mean_arr[i];
        int j, k;
        __m256i all_1 = _mm256_set1_epi8(255);

        for (j = 0; j < N; ++j) { // out_h*out_w - one channel output size [169 - 173056]
            int count = 0;
            const int bit_step = 256;
            __m256i count_sum = _mm256_set1_epi8(0);

            for (k = 0; k < K; k += bit_step) {   // l.size*l.size*l.c - one filter size [27 - 9216]
                __m256i a_bit256 = _mm256_loadu_si256((__m256i *)(A + (i*lda + k) / 8));
                __m256i b_bit256 = _mm256_loadu_si256((__m256i *)(B + (j*ldb + k) / 8));
                __m256i xor256 = _mm256_xor_si256(a_bit256, b_bit256);  // xnor = not(xor(a,b))
                __m256i c_bit256 = _mm256_andnot_si256(xor256, all_1);  // can be optimized - we can do other NOT for wegihts once and do not do this NOT

                count_sum = _mm256_add_epi64(count256(c_bit256), count_sum);    //  Mula’s algorithm

                                                                                //count += popcnt256(c_bit256);

                                                                                //binary_int64_printf(c_bit64);
                                                                                //printf(", count = %d \n\n", tmp_count);
            }

            // count of 1 bits
            //count = count_sum.m256i_i64[0] +
            //    count_sum.m256i_i64[1] +
            //    count_sum.m256i_i64[2] +
            //   count_sum.m256i_i64[3];
            count = _mm256_extract_epi64(count_sum, 0)
                + _mm256_extract_epi64(count_sum, 1)
                + _mm256_extract_epi64(count_sum, 2)
                + _mm256_extract_epi64(count_sum, 3);

            int f1 = (K % bit_step == 0) ? 0 : (bit_step - (K % bit_step));
            count = count - f1;    // remove extra bits (from empty space for align only)

            C[i*ldc + j] = (2 * count - K) * mean_val;
        }
    }
}



void float_to_bit(float *src, unsigned char *dst, size_t size)
{
    size_t dst_size = size / 8 + 1;
    memset(dst, 0, dst_size);

    size_t i;
    //__m256i all256_sing1 = _mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000);
    __m256 float_zero256 = _mm256_set1_ps(0.0);

    for (i = 0; i < size; i += 8)
    {
        //__m256i src256 = _mm256_loadu_si256((__m256i *)(&src[i]));
        //__m256i result256 = _mm256_and_si256(src256, all256_sing1); // check sign in 8 x 32-bit floats
        //uint32_t mask = _mm256_movemask_ps(_mm256_castsi256_ps(result256)); // (val >= 0) ? 0 : 1
        ////mask = ~mask;   // inverse mask,  (val >= 0) ? 1 : 0

        __m256 src256 = _mm256_loadu_ps((float *)(&src[i]));
        __m256 result256 = _mm256_cmp_ps(src256, float_zero256, _CMP_GT_OS);
        uint32_t mask = _mm256_movemask_ps(result256); // (val > 0) ? 0 : 1

        dst[i / 8] = mask;
    }
}

#else // AVX

void gemm_nn(int M, int N, int K, float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float *C, int ldc)
{
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            register float A_PART = ALPHA*A[i*lda + k];
            for (j = 0; j < N; ++j) {
                C[i*ldc + j] += A_PART*B[k*ldb + j];
            }
        }
    }
}


//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu_custom(float* data_im,
    int channels, int height, int width,
    int ksize, int stride, int pad, float* data_col)
{
    im2col_cpu(data_im, channels, height, width, ksize, stride, pad, data_col);
}


//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu_custom_bin(float* data_im,
    int channels, int height, int width,
    int ksize, int stride, int pad, float* data_col, int bit_align)
{
    int c;
    const int height_col = (height + 2 * pad - ksize) / stride + 1;
    const int width_col = (width + 2 * pad - ksize) / stride + 1;
    const int channels_col = channels * ksize * ksize;

    // optimized version
    if (height_col == height && width_col == width && stride == 1 && pad == 1)
    {
        int new_ldb = bit_align;

        #pragma omp parallel for
        for (c = 0; c < channels_col; ++c) {
            int h, w;
            int w_offset = c % ksize;
            int h_offset = (c / ksize) % ksize;
            int c_im = c / ksize / ksize;
            for (h = pad; h < height_col - pad; ++h) {
                for (w = pad; w < width_col - pad - 8; w += 1) {
                    int im_row = h_offset + h - pad;
                    int im_col = w_offset + w - pad;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    float val = data_im[im_col + width*(im_row + height*c_im)];
                    if (val > 0) set_bit(data_col, col_index);
                }

                for (; w < width_col - pad; ++w) {
                    int im_row = h_offset + h - pad;
                    int im_col = w_offset + w - pad;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    //data_col[col_index] = data_im[im_col + width*(im_row + height*c_im)];
                    float val = data_im[im_col + width*(im_row + height*c_im)];
                    if (val > 0) set_bit(data_col, col_index);
                }
            }

            {
                w = 0;
                for (h = 0; h < height_col; ++h) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    //data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    float val = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    if (val > 0) set_bit(data_col, col_index);
                }
            }

            {
                w = width_col - 1;
                for (h = 0; h < height_col; ++h) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    //data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    float val = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    if (val > 0) set_bit(data_col, col_index);
                }
            }

            {
                h = 0;
                for (w = 0; w < width_col; ++w) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    //data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    float val = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    if (val > 0) set_bit(data_col, col_index);
                }
            }

            {
                h = height_col - 1;
                for (w = 0; w < width_col; ++w) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    //data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    float val = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    if (val > 0) set_bit(data_col, col_index);
                }
            }
        }

    }
    else {
        printf("\n Error: is no non-optimized version \n");
        //im2col_cpu(data_im, channels, height, width, ksize, stride, pad, data_col); // must be aligned for transpose after float_to_bin
        // float_to_bit(b, t_input, src_size);
        // transpose_bin(t_input, *t_bit_input, k, n, bit_align, new_ldb, 8);
    }
}


void activate_array_cpu_custom(float *x, const int n, const ACTIVATION a)
{
    int i = 0;
    if (a == LINEAR)  {}
    else {
        for (i = 0; i < n; ++i) {
            x[i] = activate(x[i], a);
        }
    }
}


void forward_maxpool_layer_avx(float *src, float *dst, int *indexes, int size, int w, int h, int out_w, int out_h, int c,
    int pad, int stride, int batch)
{
    int b, k;
    const int w_offset = -pad / 2;
    const int h_offset = -pad / 2;

    for (b = 0; b < batch; ++b) {
        #pragma omp parallel for
        for (k = 0; k < c; ++k) {
            int i, j, m, n;
            for (i = 0; i < out_h; ++i) {
                for (j = 0; j < out_w; ++j) {
                    int out_index = j + out_w*(i + out_h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for (n = 0; n < size; ++n) {
                        for (m = 0; m < size; ++m) {
                            int cur_h = h_offset + i*stride + n;
                            int cur_w = w_offset + j*stride + m;
                            int index = cur_w + w*(cur_h + h*(k + b*c));
                            int valid = (cur_h >= 0 && cur_h < h &&
                                cur_w >= 0 && cur_w < w);
                            float val = (valid != 0) ? src[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max = (val > max) ? val : max;
                        }
                    }
                    dst[out_index] = max;
                    indexes[out_index] = max_i;
                }
            }
        }
    }
}

void gemm_nn_custom_bin_mean_transposed(int M, int N, int K, float ALPHA_UNUSED,
    unsigned char *A, int lda,
    unsigned char *B, int ldb,
    float *C, int ldc, float *mean_arr)
{
    int i, j, k, h;

#pragma omp parallel for
    for (i = 0; i < M; ++i) {   // l.n - filters [16 - 55 - 1024]
        float mean_val = mean_arr[i];

        for (j = 0; j < N; ++j) { // out_h*out_w - one channel output size [169 - 173056]
            int count = 0;

            for (k = 0; k < K; k += 64) {   // l.size*l.size*l.c - one filter size [27 - 9216]
                uint64_t a_bit64 = *((uint64_t *)(A + (i*lda + k) / 8));
                uint64_t b_bit64 = *((uint64_t *)(B + (j*ldb + k) / 8));
                uint64_t c_bit64 = xnor_int64(a_bit64, b_bit64);

#ifdef WIN32
                int tmp_count = __popcnt64(c_bit64);
#else
                int tmp_count = __builtin_popcountll(c_bit64);
#endif

                if (K - k < 64)  tmp_count = tmp_count - (64 - (K - k));    // remove extra bits
                count += tmp_count;
                //binary_int64_printf(c_bit64);
                //printf(", count = %d \n\n", tmp_count);
            }

            C[i*ldc + j] = (2 * count - K) * mean_val;
        }
    }
}

void float_to_bit(float *src, unsigned char *dst, size_t size)
{
    size_t dst_size = size / 8 + 1;
    memset(dst, 0, dst_size);

    size_t i;
    char *byte_arr = calloc(size, sizeof(char));
    for (i = 0; i < size; ++i) {
        if (src[i] > 0) byte_arr[i] = 1;
    }

    //for (i = 0; i < size; ++i) {
    //    dst[i / 8] |= byte_arr[i] << (i % 8);
    //}

    for (i = 0; i < size; i += 8) {
        char dst_tmp = 0;
        dst_tmp |= byte_arr[i + 0] << 0;
        dst_tmp |= byte_arr[i + 1] << 1;
        dst_tmp |= byte_arr[i + 2] << 2;
        dst_tmp |= byte_arr[i + 3] << 3;
        dst_tmp |= byte_arr[i + 4] << 4;
        dst_tmp |= byte_arr[i + 5] << 5;
        dst_tmp |= byte_arr[i + 6] << 6;
        dst_tmp |= byte_arr[i + 7] << 7;
        dst[i / 8] = dst_tmp;
    }
    free(byte_arr);
}
#endif    // __x86_64

/*
void gemm_nn(int M, int N, int K, float ALPHA,
float *A, int lda,
float *B, int ldb,
float *C, int ldc)
{
int i, j, k;
for (i = 0; i < M; ++i) {
for (k = 0; k < K; ++k) {
register float A_PART = ALPHA*A[i*lda + k];
for (j = 0; j < N; ++j) {
C[i*ldc + j] += A_PART*B[k*ldb + j];
}
}
}
}
*/

void fill_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for (i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

// -------------- utils.c --------------


// utils.c
void error(const char *s)
{
    perror(s);
    assert(0);
    exit(-1);
}

// utils.c
void malloc_error()
{
    fprintf(stderr, "Malloc error\n");
    exit(-1);
}

// utils.c
void file_error(char *s)
{
    fprintf(stderr, "Couldn't open file: %s\n", s);
    exit(0);
}

// utils.c
char *fgetl(FILE *fp)
{
    if (feof(fp)) return 0;
    size_t size = 512;
    char *line = malloc(size * sizeof(char));
    if (!fgets(line, size, fp)) {
        free(line);
        return 0;
    }

    size_t curr = strlen(line);

    while ((line[curr - 1] != '\n') && !feof(fp)) {
        if (curr == size - 1) {
            size *= 2;
            line = realloc(line, size * sizeof(char));
            if (!line) {
                printf("%ld\n", (int long)size);
                malloc_error();
            }
        }
        size_t readsize = size - curr;
        if (readsize > INT_MAX) readsize = INT_MAX - 1;
        fgets(&line[curr], readsize, fp);
        curr = strlen(line);
    }
    if (line[curr - 1] == '\n') line[curr - 1] = '\0';

    return line;
}

// utils.c
int *read_map(char *filename)
{
    int n = 0;
    int *map = 0;
    char *str;
    FILE *file = fopen(filename, "r");
    if (!file) file_error(filename);
    while ((str = fgetl(file))) {
        ++n;
        map = realloc(map, n * sizeof(int));
        map[n - 1] = atoi(str);
    }
    return map;
}

// utils.c
void del_arg(int argc, char **argv, int index)
{
    int i;
    for (i = index; i < argc - 1; ++i) argv[i] = argv[i + 1];
    argv[i] = 0;
}

// utils.c
int find_arg(int argc, char* argv[], char *arg)
{
    int i;
    for (i = 0; i < argc; ++i) {
        if (!argv[i]) continue;
        if (0 == strcmp(argv[i], arg)) {
            del_arg(argc, argv, i);
            return 1;
        }
    }
    return 0;
}

// utils.c
int find_int_arg(int argc, char **argv, char *arg, int def)
{
    int i;
    for (i = 0; i < argc - 1; ++i) {
        if (!argv[i]) continue;
        if (0 == strcmp(argv[i], arg)) {
            def = atoi(argv[i + 1]);
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}

// utils.c
float find_float_arg(int argc, char **argv, char *arg, float def)
{
    int i;
    for (i = 0; i < argc - 1; ++i) {
        if (!argv[i]) continue;
        if (0 == strcmp(argv[i], arg)) {
            def = atof(argv[i + 1]);
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}

// utils.c
char *find_char_arg(int argc, char **argv, char *arg, char *def)
{
    int i;
    for (i = 0; i < argc - 1; ++i) {
        if (!argv[i]) continue;
        if (0 == strcmp(argv[i], arg)) {
            def = argv[i + 1];
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}


// utils.c
void strip(char *s)
{
    size_t i;
    size_t len = strlen(s);
    size_t offset = 0;
    for (i = 0; i < len; ++i) {
        char c = s[i];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++offset;
        else s[i - offset] = c;
    }
    s[len - offset] = '\0';
}

// utils.c
void list_insert(list *l, void *val)
{
    node *new = malloc(sizeof(node));
    new->val = val;
    new->next = 0;

    if (!l->back) {
        l->front = new;
        new->prev = 0;
    }
    else {
        l->back->next = new;
        new->prev = l->back;
    }
    l->back = new;
    ++l->size;
}


// utils.c
float rand_uniform(float min, float max)
{
    if (max < min) {
        float swap = min;
        min = max;
        max = swap;
    }
    return ((float)rand() / RAND_MAX * (max - min)) + min;
}

// utils.c
float rand_scale(float s)
{
    float scale = rand_uniform(1, s);
    if (rand() % 2) return scale;
    return 1. / scale;
}

// utils.c
int rand_int(int min, int max)
{
    if (max < min) {
        int s = min;
        min = max;
        max = s;
    }
    int r = (rand() % (max - min + 1)) + min;
    return r;
}

// utils.c
int constrain_int(int a, int min, int max)
{
    if (a < min) return min;
    if (a > max) return max;
    return a;
}

// utils.c
float dist_array(float *a, float *b, int n, int sub)
{
    int i;
    float sum = 0;
    for (i = 0; i < n; i += sub) sum += powf(a[i] - b[i], 2);
    return sqrt(sum);
}

// utils.c
float mag_array(float *a, int n)
{
    int i;
    float sum = 0;
    for (i = 0; i < n; ++i) {
        sum += a[i] * a[i];
    }
    return sqrt(sum);
}

// utils.c
int max_index(float *a, int n)
{
    if (n <= 0) return -1;
    int i, max_i = 0;
    float max = a[0];
    for (i = 1; i < n; ++i) {
        if (a[i] > max) {
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}

// utils.c
// From http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
float rand_normal()
{
    static int haveSpare = 0;
    static double rand1, rand2;

    if (haveSpare)
    {
        haveSpare = 0;
        return sqrt(rand1) * sin(rand2);
    }

    haveSpare = 1;

    rand1 = rand() / ((double)RAND_MAX);
    if (rand1 < 1e-100) rand1 = 1e-100;
    rand1 = -2 * log(rand1);
    rand2 = (rand() / ((double)RAND_MAX)) * TWO_PI;

    return sqrt(rand1) * cos(rand2);
}

// utils.c
void free_ptrs(void **ptrs, int n)
{
    int i;
    for (i = 0; i < n; ++i) free(ptrs[i]);
    free(ptrs);
}


// -------------- tree.c --------------

// tree.c
void hierarchy_predictions(float *predictions, int n, tree *hier, int only_leaves)
{
    int j;
    for (j = 0; j < n; ++j) {
        int parent = hier->parent[j];
        if (parent >= 0) {
            predictions[j] *= predictions[parent];
        }
    }
    if (only_leaves) {
        for (j = 0; j < n; ++j) {
            if (!hier->leaf[j]) predictions[j] = 0;
        }
    }
}

// tree.c
tree *read_tree(char *filename)
{
    tree t = { 0 };
    FILE *fp = fopen(filename, "r");

    char *line;
    int last_parent = -1;
    int group_size = 0;
    int groups = 0;
    int n = 0;
    while ((line = fgetl(fp)) != 0) {
        char *id = calloc(256, sizeof(char));
        int parent = -1;
        sscanf(line, "%s %d", id, &parent);
        t.parent = realloc(t.parent, (n + 1) * sizeof(int));
        t.parent[n] = parent;

        t.name = realloc(t.name, (n + 1) * sizeof(char *));
        t.name[n] = id;
        if (parent != last_parent) {
            ++groups;
            t.group_offset = realloc(t.group_offset, groups * sizeof(int));
            t.group_offset[groups - 1] = n - group_size;
            t.group_size = realloc(t.group_size, groups * sizeof(int));
            t.group_size[groups - 1] = group_size;
            group_size = 0;
            last_parent = parent;
        }
        t.group = realloc(t.group, (n + 1) * sizeof(int));
        t.group[n] = groups;
        ++n;
        ++group_size;
    }
    ++groups;
    t.group_offset = realloc(t.group_offset, groups * sizeof(int));
    t.group_offset[groups - 1] = n - group_size;
    t.group_size = realloc(t.group_size, groups * sizeof(int));
    t.group_size[groups - 1] = group_size;
    t.n = n;
    t.groups = groups;
    t.leaf = calloc(n, sizeof(int));
    int i;
    for (i = 0; i < n; ++i) t.leaf[i] = 1;
    for (i = 0; i < n; ++i) if (t.parent[i] >= 0) t.leaf[t.parent[i]] = 0;

    fclose(fp);
    tree *tree_ptr = calloc(1, sizeof(tree));
    *tree_ptr = t;
    //error(0);
    return tree_ptr;
}


// -------------- list.c --------------


// list.c
list *make_list()
{
    list *l = malloc(sizeof(list));
    l->size = 0;
    l->front = 0;
    l->back = 0;
    return l;
}


// list.c
list *get_paths(char *filename)
{
    char *path;
    FILE *file = fopen(filename, "r");
    if (!file) file_error(filename);
    list *lines = make_list();
    while ((path = fgetl(file))) {
        list_insert(lines, path);
    }
    fclose(file);
    return lines;
}


// list.c
void **list_to_array(list *l)
{
    void **a = calloc(l->size, sizeof(void*));
    int count = 0;
    node *n = l->front;
    while (n) {
        a[count++] = n->val;
        n = n->next;
    }
    return a;
}

// list.c
void free_node(node *n)
{
    node *next;
    while (n) {
        next = n->next;
        free(n);
        n = next;
    }
}

// list.c
void free_list(list *l)
{
    free_node(l->front);
    free(l);
}

// list.c
char **get_labels(char *filename)
{
    list *plist = get_paths(filename);
    char **labels = (char **)list_to_array(plist);
    free_list(plist);
    return labels;
}



// -------------- network.c --------------

// network.c
float *get_network_output(network net)
{
#ifdef GPU
    if (gpu_index >= 0) return get_network_output_gpu(net);
#endif
    int i;
    for (i = net.n - 1; i > 0; --i) if (net.layers[i].type != COST) break;
    return net.layers[i].output;
}

// network.c
int get_network_output_size(network net)
{
    int i;
    for (i = net.n - 1; i > 0; --i) if (net.layers[i].type != COST) break;
    return net.layers[i].outputs;
}

// network.c
network make_network(int n)
{
    network net = { 0 };
    net.n = n;
    net.layers = calloc(net.n, sizeof(layer));
    net.seen = calloc(1, sizeof(uint64_t));
#ifdef GPU
    net.input_gpu = calloc(1, sizeof(float *));
    net.truth_gpu = calloc(1, sizeof(float *));
#endif
    return net;
}

void free_network(network net)
{
    int i;
    for (i = 0; i < net.n; ++i) {
        free_layer(net.layers[i]);
    }
    free(net.layers);

    free(net.scales);
    free(net.steps);
    free(net.seen);

#ifdef GPU
    if (gpu_index >= 0) cuda_free(net.workspace);
    else free(net.workspace);
    if (net.input_state_gpu) cuda_free(net.input_state_gpu);
    if (*net.input_gpu) cuda_free(*net.input_gpu);
    if (*net.truth_gpu) cuda_free(*net.truth_gpu);
    if (net.input_gpu) free(net.input_gpu);
    if (net.truth_gpu) free(net.truth_gpu);

    //if (*net.input16_gpu) cuda_free(*net.input16_gpu);
    //if (*net.output16_gpu) cuda_free(*net.output16_gpu);
    //if (net.input16_gpu) free(net.input16_gpu);
    //if (net.output16_gpu) free(net.output16_gpu);
    //if (net.max_input16_size) free(net.max_input16_size);
    //if (net.max_output16_size) free(net.max_output16_size);
#else
    free(net.workspace);
#endif
}


// network.c
#ifdef GPU
#ifdef CUDNN
void cudnn_convolutional_setup(layer *l)
{
#if(CUDNN_MAJOR >= 7)
    cudnnSetConvolutionMathType(l->convDesc, CUDNN_TENSOR_OP_MATH);
#if((CUDNN_MAJOR*10 + CUDNN_MINOR) >= 72)   // cuDNN >= 7.2
    cudnnSetConvolutionMathType(l->convDesc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);
#endif  //(CUDNN_MAJOR >= 7.2)
#endif  //(CUDNN_MAJOR >= 7)

    if (l->quantized)
    {
        cudnnDataType_t cudnn_data_type = CUDNN_DATA_INT8x4;
        cudnnTensorFormat_t tensor_format = CUDNN_TENSOR_NCHW_VECT_C;
        cudnnTensorFormat_t dst_tensor_format = CUDNN_TENSOR_NCHW;

#if((CUDNN_MAJOR*10 + CUDNN_MINOR) >= 72)
        //if (l->c % 32 == 0) cudnn_data_type = CUDNN_DATA_INT8x32;   // Tensor Cores for INT8
#endif  //(CUDNN_MAJOR >= 7.2)

        cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW_VECT_C, cudnn_data_type, l->batch, l->c, l->h, l->w);
        cudnnSetFilter4dDescriptor(l->weightDesc, cudnn_data_type, CUDNN_TENSOR_NCHW_VECT_C, l->n, l->c, l->size, l->size);
        cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w);
        cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_INT32);    // cudnn 7

        l->fw_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM; // only supported for DP4A INT8x4

                                                                       // BIAS float
        cudnnSetTensor4dDescriptor(l->biasTensorDesc, dst_tensor_format, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1);

        // https://en.wikipedia.org/wiki/Activation_function
        // CUDNN_ACTIVATION_IDENTITY
        cudnnSetActivationDescriptor(l->activationDesc, CUDNN_ACTIVATION_IDENTITY, CUDNN_NOT_PROPAGATE_NAN, 0);
        //cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.1);    // RELU or ELU can't replace LEAKY_RELU
    }
    else {
        cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w);
        cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w);
        cudnnSetFilter4dDescriptor(l->weightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c, l->size, l->size);
#if(CUDNN_MAJOR >= 6)
        cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);    // cudnn 6.0
#else
        cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION);    // cudnn 5.1
#endif
        cudnnGetConvolutionForwardAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->weightDesc,
            l->convDesc,
            l->dstTensorDesc,
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            0,
            &l->fw_algo);

        //l->fw_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM; // un-comment to use Tensor Cores for cuDNN >= 7.2
    }
}
#endif
#endif



// network.c
void set_batch_network(network *net, int b)
{
    net->batch = b;
    int i;
    for (i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        l.batch = b;
#ifdef CUDNN
        if (l.type == CONVOLUTIONAL) {
            cudnn_convolutional_setup(net->layers + i);
        }
#endif
    }
}

// -------------- layer.c --------------



void free_layer(layer l)
{
    if (l.type == DROPOUT) {
        if (l.rand)           free(l.rand);
#ifdef GPU
        if (l.rand_gpu)             cuda_free(l.rand_gpu);
#endif
        return;
    }
    if (l.cweights)           free(l.cweights);
    if (l.indexes)            free(l.indexes);
    if (l.input_layers)       free(l.input_layers);
    if (l.input_sizes)        free(l.input_sizes);
    if (l.map)                free(l.map);
    if (l.rand)               free(l.rand);
    if (l.cost)               free(l.cost);
    if (l.state)              free(l.state);
    if (l.prev_state)         free(l.prev_state);
    if (l.forgot_state)       free(l.forgot_state);
    if (l.forgot_delta)       free(l.forgot_delta);
    if (l.state_delta)        free(l.state_delta);
    if (l.concat)             free(l.concat);
    if (l.concat_delta)       free(l.concat_delta);
    if (l.binary_weights)     free(l.binary_weights);
    if (l.biases)             free(l.biases);
    if (l.biases_quant)       free(l.biases_quant);
    //if (l.bias_updates)       free(l.bias_updates);
    if (l.scales)             free(l.scales);
    //if (l.scale_updates)      free(l.scale_updates);
    if (l.weights)            free(l.weights);
    if (l.weights_int8)       free(l.weights_int8);
    //if (l.weight_updates)     free(l.weight_updates);
    //if (l.delta)              free(l.delta);
    if (l.output)             free(l.output);
    if (l.squared)            free(l.squared);
    if (l.norms)              free(l.norms);
    if (l.spatial_mean)       free(l.spatial_mean);
    if (l.mean)               free(l.mean);
    if (l.variance)           free(l.variance);
    //if (l.mean_delta)         free(l.mean_delta);
    //if (l.variance_delta)     free(l.variance_delta);
    if (l.rolling_mean)       free(l.rolling_mean);
    if (l.rolling_variance)   free(l.rolling_variance);
    if (l.x)                  free(l.x);
    if (l.x_norm)             free(l.x_norm);
    if (l.m)                  free(l.m);
    if (l.v)                  free(l.v);
    if (l.z_cpu)              free(l.z_cpu);
    if (l.r_cpu)              free(l.r_cpu);
    if (l.h_cpu)              free(l.h_cpu);
    if (l.binary_input)       free(l.binary_input);

#ifdef GPU
    if (l.indexes_gpu)           cuda_free((float *)l.indexes_gpu);

    if (l.z_gpu)                   cuda_free(l.z_gpu);
    if (l.r_gpu)                   cuda_free(l.r_gpu);
    if (l.h_gpu)                   cuda_free(l.h_gpu);
    if (l.m_gpu)                   cuda_free(l.m_gpu);
    if (l.v_gpu)                   cuda_free(l.v_gpu);
    if (l.prev_state_gpu)          cuda_free(l.prev_state_gpu);
    if (l.forgot_state_gpu)        cuda_free(l.forgot_state_gpu);
    if (l.forgot_delta_gpu)        cuda_free(l.forgot_delta_gpu);
    if (l.state_gpu)               cuda_free(l.state_gpu);
    if (l.state_delta_gpu)         cuda_free(l.state_delta_gpu);
    if (l.gate_gpu)                cuda_free(l.gate_gpu);
    if (l.gate_delta_gpu)          cuda_free(l.gate_delta_gpu);
    if (l.save_gpu)                cuda_free(l.save_gpu);
    if (l.save_delta_gpu)          cuda_free(l.save_delta_gpu);
    if (l.concat_gpu)              cuda_free(l.concat_gpu);
    if (l.concat_delta_gpu)        cuda_free(l.concat_delta_gpu);
    if (l.binary_input_gpu)        cuda_free(l.binary_input_gpu);
    if (l.binary_weights_gpu)      cuda_free(l.binary_weights_gpu);
    if (l.mean_gpu)                cuda_free(l.mean_gpu);
    if (l.variance_gpu)            cuda_free(l.variance_gpu);
    if (l.rolling_mean_gpu)        cuda_free(l.rolling_mean_gpu);
    if (l.rolling_variance_gpu)    cuda_free(l.rolling_variance_gpu);
    if (l.variance_delta_gpu)      cuda_free(l.variance_delta_gpu);
    if (l.mean_delta_gpu)          cuda_free(l.mean_delta_gpu);
    if (l.x_gpu)                   cuda_free(l.x_gpu);
    if (l.x_norm_gpu)              cuda_free(l.x_norm_gpu);
    if (l.weights_gpu)             cuda_free(l.weights_gpu);
    //if (l.weight_updates_gpu)      cuda_free(l.weight_updates_gpu);
    if (l.biases_gpu)              cuda_free(l.biases_gpu);
    //if (l.bias_updates_gpu)        cuda_free(l.bias_updates_gpu);
    if (l.scales_gpu)              cuda_free(l.scales_gpu);
    //if (l.scale_updates_gpu)       cuda_free(l.scale_updates_gpu);
    if (l.output_gpu)              cuda_free(l.output_gpu);
    if (l.output_gpu_int8)         cuda_free(l.output_gpu_int8);
    if (l.delta_gpu)               cuda_free(l.delta_gpu);
    if (l.rand_gpu)                cuda_free(l.rand_gpu);
    if (l.squared_gpu)             cuda_free(l.squared_gpu);
    if (l.norms_gpu)               cuda_free(l.norms_gpu);
#endif
}


// -------------- softmax_layer.c --------------

// softmax_layer.c
softmax_layer make_softmax_layer(int batch, int inputs, int groups)
{
    assert(inputs%groups == 0);
    fprintf(stderr, "softmax                                        %4d\n", inputs);
    softmax_layer l = { 0 };
    l.type = SOFTMAX;
    l.batch = batch;
    l.groups = groups;
    l.inputs = inputs;
    l.outputs = inputs;
    l.output = calloc(inputs*batch, sizeof(float));
    //l.delta = calloc(inputs*batch, sizeof(float));

    // commented only for this custom version of Yolo v2
    //l.forward = forward_softmax_layer;
    //l.backward = backward_softmax_layer;
#ifdef GPU
    // commented only for this custom version of Yolo v2
    //l.forward_gpu = forward_softmax_layer_gpu;
    //l.backward_gpu = backward_softmax_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch);
    cudaError_t status;
    status = cudaMalloc((void **)&(l.output_gpu_int8), sizeof(int8_t)*inputs*batch);
    //l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
#endif

#ifdef OPENCL
    l.output_ocl = ocl_make_array(l.output, inputs*batch);
#endif

    return l;
}

// -------------- upsample_layer.c --------------

// upsample_layer.c
layer make_upsample_layer(int batch, int w, int h, int c, int stride)
{
    layer l = { 0 };
    l.type = UPSAMPLE;
    l.batch = batch;
    l.w = w;
    l.h = h;
    l.c = c;
    l.out_w = w*stride;
    l.out_h = h*stride;
    l.out_c = c;
    if (stride < 0) {
        stride = -stride;
        l.reverse = 1;
        l.out_w = w / stride;
        l.out_h = h / stride;
    }
    l.stride = stride;
    l.outputs = l.out_w*l.out_h*l.out_c;
    l.inputs = l.w*l.h*l.c;
    //l.delta = calloc(l.outputs*batch, sizeof(float));
    l.output = calloc(l.outputs*batch, sizeof(float));;

    //l.forward = forward_upsample_layer;
#ifdef GPU
    //l.forward_gpu = forward_upsample_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, l.outputs*batch);
#endif
    if (l.reverse) fprintf(stderr, "downsample         %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", stride, w, h, c, l.out_w, l.out_h, l.out_c);
    else fprintf(stderr, "upsample           %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}

// -------------- shortcut_layer.c --------------

// shortcut.c
layer make_shortcut_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2)
{
    fprintf(stderr, "Shortcut Layer: %d\n", index);
    layer l = { 0 };
    l.type = SHORTCUT;
    l.batch = batch;
    l.w = w2;
    l.h = h2;
    l.c = c2;
    l.out_w = w;
    l.out_h = h;
    l.out_c = c;
    l.outputs = w*h*c;
    l.inputs = l.outputs;

    l.index = index;

    l.output = calloc(l.outputs*batch, sizeof(float));;
#ifdef GPU
    l.output_gpu = cuda_make_array(l.output, l.outputs*batch);
#endif
    return l;
}

// -------------- reorg_layer.c --------------

// reorg_layer.c
layer make_reorg_layer(int batch, int w, int h, int c, int stride, int reverse)
{
    layer l = { 0 };
    l.type = REORG;
    l.batch = batch;
    l.stride = stride;
    l.h = h;
    l.w = w;
    l.c = c;
    if (reverse) {
        l.out_w = w*stride;
        l.out_h = h*stride;
        l.out_c = c / (stride*stride);
    }
    else {
        l.out_w = w / stride;
        l.out_h = h / stride;
        l.out_c = c*(stride*stride);
    }
    l.reverse = reverse;
    fprintf(stderr, "reorg              /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", stride, w, h, c, l.out_w, l.out_h, l.out_c);
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    l.output = calloc(output_size, sizeof(float));
    l.output_int8 = calloc(output_size, sizeof(int8_t));
    //l.delta = calloc(output_size, sizeof(float));

    // commented only for this custom version of Yolo v2
    //l.forward = forward_reorg_layer;
    //l.backward = backward_reorg_layer;
#ifdef GPU
    // commented only for this custom version of Yolo v2
    //l.forward_gpu = forward_reorg_layer_gpu;
    //l.backward_gpu = backward_reorg_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, output_size);
    cudaError_t status;
    status = cudaMalloc((void **)&(l.output_gpu_int8), sizeof(int8_t)*output_size);
    //l.delta_gpu = cuda_make_array(l.delta, output_size);
#endif

#ifdef OPENCL
    l.output_ocl = ocl_make_array(l.output, output_size);
#endif
    return l;
}

// -------------- route_layer.c --------------

// route_layer.c
route_layer make_route_layer(int batch, int n, int *input_layers, int *input_sizes)
{
    fprintf(stderr, "route ");
    route_layer l = { 0 };
    l.type = ROUTE;
    l.batch = batch;
    l.n = n;
    l.input_layers = input_layers;
    l.input_sizes = input_sizes;
    int i;
    int outputs = 0;
    for (i = 0; i < n; ++i) {
        fprintf(stderr, " %d", input_layers[i]);
        outputs += input_sizes[i];
    }
    fprintf(stderr, "\n");
    l.outputs = outputs;
    l.inputs = outputs;
    //l.delta = calloc(outputs*batch, sizeof(float));
    l.output = calloc(outputs*batch, sizeof(float));
    l.output_int8 = calloc(outputs*batch, sizeof(int8_t));

    // commented only for this custom version of Yolo v2
    //l.forward = forward_route_layer;
    //l.backward = backward_route_layer;
#ifdef GPU
    // commented only for this custom version of Yolo v2
    //l.forward_gpu = forward_route_layer_gpu;
    //l.backward_gpu = backward_route_layer_gpu;

    //l.delta_gpu = cuda_make_array(l.delta, outputs*batch);
    l.output_gpu = cuda_make_array(l.output, outputs*batch);
    cudaError_t status;
    status = cudaMalloc((void **)&(l.output_gpu_int8), sizeof(int8_t)*outputs*batch);
#endif

#ifdef OPENCL
    l.output_ocl = ocl_make_array(l.output, outputs*batch);
#endif
    return l;
}

// -------------- yolo_layer.c --------------

layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int max_boxes)
{
    int i;
    layer l = { 0 };
    l.type = YOLO;

    l.n = n;
    l.total = total;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n*(classes + 4 + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.cost = calloc(1, sizeof(float));
    l.biases = calloc(total * 2, sizeof(float));
    if (mask) l.mask = mask;
    else {
        l.mask = calloc(n, sizeof(int));
        for (i = 0; i < n; ++i) {
            l.mask[i] = i;
        }
    }
    //l.bias_updates = calloc(n * 2, sizeof(float));
    l.outputs = h*w*n*(classes + 4 + 1);
    l.inputs = l.outputs;
    l.max_boxes = max_boxes;
    l.truths = l.max_boxes*(4 + 1);    // 90*(4 + 1);
    //l.delta = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));
    for (i = 0; i < total * 2; ++i) {
        l.biases[i] = .5;
    }

#ifdef GPU
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
#endif

    fprintf(stderr, "yolo\n");
    srand(0);

    return l;
}

// -------------- region_layer.c --------------

//  region_layer.c
region_layer make_region_layer(int batch, int w, int h, int n, int classes, int coords)
{
    region_layer l = { 0 };
    l.type = REGION;

    l.n = n;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.classes = classes;
    l.coords = coords;
    l.cost = calloc(1, sizeof(float));
    l.biases = calloc(n * 2, sizeof(float));
    //l.bias_updates = calloc(n * 2, sizeof(float));
    l.outputs = h*w*n*(classes + coords + 1);
    l.inputs = l.outputs;
    l.truths = 30 * (5);
    //l.delta = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));
    int i;
    for (i = 0; i < n * 2; ++i) {
        l.biases[i] = .5;
    }

    // commented only for this custom version of Yolo v2
    //l.forward = forward_region_layer;
    //l.backward = backward_region_layer;
#ifdef GPU
    // commented only for this custom version of Yolo v2
    //l.forward_gpu = forward_region_layer_gpu;
    //l.backward_gpu = backward_region_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    cudaError_t status;
    status = cudaMalloc((void **)&(l.output_gpu_int8), sizeof(int8_t)*l.outputs*batch);
    //l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

#ifdef OPENCL
    l.output_ocl = ocl_make_array(l.output, batch*l.outputs);
#endif

    fprintf(stderr, "detection\n");
    srand(0);

    return l;
}


// -------------- maxpool_layer.c --------------

// maxpool_layer.c
maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding)
{
    maxpool_layer l = { 0 };
    l.type = MAXPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;
    l.out_w = (w + padding - size) / stride + 1;
    l.out_h = (h + padding - size) / stride + 1;
    l.out_c = c;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.size = size;
    l.stride = stride;
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    l.indexes = calloc(output_size, sizeof(int));
    l.output = calloc(output_size, sizeof(float));
    l.output_int8 = calloc(output_size, sizeof(int8_t));
    //l.delta = calloc(output_size, sizeof(float));
    // commented only for this custom version of Yolo v2
    //l.forward = forward_maxpool_layer;
    //l.backward = backward_maxpool_layer;
#ifdef GPU
    // commented only for this custom version of Yolo v2
    //l.forward_gpu = forward_maxpool_layer_gpu;
    //l.backward_gpu = backward_maxpool_layer_gpu;
    l.indexes_gpu = cuda_make_int_array(output_size);
    l.output_gpu = cuda_make_array(l.output, output_size);
    cudaError_t status;
    status = cudaMalloc((void **)&(l.output_gpu_int8), sizeof(int8_t)*output_size);
    //l.delta_gpu = cuda_make_array(l.delta, output_size);

    cudnnStatus_t maxpool_status;
    maxpool_status = cudnnCreatePoolingDescriptor(&l.poolingDesc);

    maxpool_status = cudnnSetPooling2dDescriptor(
        l.poolingDesc,
        CUDNN_POOLING_MAX,
        CUDNN_PROPAGATE_NAN,    // CUDNN_PROPAGATE_NAN, CUDNN_NOT_PROPAGATE_NAN
        l.size,
        l.size,
        0, //l.pad,
        0, //l.pad,
        l.stride,
        l.stride);

    cudnnCreateTensorDescriptor(&l.srcTensorDesc);
    cudnnCreateTensorDescriptor(&l.dstTensorDesc);
    cudnnSetTensor4dDescriptor(l.srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.c, l.h, l.w);
    cudnnSetTensor4dDescriptor(l.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.out_c, l.out_h, l.out_w);
#endif

#ifdef OPENCL
    l.indexes_ocl = ocl_make_int_array(output_size);
    l.output_ocl = ocl_make_array(l.output, output_size);
#endif
    fprintf(stderr, "max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}


// -------------- convolutional_layer.c --------------

// convolutional_layer.c
size_t get_workspace_size(layer l) {
#ifdef CUDNN
    if (gpu_index >= 0) {
        size_t most = 0;
        size_t s = 0;
        cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
            l.srcTensorDesc,
            l.weightDesc,
            l.convDesc,
            l.dstTensorDesc,
            l.fw_algo,
            &s);
        /*
        if (s > most) most = s;
        cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(),
        l.srcTensorDesc,
        l.ddstTensorDesc,
        l.convDesc,
        l.dweightDesc,
        l.bf_algo,
        &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(),
        l.weightDesc,
        l.ddstTensorDesc,
        l.convDesc,
        l.dsrcTensorDesc,
        l.bd_algo,
        &s);*/
        if (s > most) most = s;
        return most;
    }
#endif
    if (l.xnor) return (size_t)l.bit_align*l.size*l.size*l.c * sizeof(float);
    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c * sizeof(float);
}

int convolutional_out_height(convolutional_layer l)
{
    return (l.h + 2 * l.pad - l.size) / l.stride + 1;
}

int convolutional_out_width(convolutional_layer l)
{
    return (l.w + 2 * l.pad - l.size) / l.stride + 1;
}


// convolutional_layer.c
convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam, int quantized, int use_bin_output)
{
    int i;
    convolutional_layer l = { 0 };
    l.type = CONVOLUTIONAL;
    l.quantized = quantized;

    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.binary = binary;
    l.xnor = xnor;
    l.use_bin_output = use_bin_output;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;

    l.weights = calloc(c*n*size*size, sizeof(float));
    l.weights_int8 = calloc(c*n*size*size, sizeof(int8_t));
    //l.weight_updates = calloc(c*n*size*size, sizeof(float));

    l.biases = calloc(n, sizeof(float));
    l.biases_quant = calloc(n, sizeof(float));
    //l.bias_updates = calloc(n, sizeof(float));

    // float scale = 1./sqrt(size*size*c);
    float scale = sqrt(2. / (size*size*c));
    for (i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_uniform(-1, 1);
    int out_h = convolutional_out_height(l);
    int out_w = convolutional_out_width(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = calloc(l.batch*l.outputs, sizeof(float));
    l.output_int8 = calloc(l.batch*l.outputs, sizeof(int8_t));
    //l.delta = calloc(l.batch*l.outputs, sizeof(float));

    // commented only for this custom version of Yolo v2
    ///l.forward = forward_convolutional_layer;
    ///l.backward = backward_convolutional_layer;
    ///l.update = update_convolutional_layer;
    if (binary) {
        l.binary_weights = calloc(c*n*size*size, sizeof(float));
        l.cweights = calloc(c*n*size*size, sizeof(char));
        l.scales = calloc(n, sizeof(float));
    }
    if (xnor) {
        l.binary_weights = calloc(c*n*size*size, sizeof(float));
        l.binary_input = calloc(l.inputs*l.batch, sizeof(float));

        int align = 32;// 8;
        int src_align = l.out_h*l.out_w;
        l.bit_align = src_align + (align - src_align % align);
    }

    if (batch_normalize) {
        l.scales = calloc(n, sizeof(float));
        //l.scale_updates = calloc(n, sizeof(float));
        for (i = 0; i < n; ++i) {
            l.scales[i] = 1;
        }

        l.mean = calloc(n, sizeof(float));
        l.variance = calloc(n, sizeof(float));

        //l.mean_delta = calloc(n, sizeof(float));
        //l.variance_delta = calloc(n, sizeof(float));

        l.rolling_mean = calloc(n, sizeof(float));
        l.rolling_variance = calloc(n, sizeof(float));
        l.x = calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = calloc(l.batch*l.outputs, sizeof(float));
    }
    if (adam) {
        l.adam = 1;
        l.m = calloc(c*n*size*size, sizeof(float));
        l.v = calloc(c*n*size*size, sizeof(float));
    }

#ifdef GPU
    // commented only for this custom version of Yolo v2
    //l.forward_gpu = forward_convolutional_layer_gpu;
    //l.backward_gpu = backward_convolutional_layer_gpu;
    //l.update_gpu = update_convolutional_layer_gpu;

    if (gpu_index >= 0) {
        //if (adam) {
        //    l.m_gpu = cuda_make_array(l.m, c*n*size*size);
        //    l.v_gpu = cuda_make_array(l.v, c*n*size*size);
        //}

        l.weights_gpu = cuda_make_array(l.weights, c*n*size*size);
        //l.weight_updates_gpu = cuda_make_array(l.weight_updates, c*n*size*size);

        l.biases_gpu = cuda_make_array(l.biases, n);
        //l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

        //l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*n);
        l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);

        cudaError_t status;
        status = cudaMalloc((void **)&(l.output_gpu_int8), sizeof(int8_t)*l.batch*out_h*out_w*n);

        //if (binary) {
        //    l.binary_weights_gpu = cuda_make_array(l.weights, c*n*size*size);
        //}
        if (xnor) {
            l.binary_weights_gpu = cuda_make_array(l.weights, c*n*size*size);
            l.binary_input_gpu = cuda_make_array(0, l.inputs*l.batch);
        }

        if (batch_normalize) {
            //l.mean_gpu = cuda_make_array(l.mean, n);
            //l.variance_gpu = cuda_make_array(l.variance, n);

            l.rolling_mean_gpu = cuda_make_array(l.mean, n);
            l.rolling_variance_gpu = cuda_make_array(l.variance, n);

            //l.mean_delta_gpu = cuda_make_array(l.mean, n);
            //l.variance_delta_gpu = cuda_make_array(l.variance, n);

            l.scales_gpu = cuda_make_array(l.scales, n);
            //l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);

            l.x_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
            //l.x_norm_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
        }
#ifdef CUDNN
        cudnnCreateTensorDescriptor(&l.biasTensorDesc);
        cudnnCreateActivationDescriptor(&l.activationDesc);
        cudnnCreateTensorDescriptor(&l.srcTensorDesc);
        cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        cudnnCreateFilterDescriptor(&l.weightDesc);
        //cudnnCreateTensorDescriptor(&l.dsrcTensorDesc);
        //cudnnCreateTensorDescriptor(&l.ddstTensorDesc);
        //cudnnCreateFilterDescriptor(&l.dweightDesc);
        cudnnCreateConvolutionDescriptor(&l.convDesc);
        cudnn_convolutional_setup(&l);
#endif
    }
#endif

#ifdef OPENCL
    //if (gpu_index >= 0) {

    l.weights_ocl = ocl_make_array(l.weights, c*n*size*size);
    l.biases_ocl = ocl_make_array(l.biases, n);
    l.output_ocl = ocl_make_array(l.output, l.batch*out_h*out_w*n);

    if (batch_normalize) {
        l.rolling_mean_ocl = ocl_make_array(l.rolling_mean, n);    // l.mean
        l.rolling_variance_ocl = ocl_make_array(l.rolling_variance, n);    // l.variance
        l.scales_ocl = ocl_make_array(l.scales, n);

        l.x_ocl = ocl_make_array(l.output, l.batch*out_h*out_w*n);
    }
    //}
#endif

    l.workspace_size = get_workspace_size(l);
    l.activation = activation;

    l.bflops = (2.0 * l.n * l.size*l.size*l.c * l.out_h*l.out_w) / 1000000000.;
    if (l.xnor && l.use_bin_output) fprintf(stderr, "convXB");
    else if (l.xnor) fprintf(stderr, "convX ");
    else fprintf(stderr, "conv  ");
    fprintf(stderr, "%5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d %5.3f BF\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c, l.bflops);

    return l;
}

// -------------- image.c --------------

// image.c
void rgbgr_image(image im)
{
    int i;
    for (i = 0; i < im.w*im.h; ++i) {
        float swap = im.data[i];
        im.data[i] = im.data[i + im.w*im.h * 2];
        im.data[i + im.w*im.h * 2] = swap;
    }
}

// image.c
image make_empty_image(int w, int h, int c)
{
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

// image.c
void free_image(image m)
{
    if (m.data) {
        free(m.data);
    }
}

// image.c
void draw_box(image a, int x1, int y1, int x2, int y2, float r, float g, float b)
{
    //normalize_image(a);
    int i;
    if (x1 < 0) x1 = 0;
    if (x1 >= a.w) x1 = a.w - 1;
    if (x2 < 0) x2 = 0;
    if (x2 >= a.w) x2 = a.w - 1;

    if (y1 < 0) y1 = 0;
    if (y1 >= a.h) y1 = a.h - 1;
    if (y2 < 0) y2 = 0;
    if (y2 >= a.h) y2 = a.h - 1;

    for (i = x1; i <= x2; ++i) {
        a.data[i + y1*a.w + 0 * a.w*a.h] = r;
        a.data[i + y2*a.w + 0 * a.w*a.h] = r;

        a.data[i + y1*a.w + 1 * a.w*a.h] = g;
        a.data[i + y2*a.w + 1 * a.w*a.h] = g;

        a.data[i + y1*a.w + 2 * a.w*a.h] = b;
        a.data[i + y2*a.w + 2 * a.w*a.h] = b;
    }
    for (i = y1; i <= y2; ++i) {
        a.data[x1 + i*a.w + 0 * a.w*a.h] = r;
        a.data[x2 + i*a.w + 0 * a.w*a.h] = r;

        a.data[x1 + i*a.w + 1 * a.w*a.h] = g;
        a.data[x2 + i*a.w + 1 * a.w*a.h] = g;

        a.data[x1 + i*a.w + 2 * a.w*a.h] = b;
        a.data[x2 + i*a.w + 2 * a.w*a.h] = b;
    }
}

// image.c
void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b)
{
    int i;
    for (i = 0; i < w; ++i) {
        draw_box(a, x1 + i, y1 + i, x2 - i, y2 - i, r, g, b);
    }
}

// image.c
image make_image(int w, int h, int c)
{
    image out = make_empty_image(w, h, c);
    out.data = calloc(h*w*c, sizeof(float));
    return out;
}

// image.c
float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}

// image.c
void set_pixel(image m, int x, int y, int c, float val)
{
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] = val;
}

// image.c
void add_pixel(image m, int x, int y, int c, float val)
{
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] += val;
}

// image.c
image resize_image(image im, int w, int h)
{
    image resized = make_image(w, h, im.c);
    image part = make_image(w, im.h, im.c);
    int r, c, k;
    float w_scale = (float)(im.w - 1) / (w - 1);
    float h_scale = (float)(im.h - 1) / (h - 1);
    for (k = 0; k < im.c; ++k) {
        for (r = 0; r < im.h; ++r) {
            for (c = 0; c < w; ++c) {
                float val = 0;
                if (c == w - 1 || im.w == 1) {
                    val = get_pixel(im, im.w - 1, r, k);
                }
                else {
                    float sx = c*w_scale;
                    int ix = (int)sx;
                    float dx = sx - ix;
                    val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix + 1, r, k);
                }
                set_pixel(part, c, r, k, val);
            }
        }
    }
    for (k = 0; k < im.c; ++k) {
        for (r = 0; r < h; ++r) {
            float sy = r*h_scale;
            int iy = (int)sy;
            float dy = sy - iy;
            for (c = 0; c < w; ++c) {
                float val = (1 - dy) * get_pixel(part, c, iy, k);
                set_pixel(resized, c, r, k, val);
            }
            if (r == h - 1 || im.h == 1) continue;
            for (c = 0; c < w; ++c) {
                float val = dy * get_pixel(part, c, iy + 1, k);
                add_pixel(resized, c, r, k, val);
            }
        }
    }

    free_image(part);
    return resized;
}

// image.c
image load_image(char *filename, int w, int h, int c)
{
#ifdef OPENCV
    image out = load_image_cv(filename, c);
#else
    image out = load_image_stb(filename, c);
#endif

    if ((h && w) && (h != out.h || w != out.w)) {
        image resized = resize_image(out, w, h);
        free_image(out);
        out = resized;
    }
    return out;
}

// image.c
image load_image_stb(char *filename, int channels)
{
    int w, h, c;
    unsigned char *data = stbi_load(filename, &w, &h, &c, channels);
    if (!data) {
        fprintf(stderr, "Cannot load image \"%s\"\nSTB Reason: %s\n", filename, stbi_failure_reason());
        exit(0);
    }
    if (channels) c = channels;
    int i, j, k;
    image im = make_image(w, h, c);
    for (k = 0; k < c; ++k) {
        for (j = 0; j < h; ++j) {
            for (i = 0; i < w; ++i) {
                int dst_index = i + w*j + w*h*k;
                int src_index = k + c*i + c*w*j;
                im.data[dst_index] = (float)data[src_index] / 255.;
            }
        }
    }
    free(data);
    return im;
}


#ifdef OPENCV

// image.c
image ipl_to_image(IplImage* src)
{
    unsigned char *data = (unsigned char *)src->imageData;
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    int step = src->widthStep;
    image out = make_image(w, h, c);
    int i, j, k, count = 0;;

    for (k = 0; k < c; ++k) {
        for (i = 0; i < h; ++i) {
            for (j = 0; j < w; ++j) {
                out.data[count++] = data[i*step + j*c + k] / 255.;
            }
        }
    }
    return out;
}

// image.c
image load_image_cv(char *filename, int channels)
{
    IplImage* src = 0;
    int flag = -1;
    if (channels == 0) flag = -1;
    else if (channels == 1) flag = 0;
    else if (channels == 3) flag = 1;
    else {
        fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);
    }

    if ((src = cvLoadImage(filename, flag)) == 0)
    {
        fprintf(stderr, "Cannot load image \"%s\"\n", filename);
        char buff[256];
        sprintf(buff, "echo %s >> bad.list", filename);
        system(buff);
        return make_image(10, 10, 3);
        //exit(0);
    }
    image out = ipl_to_image(src);
    cvReleaseImage(&src);
    rgbgr_image(out);
    return out;
}
#endif    // OPENCV

// image.c
image copy_image(image p)
{
    image copy = p;
    copy.data = calloc(p.h*p.w*p.c, sizeof(float));
    memcpy(copy.data, p.data, p.h*p.w*p.c * sizeof(float));
    return copy;
}

// image.c
void constrain_image(image im)
{
    int i;
    for (i = 0; i < im.w*im.h*im.c; ++i) {
        if (im.data[i] < 0) im.data[i] = 0;
        if (im.data[i] > 1) im.data[i] = 1;
    }
}

#ifdef OPENCV
// image.c
void show_image_cv(image p, const char *name)
{
    int x, y, k;
    image copy = copy_image(p);
    constrain_image(copy);
    if (p.c == 3) rgbgr_image(copy);
    char buff[256];
    sprintf(buff, "%s", name);

    IplImage *disp = cvCreateImage(cvSize(p.w, p.h), IPL_DEPTH_8U, p.c);
    int step = disp->widthStep;
    cvNamedWindow(buff, CV_WINDOW_NORMAL);
    for (y = 0; y < p.h; ++y) {
        for (x = 0; x < p.w; ++x) {
            for (k = 0; k < p.c; ++k) {
                disp->imageData[y*step + x*p.c + k] = (unsigned char)(get_pixel(copy, x, y, k) * 255);
            }
        }
    }
    free_image(copy);
    cvShowImage(buff, disp);

    cvReleaseImage(&disp);
}

// image.c
void show_image_cv_ipl(IplImage *disp, const char *name)
{
    if (disp == NULL) return;
    char buff[256];
    sprintf(buff, "%s", name);
    cvNamedWindow(buff, CV_WINDOW_NORMAL);
    cvShowImage(buff, disp);
}
#endif

// image.c
void save_image_png(image im, const char *name)
{
    char buff[256];
    sprintf(buff, "%s.png", name);
    unsigned char *data = calloc(im.w*im.h*im.c, sizeof(char));
    int i, k;
    for (k = 0; k < im.c; ++k) {
        for (i = 0; i < im.w*im.h; ++i) {
            data[i*im.c + k] = (unsigned char)(255 * im.data[i + k*im.w*im.h]);
        }
    }
    int success = stbi_write_png(buff, im.w, im.h, im.c, data, im.w*im.c);
    free(data);
    if (!success) fprintf(stderr, "Failed to write image %s\n", buff);
}


// image.c
void show_image(image p, const char *name)
{
#ifdef OPENCV
    show_image_cv(p, name);
#else
    fprintf(stderr, "Not compiled with OpenCV, saving to %s.png instead\n", name);
    save_image_png(p, name);
#endif
}

// image.c
float get_color(int c, int x, int max)
{
    static float colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };
    float ratio = ((float)x / max) * 5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1 - ratio) * colors[i][c] + ratio*colors[j][c];
    //printf("%f\n", r);
    return r;
}


// -------------- option_list.c --------------------


// option_list.c
typedef struct {
    char *key;
    char *val;
    int used;
} kvp;


// option_list.c
void option_insert(list *l, char *key, char *val)
{
    kvp *p = malloc(sizeof(kvp));
    p->key = key;
    p->val = val;
    p->used = 0;
    list_insert(l, p);
}

// option_list.c
int read_option(char *s, list *options)
{
    size_t i;
    size_t len = strlen(s);
    char *val = 0;
    for (i = 0; i < len; ++i) {
        if (s[i] == '=') {
            s[i] = '\0';
            val = s + i + 1;
            break;
        }
    }
    if (i == len - 1) return 0;
    char *key = s;
    option_insert(options, key, val);
    return 1;
}

// option_list.c
list *read_data_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    if (file == 0) file_error(filename);
    char *line;
    int nu = 0;
    list *options = make_list();
    while ((line = fgetl(file)) != 0) {
        ++nu;
        strip(line);
        switch (line[0]) {
        case '\0':
        case '#':
        case ';':
            free(line);
            break;
        default:
            if (!read_option(line, options)) {
                fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                free(line);
            }
            break;
        }
    }
    fclose(file);
    return options;
}

// option_list.c
void option_unused(list *l)
{
    node *n = l->front;
    while (n) {
        kvp *p = (kvp *)n->val;
        if (!p->used) {
            fprintf(stderr, "Unused field: '%s = %s'\n", p->key, p->val);
        }
        n = n->next;
    }
}

// option_list.c
char *option_find(list *l, char *key)
{
    node *n = l->front;
    while (n) {
        kvp *p = (kvp *)n->val;
        if (strcmp(p->key, key) == 0) {
            p->used = 1;
            return p->val;
        }
        n = n->next;
    }
    return 0;
}

// option_list.c
char *option_find_str(list *l, char *key, char *def)
{
    char *v = option_find(l, key);
    if (v) return v;
    if (def) fprintf(stderr, "%s: Using default '%s'\n", key, def);
    return def;
}

// option_list.c
int option_find_int(list *l, char *key, int def)
{
    char *v = option_find(l, key);
    if (v) return atoi(v);
    fprintf(stderr, "%s: Using default '%d'\n", key, def);
    return def;
}

// option_list.c
int option_find_int_quiet(list *l, char *key, int def)
{
    char *v = option_find(l, key);
    if (v) return atoi(v);
    return def;
}

// option_list.c
float option_find_float_quiet(list *l, char *key, float def)
{
    char *v = option_find(l, key);
    if (v) return atof(v);
    return def;
}

// option_list.c
float option_find_float(list *l, char *key, float def)
{
    char *v = option_find(l, key);
    if (v) return atof(v);
    fprintf(stderr, "%s: Using default '%lf'\n", key, def);
    return def;
}


// -------------- parser.c --------------------

// parser.c
typedef struct size_params {
    int quantized;
    int batch;
    int inputs;
    int h;
    int w;
    int c;
    int index;
    int time_steps;
    network net;
} size_params;

// parser.c
typedef struct {
    char *type;
    list *options;
}section;

// parser.c
list *read_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    if (file == 0) file_error(filename);
    char *line;
    int nu = 0;
    list *sections = make_list();
    section *current = 0;
    while ((line = fgetl(file)) != 0) {
        ++nu;
        strip(line);
        switch (line[0]) {
        case '[':
            current = malloc(sizeof(section));
            list_insert(sections, current);
            current->options = make_list();
            current->type = line;
            break;
        case '\0':
        case '#':
        case ';':
            free(line);
            break;
        default:
            if (!read_option(line, current->options)) {
                fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                free(line);
            }
            break;
        }
    }
    fclose(file);
    return sections;
}

// parser.c
void load_convolutional_weights_cpu(layer l, FILE *fp)
{
    int num = l.n*l.c*l.size*l.size;
    fread(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize && (!l.dontloadscales)) {
        fread(l.scales, sizeof(float), l.n, fp);
        fread(l.rolling_mean, sizeof(float), l.n, fp);
        fread(l.rolling_variance, sizeof(float), l.n, fp);
    }
    fread(l.weights, sizeof(float), num, fp);
    /*    if (l.adam) {
    fread(l.m, sizeof(float), num, fp);
    fread(l.v, sizeof(float), num, fp);
    }
    if (l.flipped) {
    transpose_matrix(l.weights, l.c*l.size*l.size, l.n);
    }*/
    //if (l.binary) binarize_weights(l.weights, l.n, l.c*l.size*l.size, l.weights);
#ifdef GPU
    if (gpu_index >= 0) {
        push_convolutional_layer(l);
    }
#endif

#ifdef OPENCL
    //if (gpu_index >= 0) {
    ocl_push_convolutional_layer(l);
    //}
#endif
}

// parser.c
void load_weights_upto_cpu(network *net, char *filename, int cutoff)
{
#ifdef GPU
    if (net->gpu_index >= 0) {
        cuda_set_device(net->gpu_index);
    }
#endif
    fprintf(stderr, "Loading weights from %s...", filename);
    fflush(stdout);
    FILE *fp = fopen(filename, "rb");
    if (!fp) file_error(filename);

    int major;
    int minor;
    int revision;
    fread(&major, sizeof(int), 1, fp);
    fread(&minor, sizeof(int), 1, fp);
    fread(&revision, sizeof(int), 1, fp);
    if ((major * 10 + minor) >= 2) {
        fread(net->seen, sizeof(uint64_t), 1, fp);
    }
    else {
        int iseen = 0;
        fread(&iseen, sizeof(int), 1, fp);
        *net->seen = iseen;
    }
    //int transpose = (major > 1000) || (minor > 1000);

    int i;
    for (i = 0; i < net->n && i < cutoff; ++i) {
        layer l = net->layers[i];
        if (l.dontload) continue;
        if (l.type == CONVOLUTIONAL) {
            load_convolutional_weights_cpu(l, fp);
        }
    }
    fprintf(stderr, "Done!\n");
    fclose(fp);
}



// parser.c
convolutional_layer parse_convolutional(list *options, size_params params)
{
    int n = option_find_int(options, "filters", 1);
    int size = option_find_int(options, "size", 1);
    int stride = option_find_int(options, "stride", 1);
    int pad = option_find_int_quiet(options, "pad", 0);
    int padding = option_find_int_quiet(options, "padding", 0);
    if (pad) padding = size / 2;

    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch, h, w, c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch = params.batch;
    if (!(h && w && c)) error("Layer before convolutional layer must output image.");
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int binary = option_find_int_quiet(options, "binary", 0);
    int xnor = option_find_int_quiet(options, "xnor", 0);
    int use_bin_output = option_find_int_quiet(options, "bin_output", 0);

    int quantized = params.quantized;
    if (params.index == 0 || activation == LINEAR || (params.index > 1 && stride>1) || size==1)
        quantized = 0; // disable Quantized for 1st and last layers
    convolutional_layer layer = make_convolutional_layer(batch, h, w, c, n, size, stride, padding, activation, batch_normalize, binary, xnor, params.net.adam, quantized, use_bin_output);
    layer.flipped = option_find_int_quiet(options, "flipped", 0);
    layer.dot = option_find_float_quiet(options, "dot", 0);
    if (params.net.adam) {
        layer.B1 = params.net.B1;
        layer.B2 = params.net.B2;
        layer.eps = params.net.eps;
    }

    return layer;
}

// parser.c
layer parse_region(list *options, size_params params)
{
    int coords = option_find_int(options, "coords", 4);
    int classes = option_find_int(options, "classes", 20);
    int num = option_find_int(options, "num", 1);

    layer l = make_region_layer(params.batch, params.w, params.h, num, classes, coords);
    assert(l.outputs == params.inputs);

    l.log = option_find_int_quiet(options, "log", 0);
    l.sqrt = option_find_int_quiet(options, "sqrt", 0);

    l.softmax = option_find_int(options, "softmax", 0);
    l.max_boxes = option_find_int_quiet(options, "max", 30);
    l.jitter = option_find_float(options, "jitter", .2);
    l.rescore = option_find_int_quiet(options, "rescore", 0);

    l.thresh = option_find_float(options, "thresh", .5);
    l.classfix = option_find_int_quiet(options, "classfix", 0);
    l.absolute = option_find_int_quiet(options, "absolute", 0);
    l.random = option_find_int_quiet(options, "random", 0);

    l.coord_scale = option_find_float(options, "coord_scale", 1);
    l.object_scale = option_find_float(options, "object_scale", 1);
    l.noobject_scale = option_find_float(options, "noobject_scale", 1);
    l.class_scale = option_find_float(options, "class_scale", 1);
    l.bias_match = option_find_int_quiet(options, "bias_match", 0);

    char *tree_file = option_find_str(options, "tree", 0);
    if (tree_file) l.softmax_tree = read_tree(tree_file);
    char *map_file = option_find_str(options, "map", 0);
    if (map_file) l.map = read_map(map_file);

    char *a = option_find_str(options, "anchors", 0);
    if (a) {
        int len = strlen(a);
        int n = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (a[i] == ',') ++n;
        }
        for (i = 0; i < n; ++i) {
            float bias = atof(a);
            l.biases[i] = bias;
            a = strchr(a, ',') + 1;
        }
    }
    return l;
}

// parser.c
int *parse_yolo_mask(char *a, int *num)
{
    int *mask = 0;
    if (a) {
        int len = strlen(a);
        int n = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (a[i] == ',') ++n;
        }
        mask = calloc(n, sizeof(int));
        for (i = 0; i < n; ++i) {
            int val = atoi(a);
            mask[i] = val;
            a = strchr(a, ',') + 1;
        }
        *num = n;
    }
    return mask;
}

// parser.c
layer parse_yolo(list *options, size_params params)
{
    int classes = option_find_int(options, "classes", 20);
    int total = option_find_int(options, "num", 1);
    int num = total;

    char *a = option_find_str(options, "mask", 0);
    int *mask = parse_yolo_mask(a, &num);
    int max_boxes = option_find_int_quiet(options, "max", 90);
    layer l = make_yolo_layer(params.batch, params.w, params.h, num, total, mask, classes, max_boxes);
    if (l.outputs != params.inputs) {
        printf("Error: l.outputs == params.inputs \n");
        printf("filters= in the [convolutional]-layer doesn't correspond to classes= or mask= in [yolo]-layer \n");
        exit(EXIT_FAILURE);
    }
    //assert(l.outputs == params.inputs);
    char *map_file = option_find_str(options, "map", 0);
    if (map_file) l.map = read_map(map_file);

    l.jitter = option_find_float(options, "jitter", .2);
    l.focal_loss = option_find_int_quiet(options, "focal_loss", 0);

    l.ignore_thresh = option_find_float(options, "ignore_thresh", .5);
    l.truth_thresh = option_find_float(options, "truth_thresh", 1);
    l.random = option_find_int_quiet(options, "random", 0);

    a = option_find_str(options, "anchors", 0);
    if (a) {
        int len = strlen(a);
        int n = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (a[i] == ',') ++n;
        }
        for (i = 0; i < n && i < total * 2; ++i) {
            float bias = atof(a);
            l.biases[i] = bias;
            a = strchr(a, ',') + 1;
        }
    }
    return l;
}

// parser.c
softmax_layer parse_softmax(list *options, size_params params)
{
    int groups = option_find_int_quiet(options, "groups", 1);
    softmax_layer layer = make_softmax_layer(params.batch, params.inputs, groups);
    layer.temperature = option_find_float_quiet(options, "temperature", 1);
    char *tree_file = option_find_str(options, "tree", 0);
    if (tree_file) layer.softmax_tree = read_tree(tree_file);
    return layer;
}

// parser.c
maxpool_layer parse_maxpool(list *options, size_params params)
{
    int stride = option_find_int(options, "stride", 1);
    int size = option_find_int(options, "size", stride);
    int padding = option_find_int_quiet(options, "padding", size - 1);

    int batch, h, w, c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch = params.batch;
    if (!(h && w && c)) error("Layer before maxpool layer must output image.");

    maxpool_layer layer = make_maxpool_layer(batch, h, w, c, size, stride, padding);
    return layer;
}

// parser.c
layer parse_reorg(list *options, size_params params)
{
    int stride = option_find_int(options, "stride", 1);
    int reverse = option_find_int_quiet(options, "reverse", 0);

    int batch, h, w, c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch = params.batch;
    if (!(h && w && c)) error("Layer before reorg layer must output image.");

    layer layer = make_reorg_layer(batch, w, h, c, stride, reverse);
    return layer;
}

// parser.c
layer parse_upsample(list *options, size_params params, network net)
{

    int stride = option_find_int(options, "stride", 2);
    layer l = make_upsample_layer(params.batch, params.w, params.h, params.c, stride);
    l.scale = option_find_float_quiet(options, "scale", 1);
    return l;
}

// parser.c
layer parse_shortcut(list *options, size_params params, network net)
{
    char *l = option_find(options, "from");
    int index = atoi(l);
    if (index < 0) index = params.index + index;

    int batch = params.batch;
    layer from = net.layers[index];

    layer s = make_shortcut_layer(batch, index, params.w, params.h, params.c, from.out_w, from.out_h, from.out_c);

    char *activation_s = option_find_str(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);
    s.activation = activation;
    return s;
}

// parser.c
route_layer parse_route(list *options, size_params params, network net)
{
    char *l = option_find(options, "layers");
    int len = strlen(l);
    if (!l) error("Route Layer must specify input layers");
    int n = 1;
    int i;
    for (i = 0; i < len; ++i) {
        if (l[i] == ',') ++n;
    }

    int *layers = calloc(n, sizeof(int));
    int *sizes = calloc(n, sizeof(int));
    for (i = 0; i < n; ++i) {
        int index = atoi(l);
        l = strchr(l, ',') + 1;
        if (index < 0) index = params.index + index;
        layers[i] = index;
        sizes[i] = net.layers[index].outputs;
    }
    int batch = params.batch;

    route_layer layer = make_route_layer(batch, n, layers, sizes);

    convolutional_layer first = net.layers[layers[0]];
    layer.out_w = first.out_w;
    layer.out_h = first.out_h;
    layer.out_c = first.out_c;
    for (i = 1; i < n; ++i) {
        int index = layers[i];
        convolutional_layer next = net.layers[index];
        if (next.out_w == first.out_w && next.out_h == first.out_h) {
            layer.out_c += next.out_c;
        }
        else {
            layer.out_h = layer.out_w = layer.out_c = 0;
        }
    }

    return layer;
}

// parser.c
void free_section(section *s)
{
    free(s->type);
    node *n = s->options->front;
    while (n) {
        kvp *pair = (kvp *)n->val;
        free(pair->key);
        free(pair);
        node *next = n->next;
        free(n);
        n = next;
    }
    free(s->options);
    free(s);
}

// parser.c
LAYER_TYPE string_to_layer_type(char * type)
{
    if (strcmp(type, "[yolo]") == 0) return YOLO;
    if (strcmp(type, "[region]") == 0) return REGION;
    if (strcmp(type, "[conv]") == 0
        || strcmp(type, "[convolutional]") == 0) return CONVOLUTIONAL;
    if (strcmp(type, "[net]") == 0
        || strcmp(type, "[network]") == 0) return NETWORK;
    if (strcmp(type, "[max]") == 0
        || strcmp(type, "[maxpool]") == 0) return MAXPOOL;
    if (strcmp(type, "[reorg]") == 0) return REORG;
    if (strcmp(type, "[upsample]") == 0) return UPSAMPLE;
    if (strcmp(type, "[shortcut]") == 0) return SHORTCUT;
    if (strcmp(type, "[soft]") == 0
        || strcmp(type, "[softmax]") == 0) return SOFTMAX;
    if (strcmp(type, "[route]") == 0) return ROUTE;
    return BLANK;
}

// parser.c
learning_rate_policy get_policy(char *s)
{
    if (strcmp(s, "random") == 0) return RANDOM;
    if (strcmp(s, "poly") == 0) return POLY;
    if (strcmp(s, "constant") == 0) return CONSTANT;
    if (strcmp(s, "step") == 0) return STEP;
    if (strcmp(s, "exp") == 0) return EXP;
    if (strcmp(s, "sigmoid") == 0) return SIG;
    if (strcmp(s, "steps") == 0) return STEPS;
    fprintf(stderr, "Couldn't find policy %s, going with constant\n", s);
    return CONSTANT;
}

// parser.c
void parse_net_options(list *options, network *net)
{
    net->batch = option_find_int(options, "batch", 1);
    net->learning_rate = option_find_float(options, "learning_rate", .001);
    net->momentum = option_find_float(options, "momentum", .9);
    net->decay = option_find_float(options, "decay", .0001);
    int subdivs = option_find_int(options, "subdivisions", 1);
    net->time_steps = option_find_int_quiet(options, "time_steps", 1);
    net->batch /= subdivs;
    net->batch *= net->time_steps;
    net->subdivisions = subdivs;

    char *a = option_find_str(options, "input_calibration", 0);
    if (a) {
        int len = strlen(a);
        int n = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (a[i] == ',') ++n;
        }
        net->input_calibration_size = n;
        net->input_calibration = (float *)calloc(n, sizeof(float));
        for (i = 0; i < n; ++i) {
            float coef = atof(a);
            net->input_calibration[i] = coef;
            a = strchr(a, ',') + 1;
        }
    }

    net->adam = option_find_int_quiet(options, "adam", 0);
    if (net->adam) {
        net->B1 = option_find_float(options, "B1", .9);
        net->B2 = option_find_float(options, "B2", .999);
        net->eps = option_find_float(options, "eps", .000001);
    }

    net->h = option_find_int_quiet(options, "height", 0);
    net->w = option_find_int_quiet(options, "width", 0);
    net->c = option_find_int_quiet(options, "channels", 0);
    net->inputs = option_find_int_quiet(options, "inputs", net->h * net->w * net->c);
    net->max_crop = option_find_int_quiet(options, "max_crop", net->w * 2);
    net->min_crop = option_find_int_quiet(options, "min_crop", net->w);

    net->angle = option_find_float_quiet(options, "angle", 0);
    net->aspect = option_find_float_quiet(options, "aspect", 1);
    net->saturation = option_find_float_quiet(options, "saturation", 1);
    net->exposure = option_find_float_quiet(options, "exposure", 1);
    net->hue = option_find_float_quiet(options, "hue", 0);

    if (!net->inputs && !(net->h && net->w && net->c)) error("No input parameters supplied");

    char *policy_s = option_find_str(options, "policy", "constant");
    net->policy = get_policy(policy_s);
    net->burn_in = option_find_int_quiet(options, "burn_in", 0);
    if (net->policy == STEP) {
        net->step = option_find_int(options, "step", 1);
        net->scale = option_find_float(options, "scale", 1);
    }
    else if (net->policy == STEPS) {
        char *l = option_find(options, "steps");
        char *p = option_find(options, "scales");
        if (!l || !p) error("STEPS policy must have steps and scales in cfg file");

        int len = strlen(l);
        int n = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (l[i] == ',') ++n;
        }
        int *steps = calloc(n, sizeof(int));
        float *scales = calloc(n, sizeof(float));
        for (i = 0; i < n; ++i) {
            int step = atoi(l);
            float scale = atof(p);
            l = strchr(l, ',') + 1;
            p = strchr(p, ',') + 1;
            steps[i] = step;
            scales[i] = scale;
        }
        net->scales = scales;
        net->steps = steps;
        net->num_steps = n;
    }
    else if (net->policy == EXP) {
        net->gamma = option_find_float(options, "gamma", 1);
    }
    else if (net->policy == SIG) {
        net->gamma = option_find_float(options, "gamma", 1);
        net->step = option_find_int(options, "step", 1);
    }
    else if (net->policy == POLY || net->policy == RANDOM) {
        net->power = option_find_float(options, "power", 1);
    }
    net->max_batches = option_find_int(options, "max_batches", 0);
}

// parser.c
network parse_network_cfg(char *filename, int batch, int quantized)
{
    list *sections = read_cfg(filename);
    node *n = sections->front;
    if (!n) error("Config file has no sections");
    network net = make_network(sections->size - 1);
    net.quantized = quantized;
    net.do_input_calibration = 0;
    net.gpu_index = gpu_index;
    size_params params;
    params.quantized = quantized;

    section *s = (section *)n->val;
    list *options = s->options;
    if (strcmp(s->type, "[net]") == 0 && strcmp(s->type, "[network]") == 0)
        error("First section must be [net] or [network]");
    parse_net_options(options, &net);

    params.h = net.h;
    params.w = net.w;
    params.c = net.c;
    params.inputs = net.inputs;
    if (batch > 0) net.batch = batch;
    params.batch = net.batch;
    params.time_steps = net.time_steps;
    params.net = net;

    size_t workspace_size = 0;
    n = n->next;
    int count = 0;
    free_section(s);
    fprintf(stderr, "layer     filters    size              input                output\n");
    while (n) {
        params.index = count;
        fprintf(stderr, "%5d ", count);
        s = (section *)n->val;
        options = s->options;
        layer l = { 0 };
        LAYER_TYPE lt = string_to_layer_type(s->type);
        if (lt == CONVOLUTIONAL) {
            // if(count == 80) params.quantized = 0;    // doesn't lost GPU - mAP = 45.61%
            node *tmp = n->next;
            if(tmp) tmp = tmp->next;
            if (tmp)
            {
                if (string_to_layer_type(((section *)tmp->val)->type) == YOLO) {
                    params.quantized = 0;    // mAP = 53.60%
                    //printf("\n\n i = %d \n\n", count);
                }
            }

            l = parse_convolutional(options, params);
        }
        else if (lt == REGION) {
            l = parse_region(options, params);
        }
        else if (lt == YOLO) {
            l = parse_yolo(options, params);
        }
        else if (lt == SOFTMAX) {
            l = parse_softmax(options, params);
            net.hierarchy = l.softmax_tree;
        }
        else if (lt == MAXPOOL) {
            l = parse_maxpool(options, params);
        }
        else if (lt == REORG) {
            l = parse_reorg(options, params);
        }
        else if (lt == ROUTE) {
            l = parse_route(options, params, net);
        }
        else if (lt == UPSAMPLE) {
            l = parse_upsample(options, params, net);
        }
        else if (lt == SHORTCUT) {
            l = parse_shortcut(options, params, net);
        }
        else {
            fprintf(stderr, "Type not recognized: %s\n", s->type);
        }
        l.dontload = option_find_int_quiet(options, "dontload", 0);
        l.dontloadscales = option_find_int_quiet(options, "dontloadscales", 0);
        option_unused(options);
        net.layers[count] = l;
        if (l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        free_section(s);
        n = n->next;
        ++count;
        if (n) {
            params.h = l.out_h;
            params.w = l.out_w;
            params.c = l.out_c;
            params.inputs = l.outputs;
        }
    }
    free_list(sections);
    net.outputs = get_network_output_size(net);
    net.output = get_network_output(net);
    if (workspace_size) {
        //printf("%ld\n", workspace_size);
#ifdef GPU
        if (gpu_index >= 0) {
            net.workspace = cuda_make_array(0, (workspace_size - 1) / sizeof(float) + 1);
            int size = net.layers[0].inputs * net.batch;    //get_network_input_size(net) * net.batch;
            net.input_state_gpu = cuda_make_array(0, size);
        }
        else {
            net.workspace = calloc(1, workspace_size);
        }
#else    // GPU
        net.workspace = calloc(1, workspace_size);
#endif    // GPU

#ifdef OPENCL
        //if (gpu_index >= 0) {
        net.workspace_ocl = ocl_make_array(0, workspace_size / sizeof(float));
        //net.workspace_ocl = ocl_make_array(0, (workspace_size - 1) / sizeof(float) + 1);
        //net.workspace_ocl = ocl_make_array(NULL, 1024*1024*1024);
        //}
#endif    // OPENCL
    }
    return net;
}

// -------------- gettimeofday for Windows--------------------

#if defined(_MSC_VER)
int gettimeofday(struct timeval *tv, struct timezone *tz)
{
    FILETIME ft;
    unsigned __int64 tmpres = 0;
    static int tzflag;

    if (NULL != tv)
    {
        GetSystemTimeAsFileTime(&ft);

        tmpres |= ft.dwHighDateTime;
        tmpres <<= 32;
        tmpres |= ft.dwLowDateTime;

        /*converting file time to unix epoch*/
        tmpres -= DELTA_EPOCH_IN_MICROSECS;
        tmpres /= 10;  /*convert into microseconds*/
        tv->tv_sec = (long)(tmpres / 1000000UL);
        tv->tv_usec = (long)(tmpres % 1000000UL);
    }

    if (NULL != tz)
    {
        if (!tzflag)
        {
            _tzset();
            tzflag++;
        }
        tz->tz_minuteswest = _timezone / 60;
        tz->tz_dsttime = _daylight;
    }

    return 0;
}
#endif    // _MSC_VER




// ------------------------------------------------------
// Calculate mAP and TP/FP/FN, IoU, F1

#include "pthread.h"
//#include "box.h"

/*
// from: box.h
typedef struct {
    float x, y, w, h;
} box;
*/

float box_iou(box a, box b);

typedef enum {
    CLASSIFICATION_DATA, DETECTION_DATA, CAPTCHA_DATA, REGION_DATA, IMAGE_DATA, LETTERBOX_DATA, COMPARE_DATA, WRITING_DATA, SWAG_DATA, TAG_DATA, OLD_CLASSIFICATION_DATA, STUDY_DATA, DET_DATA, SUPER_DATA
} data_type;

typedef struct matrix {
    int rows, cols;
    float **vals;
} matrix;

typedef struct {
    int w, h;
    matrix X;
    matrix y;
    int shallow;
    int *num_boxes;
    box **boxes;
} data;

typedef struct {
    int id;
    float x, y, w, h;
    float left, right, top, bottom;
} box_label;

typedef struct load_args {
    int threads;
    char **paths;
    char *path;
    int n;
    int m;
    char **labels;
    int h;
    int w;
    int out_w;
    int out_h;
    int nh;
    int nw;
    int num_boxes;
    int min, max, size;
    int classes;
    int background;
    int scale;
    int small_object;
    float jitter;
    int flip;
    float angle;
    float aspect;
    float saturation;
    float exposure;
    float hue;
    data *d;
    image *im;
    image *resized;
    data_type type;
    tree *hierarchy;
} load_args;

int entry_index(layer l, int batch, int location, int entry)
{
    int n = location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(4 + l.classes + 1) + entry*l.w*l.h + loc;
}

int yolo_num_detections(layer l, float thresh)
{
    int i, n;
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i) {
        for (n = 0; n < l.n; ++n) {
            int obj_index = entry_index(l, 0, n*l.w*l.h + i, 4);
            if (l.output[obj_index] > thresh) {
                ++count;
            }
        }
    }
    return count;
}

int num_detections(network *net, float thresh)
{
    int i;
    int s = 0;
    for (i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        if (l.type == YOLO) {
            s += yolo_num_detections(l, thresh);
        }
        if (l.type == DETECTION || l.type == REGION) {
            s += l.w*l.h*l.n;
        }
    }
    return s;
}

detection *make_network_boxes(network *net, float thresh, int *num)
{
    layer l = net->layers[net->n - 1];
    int i;
    int nboxes = num_detections(net, thresh);
    if (num) *num = nboxes;
    detection *dets = calloc(nboxes, sizeof(detection));
    for (i = 0; i < nboxes; ++i) {
        dets[i].prob = calloc(l.classes, sizeof(float));
        if (l.coords > 4) {
            dets[i].mask = calloc(l.coords - 4, sizeof(float));
        }
    }
    return dets;
}


void free_detections(detection *dets, int n)
{
    int i;
    for (i = 0; i < n; ++i) {
        free(dets[i].prob);
        if (dets[i].mask) free(dets[i].mask);
    }
    free(dets);
}

void find_replace(char *str, char *orig, char *rep, char *output)
{
    char buffer[4096] = { 0 };
    char *p;

    sprintf(buffer, "%s", str);
    if (!(p = strstr(buffer, orig))) {  // Is 'orig' even in 'str'?
        sprintf(output, "%s", str);
        return;
    }

    *p = '\0';

    sprintf(output, "%s%s%s", buffer, rep, p + strlen(orig));
}

void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative, int letter)
{
    int i;
    int new_w = 0;
    int new_h = 0;
    if (letter) {
        if (((float)netw / w) < ((float)neth / h)) {
            new_w = netw;
            new_h = (h * netw) / w;
        }
        else {
            new_h = neth;
            new_w = (w * neth) / h;
        }
    }
    else {
        new_w = netw;
        new_h = neth;
    }
    for (i = 0; i < n; ++i) {
        box b = dets[i].bbox;
        b.x = (b.x - (netw - new_w) / 2. / netw) / ((float)new_w / netw);
        b.y = (b.y - (neth - new_h) / 2. / neth) / ((float)new_h / neth);
        b.w *= (float)netw / new_w;
        b.h *= (float)neth / new_h;
        if (!relative) {
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

// yolo_layer.c
box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0 * stride]) / lw;
    b.y = (j + x[index + 1 * stride]) / lh;
    b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
    b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
    return b;
}

// yolo_layer.c
int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets, int letter)
{
    int i, j, n;
    float *predictions = l.output;
    //if (l.batch == 2) avg_flipped_yolo(l);
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i) {
        int row = i / l.w;
        int col = i % l.w;
        for (n = 0; n < l.n; ++n) {
            int obj_index = entry_index(l, 0, n*l.w*l.h + i, 4);
            float objectness = predictions[obj_index];
            if (objectness <= thresh) continue;
            int box_index = entry_index(l, 0, n*l.w*l.h + i, 0);
            dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h);
            dets[count].objectness = objectness;
            dets[count].classes = l.classes;
            for (j = 0; j < l.classes; ++j) {
                int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
                float prob = objectness*predictions[class_index];
                dets[count].prob[j] = (prob > thresh) ? prob : 0;
            }
            ++count;
        }
    }
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative, letter);
    return count;
}

// get prediction boxes: yolov2_forward_network.c
void get_region_boxes_cpu(layer l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness, int *map);


void custom_get_region_detections(layer l, int w, int h, int net_w, int net_h, float thresh, int *map, float hier, int relative, detection *dets, int letter)
{
    box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
    float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
    int i, j;
    for (j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
    get_region_boxes_cpu(l, 1, 1, thresh, probs, boxes, 0, map);
    for (j = 0; j < l.w*l.h*l.n; ++j) {
        dets[j].classes = l.classes;
        dets[j].bbox = boxes[j];
        dets[j].objectness = 1;
        for (i = 0; i < l.classes; ++i) {
            dets[j].prob[i] = probs[j][i];
        }
    }

    free(boxes);
    free_ptrs((void **)probs, l.w*l.h*l.n);

    //correct_region_boxes(dets, l.w*l.h*l.n, w, h, net_w, net_h, relative);
    correct_yolo_boxes(dets, l.w*l.h*l.n, w, h, net_w, net_h, relative, letter);
}

void fill_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, detection *dets, int letter)
{
    int j;
    for (j = 0; j < net->n; ++j) {
        layer l = net->layers[j];
        if (l.type == YOLO) {
            int count = get_yolo_detections(l, w, h, net->w, net->h, thresh, map, relative, dets, letter);
            dets += count;
        }
        if (l.type == REGION) {
            custom_get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets, letter);
            //get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets);
            dets += l.w*l.h*l.n;
        }
    }
}

detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num, int letter)
{
    detection *dets = make_network_boxes(net, thresh, num);
    fill_network_boxes(net, w, h, thresh, hier, map, relative, dets, letter);
    return dets;
}

void *load_thread(void *ptr)
{
    load_args a = *(struct load_args*)ptr;
    if (a.type == IMAGE_DATA) {
        *(a.im) = load_image(a.path, 0, 0, 3);
        *(a.resized) = resize_image(*(a.im), a.w, a.h);
        //printf(" a.path = %s, a.w = %d, a.h = %d \n", a.path, a.w, a.h);
    }
    else if (a.type == LETTERBOX_DATA) {
        printf(" LETTERBOX_DATA isn't implemented \n");
        getchar();
        //*(a.im) = load_image(a.path, 0, 0, 0);
        //*(a.resized) = letterbox_image(*(a.im), a.w, a.h);
    }
    else {
        printf("unknown DATA type = %d \n", a.type);
        getchar();
    }
    free(ptr);
    return 0;
}

pthread_t load_data_in_thread(load_args args)
{
    pthread_t thread;
    struct load_args *ptr = calloc(1, sizeof(struct load_args));
    *ptr = args;
    if (pthread_create(&thread, 0, load_thread, ptr)) error("Thread creation failed");
    return thread;
}

box_label *read_boxes(char *filename, int *n)
{
    box_label *boxes = calloc(1, sizeof(box_label));
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        //file_error(filename);
        *n = 0;
        return boxes;
    }
    float x, y, h, w;
    int id;
    int count = 0;
    while (fscanf(file, "%d %f %f %f %f", &id, &x, &y, &w, &h) == 5) {
        boxes = realloc(boxes, (count + 1) * sizeof(box_label));
        boxes[count].id = id;
        boxes[count].x = x;
        boxes[count].y = y;
        boxes[count].h = h;
        boxes[count].w = w;
        boxes[count].left = x - w / 2;
        boxes[count].right = x + w / 2;
        boxes[count].top = y - h / 2;
        boxes[count].bottom = y + h / 2;
        ++count;
    }
    fclose(file);
    *n = count;
    return boxes;
}

typedef struct {
    box b;
    float p;
    int class_id;
    int image_index;
    int truth_flag;
    int unique_truth_index;
} box_prob;

int detections_comparator(const void *pa, const void *pb)
{
    box_prob a = *(box_prob *)pa;
    box_prob b = *(box_prob *)pb;
    float diff = a.p - b.p;
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}

int nms_comparator_v3(const void *pa, const void *pb)
{
    detection a = *(detection *)pa;
    detection b = *(detection *)pb;
    float diff = 0;
    if (b.sort_class >= 0) {
        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
    }
    else {
        diff = a.objectness - b.objectness;
    }
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}

void do_nms_sort_v3(detection *dets, int total, int classes, float thresh)
{
    int i, j, k;
    k = total - 1;
    for (i = 0; i <= k; ++i) {
        if (dets[i].objectness == 0) {
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k + 1;

    for (k = 0; k < classes; ++k) {
        for (i = 0; i < total; ++i) {
            dets[i].sort_class = k;
        }
        qsort(dets, total, sizeof(detection), nms_comparator_v3);
        for (i = 0; i < total; ++i) {
            //printf("  k = %d, \t i = %d \n", k, i);
            if (dets[i].prob[k] == 0) continue;
            box a = dets[i].bbox;
            for (j = i + 1; j < total; ++j) {
                box b = dets[j].bbox;
                if (box_iou(a, b) > thresh) {
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}

void validate_detector_map(char *datacfg, char *cfgfile, char *weightfile, float thresh_calc_avg_iou, int quantized)
{
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.txt");
    char *difficult_valid_images = option_find_str(options, "difficult", NULL);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);

    network net = parse_network_cfg(cfgfile, 1, quantized);
    //parse_network_cfg_custom(cfgfile, 1);    // set batch=1
    if (weightfile) {
        load_weights_upto_cpu(&net, weightfile, net.n);
    }
    //set_batch_network(&net, 1);
    yolov2_fuse_conv_batchnorm(net);
    calculate_binary_weights(net);
    if (quantized) quantinization_and_get_multipliers(net);
    srand(time(0));

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    char **paths_dif = NULL;
    if (difficult_valid_images) {
        list *plist_dif = get_paths(difficult_valid_images);
        paths_dif = (char **)list_to_array(plist_dif);
    }


    layer l = net.layers[net.n - 1];
    int classes = l.classes;

    int m = plist->size;
    int i = 0;
    int t;

    const float thresh = .005;
    float nms = .45;
    const float iou_thresh = 0.5;

    int nthreads = 4;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = { 0 };
    args.w = net.w;
    args.h = net.h;
    args.type = IMAGE_DATA;
    //args.type = LETTERBOX_DATA;

    //const float thresh_calc_avg_iou = 0.24;
    float avg_iou = 0;
    int tp_for_thresh = 0;
    int fp_for_thresh = 0;

    box_prob *detections = calloc(1, sizeof(box_prob));
    int detections_count = 0;
    int unique_truth_count = 0;

    int *truth_classes_count = calloc(classes, sizeof(int));

    for (t = 0; t < nthreads; ++t) {
        args.path = paths[i + t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
    for (i = nthreads; i < m + nthreads; i += nthreads) {
        fprintf(stderr, "%d\n", i);
        for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for (t = 0; t < nthreads && i + t < m; ++t) {
            args.path = paths[i + t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for (t = 0; t < nthreads && i + t - nthreads < m; ++t)
        {
#ifndef WIN32
            //if (!quantized) usleep(50000);
                usleep(50000); // 0.005 sec
#endif
            const int image_index = i + t - nthreads;
            char *path = paths[image_index];
            //char *id = basecfg(path);
            float *X = val_resized[t].data;
            //network_predict(net, X);
#ifdef GPU
            if (quantized) {
                network_predict_gpu_cudnn_quantized(net, X);    // quantized
                //nms = 0.2;
            }
            else {
                network_predict_gpu_cudnn(net, X);
            }
#else    // GPU
#ifdef OPENCL
            network_predict_opencl(net, X);
#else    // OPENCL
            if (quantized) {
                network_predict_quantized(net, X);    // quantized
                //nms = 0.2;
            }
            else {
                network_predict_cpu(net, X);
            }
#endif    // OPENCL
#endif    // GPU

            int nboxes = 0;
            int letterbox = (args.type == LETTERBOX_DATA);
            float hier_thresh = 0;
            detection *dets = get_network_boxes(&net, 1, 1, thresh, hier_thresh, 0, 0, &nboxes, letterbox);
            //detection *dets = get_network_boxes(&net, val[t].w, val[t].h, thresh, hier_thresh, 0, 1, &nboxes, letterbox); // for letterbox=1
            if (nms) do_nms_sort_v3(dets, nboxes, l.classes, nms);

            char labelpath[4096];
            find_replace(path, "images", "labels", labelpath);
            find_replace(labelpath, "JPEGImages", "labels", labelpath);
            find_replace(labelpath, ".jpg", ".txt", labelpath);
            find_replace(labelpath, ".png", ".txt", labelpath);
            find_replace(labelpath, ".bmp", ".txt", labelpath);
            find_replace(labelpath, ".JPG", ".txt", labelpath);
            find_replace(labelpath, ".JPEG", ".txt", labelpath);
            int num_labels = 0;
            box_label *truth = read_boxes(labelpath, &num_labels);
            //printf(" labelpath = %s \n", labelpath);
            int i, j;
            for (j = 0; j < num_labels; ++j) {
                truth_classes_count[truth[j].id]++;
            }

            // difficult
            box_label *truth_dif = NULL;
            int num_labels_dif = 0;
            if (paths_dif)
            {
                char *path_dif = paths_dif[image_index];

                char labelpath_dif[4096];
                find_replace(path_dif, "images", "labels", labelpath_dif);
                find_replace(labelpath_dif, "JPEGImages", "labels", labelpath_dif);
                find_replace(labelpath_dif, ".jpg", ".txt", labelpath_dif);
                find_replace(labelpath_dif, ".JPEG", ".txt", labelpath_dif);
                find_replace(labelpath_dif, ".png", ".txt", labelpath_dif);
                truth_dif = read_boxes(labelpath_dif, &num_labels_dif);
            }

            const int checkpoint_detections_count = detections_count;

            for (i = 0; i < nboxes; ++i) {

                int class_id;
                for (class_id = 0; class_id < classes; ++class_id) {
                    float prob = dets[i].prob[class_id];
                    if (prob > 0) {
                        detections_count++;
                        detections = realloc(detections, detections_count * sizeof(box_prob));
                        detections[detections_count - 1].b = dets[i].bbox;
                        detections[detections_count - 1].p = prob;
                        detections[detections_count - 1].image_index = image_index;
                        detections[detections_count - 1].class_id = class_id;
                        detections[detections_count - 1].truth_flag = 0;
                        detections[detections_count - 1].unique_truth_index = -1;

                        int truth_index = -1;
                        float max_iou = 0;
                        for (j = 0; j < num_labels; ++j)
                        {
                            box t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h };
                            //printf(" IoU = %f, prob = %f, class_id = %d, truth[j].id = %d \n",
                            //    box_iou(dets[i].bbox, t), prob, class_id, truth[j].id);
                            float current_iou = box_iou(dets[i].bbox, t);
                            if (current_iou > iou_thresh && class_id == truth[j].id) {
                                if (current_iou > max_iou) {
                                    max_iou = current_iou;
                                    truth_index = unique_truth_count + j;
                                }
                            }
                        }

                        // best IoU
                        if (truth_index > -1) {
                            detections[detections_count - 1].truth_flag = 1;
                            detections[detections_count - 1].unique_truth_index = truth_index;
                        }
                        else {
                            // if object is difficult then remove detection
                            for (j = 0; j < num_labels_dif; ++j) {
                                box t = { truth_dif[j].x, truth_dif[j].y, truth_dif[j].w, truth_dif[j].h };
                                float current_iou = box_iou(dets[i].bbox, t);
                                if (current_iou > iou_thresh && class_id == truth_dif[j].id) {
                                    --detections_count;
                                    break;
                                }
                            }
                        }

                        // calc avg IoU, true-positives, false-positives for required Threshold
                        if (prob > thresh_calc_avg_iou) {
                            int z, found = 0;
                            for (z = checkpoint_detections_count; z < detections_count - 1; ++z)
                                if (detections[z].unique_truth_index == truth_index) {
                                    found = 1; break;
                                }

                            if (truth_index > -1 && found == 0) {
                                avg_iou += max_iou;
                                ++tp_for_thresh;
                            }
                            else
                                fp_for_thresh++;
                        }
                    }
                }
            }

            unique_truth_count += num_labels;

            free_detections(dets, nboxes);
            //free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }

    if ((tp_for_thresh + fp_for_thresh) > 0) {
        avg_iou = avg_iou / (tp_for_thresh + fp_for_thresh);
    }


    // SORT(detections)
    qsort(detections, detections_count, sizeof(box_prob), detections_comparator);

    typedef struct {
        double precision;
        double recall;
        int tp, fp, fn;
    } pr_t;

    // for PR-curve
    pr_t **pr = calloc(classes, sizeof(pr_t*));
    for (i = 0; i < classes; ++i) {
        pr[i] = calloc(detections_count, sizeof(pr_t));
    }
    printf("detections_count = %d, unique_truth_count = %d  \n", detections_count, unique_truth_count);


    int *truth_flags = calloc(unique_truth_count, sizeof(int));

    int rank;
    for (rank = 0; rank < detections_count; ++rank) {
        if (rank % 100 == 0)
            printf(" rank = %d of ranks = %d \r", rank, detections_count);

        if (rank > 0) {
            int class_id;
            for (class_id = 0; class_id < classes; ++class_id) {
                pr[class_id][rank].tp = pr[class_id][rank - 1].tp;
                pr[class_id][rank].fp = pr[class_id][rank - 1].fp;
            }
        }

        box_prob d = detections[rank];
        // if (detected && isn't detected before)
        if (d.truth_flag == 1) {
            if (truth_flags[d.unique_truth_index] == 0)
            {
                truth_flags[d.unique_truth_index] = 1;
                pr[d.class_id][rank].tp++;    // true-positive
            }
        }
        else {
            pr[d.class_id][rank].fp++;    // false-positive
        }

        for (i = 0; i < classes; ++i)
        {
            const int tp = pr[i][rank].tp;
            const int fp = pr[i][rank].fp;
            const int fn = truth_classes_count[i] - tp;    // false-negative = objects - true-positive
            pr[i][rank].fn = fn;

            if ((tp + fp) > 0) pr[i][rank].precision = (double)tp / (double)(tp + fp);
            else pr[i][rank].precision = 0;

            if ((tp + fn) > 0) pr[i][rank].recall = (double)tp / (double)(tp + fn);
            else pr[i][rank].recall = 0;
        }
    }

    free(truth_flags);


    double mean_average_precision = 0;

    for (i = 0; i < classes; ++i) {
        double avg_precision = 0;
        int point;
        for (point = 0; point < 11; ++point) {
            double cur_recall = point * 0.1;
            double cur_precision = 0;
            for (rank = 0; rank < detections_count; ++rank)
            {
                if (pr[i][rank].recall >= cur_recall) {    // > or >=
                    if (pr[i][rank].precision > cur_precision) {
                        cur_precision = pr[i][rank].precision;
                    }
                }
            }
            //printf("class_id = %d, point = %d, cur_recall = %.4f, cur_precision = %.4f \n", i, point, cur_recall, cur_precision);

            avg_precision += cur_precision;
        }
        avg_precision = avg_precision / 11;
        printf("class_id = %d, name = %s, \t ap = %2.2f %% \n", i, names[i], avg_precision * 100);
        mean_average_precision += avg_precision;
    }

    const float cur_precision = (float)tp_for_thresh / ((float)tp_for_thresh + (float)fp_for_thresh);
    const float cur_recall = (float)tp_for_thresh / ((float)tp_for_thresh + (float)(unique_truth_count - tp_for_thresh));
    const float f1_score = 2.F * cur_precision * cur_recall / (cur_precision + cur_recall);
    printf(" for thresh = %1.2f, precision = %1.2f, recall = %1.2f, F1-score = %1.2f \n",
        thresh_calc_avg_iou, cur_precision, cur_recall, f1_score);

    printf(" for thresh = %0.2f, TP = %d, FP = %d, FN = %d, average IoU = %2.2f %% \n",
        thresh_calc_avg_iou, tp_for_thresh, fp_for_thresh, unique_truth_count - tp_for_thresh, avg_iou * 100);

    mean_average_precision = mean_average_precision / classes;
    printf("\n mean average precision (mAP) = %f, or %2.2f %% \n", mean_average_precision, mean_average_precision * 100);


    for (i = 0; i < classes; ++i) {
        free(pr[i]);
    }
    free(pr);
    free(detections);
    free(truth_classes_count);

    fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));

    //getchar();
}



void validate_calibrate_valid(char *datacfg, char *cfgfile, char *weightfile, int input_calibration)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.txt");
    printf("valid=%s \n", valid_images);

    network net = parse_network_cfg(cfgfile, 1, 0);    // batch=1, quantized=0

    if (!input_calibration) {
        printf("\n -input_calibration <number> - isn't specified in command line, will be used 1000 images \n\n");
        input_calibration = 1000;
    }
    net.do_input_calibration = input_calibration;

    if (weightfile) {
        load_weights_upto_cpu(&net, weightfile, net.n);
    }
    //set_batch_network(&net, 1);
    yolov2_fuse_conv_batchnorm(net);
    srand(time(0));

#ifdef GPU
    size_t workspace_size = 0;
    for (j = 0; j < net.n; ++j) {
        layer l = net.layers[j];
        size_t cur_workspace_size = (size_t)l.out_h*l.out_w*l.size*l.size*l.c * sizeof(float);
        if (cur_workspace_size > workspace_size) workspace_size = cur_workspace_size;
    }
    cudaFree(net.workspace);
    net.workspace = calloc(1, workspace_size);
#endif // GPU

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n - 1];
    int classes = l.classes;

    int m = plist->size;
    int i = 0;
    int t;

    const float thresh = .005;
    float nms = .45;

    int nthreads = 4;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = { 0 };
    args.w = net.w;
    args.h = net.h;
    args.type = IMAGE_DATA;
    //args.type = LETTERBOX_DATA;

    //const float thresh_calc_avg_iou = 0.24;
    float avg_iou = 0;
    int tp_for_thresh = 0;
    int fp_for_thresh = 0;

    box_prob *detections = calloc(1, sizeof(box_prob));
    int detections_count = 0;
    int unique_truth_count = 0;

    int *truth_classes_count = calloc(classes, sizeof(int));

    for (t = 0; t < nthreads; ++t) {
        args.path = paths[i + t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
    for (i = nthreads; i < m + nthreads; i += nthreads) {
        fprintf(stderr, "%d\n", i);
        for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for (t = 0; t < nthreads && i + t < m; ++t) {
            args.path = paths[i + t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
            const int image_index = i + t - nthreads;
            char *path = paths[image_index];
            //char *id = basecfg(path);
            float *X = val_resized[t].data;

            network_calibrate_cpu(net, X);
        }
    }
}