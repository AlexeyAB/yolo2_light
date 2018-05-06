#include "additionally.h"	// some definitions from: im2col.h, blas.h, list.h, utils.h, activations.h, tree.h, layer.h, network.h
							// softmax_layer.h, reorg_layer.h, route_layer.h, region_layer.h, maxpool_layer.h, convolutional_layer.h

#define GEMMCONV



#define W_MAX_VAL (256/2 - 1)	// 7-bit (1-bit sign)
#define I_MAX_VAL (256/2 - 1)	// 7-bit (1-bit sign)
#define R_MAX_VAL (256*256/2 - 1)	// 31-bit (1-bit sign)


#define R_MULT (32)	// 4 - 32

// from: box.h
typedef struct {
	float x, y, w, h;
} box;


int max_abs(int src, int max_val) 
{
	if (abs(src) > abs(max_val)) src = (src > 0)?max_val:-max_val;
	return src;
}

short int max_abs_short(short int src, short int max_val)
{
	if (abs(src) > abs(max_val)) src = (src > 0) ? max_val : -max_val;
	return src;
}

int * get_distribution(float *arr_ptr, int arr_size, int number_of_ranges, float start_range)
{
	//const int number_of_ranges = 32;
	//const float start_range = 1.F / 65536;
	int *count = calloc(number_of_ranges, sizeof(int));
	float min_val = 10000, max_val = 0;

	int i, j;
	for (i = 0; i < arr_size; ++i) {
		float w = arr_ptr[i];

		float cur_range = start_range;
		for (j = 0; j < number_of_ranges; ++j) {
			if (fabs(cur_range) <= w && w < fabs(cur_range * 2))
				count[j]++;// , printf("found \n");
			cur_range *= 2;
			//printf("%f, ", w);
		}
	}

	return count;
}


float get_multiplier(float *arr_ptr, int arr_size, int bits_length)
{
	const int number_of_ranges = 32;
	const float start_range = 1.F / 65536;

	int i, j;
	int *count = get_distribution(arr_ptr, arr_size, number_of_ranges, start_range);

	int max_count_range = 0;
	int index_max_count = 0;
	for (j = 0; j < number_of_ranges; ++j) {
		int counter = 0;
		for (i = j; i < (j + bits_length) && i < number_of_ranges; ++i)
		{
			counter += count[i];
			//counter += log2(count[i]);
		}
		if (max_count_range < counter) {
			max_count_range = counter;
			index_max_count = j;
		}
	}
	//index_max_count = index_max_count + 2;	// optimal shift multipler
	float multiplier = 1 / (start_range * powf(2., (float)index_max_count));
	//printf(" max_count_range = %d, index_max_count = %d, multiplier = %g \n", 
	//	max_count_range, index_max_count, multiplier);
	free(count);
	return multiplier;
}

#ifdef OPENCV
#include <opencv2/core/fast_math.hpp>
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/core/core_c.h"
#include "opencv2/core/version.hpp"

void draw_distribution(float *arr_ptr, int arr_size, char *name)
{
	int img_w = 1200, img_h = 800;
	const int number_of_ranges = 32;
	const float start_range = 1.F / 65536;
	//int *count = calloc(number_of_ranges, sizeof(int));
	//float min_val = 100, max_val = 0;

	int i, j;
	int *count = get_distribution(arr_ptr, arr_size, number_of_ranges, start_range);

	float multiplier = get_multiplier(arr_ptr, arr_size, 8);

	int max_count_range = 0;
	for (j = 0; j < number_of_ranges; ++j) {
		count[j] = log2(count[j]);
		if (max_count_range < count[j])
			max_count_range = count[j];
	}

	cvNamedWindow("Wights", CV_WINDOW_NORMAL);
	cvResizeWindow("Wights", img_w, img_h);

	IplImage *img = cvCreateImage(cvSize(img_w, img_h), IPL_DEPTH_8U, 3);

	if (max_count_range > 0) {
		for (j = 0; j < number_of_ranges; ++j) {
			//printf("count[j] = %d, max_count_range = %d, img_w = %d, img_h = %d, j = %d, number_of_ranges = %d \n",
			//	count[j], max_count_range, img_w, img_h, j, number_of_ranges);
			CvPoint pt1, pt2;
			pt1.x = j*img_w / number_of_ranges;
			pt2.x = (j + 1)*img_w / number_of_ranges;
			pt1.y = img_h;
			pt2.y = img_h - img_h*count[j] / max_count_range;
			//printf("pt1.x = %d, pt1.y = %d, pt2.x = %d, pt2.y = %d \n", pt1.x, pt1.y, pt2.x, pt2.y);

			//if(pt2.y < pt1.y)
			cvRectangle(img, pt1, pt2, CV_RGB(128, 64, 32), CV_FILLED, 8, 0);
			cvRectangle(img, pt1, pt2, CV_RGB(32, 32, 32), 1, 8, 0);
		}
	}

	int index_multiplier = log2(1 / (multiplier*start_range));
	int x_coord_multiplier = index_multiplier*img_w / number_of_ranges;
	cvLine(img, cvPoint(x_coord_multiplier, 0), cvPoint(x_coord_multiplier, img_h), CV_RGB(255, 32, 32), 1, 8, 0);

	char buff[256];
	//sprintf(buff, "[%g - %g]", min_val, max_val);
	sprintf(buff, "optimal multiplier = %g", multiplier);
	//printf("[%g - %g]", min_val, max_val);
	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX, 1, 1, 0, 2, 8);
	cvPutText(img, buff, cvPoint(100, 50), &font, CV_RGB(32, 64, 128));
	
	if(name)
		cvPutText(img, name, cvPoint(0, 50), &font, CV_RGB(32, 64, 128));

	float cur_range = start_range;
	cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX, 0.5, 0.5, 0, 1, 8);
	for (j = 0; j < number_of_ranges; ++j) {
		CvPoint pt_text = cvPoint(j*img_w / number_of_ranges, img_h - 50);
		int lg = log2(cur_range);
		sprintf(buff, "%d", lg);
		cvPutText(img, buff, pt_text, &font, CV_RGB(32, 64, 128));
		cur_range *= 2;
	}
	cvPutText(img, "X and Y are log2", cvPoint(img_w / 2 - 100, img_h - 10), &font, CV_RGB(32, 64, 128));
	
	cvShowImage("Wights", img);
	cvWaitKey(0);

	free(count);
}
#endif // OPENCV

// im2col.c
int8_t im2col_get_pixel_int8(int8_t *im, int height, int width, int channels,
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
void im2col_cpu_int8(int8_t* data_im,
	int channels, int height, int width,
	int ksize, int stride, int pad, int8_t* data_col)
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
				data_col[col_index] = im2col_get_pixel_int8(data_im, height, width, channels,
					im_row, im_col, c_im, pad);
			}
		}
	}
}

// Use to enable AVX or SSE41
//#define AVX	// 1.35 sec (0.8 FPS) 2.3x - GCC -mavx -mavx2 -mfma -ffp-contract=fast
//#define SSE41	// 1.55 sec (0.7 FPS) 2x
// default 3.10 sec (0.3 FPS)


#if defined(AVX) || defined(SSE41)

#ifdef _WIN64
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#include <ammintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <emmintrin.h>
// https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=broad&expand=561
#endif	// AVX or SSE41


#if defined(AVX)


inline __m256i _mm256_div_epi16(const __m256i va, const int b)
{
	__m256i vb = _mm256_set1_epi16(32768 / b);
	return _mm256_mulhrs_epi16(va, vb);
}

#define INTERMEDIATE_MULT 15	// 8 or 15
#define FINAL_MULT (R_MULT / INTERMEDIATE_MULT)

// 0.89 sec
void gemm_nn_int8_int16_conv16(int M, int N, int K, int8_t ALPHA,
	int8_t *A, int lda,
	int8_t *B, int ldb,
	int16_t *C, int ldc)
{
	__m256i res;
	__m256i a, b, d;
	__m128i tmp128;
	__m256i div256 = _mm256_set1_epi16(INTERMEDIATE_MULT);

	int16_t *c_tmp = calloc(N, sizeof(int16_t));
	int i, j, k;
	for (i = 0; i < M; ++i) {
		for (k = 0; k < K; ++k) {
			register int16_t A_PART = ALPHA*A[i*lda + k];
			a = _mm256_set1_epi16(A_PART);
			for (j = 0; j < N - 32; j += 32) {
				int index = k*ldb + j;
				d = _mm256_loadu_si256((__m256i*)&B[index]);


				tmp128 = _mm256_extractf128_si256(d, 0);// get low 128 bit
				b = _mm256_cvtepi8_epi16(tmp128);		// int8 -> int16	

				b = _mm256_mullo_epi16(a, b);	// B = A * B

				b = _mm256_div_epi16(b, INTERMEDIATE_MULT);	// B = (A * B) / INTERMEDIATE_MULL

				res = _mm256_loadu_si256(&c_tmp[j]);		// load temp C
				res = _mm256_add_epi16(b, res);				// (A*B) + C
				_mm256_storeu_si256(&c_tmp[j], res);		// store temp C


				tmp128 = _mm256_extractf128_si256(d, 1);// get high 128 bit
				b = _mm256_cvtepi8_epi16(tmp128);		// int8 -> int16 (for low 8 bytes)

				b = _mm256_mullo_epi16(a, b);	// B = A * B

				b = _mm256_div_epi16(b, INTERMEDIATE_MULT);	// B = (A * B) / INTERMEDIATE_MULL

				res = _mm256_loadu_si256(&c_tmp[j + 16]);	// Load next temp C
				res = _mm256_add_epi16(b, res);				// (A*B) + C
				_mm256_storeu_si256(&c_tmp[j + 16], res);	// store temp C

				//c_tmp[j] += A_PART*B[k*ldb + j];
				//C[i*ldc + j] += max_abs(A_PART*B[k*ldb + j] / (INTERMEDIATE_MULL), (256 * 128 - 1));
			}

			int prev_end = (N % 32 == 0) ? (N - 32) : (N / 32) * 32;
			for (j = prev_end; j < N; ++j) {
				c_tmp[j] += A_PART*B[k*ldb + j] / (INTERMEDIATE_MULT);
			}
		}
		for (j = 0; j < N; ++j) {
			C[i*ldc + j] += c_tmp[j] / FINAL_MULT;
			c_tmp[j] = 0;
		}
	}
	free(c_tmp);
}


// 1.15 sec
void gemm_nn_int8_int16(int M, int N, int K, int8_t ALPHA,
	int8_t *A, int lda,
	int8_t *B, int ldb,
	int16_t *C, int ldc)
{
	__m256i multyplied_i32, res;
	__m256i a, b, d;
	__m128i tmp128;

	int32_t *c_tmp = calloc(N, sizeof(int32_t));
	int i, j, k;
	for (i = 0; i < M; ++i) {
		for (k = 0; k < K; ++k) {
			register int16_t A_PART = ALPHA*A[i*lda + k];
			a = _mm256_set1_epi16(A_PART);
			for (j = 0; j < N - 32; j += 32) {
				int index = k*ldb + j;
				d = _mm256_loadu_si256((__m256i*)&B[index]);

				tmp128 = _mm256_extractf128_si256(d, 0);// get low 128 bit
				b = _mm256_cvtepi8_epi16(tmp128);		// int8 -> int16	

				b = _mm256_mullo_epi16(a, b);	// B = A * B

				tmp128 = _mm256_extractf128_si256(b, 0);		// get low 128 bit
				multyplied_i32 = _mm256_cvtepi16_epi32(tmp128);	// int16 -> int32

				res = _mm256_loadu_si256(&c_tmp[j]);		// load temp C
				res = _mm256_add_epi32(multyplied_i32, res);// (A*B) + C
				_mm256_storeu_si256(&c_tmp[j], res);		// store temp C

				tmp128 = _mm256_extractf128_si256(b, 1);		// get high 128 bit
				multyplied_i32 = _mm256_cvtepi16_epi32(tmp128);	// int16 -> int32

				res = _mm256_loadu_si256(&c_tmp[j + 8]);	// Load next temp C
				res = _mm256_add_epi32(multyplied_i32, res);// (A*B) + C
				_mm256_storeu_si256(&c_tmp[j + 8], res);	// store temp C

				tmp128 = _mm256_extractf128_si256(d, 1);// get high 128 bit
				b = _mm256_cvtepi8_epi16(tmp128);		// int8 -> int16 (for low 8 bytes)

				b = _mm256_mullo_epi16(a, b);	// B = A * B

				tmp128 = _mm256_extractf128_si256(b, 0);		// get low 128 bit
				multyplied_i32 = _mm256_cvtepi16_epi32(tmp128);	// int16 -> int32

				res = _mm256_loadu_si256(&c_tmp[j + 16]);	// Load next temp C
				res = _mm256_add_epi32(multyplied_i32, res);// (A*B) + C
				_mm256_storeu_si256(&c_tmp[j + 16], res);	// store temp C

				tmp128 = _mm256_extractf128_si256(b, 1);		// get high 128 bit
				multyplied_i32 = _mm256_cvtepi16_epi32(tmp128);	// int16 -> int32

				res = _mm256_loadu_si256(&c_tmp[j + 24]);	// Load next temp C
				res = _mm256_add_epi32(multyplied_i32, res);// (A*B) + C
				_mm256_storeu_si256(&c_tmp[j + 24], res);	// store temp C

				//c_tmp[j] += A_PART*B[k*ldb + j];
				//C[i*ldc + j] += max_abs(A_PART*B[k*ldb + j] / (32), (256 * 128 - 1));
			}

			int prev_end = (N % 32 == 0) ? (N - 32) : (N / 32) * 32;
			for (j = prev_end; j < N; ++j) {
				c_tmp[j] += A_PART*B[k*ldb + j];
			}
		}
		for (j = 0; j < N; ++j) {
			C[i*ldc + j] += max_abs(c_tmp[j] / (R_MULT), (256 * 128 - 1));
			c_tmp[j] = 0;
		}
		//for (j = 0; j < N; ++j) C[i*ldc + j] += c_tmp[j] / (R_MULT);
	}
	free(c_tmp);
}

#elif defined(SSE41)
// 1.3 sec
void gemm_nn_int8_int16(int M, int N, int K, int8_t ALPHA,
	int8_t *A, int lda,
	int8_t *B, int ldb,
	int16_t *C, int ldc)
{
	__m128i multyplied_i32, res;
	__m128i a, b, d;
	//c = _mm_set1_epi16(32);

	int32_t *c_tmp = calloc(N, sizeof(int32_t));
	int i, j, k;
	for (i = 0; i < M; ++i) {
		for (k = 0; k < K; ++k) {
			register int16_t A_PART = ALPHA*A[i*lda + k];
			a = _mm_set1_epi16(A_PART);
			for (j = 0; j < N - 16; j += 16) {
				int index = k*ldb + j;
				d = _mm_loadu_si128((__m128i*)&B[index]);
					
				b = _mm_cvtepi8_epi16(d);	// int8 -> int16	

				b = _mm_mullo_epi16(a, b);	// B = A * B
					
				multyplied_i32 = _mm_cvtepi16_epi32(b);	// int16 -> int32

				res = _mm_loadu_si128(&c_tmp[j]);		// load temp C
				res = _mm_add_epi32(multyplied_i32, res);// (A*B) + C
				_mm_store_si128(&c_tmp[j], res);		// store temp C

				b = _mm_srli_si128(b, 8);				// Shift Right -> 8 bytes
				multyplied_i32 = _mm_cvtepi16_epi32(b);	// int16 -> int32

				res = _mm_loadu_si128(&c_tmp[j + 4]);	// Load next temp C
				res = _mm_add_epi32(multyplied_i32, res);// (A*B) + C
				_mm_store_si128(&c_tmp[j + 4], res);	// store temp C
					
				d = _mm_srli_si128(d, 8);	// Shift Right -> 8 bytes
				b = _mm_cvtepi8_epi16(d);	// int8 -> int16 (for low 8 bytes)

				b = _mm_mullo_epi16(a, b);	// B = A * B

				multyplied_i32 = _mm_cvtepi16_epi32(b);	// int16 -> int32

				res = _mm_loadu_si128(&c_tmp[j + 8]);	// Load next temp C
				res = _mm_add_epi32(multyplied_i32, res);// (A*B) + C
				_mm_store_si128(&c_tmp[j + 8], res);	// store temp C
					
				b = _mm_srli_si128(b, 8);				// Shift Right -> 8 bytes
				multyplied_i32 = _mm_cvtepi16_epi32(b);	// int16 -> int32

				res = _mm_loadu_si128(&c_tmp[j + 12]);	// Load next temp C
				res = _mm_add_epi32(multyplied_i32, res);// (A*B) + C
				_mm_store_si128(&c_tmp[j + 12], res);	// store temp C
				
				//c_tmp[j] += A_PART*B[k*ldb + j];
				//C[i*ldc + j] += max_abs(A_PART*B[k*ldb + j] / (32), (256 * 128 - 1));
			}

			int prev_end = (N % 16 == 0)?(N - 16) : (N / 16) * 16;
			for (j = prev_end; j < N; ++j) {
				c_tmp[j] += A_PART*B[k*ldb + j];
			}
		}
		for (j = 0; j < N; ++j) {
			C[i*ldc + j] += max_abs(c_tmp[j] / (R_MULT), (256 * 128 - 1));
			c_tmp[j] = 0;
		}
		//for (j = 0; j < N; ++j) C[i*ldc + j] += c_tmp[j] / (R_MULT);
	}
	free(c_tmp);
}

void gemm_nn_int8_int16_conv16(int M, int N, int K, int8_t ALPHA,
	int8_t *A, int lda,
	int8_t *B, int ldb,
	int16_t *C, int ldc)
{
	printf(" gemm_nn_int8_int16_conv16() isn't implemented for SSE4.1 \n");
}

#else

// 2.9 sec
void gemm_nn_int8_int16(int M, int N, int K, int8_t ALPHA,
	int8_t *A, int lda,
	int8_t *B, int ldb,
	int16_t *C, int ldc)
{
	int32_t *c_tmp = calloc(N, sizeof(int32_t));
	int i, j, k;
	for (i = 0; i < M; ++i) {
		for (k = 0; k < K; ++k) {
			register int16_t A_PART = ALPHA*A[i*lda + k];
			//#pragma simd parallel for
			for (j = 0; j < N; ++j) {
				c_tmp[j] += A_PART*B[k*ldb + j];
				//C[i*ldc + j] += max_abs(A_PART*B[k*ldb + j] / (R_MULT), (256 * 128 - 1));
			}
		}
		for (j = 0; j < N; ++j) {
			C[i*ldc + j] += max_abs(c_tmp[j] / (R_MULT), (256 * 128 - 1));
			c_tmp[j] = 0;
		}
	}
	free(c_tmp);
}

void gemm_nn_int8_int16_conv16(int M, int N, int K, int8_t ALPHA,
	int8_t *A, int lda,
	int8_t *B, int ldb,
	int16_t *C, int ldc)
{
	printf(" gemm_nn_int8_int16_conv16() isn't implemented \n");
}
#endif	// SSE41 or AVX


// 4 layers in 1: convolution, batch-normalization, BIAS and activation
void forward_convolutional_layer_q(layer l, network_state state, int return_float)
{

	int out_h = (l.h + 2 * l.pad - l.size) / l.stride + 1;	// output_height=input_height for stride=1 and pad=1 
	int out_w = (l.w + 2 * l.pad - l.size) / l.stride + 1;	// output_width=input_width for stride=1 and pad=1 
	int i, f, j;
	int const out_size = out_h*out_w;
	size_t const weights_size = l.size*l.size*l.c*l.n;

	// fill zero (ALPHA)
	//for (i = 0; i < l.outputs; ++i) l.output[i] = 0;

	// l.n - number of filters on this layer
	// l.c - channels of input-array
	// l.h - height of input-array
	// l.w - width of input-array
	// l.size - width and height of filters (the same size for all filters)


	//draw_distribution(l.weights, weights_size, NULL);
	//draw_distribution(state.input, l.inputs, NULL);

	typedef int16_t conv_t;
	conv_t *output_q = calloc(l.outputs, sizeof(conv_t));	// l.output

	////////////////////////////////////
	// cudnnConvolutionBiasActivationForward()
	// y = act ( alpha1 * conv(x) + alpha2 * z + bias )
	// int8 = activation( float * conv(int8) + float * int8 + float )
	// int8 = activation( conv(input_int8) + bias_float ) // X_INT8x4 or X_INT8
	// https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBiasActivationForward
	///////////////////////////////////


	// 1. Convolution !!!
#ifndef GEMMCONV
	int fil;
	// filter index 
	#pragma omp parallel for  	// "omp parallel for" - automatic parallelization of loop by using OpenMP
	for (fil = 0; fil < l.n; ++fil) {
		
		int chan, y, x, f_y, f_x;
		// channel index
		for (chan = 0; chan < l.c; ++chan)
			// input - y
			for (y = 0; y < l.h; ++y)
				// input - x
				for (x = 0; x < l.w; ++x)
				{
					int const output_index = fil*l.w*l.h + y*l.w + x;
					int const weights_pre_index = fil*l.c*l.size*l.size + chan*l.size*l.size;
					int const input_pre_index = chan*l.w*l.h;
					//float sum = 0;

					//int16_t sum = 0;
					int32_t sum = 0;
					//conv_t sum = 0;

					// filter - y
					for (f_y = 0; f_y < l.size; ++f_y)
					{
						int input_y = y + f_y - l.pad;
						// filter - x
						for (f_x = 0; f_x < l.size; ++f_x)
						{
							int input_x = x + f_x - l.pad;
							if (input_y < 0 || input_x < 0 || input_y >= l.h || input_x >= l.w) continue;

							int input_index = input_pre_index + input_y*l.w + input_x;
							int weights_index = weights_pre_index + f_y*l.size + f_x;

							//sum += state.input[input_index] * l.weights[weights_index];
							// int16 += int8 * int8;
							sum += (int32_t)state.input_int8[input_index] * (int32_t)l.weights_int8[weights_index];
						}
					}
					// l.output[filters][width][height] += 
					//		state.input[channels][width][height] * 
					//		l.weights[filters][channels][filter_width][filter_height];
					
					
					//output_q[output_index] += max_abs(sum, R_MAX_VAL);
					output_q[output_index] += max_abs(sum / R_MULT, R_MAX_VAL);
					//output_q[output_index] += sum / R_MULT;

					//if (fabs(output_q[output_index]) > 65535) printf(" fabs(output_q[output_index]) > 65535 \n");
				}
	}
#else
	int fil;

	// cuDNN: y = conv(x)
	int m = l.n;
	int k = l.size*l.size*l.c;
	int n = out_h*out_w;
	int8_t *a = l.weights_int8;
	int8_t *b = (int8_t *)state.workspace;
	int16_t *c = output_q;

	// convolution as GEMM (as part of BLAS)
	//for (i = 0; i < l.batch; ++i) {		
		im2col_cpu_int8(state.input_int8, l.c, l.h, l.w, l.size, l.stride, l.pad, b);	// here
		//gemm_nn_int8_int16(m, n, k, 1, a, k, b, n, c, n);	// single-thread gemm
		
		int t;	// multi-thread gemm
		#pragma omp parallel for
		for (t = 0; t < m; ++t) {
			//gemm_nn_int8_int16(1, n, k, 1, a + t*k, k, b, n, c + t*n, n);
			gemm_nn_int8_int16_conv16(1, n, k, 1, a + t*k, k, b, n, c + t*n, n);
		}		
	//}
#endif

	// cuDNN: y = alpha1 * conv(x)
	for (i = 0; i < l.outputs; ++i) {
		output_q[i] = output_q[i] * l.output_multipler;	// cuDNN: alpha1
	}

	// cuDNN: y = alpha1 * conv(x) + bias 
	for (fil = 0; fil < l.n; ++fil) {
		for (j = 0; j < out_size; ++j) {
			output_q[fil*out_size + j] += l.biases_quant[fil];
		}
	}


	// cuDNN: y = act ( alpha1 * conv(x) + bias )
	// bias is always FLOAT
	if (l.activation == LEAKY) {
		for (i = 0; i < l.n*out_size; ++i) {
			output_q[i] = (output_q[i]>0) ? output_q[i] : output_q[i] / 10; //leaky_activate(l.output[i]);
		}
	}	
	
	// cuDNN: y = act ( alpha1 * conv(x) + alpha2 * z + bias ), where: alpha2=0, z=NULL
	if (return_float) {
		// y - FLOAT, x,w - X_INT8 / X_INT8x4
		for (i = 0; i < l.outputs; ++i) {
			l.output[i] = (float)output_q[i] / 8.F;	// /8	// float32
		}
	}
	else
	{
		// y - X_INT8 / X_INT8x4, x,w - X_INT8 / X_INT8x4
		for (i = 0; i < l.outputs; ++i) {
			l.output_int8[i] = max_abs(output_q[i], I_MAX_VAL);	// int8
		}
	}

	free(output_q);
}

#define MIN_INT8 -128

// MAX pooling layer
void forward_maxpool_layer_q(const layer l, network_state state)
{
	int b, i, j, k, m, n;
	int w_offset = -l.pad;
	int h_offset = -l.pad;

	int h = l.out_h;
	int w = l.out_w;
	int c = l.c;

	// batch index
	for (b = 0; b < l.batch; ++b) {
		// channel index
		for (k = 0; k < c; ++k) {
			// y - input
			for (i = 0; i < h; ++i) {
				// x - input
				for (j = 0; j < w; ++j) {
					int out_index = j + w*(i + h*(k + c*b));
					int8_t max = MIN_INT8;
					int max_i = -1;
					// pooling x-index
					for (n = 0; n < l.size; ++n) {
						// pooling y-index
						for (m = 0; m < l.size; ++m) {
							int cur_h = h_offset + i*l.stride + n;
							int cur_w = w_offset + j*l.stride + m;
							int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
							int valid = (cur_h >= 0 && cur_h < l.h &&
								cur_w >= 0 && cur_w < l.w);
							int8_t val = (valid != 0) ? state.input_int8[index] : MIN_INT8;
							max_i = (val > max) ? index : max_i;	// get max index
							max = (val > max) ? val : max;			// get max value
						}
					}
					//l.output[out_index] = max;		// store max value
					l.output_int8[out_index] = max;		// store max value
					l.indexes[out_index] = max_i;	// store max index
				}
			}
		}
	}
}


// Route layer - just copy 1 or more layers into the current layer
void forward_route_layer_q(const layer l, network_state state)
{
	int i, j;
	int offset = 0;
	// number of merged layers
	for (i = 0; i < l.n; ++i) {
		int index = l.input_layers[i];					// source layer index
		//float *input = state.net.layers[index].output;	// source layer output ptr
		int8_t *input = state.net.layers[index].output_int8;	// source layer output ptr
		int input_size = l.input_sizes[i];				// source layer size
		// batch index
		for (j = 0; j < l.batch; ++j) {
			memcpy( l.output_int8 + offset + j*l.outputs, input + j*input_size, input_size*sizeof(int8_t) );
		}
		offset += input_size;
	}
}

// Reorg layer - just change dimension sizes of the previous layer (some dimension sizes are increased by decreasing other)
void forward_reorg_layer_q(const layer l, network_state state)
{
	//float *out = l.output;
	//float *x = state.input;
	int8_t *out = l.output_int8;
	int8_t *x = state.input_int8;
	int out_w = l.out_w;
	int out_h = l.out_h;
	int out_c = l.out_c;
	int batch = l.batch;

	int stride = l.stride;
	int b, i, j, k;
	int in_c = out_c / (stride*stride);

	int out_w_X_stride = out_w*stride;
	int out_h_X_stride = out_h*stride;

	//printf("\n out_c = %d, out_w = %d, out_h = %d, stride = %d, forward = %d \n", out_c, out_w, out_h, stride, forward);
	//printf("  in_c = %d,  in_w = %d,  in_h = %d \n", in_c, out_w*stride, out_h*stride);

	// batch
	for (b = 0; b < batch; ++b) {
		// channel
		for (k = 0; k < out_c; ++k) {
			int c2 = k % in_c;
			int pre_out_index = out_h_X_stride*(c2 + in_c*b);
			int offset = k / in_c;
			int offset_mod_stride = offset % stride;
			int offset_div_stride = offset / stride;
			// y
			for (j = 0; j < out_h; ++j) {
				int pre_in_index = out_w*(j + out_h*(k + out_c*b));
				// x
				for (i = 0; i < out_w; ++i) {
					int in_index = i + pre_in_index;
					int w2 = i*stride + offset_mod_stride;
					int h2 = j*stride + offset_div_stride;
					int out_index = w2 + out_w_X_stride*(h2 + pre_out_index);
					out[in_index] = x[out_index];
				}
			}
		}
	}
}




// ---- region layer ----

static void softmax_q(float *input, int n, float temp, float *output)
{
	int i;
	float sum = 0;
	float largest = -FLT_MAX;
	for (i = 0; i < n; ++i) {
		if (input[i] > largest) largest = input[i];
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

static void softmax_tree(float *input, int batch, int inputs, float temp, tree *hierarchy, float *output)
{
	int b;
	for (b = 0; b < batch; ++b) {
		int i;
		int count = 0;
		for (i = 0; i < hierarchy->groups; ++i) {
			int group_size = hierarchy->group_size[i];
			softmax_q(input + b*inputs + count, group_size, temp, output + b*inputs + count);
			count += group_size;
		}
	}
}
// ---


// Region layer - just change places of array items, then do logistic_activate and softmax 
void forward_region_layer_q(const layer l, network_state state)
{
	int i, b;
	int size = l.coords + l.classes + 1;	// 4 Coords(x,y,w,h) + Classes + 1 Probability-t0
	//printf("\n l.coords = %d \n", l.coords);
	memcpy(l.output, state.input, l.outputs*l.batch * sizeof(float));

	//flatten(l.output, l.w*l.h, size*l.n, l.batch, 1);
	// convert many channels to the one channel (depth=1)
	// (each grid cell will have a number of float-variables equal = to the initial number of channels)
	{
		float *x = l.output;
		int layer_size = l.w*l.h;	// W x H - size of layer
		int layers = size*l.n;		// number of channels (where l.n = number of anchors)
		int batch = l.batch;

		float *swap = calloc(layer_size*layers*batch, sizeof(float));
		int i, c, b;
		// batch index
		for (b = 0; b < batch; ++b) {
			// channel index
			for (c = 0; c < layers; ++c) {
				// layer grid index
				for (i = 0; i < layer_size; ++i) {
					int i1 = b*layers*layer_size + c*layer_size + i;
					int i2 = b*layers*layer_size + i*layers + c;
					swap[i2] = x[i1];	
				}
			}
		}
		memcpy(x, swap, layer_size*layers*batch * sizeof(float));
		free(swap);
	}


	// logistic activation only for: t0 (where is t0 = Probability * IoU(box, object))
	for (b = 0; b < l.batch; ++b) {
		// for each item (x, y, anchor-index)
		for (i = 0; i < l.h*l.w*l.n; ++i) {
			int index = size*i + b*l.outputs;
			float x = l.output[index + 4];
			l.output[index + 4] = 1.0F / (1.0F + expf(-x));	// logistic_activate_q(l.output[index + 4]);
		}
	}

	
	if (l.softmax_tree) {	// Yolo 9000
		for (b = 0; b < l.batch; ++b) {
			for (i = 0; i < l.h*l.w*l.n; ++i) {
				int index = size*i + b*l.outputs;
				softmax_tree(l.output + index + 5, 1, 0, 1, l.softmax_tree, l.output + index + 5);
			}
		}
	}
	else if (l.softmax) {	// Yolo v2
		// softmax activation only for Classes probability
		for (b = 0; b < l.batch; ++b) {
			// for each item (x, y, anchor-index)
			//#pragma omp parallel for
			for (i = 0; i < l.h*l.w*l.n; ++i) {
				int index = size*i + b*l.outputs;
				softmax_q(l.output + index + 5, l.classes, 1, l.output + index + 5);
			}
		}
	}

}





void yolov2_forward_network_q(network net, network_state state)
{
	state.workspace = net.workspace;
	int i, k;
	for (i = 0; i < net.n; ++i) {
		state.index = i;
		layer l = net.layers[i];

		if (l.type == CONVOLUTIONAL) {
			int return_float = (i == (net.n - 2));	// if this is the last convolutional layer

			forward_convolutional_layer_q(l, state, return_float);
			//printf("\n CONVOLUTIONAL \t\t l.size = %d  \n", l.size);
		}
		else if (l.type == MAXPOOL) {
			forward_maxpool_layer_q(l, state);
			//printf("\n MAXPOOL \t\t l.size = %d  \n", l.size);
		}
		else if (l.type == ROUTE) {
			forward_route_layer_q(l, state);
			//printf("\n ROUTE \t\t\t l.n = %d  \n", l.n);
		}
		else if (l.type == REORG) {
			forward_reorg_layer_q(l, state);
			//printf("\n REORG \n");
		}
		else if (l.type == REGION) {
			forward_region_layer_q(l, state);
			//printf("\n REGION \n");
		}
		else {
			printf("\n layer: %d \n", l.type);
		}


		state.input = l.output;
		state.input_int8 = l.output_int8;
	}
}


// detect on CPU
float *network_predict_quantized(network net, float *input)
{
	network_state state;
	state.net = net;
	state.index = 0;
	state.input = input;
	state.input_int8 = calloc(net.w*net.h*net.c, sizeof(int8_t));
	state.truth = 0;
	state.train = 0;
	state.delta = 0;
	int k;
	for (k = 0; k < net.w*net.h*net.c; ++k){
		int16_t src = state.input[k] * net.layers[0].input_quant_multipler;
		state.input_int8[k] = max_abs(src, I_MAX_VAL);
	}

	yolov2_forward_network_q(net, state);	// network on CPU
	//float *out = get_network_output(net);
	int i;
	for (i = net.n - 1; i > 0; --i) if (net.layers[i].type != COST) break;
	free(state.input_int8);
	return net.layers[i].output;
}


// --------------------
// x - last conv-layer output
// biases - anchors from cfg-file
// n - number of anchors from cfg-file
box get_region_box_q(float *x, float *biases, int n, int index, int i, int j, int w, int h)
{
	box b;
	b.x = (i + logistic_activate(x[index + 0])) / w;	// (col + 1./(1. + exp(-x))) / width_last_layer
	b.y = (j + logistic_activate(x[index + 1])) / h;	// (row + 1./(1. + exp(-x))) / height_last_layer
	b.w = expf(x[index + 2]) * biases[2 * n] / w;		// exp(x) * anchor_w / width_last_layer
	b.h = expf(x[index + 3]) * biases[2 * n + 1] / h;	// exp(x) * anchor_h / height_last_layer
	return b;
}

// get prediction boxes
void get_region_boxes_q(layer l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness, int *map)
{
	int i, j, n;
	float *predictions = l.output;
	// grid index
	for (i = 0; i < l.w*l.h; ++i) {
		int row = i / l.w;
		int col = i % l.w;
		// anchor index
		for (n = 0; n < l.n; ++n) {
			int index = i*l.n + n;	// index for each grid-cell & anchor
			int p_index = index * (l.classes + 5) + 4;
			float scale = predictions[p_index];				// scale = t0 = Probability * IoU(box, object)
			if (l.classfix == -1 && scale < .5) scale = 0;	// if(t0 < 0.5) t0 = 0;
			int box_index = index * (l.classes + 5);
			boxes[index] = get_region_box_q(predictions, l.biases, n, box_index, col, row, l.w, l.h);
			boxes[index].x *= w;
			boxes[index].y *= h;
			boxes[index].w *= w;
			boxes[index].h *= h;

			int class_index = index * (l.classes + 5) + 5;

			// Yolo 9000 or Yolo v2
			if (l.softmax_tree) {
				// Yolo 9000
				hierarchy_predictions(predictions + class_index, l.classes, l.softmax_tree, 0);
				int found = 0;
				if (map) {
					for (j = 0; j < 200; ++j) {
						float prob = scale*predictions[class_index + map[j]];
						probs[index][j] = (prob > thresh) ? prob : 0;
					}
				}
				else {
					for (j = l.classes - 1; j >= 0; --j) {
						if (!found && predictions[class_index + j] > .5) {
							found = 1;
						}
						else {
							predictions[class_index + j] = 0;
						}
						float prob = predictions[class_index + j];
						probs[index][j] = (scale > thresh) ? prob : 0;
					}
				}
			}
			else
			{
				// Yolo v2
				for (j = 0; j < l.classes; ++j) {
					float prob = scale*predictions[class_index + j];	// prob = IoU(box, object) = t0 * class-probability
					probs[index][j] = (prob > thresh) ? prob : 0;		// if (IoU < threshold) IoU = 0;
				}
			}
			if (only_objectness) {
				probs[index][0] = scale;
			}
		}
	}
}

// Quantinization and get multiplers for convolutional weights for quantinization
void quantinization_and_get_multipliers(network net)
{
	//float input_mult[] = { 16, 16, 16, 16, 16, 16, 16, 16, 16 };
	//float input_mult[] = { 127, 2, 16, 16, 16, 32, 32, 16, 32 };
	//float input_mult[] = { 127, 2, 32, 32, 32, 32, 32, 32, 32 };
	//float input_mult[] = { 127, 2, 16, 16, 16, 16, 16, 16, 16 };	// good

	//float input_mult[] = { 32, 4, 16, 16, 16, 16, 16, 16, 16 };
	//float input_mult[] = { 128, 4, 32, 32, 32, 64, 64, 32, 64 };

	// full
	//float input_mult[] = { 256, 4,32,64,32,32,32,32,32,64,64,64,64,64,128,64,128,128,64,128,64,128,128 };	// divided 4 - full works
	int counter = 0;

	int j;
	for (j = 0; j < net.n; ++j) {
		layer *l = &net.layers[j];

		if (l->type == CONVOLUTIONAL) {
			size_t const weights_size = l->size*l->size*l->c*l->n;
			int k;

			// get optimal multipliers - for Weights
			float weights_multiplier = get_multiplier(l->weights, weights_size, 8);
			l->weights_quant_multipler = weights_multiplier / 4;	// manual shift 2 bits

			for (k = 0; k < weights_size; ++k) {
				float w = l->weights[k] * l->weights_quant_multipler;
				l->weights_int8[k] = max_abs(w, W_MAX_VAL);
			}
			
			// set manually optimal multipliers - for Inputs
			// tiny
			if (counter == 0) l->input_quant_multipler = 128;
			else if(counter == 1) l->input_quant_multipler = 2;
			else  l->input_quant_multipler = 16;
			++counter;

			//if (counter == 0) l->output_multipler = 128 / (l->weights_quant_multipler * l->input_quant_multipler / R_MULT);
			//else 
			if (counter == 1) l->output_multipler = 2 / (l->weights_quant_multipler * l->input_quant_multipler / R_MULT);
			else if (counter >= 1) l->output_multipler = 16 / (l->weights_quant_multipler * l->input_quant_multipler / R_MULT);
			
			// calculate optimal multipliers - for Biases
			float biases_multipler = (l->output_multipler * l->weights_quant_multipler * l->input_quant_multipler / R_MULT);
			for (k = 0; k < l->n; ++k)
				l->biases_quant[k] = l->biases[k] * biases_multipler;

			printf(" Get optimal multiplers for Input and Conv-weights %g, input %g \n", 
				l->weights_quant_multipler, l->input_quant_multipler);
		}
		else {
			printf(" Skip layer: %d \n", l->type);
		}
	}



}