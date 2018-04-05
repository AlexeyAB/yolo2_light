#include "additionally.h"	// some definitions from: im2col.h, blas.h, list.h, utils.h, activations.h, tree.h, layer.h, network.h
							// softmax_layer.h, reorg_layer.h, route_layer.h, region_layer.h, maxpool_layer.h, convolutional_layer.h

#define GEMMCONV

#include "opencv2/highgui/highgui_c.h"
#include "opencv2/core/core_c.h"
#include "opencv2/core/version.hpp"

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



void gemm_nn_int8(int M, int N, int K, int8_t ALPHA,
	int8_t *A, int lda,
	int8_t *B, int ldb,
	int16_t *C, int ldc)
{
	int32_t *tmp = calloc(N, sizeof(int32_t));
	int i, j, k;
	for (i = 0; i < M; ++i) {
		for (k = 0; k < K; ++k) {
			register int16_t A_PART = ALPHA*A[i*lda + k];
			//#pragma simd parallel for
			for (j = 0; j < N; ++j) {
				tmp[j] += A_PART*B[k*ldb + j];
				//C[i*ldc + j] += max_abs(A_PART*B[k*ldb + j] / (32), (256 * 128 - 1));
			}
		}
		for (j = 0; j < N; ++j) C[i*ldc + j] += max_abs(tmp[j] / (32), (256 * 128 - 1));
	}
	free(tmp);
}

// 4 layers in 1: convolution, batch-normalization, BIAS and activation
void forward_convolutional_layer_q(layer l, network_state state)
{

	int out_h = (l.h + 2 * l.pad - l.size) / l.stride + 1;	// output_height=input_height for stride=1 and pad=1 
	int out_w = (l.w + 2 * l.pad - l.size) / l.stride + 1;	// output_width=input_width for stride=1 and pad=1 
	int i, f, j;
	int const out_size = out_h*out_w;
	size_t const weights_size = l.size*l.size*l.c*l.n;

	// fill zero (ALPHA)
	for (i = 0; i < l.outputs; ++i) l.output[i] = 0;

	// l.n - number of filters on this layer
	// l.c - channels of input-array
	// l.h - height of input-array
	// l.w - width of input-array
	// l.size - width and height of filters (the same size for all filters)

//#define MAX_VAL (65535*16)
//#define MULT 256.F
	
	/*
	for (f = 0; f < l.n; ++f)
	{
		const int filter_size = l.size*l.size*l.c;
		int w_index = f*filter_size;
		char buff[256];
		sprintf(buff, "%d", f);
		//printf("\n f = %d, filter_size = %d, w_index = %d, weights_size = %d, l.weights + w_index = %p \n",
		//	f, filter_size, w_index, weights_size, l.weights + w_index);
		draw_distribution(&l.weights[w_index], filter_size, buff);
	}
	*/

	//draw_distribution(l.weights, weights_size, NULL);
	//draw_distribution(state.input, l.inputs, NULL);

/*	
#define W_MAX_VAL (256/2 - 1)	// 31-bit (32)
#define I_MAX_VAL (256/2 - 1)	// 31-bit (32)
#define R_MAX_VAL (256*128 - 1)	// 31-bit (32)
//#define W_MULT (256.F*4)
//#define I_MULT (16.F)
#define I_MULT (l.input_quant_multipler)
#define R_MULT (32)	// 4 - 32

	typedef int16_t conv_t;
	typedef int8_t weight_t;
	typedef int8_t input_t;
		
	l.weights_quant_multipler /= 4;	// (int8 = optimal 4) (int16 = optimal 32) 8 - 32
*/

#define W_MAX_VAL (256/2 - 1)	// 31-bit (32)
#define I_MAX_VAL (256/2 - 1)	// 31-bit (32)
#define R_MAX_VAL (256*128 - 1)	// 31-bit (32)
//#define W_MULT (256.F*4)
//#define I_MULT (16.F)
#define I_MULT (l.input_quant_multipler)
#define R_MULT (32)	// 4 - 32

	typedef int16_t conv_t;
	typedef int8_t weight_t;
	typedef int8_t input_t;

	l.weights_quant_multipler /= 4;	// (int8 = optimal 4) (int16 = optimal 32) 8 - 32

	// for int8 required 7-bit (1-bit for sign)

	/*
#define W_MAX_VAL (256*16 - 1)	// 12-bit (13)
#define I_MAX_VAL (256/2 - 1)	// 7-bit (8)
#define R_MAX_VAL (256*4 - 1)	// 10-bit (11)
#define W_MULT (256.F/1)
#define I_MULT (256.F/16)
#define R_MULT (256.F/8)
	typedef int16_t conv_t;
	typedef int8_t input_t;
	*/
	
	// Tiny-Yolo works successfully:
	// int - with MULT=[256,512,1024,2048,4096] without MAX_VAL
	// int - with MULT=256 with MAX_VAL=65535*16 and higher
	// int - with W_MULT=512, I_MULT=8 with MAX_VAL=65535 and higher
	// short - with W_MULT=256, I_MULT=16, R_MULT=8 with MAX_VAL=65535 and higher
	// short - with W_MULT=512, I_MULT=16, R_MULT=16 with MAX_VAL=65535 and higher
	// short - with W_MULT=2048, I_MULT=16, R_MULT=64 with MAX_VAL=65535 and higher
	// short VOC+COCO - with W_MULT=1024, I_MULT=16, R_MULT=64 with MAX_VAL=65535 and higher
	// FUSED short VOC+COCO - with W_MULT=256, I_MULT=256, R_MULT=128 with MAX_VAL=4096 and higher
	// FUSED short VOC+COCO - with W_MULT=256, I_MULT=256, R_MULT=256 with W_MAX_VAL=I_MAX_VAL=4096, R_MAX_VAL=1024 and higher
	// FUSED short VOC+COCO - with W_MULT=256, I_MULT=64, R_MULT=256 with W_MAX_VAL=4096, I_MAX_VAL=1024, R_MAX_VAL=1024 and higher
	// FUSED short VOC+COCO - with W_MULT=256, I_MULT=16, R_MULT=32 with W_MAX_VAL=4096, I_MAX_VAL=256, R_MAX_VAL=1024 and higher

	weight_t *weights_q = calloc(weights_size, sizeof(conv_t));	// l.weights
	input_t *input_q = calloc(l.inputs, sizeof(conv_t));	// state.input
	conv_t *output_q = calloc(l.outputs, sizeof(conv_t));	// l.output

	float *biases_q = calloc(l.n, sizeof(float));	// l.biases


    //l.output[index] = ((l.output[index] - l.rolling_mean[f]) / (sqrtf(l.rolling_variance[f]) + .000001f)) * l.scales[i];
	//l.output[i*out_size + j] += l.biases[i];

	for (f = 0; f < l.n; ++f) biases_q[f] = l.biases[f];
		
	for (i = 0; i < weights_size; ++i) {
		//float w = l.weights[i] * W_MULT;	// can be multiplied more
		float w = l.weights[i] * l.weights_quant_multipler;
		weights_q[i] = max_abs(w, W_MAX_VAL);
		//if (fabs(weights_q[i]) > 65535) printf(" fabs(weights_q[i]) > 65535 \n");
	}

	for (i = 0; i < l.inputs; ++i){
		int32_t src = state.input[i] * I_MULT;	// can't be multiplied more
		input_q[i] = max_abs(src, I_MAX_VAL);
		//if (fabs(input_q[i]) > 127) printf(" fabs(input_q[i]) > 127 \n");
	}

	//for (i = 0; i < l.inputs; ++i) state.input[i] = input_q[i];
	//draw_distribution(state.input, l.inputs, NULL);

	// 1. Convolution !!!
#ifndef GEMMCONV
	int fil;
	// filter index 
	#pragma omp parallel for  	// "omp parallel for" - automatic parallelization of loop by using OpenMP
	for (fil = 0; fil < l.n; ++fil) {
		for (j = 0; j < out_size; ++j)
			output_q[fil*out_size + j] = biases_q[fil] * (l.weights_quant_multipler*I_MULT / R_MULT);
			//output_q[fil*out_size + j] = biases_q[fil] * (W_MULT*I_MULT / R_MULT);
		
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

					int16_t sum = 0;
					//int32_t sum = 0;

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
							sum += (int16_t)input_q[input_index] * (int16_t)weights_q[weights_index];
						}
					}
					// l.output[filters][width][height] += 
					//		state.input[channels][width][height] * 
					//		l.weights[filters][channels][filter_width][filter_height];
					
					
					//l.output[output_index] += sum;
					//output_q[output_index] += sum;
					//output_q[output_index] += max_abs(sum, R_MAX_VAL);
					output_q[output_index] += max_abs(sum / R_MULT, R_MAX_VAL);
					//output_q[output_index] += sum / R_MULT;
					//if (fabs(output_q[output_index]) > 65535) printf(" fabs(output_q[output_index]) > 65535 \n");
				}
	}
#else
	int fil;
	// filter index 
	for (fil = 0; fil < l.n; ++fil) {
		for (j = 0; j < out_size; ++j)
			//output_q[fil*out_size + j] = output_q[fil*out_size + j]/R_MULT + biases_q[fil] * (l.weights_quant_multipler*I_MULT / R_MULT);
			output_q[fil*out_size + j] = biases_q[fil] * (l.weights_quant_multipler*I_MULT / R_MULT);
	}

	int m = l.n;
	int k = l.size*l.size*l.c;
	int n = out_h*out_w;
	int8_t *a = weights_q;
	int8_t *b = (int8_t *)state.workspace;
	int16_t *c = output_q;

	// convolution as GEMM (as part of BLAS)
	//for (i = 0; i < l.batch; ++i) {		
		im2col_cpu_int8(input_q, l.c, l.h, l.w, l.size, l.stride, l.pad, b);	// here
		//gemm_nn_int8(m, n, k, 1, a, k, b, n, c, n);	// single-thread gemm
		
		int t;	// multi-thread gemm
		#pragma omp parallel for
		for (t = 0; t < m; ++t) {
			gemm_nn_int8(1, n, k, 1, a + t*k, k, b, n, c + t*n, n);
		}		
	//}
#endif


	// 4. Activation function (LEAKY or LINEAR)
	if (l.activation == LEAKY) {
		for (i = 0; i < l.n*out_size; ++i) {
			output_q[i] = (output_q[i]>0) ? output_q[i] : 0.1F * output_q[i]; //leaky_activate(l.output[i]);
		}
	}	
	
	//for (i = 0; i < l.outputs; ++i) l.output[i] = output_q[i] / (W_MULT*I_MULT / R_MULT);
	for (i = 0; i < l.outputs; ++i) l.output[i] = output_q[i] / (l.weights_quant_multipler*I_MULT / R_MULT);



	free(weights_q);
	free(input_q);
	free(output_q);
	free(biases_q);
}



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
					float max = -FLT_MAX;
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
							float val = (valid != 0) ? state.input[index] : -FLT_MAX;
							max_i = (val > max) ? index : max_i;	// get max index
							max = (val > max) ? val : max;			// get max value
						}
					}
					l.output[out_index] = max;		// store max value
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
		float *input = state.net.layers[index].output;	// source layer output ptr
		int input_size = l.input_sizes[i];				// source layer size
		// batch index
		for (j = 0; j < l.batch; ++j) {
			memcpy( l.output + offset + j*l.outputs, input + j*input_size, input_size*sizeof(float) );
		}
		offset += input_size;
	}
}

// Reorg layer - just change dimension sizes of the previous layer (some dimension sizes are increased by decreasing other)
void forward_reorg_layer_q(const layer l, network_state state)
{
	float *out = l.output;
	float *x = state.input;
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

	for (b = 0; b < batch; ++b) {
		for (k = 0; k < out_c; ++k) {
			int c2 = k % in_c;
			int pre_out_index = out_h_X_stride*(c2 + in_c*b);
			int offset = k / in_c;
			int offset_mod_stride = offset % stride;
			int offset_div_stride = offset / stride;
			for (j = 0; j < out_h; ++j) {
				int pre_in_index = out_w*(j + out_h*(k + out_c*b));
				for (i = 0; i < out_w; ++i) {
					int in_index = i + pre_in_index;
					int w2 = i*stride + offset_mod_stride;
					int h2 = j*stride + offset_div_stride;
					int out_index = w2 + out_w_X_stride*(h2 + pre_out_index);
					out[out_index] = x[in_index];	// used by default for forward (i.e. forward = 0)
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
	printf("\n l.coords = %d \n", l.coords);
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
	int i;
	for (i = 0; i < net.n; ++i) {
		state.index = i;
		layer l = net.layers[i];

		if (l.type == CONVOLUTIONAL) {
			forward_convolutional_layer_q(l, state);
			printf("\n CONVOLUTIONAL \t\t l.size = %d  \n", l.size);
		}
		else if (l.type == MAXPOOL) {
			forward_maxpool_layer_q(l, state);
			printf("\n MAXPOOL \t\t l.size = %d  \n", l.size);
		}
		else if (l.type == ROUTE) {
			forward_route_layer_q(l, state);
			printf("\n ROUTE \t\t\t l.n = %d  \n", l.n);
		}
		else if (l.type == REORG) {
			forward_reorg_layer_q(l, state);
			printf("\n REORG \n");
		}
		else if (l.type == REGION) {
			forward_region_layer_q(l, state);
			printf("\n REGION \n");
		}
		else {
			printf("\n layer: %d \n", l.type);
		}


		state.input = l.output;
	}
}


// detect on CPU
float *network_predict_quantized(network net, float *input)
{
	network_state state;
	state.net = net;
	state.index = 0;
	state.input = input;
	state.truth = 0;
	state.train = 0;
	state.delta = 0;
	yolov2_forward_network_q(net, state);	// network on CPU
	//float *out = get_network_output(net);
	int i;
	for (i = net.n - 1; i > 0; --i) if (net.layers[i].type != COST) break;
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
					push_convolutional_layer(l);
				}
#endif

#ifdef OPENCL
				//if (gpu_index >= 0) {
				ocl_push_convolutional_layer(l);
				//}
#endif
			}
		}
		else {
			printf(" Skip layer: %d \n", l->type);
		}
	}
}

// get multiplers for convolutional weights for quantinization
void get_conv_weight_optimal_multipliers(network net)
{
	//float input_mult[] = { 16, 16, 16, 16, 16, 16, 16, 16, 16 };
	//float input_mult[] = { 127, 2, 16, 16, 16, 32, 32, 16, 32 };
	//float input_mult[] = { 127, 2, 32, 32, 32, 32, 32, 32, 32 };
	//float input_mult[] = { 127, 2, 16, 16, 16, 16, 16, 16, 16 };	// good

	//float input_mult[] = { 32, 4, 16, 16, 16, 16, 16, 16, 16 };
	//float input_mult[] = { 128, 4, 32, 32, 32, 64, 64, 32, 64 };

	// full
	//float input_mult[] = { 256, 4,32,64,32,32,32,32,32,64,64,64,64,64,128,64,128,128,64,128,64,128,128 };	// divided 4 - full works
	int couter = 0;

	int j;
	for (j = 0; j < net.n; ++j) {
		layer *l = &net.layers[j];

		if (l->type == CONVOLUTIONAL) {
			size_t const weights_size = l->size*l->size*l->c*l->n;

			float multiplier = get_multiplier(l->weights, weights_size, 8);
			l->weights_quant_multipler = multiplier;
			
			//l->input_quant_multipler = input_mult[couter++];

			
			// tiny
			if (couter == 0) l->input_quant_multipler = 128;
			else if(couter == 1) l->input_quant_multipler = 2;
			else  l->input_quant_multipler = 16;
			++couter;
			
			printf(" Get optimal multiplers for Input and Conv-weights for quantinization %g, input %g \n", 
				l->weights_quant_multipler, l->input_quant_multipler);
		}
		else {
			printf(" Skip layer: %d \n", l->type);
		}
	}



}