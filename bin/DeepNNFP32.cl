
/*Copyright 2017 Sateesh Pedagadi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.*/

#pragma OPENCL EXTENSION cl_nv_pragma_unroll : enable

typedef enum {
	LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN
}ACTIVATION;

__kernel void image2columarray3x3(int n, __global float* data_im,
	int height, int width, int ksize, const int pad, int stride, int height_col, int width_col, 
	__global float *data_col) { 

	int index = get_global_id(0);
	int stepSize = get_local_size(0) * get_num_groups(0);
	int w_out, h_index, h_out, channel_in, channel_out, h_in, w_in, i, j, h, w;
	
#pragma unroll
	for (; index < n; index += stepSize) {

		w_out = index % width_col;
		h_index = index / width_col;
		h_out = h_index % height_col;
		channel_in = h_index / height_col;
		channel_out = channel_in * ksize * ksize;
		h_in = h_out * stride - pad;
		w_in = w_out * stride - pad;

		__global float* data_col_ptr = data_col;
		data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;

		const __global float* data_im_ptr = data_im; 
		data_im_ptr += (channel_in * height + h_in) * width + w_in;

		#pragma unroll 3
		for (i = 0; i < 3; ++i) {
			#pragma unroll 3
			for (j = 0; j < 3; ++j) {

				h = h_in + i;
				w = w_in + j;

				*data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
					data_im_ptr[i * width + j] : 0;
				data_col_ptr += height_col * width_col;
			}
		}
	}
}


__kernel void image2columarray1x1(int n, __global float* data_im,
	int height, int width, int ksize, const int pad, int stride, int height_col, int width_col, 
	__global float *data_col) { 

	int index = get_global_id(0);
	int stepSize = get_local_size(0) * get_num_groups(0);
	int w_out, h_index, h_out, channel_in, channel_out, h_in, w_in, i = 0, j = 0, h, w, count = 0;


#pragma unroll
	for (; index < n; index += stepSize) {

		w_out = index % width_col;
		h_index = index / width_col;
		h_out = h_index % height_col;
		channel_in = h_index / height_col;
		channel_out = channel_in * ksize * ksize;
		h_in = h_out * stride - pad;
		w_in = w_out * stride - pad;

		__global float* data_col_ptr = data_col;
		data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;

		const __global float* data_im_ptr = data_im;
		data_im_ptr += (channel_in * height + h_in) * width + w_in;

		h = h_in;
		w = w_in;

		*data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
					data_im_ptr[i * width + j] : 0;
		data_col_ptr += height_col * width_col;
	}
}

__kernel void resetarray(__global float *output, int size) {

	int filter = get_group_id(1);
	int resetStepSize = get_local_size(0);
	int offset = get_global_id(0) * resetStepSize;
	float8 localOutput = (float8)0;
	vstore8(localOutput, 0, &output[offset]);
}


__kernel void addbias(__global float *output, __global float *biases, int filters, int spatial, int size) 
{
	int id = get_global_id(0);
	if (id >= size) return;
	int filter_index = id / spatial;

	output[id] = output[id] + biases[filter_index];
}


__kernel void normscaleaddbias(__global float *output, __global float *scales, __global float *biases,
	__global float *mean, __global float *variance, int filters, int spatial, int size) 
{
	int id = get_global_id(0);
	if (id >= size) return;
	int filter_index = (id / spatial) % filters;

	output[id] = ((output[id] - mean[filter_index]) / (sqrt(variance[filter_index]) + .000001f))
		* scales[filter_index] + biases[filter_index];
}


__kernel void leakyactivatearray(__global float *x, __global float *swapBuffer, int n, ACTIVATION a){

	int stepSize = get_local_size(0);
	int index = get_global_id(0) * stepSize;
	int count = 0;
	int idL = get_local_id(0);

	float8 xLocal = vload8(0, &x[index]);
	xLocal = (xLocal > (float8)0) ? xLocal : ((float8)0.1 * xLocal);
	vstore8(xLocal, 0, &x[index]);
}


__kernel void flattenarray(int N, __global float *x, int spatial, int layers, int batch, __global float *out) {

	int stepSize = get_local_size(0);
	int idx = get_global_id(0) * stepSize;
	int count = 0, i, in_s, in_c, i2, i1;

#pragma unroll 8
	for (; count < 8; count++) {

		i = idx;
		in_s = i%spatial;
		i = i / spatial;
		in_c = i%layers;
		i = i / layers;

		i1 = in_c*spatial + in_s;
		i2 = in_s*layers + in_c;

		out[i2] = x[i1];
		idx = idx + 1;
	}
}

float maxvalue_device(__global float *x) {

	float8 localInput[10];
	localInput[0] = vload8(0, &x[0]);
	localInput[1] = vload8(0, &x[8]);
	localInput[2] = vload8(0, &x[16]);
	localInput[3] = vload8(0, &x[24]);
	localInput[4] = vload8(0, &x[32]);
	localInput[5] = vload8(0, &x[40]);
	localInput[6] = vload8(0, &x[48]);
	localInput[7] = vload8(0, &x[56]);
	localInput[8] = vload8(0, &x[64]);
	localInput[9] = vload8(0, &x[72]);

	float8 maxValue;
	float largest = -INFINITY;

	maxValue = max(localInput[0], localInput[1]);
	maxValue = max(maxValue, localInput[2]);
	maxValue = max(maxValue, localInput[3]);
	maxValue = max(maxValue, localInput[4]);
	maxValue = max(maxValue, localInput[5]);
	maxValue = max(maxValue, localInput[6]);
	maxValue = max(maxValue, localInput[7]);
	maxValue = max(maxValue, localInput[8]);
	maxValue = max(maxValue, localInput[9]);
	maxValue = max(maxValue, localInput[10]);

	largest = (maxValue.s0 > largest) ? maxValue.s0 : largest;
	largest = (maxValue.s1 > largest) ? maxValue.s1 : largest;
	largest = (maxValue.s2 > largest) ? maxValue.s2 : largest;
	largest = (maxValue.s3 > largest) ? maxValue.s3 : largest;
	largest = (maxValue.s4 > largest) ? maxValue.s4 : largest;
	largest = (maxValue.s5 > largest) ? maxValue.s5 : largest;
	largest = (maxValue.s6 > largest) ? maxValue.s6 : largest;
	largest = (maxValue.s7 > largest) ? maxValue.s7 : largest;

	return largest;
}

void softmax_device_optimized(__global float *x, __global float *xOut) {

	int i = 0;
	float8 localInput[10];
	localInput[0] = vload8(0, &x[0]);
	localInput[1] = vload8(0, &x[8]);
	localInput[2] = vload8(0, &x[16]);
	localInput[3] = vload8(0, &x[24]);
	localInput[4] = vload8(0, &x[32]);
	localInput[5] = vload8(0, &x[40]);
	localInput[6] = vload8(0, &x[48]);
	localInput[7] = vload8(0, &x[56]);
	localInput[8] = vload8(0, &x[64]);
	localInput[9] = vload8(0, &x[72]);

	float8 maxValue;
	float largest = -INFINITY;

	maxValue = max(localInput[0], localInput[1]);
	maxValue = max(maxValue, localInput[2]);
	maxValue = max(maxValue, localInput[3]);
	maxValue = max(maxValue, localInput[4]);
	maxValue = max(maxValue, localInput[5]);
	maxValue = max(maxValue, localInput[6]);
	maxValue = max(maxValue, localInput[7]);
	maxValue = max(maxValue, localInput[8]);
	maxValue = max(maxValue, localInput[9]);
	maxValue = max(maxValue, localInput[10]);

	largest = (maxValue.s0 > largest) ? maxValue.s0 : largest;
	largest = (maxValue.s1 > largest) ? maxValue.s1 : largest;
	largest = (maxValue.s2 > largest) ? maxValue.s2 : largest;
	largest = (maxValue.s3 > largest) ? maxValue.s3 : largest;
	largest = (maxValue.s4 > largest) ? maxValue.s4 : largest;
	largest = (maxValue.s5 > largest) ? maxValue.s5 : largest;
	largest = (maxValue.s6 > largest) ? maxValue.s6 : largest;
	largest = (maxValue.s7 > largest) ? maxValue.s7 : largest;

	float8 largest8 = (float8)largest;
	float8 sum8 = (float8)0;
	float sum = 0;

#pragma unroll 10
	for (i = 0; i < 10; i++) {

		localInput[i] = exp(localInput[i] - largest8);
		sum8 += localInput[i];
	}

	sum = sum8.s0 + sum8.s1 + sum8.s2 + sum8.s3 + sum8.s4 + sum8.s5 + sum8.s6 + sum8.s7;

#pragma unroll 10
	for (i = 0; i < 10; i++) {

		localInput[i] /= sum;
		vstore8(localInput[i], 0, &x[i * 8]);
	}
}


void softmax_device_optimized2(__global float *x, __global float *xOut) {

	int i = 0;
	float16 localInput[5];
	localInput[0] = vload16(0, &x[0]);
	localInput[1] = vload16(0, &x[16]);
	localInput[2] = vload16(0, &x[32]);
	localInput[3] = vload16(0, &x[48]);
	localInput[4] = vload16(0, &x[64]);

	float16 maxValue;
	float largest = -INFINITY;

	maxValue = max(localInput[0], localInput[1]);
	maxValue = max(maxValue, localInput[2]);
	maxValue = max(maxValue, localInput[3]);
	maxValue = max(maxValue, localInput[4]);

	largest = (maxValue.s0 > largest) ? maxValue.s0 : largest;
	largest = (maxValue.s1 > largest) ? maxValue.s1 : largest;
	largest = (maxValue.s2 > largest) ? maxValue.s2 : largest;
	largest = (maxValue.s3 > largest) ? maxValue.s3 : largest;
	largest = (maxValue.s4 > largest) ? maxValue.s4 : largest;
	largest = (maxValue.s5 > largest) ? maxValue.s5 : largest;
	largest = (maxValue.s6 > largest) ? maxValue.s6 : largest;
	largest = (maxValue.s7 > largest) ? maxValue.s7 : largest;
	largest = (maxValue.s8 > largest) ? maxValue.s8 : largest;
	largest = (maxValue.s9 > largest) ? maxValue.s9 : largest;
	largest = (maxValue.sa > largest) ? maxValue.sa : largest;
	largest = (maxValue.sb > largest) ? maxValue.sb : largest;
	largest = (maxValue.sc > largest) ? maxValue.sc : largest;
	largest = (maxValue.sd > largest) ? maxValue.sd : largest;
	largest = (maxValue.se > largest) ? maxValue.se : largest;
	largest = (maxValue.sf > largest) ? maxValue.sf : largest;


	float16 largest16 = (float16)largest;
	float16 sum16 = (float16)0;
	float sum = 0;

#pragma unroll 5
	for (i = 0; i < 5; i++) {

		localInput[i] = exp(localInput[i] - largest16);
		sum16 += localInput[i];
	}

	sum = sum16.s0 + sum16.s1 + sum16.s2 + sum16.s3 + sum16.s4 + sum16.s5 + sum16.s6 + sum16.s7 + sum16.s8 +
		sum16.s9 + sum16.sa + sum16.sb + sum16.sc + sum16.sd + sum16.se + sum16.sf;

#pragma unroll 5
	for (i = 0; i < 5; i++) {

		localInput[i] /= sum;
		vstore16(localInput[i], 0, &x[i * 16]);
	}
}

void softmax_device(int baseIdxOffset, __global float *input, float temp, __global float *output) {

	int i;
	float sum = 0, e = 0;
	float largest = -INFINITY;
	
	largest = maxvalue_device(input);

#pragma unroll 80	
	for (i = 0; i < 80; ++i) {

		output[i] = exp(input[i]- largest);
		sum += output[i];
	}

#pragma unroll 80	
	for (i = 0; i < 80; ++i)
		output[i] /= sum;
}

__kernel void softmax(int n, int offset, int batch, __global float *input, float temp, __global float *output, int base) {

	int stepSize = get_local_size(0);
	int idx = get_global_id(0) * stepSize;
	int count = 0, i = 0;
	int baseIdxOffset = 0; 

#pragma unroll 2
	for (; count < 2; count++) {
	
		baseIdxOffset = base + idx*offset;
		softmax_device_optimized2(input + baseIdxOffset, output + baseIdxOffset);
		idx = idx + 1;
	}
}

__kernel void maxpool(int n, int in_h, int in_w, int in_c, int stride, int size, 
				int pad, __global float *input, __global float *output) { 

	int h = (in_h + 2 * pad) / stride;
	int w = (in_w + 2 * pad) / stride;
	int c = in_c;

	int stepSize = get_local_size(0);
	int idx = get_global_id(0) * stepSize;

	int count = 0, id = 0;
	int j, i, k, w_offset, h_offset, out_index, l, m, cur_h, cur_w, index, valid;
	float maxVal, val;

#pragma unroll 8
	for (; count < 8; count++) {

		id = idx;
		j = id % w;
		id /= w;
		i = id % h;
		id /= h;
		k = id % c;
		id /= c;

		w_offset = -pad;
		h_offset = -pad;

		out_index = j + w * (i + h * k);
		maxVal = -INFINITY;
		
		#pragma unroll 2
		for (l = 0; l < 2; ++l) {
			#pragma unroll 2
			for (m = 0; m < 2; ++m) {
				cur_h = h_offset + i*stride + l;
				cur_w = w_offset + j*stride + m;
				index = cur_w + in_w*(cur_h + in_h * k);
				valid = (cur_h >= 0 && cur_h < in_h && cur_w >= 0 && cur_w < in_w);
				val = (valid != 0) ? input[index] : -INFINITY;
				maxVal = (val > maxVal) ? val : maxVal;
			}
		}
		output[out_index] = maxVal;
		idx = idx + 1;
	}
}

//resetIdx = index * resetStepSize;
//float8 localOutput = (float8)0;
//vstore8(localOutput, 0, &reset_arr[resetIdx]);
