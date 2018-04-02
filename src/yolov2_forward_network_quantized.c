#include "additionally.h"	// some definitions from: im2col.h, blas.h, list.h, utils.h, activations.h, tree.h, layer.h, network.h
							// softmax_layer.h, reorg_layer.h, route_layer.h, region_layer.h, maxpool_layer.h, convolutional_layer.h

//#define GEMMCONV

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


// 4 layers in 1: convolution, batch-normalization, BIAS and activation
void forward_convolutional_layer_q(layer l, network_state state)
{

	int out_h = (l.h + 2 * l.pad - l.size) / l.stride + 1;	// output_height=input_height for stride=1 and pad=1 
	int out_w = (l.w + 2 * l.pad - l.size) / l.stride + 1;	// output_width=input_width for stride=1 and pad=1 
	int i, f, j;

	// fill zero (ALPHA)
	for (i = 0; i < l.outputs; ++i) l.output[i] = 0;

	// l.n - number of filters on this layer
	// l.c - channels of input-array
	// l.h - height of input-array
	// l.w - width of input-array
	// l.size - width and height of filters (the same size for all filters)

//#define MAX_VAL (65535*16)
//#define MULT 256.F
#define MAX_VAL (65535)
#define W_MULT (256.F*4)
#define I_MULT (16.F)
#define R_MULT (64)
	typedef short int conv_t;
	
	// Tiny-Yolo works successfully:
	// int - with MULT=[256,512,1024,2048,4096] without MAX_VAL
	// int - with MULT=256 with MAX_VAL=65535*16 and higher
	// int - with W_MULT=512, I_MULT=8 with MAX_VAL=65535 and higher
	// short - with W_MULT=256, I_MULT=16, R_MULT=8 with MAX_VAL=65535 and higher
	// short - with W_MULT=512, I_MULT=16, R_MULT=16 with MAX_VAL=65535 and higher
	// short - with W_MULT=2048, I_MULT=16, R_MULT=64 with MAX_VAL=65535 and higher
	// short VOC+COCO - with W_MULT=1024, I_MULT=16, R_MULT=64 with MAX_VAL=65535 and higher
	size_t const weights_size = l.size*l.size*l.c*l.n;
	conv_t *weights_q = calloc(weights_size, sizeof(int));	// l.weights
	conv_t *input_q = calloc(l.inputs, sizeof(int));	// state.input
	//conv_t *output_q = calloc(l.outputs, sizeof(int));	// l.output
	conv_t *output_q = calloc(l.outputs, sizeof(int));	// l.output



	//for (i = 0; i < weights_size; ++i) weights_q[i] = l.weights[i] * MULT;
	//for (i = 0; i < l.inputs; ++i) input_q[i] = state.input[i] * MULT;

	for (i = 0; i < weights_size; ++i) {
		float w = l.weights[i] * W_MULT;	// can be multiplied more
		weights_q[i] = w;// (w > MAX_VAL) ? MAX_VAL : w;
		//if (abs(weights_q[i]) > 65535) printf(" abs(weights_q[i]) > 65535 \n");
	}
	for (i = 0; i < l.inputs; ++i){
		float src = state.input[i] * I_MULT;	// can't be multiplied more
		input_q[i] = src;// (src > MAX_VAL) ? MAX_VAL : src;
		//if (abs(input_q[i]) > 65535) printf(" abs(input_q[i]) > 65535 \n");
	}



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

					int sum = 0;

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
							// int32 = int16 * int16;
							sum += input_q[input_index] * weights_q[weights_index];
						}
					}
					// l.output[filters][width][height] += 
					//		state.input[channels][width][height] * 
					//		l.weights[filters][channels][filter_width][filter_height];
					
					
					//l.output[output_index] += sum;
					//output_q[output_index] += sum;
					//output_q[output_index] += max_abs(sum, MAX_VAL);
					//output_q[output_index] += max_abs(sum / R_MULT, MAX_VAL);
					output_q[output_index] += sum / R_MULT;
					//if (abs(output_q[output_index]) > 65535) printf(" abs(output_q[output_index]) > 65535 \n");
				}
	}
#else


	int m = l.n;
	int k = l.size*l.size*l.c;
	int n = out_h*out_w;
	float *a = l.weights;
	float *b = state.workspace;
	float *c = l.output;

	// convolution as GEMM (as part of BLAS)
	for (i = 0; i < l.batch; ++i) {		
		im2col_q(state.input, l.c, l.h, l.w, l.size, l.stride, l.pad, b);	// im2col.c
		gemm_nn(m, n, k, 1, a, k, b, n, c, n);	// gemm.c
		c += n*m;
		state.input += l.c*l.h*l.w;
	}
#endif

	
	//for (i = 0; i < l.outputs; ++i) l.output[i] = output_q[i] / (W_MULT*I_MULT);
	for (i = 0; i < l.outputs; ++i) l.output[i] = output_q[i] / (W_MULT*I_MULT/ R_MULT);
	
	int const out_size = out_h*out_w;

	// 2. Batch normalization
	if (l.batch_normalize) {
		for (f = 0; f < l.out_c; ++f) {
			for (i = 0; i < out_size; ++i) {
				int index = f*out_size + i;
				l.output[index] = (l.output[index] - l.rolling_mean[f]) / (sqrtf(l.rolling_variance[f]) + .000001f);
			}
		}

		// scale_bias
		for (i = 0; i < l.out_c; ++i) {
			for (j = 0; j < out_size; ++j) {
				l.output[i*out_size + j] *= l.scales[i];
			}
		}
	}


	// 3. Add BIAS
	//if (l.batch_normalize)
	for (i = 0; i < l.n; ++i) {
		for (j = 0; j < out_size; ++j) {
			l.output[i*out_size + j] += l.biases[i];
		}
	}

	// 4. Activation function (LEAKY or LINEAR)
	if (l.activation == LEAKY) {
		for (i = 0; i < l.n*out_size; ++i) {
			l.output[i] = leaky_activate(l.output[i]);
		}
	}


	free(weights_q);
	free(input_q);
	free(output_q);
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
	int w = l.w;
	int h = l.h;
	int c = l.c;
	int batch = l.batch;
	
	int stride = l.stride;
	int b, i, j, k;
	int out_c = c / (stride*stride);

	// batch index
	for (b = 0; b < batch; ++b) {
		// channel index
		for (k = 0; k < c; ++k) {
			// y
			for (j = 0; j < h; ++j) {
				// x
				for (i = 0; i < w; ++i) {
					int in_index = i + w*(j + h*(k + c*b));
					int c2 = k % out_c;
					int offset = k / out_c;
					int w2 = i*stride + offset % stride;
					int h2 = j*stride + offset / stride;
					int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));
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

