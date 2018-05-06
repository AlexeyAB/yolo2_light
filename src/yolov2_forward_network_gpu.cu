#include "additionally.h"	// some definitions from: im2col.h, blas.h, list.h, utils.h, activations.h, tree.h, layer.h, network.h
// softmax_layer.h, reorg_layer.h, route_layer.h, region_layer.h, maxpool_layer.h, convolutional_layer.h

#include "gpu.h"


// from: box.h
typedef struct {
	float x, y, w, h;
} box;

// ------------- GPU cuDNN ---------------

// 4 layers in 1: convolution, batch-normalization, BIAS and activation
void forward_convolutional_layer_gpu_cudnn(layer l, network_state state)
{
	// blas_kernels.cu
	fill_ongpu(l.outputs*l.batch, 0, l.output_gpu, 1);

	float one = 1;
	// cuDNN >= v5.1
	cudnnConvolutionForward(cudnn_handle(),
		&one,
		l.srcTensorDesc,
		state.input,
		l.weightDesc,
		l.weights_gpu,
		l.convDesc,
		l.fw_algo,
		state.workspace,
		l.workspace_size,
		&one,
		l.dstTensorDesc,
		l.output_gpu);


	if (l.batch_normalize) {
		// blas_kernels.cu
		normalize_gpu(l.output_gpu, l.rolling_mean_gpu, l.rolling_variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
		scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
	}

	// blas_kernels.cu
	add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h); 

	// blas_kernels.cu
	activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);
}


// MAX pooling layer
void forward_maxpool_layer_gpu_cuda(const layer l, network_state state)
{
	// maxpool_layer_kernels.cu
	forward_maxpool_layer_gpu(l, state);
}


// route layer
void forward_route_layer_gpu_cuda(const layer l, network_state state)
{
	int i, j;
	int offset = 0;
	for (i = 0; i < l.n; ++i) {
		int index = l.input_layers[i];
		float *input = state.net.layers[index].output_gpu;
		int input_size = l.input_sizes[i];
		for (j = 0; j < l.batch; ++j) {
			// CUDA
			cudaMemcpy(l.output_gpu + offset + j*l.outputs, input + j*input_size, sizeof(float)*input_size, cudaMemcpyDeviceToDevice);
		}
		offset += input_size;
	}
}


// reorg layer
void forward_reorg_layer_gpu_cuda(layer l, network_state state)
{
	// blas_kernels.cu
	//reorg_ongpu(state.input, l.w, l.h, l.c, l.batch, l.stride, 1, l.output_gpu);
	reorg_ongpu(state.input, l.out_w, l.out_h, l.out_c, l.batch, l.stride, 0, l.output_gpu);
}


// region layer
void forward_region_layer_gpu_cuda(const layer l, network_state state)
{
	// blas_kernels.cu
	flatten_ongpu(state.input, l.h*l.w, l.n*(l.coords + l.classes + 1), l.batch, 1, l.output_gpu);
	if (l.softmax_tree) {	// Yolo 9000
		int i;
		int count = 5;
		for (i = 0; i < l.softmax_tree->groups; ++i) {
			int group_size = l.softmax_tree->group_size[i];
			// blas_kernels.cu
			softmax_gpu(l.output_gpu + count, group_size, l.classes + 5, l.w*l.h*l.n*l.batch, 1, l.output_gpu + count);
			count += group_size;
		}
	}
	else if (l.softmax) {	// Yolo v2
		// blas_kernels.cu
		softmax_gpu(l.output_gpu + 5, l.classes, l.classes + 5, l.w*l.h*l.n*l.batch, 1, l.output_gpu + 5);
	}

	float *in_cpu = (float *)calloc(l.batch*l.inputs, sizeof(float));
	float *truth_cpu = 0;
	if (state.truth) {
		int num_truth = l.batch*l.truths;
		truth_cpu = (float *)calloc(num_truth, sizeof(float));
		cudaError_t status = cudaMemcpy(state.truth, truth_cpu, num_truth*sizeof(float), cudaMemcpyDeviceToHost);
	}

	cudaError_t status = cudaMemcpy(in_cpu, l.output_gpu, l.batch*l.inputs*sizeof(float), cudaMemcpyDeviceToHost);
	network_state cpu_state = state;
	cpu_state.train = state.train;
	cpu_state.truth = truth_cpu;
	cpu_state.input = in_cpu;

	int i, b;
	int size = l.coords + l.classes + 1;
	memcpy(l.output, cpu_state.input, l.outputs*l.batch * sizeof(float));
	for (b = 0; b < l.batch; ++b) {
		for (i = 0; i < l.h*l.w*l.n; ++i) {
			int index = size*i + b*l.outputs;
			float x = l.output[index + 4];
			l.output[index + 4] = 1.0F / (1.0F + expf(-x));	// logistic_activate_cpu(l.output[index + 4]);
		}
	}

	free(cpu_state.input);
}



void forward_network_gpu_cudnn(network net, network_state state)
{
	state.workspace = net.workspace;
	int i;
	for (i = 0; i < net.n; ++i) {
		state.index = i;
		layer l = net.layers[i];

		if (l.type == CONVOLUTIONAL) {
			forward_convolutional_layer_gpu_cudnn(l, state);
			//printf("\n CONVOLUTIONAL \t\t l.size = %d  \n", l.size);
		}
		else if (l.type == MAXPOOL) {
			forward_maxpool_layer_gpu_cuda(l, state);
			//printf("\n MAXPOOL \t\t l.size = %d  \n", l.size);
		}
		else if (l.type == ROUTE) {
			forward_route_layer_gpu_cuda(l, state);
			//printf("\n ROUTE \n");
		}
		else if (l.type == REORG) {
			forward_reorg_layer_gpu_cuda(l, state);
			//printf("\n REORG \n");
		}
		else if (l.type == REGION) {
			forward_region_layer_gpu_cuda(l, state);
			//printf("\n REGION \n");
		}
		else {
			printf("\n layer: %d \n", l.type);
		}
		state.input = l.output_gpu;
	}
}

// detect on GPU
float *network_predict_gpu_cudnn(network net, float *input)
{
	cudaError_t status = cudaSetDevice(net.gpu_index);
	//check_error(status);
	int size = net.layers[0].inputs * net.batch;
	network_state state;
	state.index = 0;
	state.net = net;
	status = cudaMalloc((void **)&(state.input), sizeof(float)*size);
	status = cudaMemcpy(state.input, input, sizeof(float)*size, cudaMemcpyHostToDevice);
	state.truth = 0;
	state.train = 0;
	state.delta = 0;
	forward_network_gpu_cudnn(net, state); // network on GPU
	status = cudaFree(state.input);
	//float *out = get_network_output_gpu(net);
	int i;
	for (i = net.n - 1; i > 0; --i) if (net.layers[i].type != COST) break;
	layer l = net.layers[i];
	if (l.type != REGION) status = cudaMemcpy(l.output, l.output_gpu, l.outputs*l.batch*sizeof(float), cudaMemcpyDeviceToHost); 
	return l.output;
}


