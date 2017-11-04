#include <istream>
#include <memory>

#include "additionally.h"
#include "ocl.h"

#include "OpenCL/include/OCLManager.h"
#include "OpenCL/include/clutils.h"

OCLManager* m_OCLManager;

bool ocl_initialize() {

	m_OCLManager = new OCLManager();
	if (m_OCLManager->Initialize() != OCL_STATUS_READY) {
		//Log error
		return false;
	}

	char m_OCLDeviceName[128];
	strcpy(m_OCLDeviceName, m_OCLManager->GetDeviceName());

	srand(2222222);


	return true;
}


void ocl_push_array(cl_mem x_gpu, float *x, size_t n)
{
	size_t size = sizeof(float)*n;

	float *tmp = (float *)calloc(n, sizeof(float));
	memcpy(tmp, x, size);
	//getchar();
	cl_int status;
	status = clEnqueueWriteBuffer(*m_OCLManager->m_OpenCLSetup.getQueue(), x_gpu, CL_TRUE, 0, size, x, 0, NULL, NULL);
	//DEBUG_CL(status);
	if (status != CL_SUCCESS) {
		printf("\n clWrite buffer error : %s", getCLErrorString(status));
		exit(-1);
	}
}

cl_mem ocl_make_array(float *x, size_t n)
{
	size_t size = sizeof(float)*n;
	cl_int status;
	cl_mem x_gpu = clCreateBuffer(*m_OCLManager->m_OpenCLSetup.getContext(), CL_MEM_READ_WRITE, size, NULL, &status);
	if (status != CL_SUCCESS) {
		printf("\n createBuffer error : %s", getCLErrorString(status));
		exit(-1);
	}
	if (x) {
		ocl_push_array(x_gpu, x, n);
	}
	return x_gpu;
}

cl_mem ocl_make_int_array(size_t n)
{
	size_t size = sizeof(int)*n;
	cl_int status;
	cl_mem x_gpu = clCreateBuffer(*m_OCLManager->m_OpenCLSetup.getContext(), CL_MEM_READ_WRITE, size, NULL, &status);
	if (status != CL_SUCCESS) {
		printf("\n createBuffer error : %s", getCLErrorString(status));
		exit(-1);
	}
	return x_gpu;
}

#define OCL_BLOCK 512

size_t ocl_blocks(size_t size) 
{
	size_t global_size = size;
	if (global_size % OCL_BLOCK != 0)
		global_size = (global_size / OCL_BLOCK + 1)*OCL_BLOCK;

	return global_size;
}


void ocl_push_convolutional_layer(convolutional_layer layer)
{
	ocl_push_array(layer.weights_ocl, layer.weights, layer.c*layer.n*layer.size*layer.size);
	ocl_push_array(layer.biases_ocl, layer.biases, layer.n);
	if (layer.batch_normalize) {
		ocl_push_array(layer.scales_ocl, layer.scales, layer.n);
		ocl_push_array(layer.rolling_mean_ocl, layer.rolling_mean, layer.n);
		ocl_push_array(layer.rolling_variance_ocl, layer.rolling_variance, layer.n);
	}
}


#define BLOCK 8

void forward_convolutional_layer_opencl(layer l, network_state state)
{
	int LOCAL_BLOCK = BLOCK;

	int out_h = (l.h + 2 * l.pad - l.size) / l.stride + 1;	// output_height=input_height for stride=1 and pad=1 
	int out_w = (l.w + 2 * l.pad - l.size) / l.stride + 1;	// output_width=input_width for stride=1 and pad=1 

	int num_kernels = l.c * out_h * out_w;

	int globalDimX = l.outputs / BLOCK;
	if (globalDimX % BLOCK != 0)
		globalDimX = ((globalDimX + BLOCK) / BLOCK) * BLOCK;

	m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_RESETARR]->pGlobal(globalDimX)->pLocal(BLOCK);
	m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_RESETARR]->arg(0, l.output_ocl);
	m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_RESETARR]->arg(1, l.outputs);
	m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_RESETARR]->run(PROFILE_KERNELS, BLOCK_KERNEL_EXEC);



	globalDimX = ((num_kernels + LOCAL_BLOCK) / LOCAL_BLOCK) * LOCAL_BLOCK;
	
	int KERNEL_IDX = (l.size == 3) ? NN_KERNEL_IDX_IM2COL3X3 : NN_KERNEL_IDX_IM2COL1X1;

	
	m_OCLManager->m_OpenCLKernels[KERNEL_IDX]->pGlobal(globalDimX)->pLocal(LOCAL_BLOCK);
	m_OCLManager->m_OpenCLKernels[KERNEL_IDX]->arg(0, num_kernels);
	m_OCLManager->m_OpenCLKernels[KERNEL_IDX]->arg(1, state.input_ocl);
	m_OCLManager->m_OpenCLKernels[KERNEL_IDX]->arg(2, l.h);
	m_OCLManager->m_OpenCLKernels[KERNEL_IDX]->arg(3, l.w);
	m_OCLManager->m_OpenCLKernels[KERNEL_IDX]->arg(4, l.size);
	m_OCLManager->m_OpenCLKernels[KERNEL_IDX]->arg(5, l.pad);
	m_OCLManager->m_OpenCLKernels[KERNEL_IDX]->arg(6, l.stride);
	m_OCLManager->m_OpenCLKernels[KERNEL_IDX]->arg(7, out_h);
	m_OCLManager->m_OpenCLKernels[KERNEL_IDX]->arg(8, out_w);
	m_OCLManager->m_OpenCLKernels[KERNEL_IDX]->arg(9, state.workspace_ocl);
	m_OCLManager->m_OpenCLKernels[KERNEL_IDX]->run(PROFILE_KERNELS, BLOCK_KERNEL_EXEC);
	

	// Conv as GEMM from clBLAS
	{
		int m = l.n;
		int k = l.size*l.size*l.c;
		int n = out_h*out_w;
		cl_mem a = l.weights_ocl;
		cl_mem b = state.workspace_ocl;
		cl_mem c = l.output_ocl;

		clblast::StatusCode status = clblast::Gemm(clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kNo,
			m, n, k,
			1.0f, a, 0, k, b, 0, n, 1.0f, c, 0, n, m_OCLManager->m_OpenCLSetup.getQueue());

		if (status != clblast::StatusCode::kSuccess) {
			//printf("\n clblast::Gemm error : %d", status);
			printf("\n clblast::Gemm error \n");
			exit(-1);
		}
	}
	

	if (l.batch_normalize) {
		
		int spatial = out_w*out_h;

		m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_NORMSCALEADDBIAS]->pGlobal(ocl_blocks(l.outputs))->pLocal(OCL_BLOCK);
		m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_NORMSCALEADDBIAS]->arg(0, l.output_ocl);
		m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_NORMSCALEADDBIAS]->arg(1, l.scales_ocl);
		m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_NORMSCALEADDBIAS]->arg(2, l.biases_ocl);
		m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_NORMSCALEADDBIAS]->arg(3, l.rolling_mean_ocl);
		m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_NORMSCALEADDBIAS]->arg(4, l.rolling_variance_ocl);
		m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_NORMSCALEADDBIAS]->arg(5, l.n);
		m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_NORMSCALEADDBIAS]->arg(6, spatial);
		m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_NORMSCALEADDBIAS]->arg(7, l.outputs);
		m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_NORMSCALEADDBIAS]->run(PROFILE_KERNELS, BLOCK_KERNEL_EXEC);
		
	}
	else {
		int spatial = out_w*out_h;

		m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_ADDBIAS]->pGlobal(ocl_blocks(l.outputs))->pLocal(OCL_BLOCK);
		m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_ADDBIAS]->arg(0, l.output_ocl);
		m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_ADDBIAS]->arg(1, l.biases_ocl);
		m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_ADDBIAS]->arg(2, l.n);
		m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_ADDBIAS]->arg(3, spatial);
		m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_ADDBIAS]->arg(4, l.outputs);
		m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_ADDBIAS]->run(PROFILE_KERNELS, BLOCK_KERNEL_EXEC);
	
	}

	if (l.activation == LEAKY) {
		globalDimX = l.outputs / BLOCK;

		if (globalDimX % BLOCK != 0)
			globalDimX = ((globalDimX + BLOCK) / BLOCK) * BLOCK;

		int KERNEL_IDX = NN_KERNEL_IDX_LEAKY_ACTIVATE;
		int actType = 7;

		m_OCLManager->m_OpenCLKernels[KERNEL_IDX]->pGlobal(globalDimX)->pLocal(BLOCK);
		m_OCLManager->m_OpenCLKernels[KERNEL_IDX]->arg(0, l.output_ocl);
		m_OCLManager->m_OpenCLKernels[KERNEL_IDX]->arg(1, l.output_ocl);
		m_OCLManager->m_OpenCLKernels[KERNEL_IDX]->arg(2, l.outputs);
		m_OCLManager->m_OpenCLKernels[KERNEL_IDX]->arg(3, actType);
		m_OCLManager->m_OpenCLKernels[KERNEL_IDX]->run(PROFILE_KERNELS, BLOCK_KERNEL_EXEC);
	}
}

void forward_maxpool_layer_opencl(const layer l, network_state state)
{
	int globalDimX = l.outputs / BLOCK;

	if (globalDimX % BLOCK != 0)
		globalDimX = ((globalDimX + BLOCK) / BLOCK) * BLOCK;

	float execTime = 0.0f;

	m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_MAXPOOL]->pGlobal(globalDimX)->pLocal(BLOCK);
	m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_MAXPOOL]->arg(0, l.outputs);
	m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_MAXPOOL]->arg(1, l.h);
	m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_MAXPOOL]->arg(2, l.w);
	m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_MAXPOOL]->arg(3, l.c);
	m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_MAXPOOL]->arg(4, l.stride);
	m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_MAXPOOL]->arg(5, l.size);
	m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_MAXPOOL]->arg(6, l.pad);
	m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_MAXPOOL]->arg(7, state.input_ocl);
	m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_MAXPOOL]->arg(8, l.output_ocl);
	m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_MAXPOOL]->run(PROFILE_KERNELS, BLOCK_KERNEL_EXEC);
}




// Route layer - just copy 1 or more layers into the current layer
void forward_route_layer_opencl(const layer l, network_state state)
{
	int i, j;
	size_t offset = 0;
	// number of merged layers
	for (i = 0; i < l.n; ++i) {
		int index = l.input_layers[i];					// source layer index
		cl_mem input_ocl = state.net.layers[index].output_ocl;
		size_t input_size_bytes = l.input_sizes[i] * sizeof(float);

		cl_int status = clEnqueueCopyBuffer(*m_OCLManager->m_OpenCLSetup.getQueue(),
			input_ocl, l.output_ocl, 0, offset, input_size_bytes, 0, NULL, NULL);

		offset += input_size_bytes;
	}
}


// Reorg layer - just change dimension sizes of the previous layer (some dimension sizes are increased by decreasing other)
void forward_reorg_layer_opencl(const layer l, network_state state)
{
	m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_REORG]->pGlobal(ocl_blocks(l.outputs))->pLocal(OCL_BLOCK);
	m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_REORG]->arg(0, l.outputs);
	m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_REORG]->arg(1, state.input_ocl);
	m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_REORG]->arg(2, l.w);
	m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_REORG]->arg(3, l.h);
	m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_REORG]->arg(4, l.c);
	m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_REORG]->arg(5, l.batch);
	m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_REORG]->arg(6, l.stride);
	m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_REORG]->arg(7, l.output_ocl);
	m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_REORG]->run(PROFILE_KERNELS, BLOCK_KERNEL_EXEC);

}




// ---- region layer ----

static void softmax_cpu(float *input, int n, float temp, float *output)
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
			softmax_cpu(input + b*inputs + count, group_size, temp, output + b*inputs + count);
			count += group_size;
		}
	}
}
// ---


// Region layer - just change places of array items, then do logistic_activate and softmax 
void forward_region_layer_opencl(const layer l, network_state state)
{
	int i, b;
	int size = l.coords + l.classes + 1;	// 4 Coords(x,y,w,h) + Classes + 1 Probability-t0

	int spatial = l.w * l.h;
	int slices = l.n * size;
	int output_size = spatial * slices;

	int globalDimX = output_size / BLOCK;
	if (globalDimX % BLOCK != 0)
		globalDimX = ((globalDimX + BLOCK) / BLOCK) * BLOCK;

	m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_FLATARR]->pGlobal(globalDimX)->pLocal(BLOCK);
	m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_FLATARR]->arg(0, output_size);
	m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_FLATARR]->arg(1, state.input_ocl);
	m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_FLATARR]->arg(2, spatial);
	m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_FLATARR]->arg(3, slices);
	m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_FLATARR]->arg(4, 1);
	m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_FLATARR]->arg(5, l.output_ocl);
	m_OCLManager->m_OpenCLKernels[NN_KERNEL_IDX_FLATARR]->run(PROFILE_KERNELS, BLOCK_KERNEL_EXEC);
	
	cl_int status =
		clEnqueueReadBuffer(*m_OCLManager->m_OpenCLSetup.getQueue(), l.output_ocl, CL_TRUE, 0,
			l.outputs*l.batch * sizeof(float), l.output, 0, NULL, NULL);

	clFinish(*m_OCLManager->m_OpenCLSetup.getQueue());




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
				softmax_cpu(l.output + index + 5, l.classes, 1, l.output + index + 5);
			}
		}
	}


	// logistic activation only for: t0 (where is t0 = Probability * IoU(box, object))
	for (b = 0; b < l.batch; ++b) {
		// for each item (x, y, anchor-index)
		for (i = 0; i < l.h*l.w*l.n; ++i) {
			int index = size*i + b*l.outputs;
			float x = l.output[index + 4];
			l.output[index + 4] = 1.0F / (1.0F + expf(-x));	// logistic_activate_cpu(l.output[index + 4]);
		}
	}
}


void yolov2_forward_network_cpu(network net, network_state state)
{
	state.workspace_ocl = net.workspace_ocl;
	state.workspace = net.workspace;
	int i;
	for (i = 0; i < net.n; ++i) {
		state.index = i;
		layer l = net.layers[i];

		if (l.type == CONVOLUTIONAL) {
			forward_convolutional_layer_opencl(l, state);
			//printf("\n CONVOLUTIONAL \t\t l.size = %d  \n", l.size);
		}
		else if (l.type == MAXPOOL) {
			forward_maxpool_layer_opencl(l, state);
			//printf("\n MAXPOOL \t\t l.size = %d  \n", l.size);
		}
		else if (l.type == ROUTE) {
			forward_route_layer_opencl(l, state);
			//printf("\n ROUTE \t\t\t l.n = %d  \n", l.n);
		}
		else if (l.type == REORG) {
			forward_reorg_layer_opencl(l, state);
			//printf("\n REORG \n");
		}
		else if (l.type == REGION) {
			forward_region_layer_opencl(l, state);
			//printf("\n REGION \n");
		}
		else {
			printf("\n layer: %d \n", l.type);
		}

		state.input_ocl = l.output_ocl;
		state.input = l.output;
	}
}


// detect on CPU
float *network_predict_opencl(network net, float *input)
{
	network_state state;
	state.net = net;
	state.index = 0;
	state.input = input;
	int size = net.layers[0].inputs * net.batch;
	cl_mem input_array = ocl_make_array(input, size);
	state.input_ocl = input_array;
	state.truth = 0;
	state.train = 0;
	state.delta = 0;
	yolov2_forward_network_cpu(net, state);	// network on CPU

	cl_int status = clReleaseMemObject(input_array);

	int i;
	for (i = net.n - 1; i > 0; --i) if (net.layers[i].type != COST) break;
	return net.layers[i].output;
}



