#include "additionally.h"    // some definitions from: im2col.h, blas.h, list.h, utils.h, activations.h, tree.h, layer.h, network.h
// softmax_layer.h, reorg_layer.h, route_layer.h, region_layer.h, maxpool_layer.h, convolutional_layer.h

#include "gpu.h"

/*
// from: box.h
typedef struct {
    float x, y, w, h;
} box;
*/

// ------------- GPU cuDNN ---------------

#define checkCUDNN(status) {                                           \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      printf("CUDNN failure\nError: %d - %s \n", status, cudnnGetErrorString(status)); \
      getchar();                                        \
    }                                                                  \
}

// 4 layers in 1: convolution, batch-normalization, BIAS and activation
void forward_convolutional_layer_gpu_cudnn(layer l, network_state state)
{
    // XNOR-net
    if (l.xnor) {

        if (l.align_bit_weights_gpu && l.c >= 256 && l.size > 1)
        {
            cudaError_t status = cudaSuccess;
            int input_size = l.c*l.h*l.w*l.batch;

            int m = l.n;
            int k = l.size*l.size*l.c;
            int n = l.out_w*l.out_h;
            //float * a = l.binary_weights_gpu;

            int ldb_align = l.lda_align;
            size_t new_ldb = k + (ldb_align - k%ldb_align); // (k / 8 + 1) * 8;
            size_t t_intput_size = new_ldb * n;
            size_t t_bit_input_size = t_intput_size / 8;// +1;

            {
                int i = 0;
                if (l.stride == 1 && l.c >= 256 && l.w >= 13 && l.size > 1 && 0)    // disable
                {
                    // stride=1 only
                    im2col_align_bin_ongpu(state.input + i*l.c*l.h*l.w, l.c, l.h, l.w, l.size, l.stride, l.pad, state.workspace, l.bit_align);
                    //cudaDeviceSynchronize();
                }
                else
                {
                    im2col_align_ongpu(state.input + i*l.c*l.h*l.w, l.c, l.h, l.w, l.size, l.stride, l.pad, l.align_workspace_gpu, l.bit_align);
                    //cudaDeviceSynchronize();

                    // should be optimized
                    float_to_bit_gpu(l.align_workspace_gpu, (unsigned char *)state.workspace, l.align_workspace_size);
                    //cudaDeviceSynchronize();
                }

                transpose_bin_gpu((unsigned char *)state.workspace, (unsigned char *)l.transposed_align_workspace_gpu, k, n, l.bit_align, new_ldb, 8);
                //cudaDeviceSynchronize();

                // should be optimized
                gemm_nn_custom_bin_mean_transposed_gpu(m, n, k,
                    (unsigned char *)l.align_bit_weights_gpu, new_ldb, (unsigned char *)l.transposed_align_workspace_gpu, new_ldb, l.output_gpu, n, l.mean_arr_gpu, l.biases_gpu);

                //gemm_nn_custom_bin_mean_transposed_sequentially_gpu(m, n, k,
                //    (unsigned char *)l.align_bit_weights_gpu, new_ldb, (unsigned char *)l.transposed_align_workspace_gpu, new_ldb, l.output_gpu, n, l.mean_arr_gpu);

                //cudaDeviceSynchronize();
                //check_error(status);
            }

            //add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
            if (l.activation != LINEAR) activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);
            //cudaDeviceSynchronize();
            return;
        }

        if (!l.align_bit_weights_gpu) {
            binarize_weights_gpu(l.weights_gpu, l.n, l.c*l.size*l.size, l.binary_weights_gpu);
        }
        l.weights_gpu = l.binary_weights_gpu;

        binarize_gpu(state.input, l.c*l.h*l.w*l.batch, l.binary_input_gpu);
        state.input = l.binary_input_gpu;

    }


    // blas_kernels.cu
    fill_ongpu(l.outputs*l.batch, 0, l.output_gpu, 1);

    int size = l.inputs * l.batch;

    float one = 1;
    float zero = 0;
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
        &zero,
        l.dstTensorDesc,
        l.output_gpu);


    if (l.batch_normalize) {
        // blas_kernels.cu
        //normalize_gpu(l.output_gpu, l.rolling_mean_gpu, l.rolling_variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
        //scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
    }

    // blas_kernels.cu
    add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);

    // blas_kernels.cu
    if (l.activation != LINEAR) activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);
}


// 4 layers in 1: convolution, batch-normalization, BIAS and activation
void forward_convolutional_layer_gpu_cudnn_quantized(layer l, network_state state)
{
    int i;

    // blas_kernels.cu
    //fill_ongpu(l.outputs*l.batch, 0, l.output_gpu, 1);

    int size = l.inputs * l.batch;
    cudaError_t status;

    /*
    static int once = 1;
    if (once) {
    //printf(" l.input_quant_multipler = %f \n", l.input_quant_multipler);
    once = 0;
    cuda_convert_f32_to_int8(state.input, size, state.input_int8, l.input_quant_multipler, (256 / 2 - 1)); // 7-bit (1-bit sign)
    //cuda_convert_int8_to_f32(state.input_int8, size, state.input, 1.0F / l.input_quant_multipler);
    }
    else {
    //printf(" NEXT!!! \n");
    //cuda_convert_f32_to_int8(state.input, size, state.input_int8, l.input_quant_multipler, (256 / 2 - 1)); // 7-bit (1-bit sign)
    cuda_convert_f32_to_int8_nomax(state.input, size, state.input_int8, l.input_quant_multipler); // 7-bit (1-bit sign)
    }
    */


    //#if(CUDNN_MAJOR >= 7 )
#define INT8CONV
    //#endif    // #if(CUDNN_MAJOR >= 7 )

#ifdef INT8CONV
    {
        float one = 1;
        float zero = 0;


        // input
        cudnnTensorDescriptor_t srcTransformDesc;
        cudnnCreateTensorDescriptor(&srcTransformDesc);
        cudnnSetTensor4dDescriptor(srcTransformDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_INT8, l.batch, l.c, l.h, l.w);

        cuda_convert_f32_to_int8(state.input, size, state.input_int8, l.input_quant_multipler, (256 / 2 - 1)); // 7-bit (1-bit sign)
                                                                                                               //cuda_convert_f32_to_int8_nomax(state.input, size, state.input_int8, l.input_quant_multipler); // 7-bit (1-bit sign)

        //printf("\n l.input_quant_multipler = %f \n", l.input_quant_multipler);

        cudnnStatus_t transform_status =
            cudnnTransformTensor(
                cudnn_handle(),
                &one,
                srcTransformDesc,
                state.input_int8, //input_init_int8,
                &zero,
                l.srcTensorDesc,
                state.input);

        checkCUDNN(transform_status);

        //float ALPHA1 = l.output_multipler / R_MULT;
        float ALPHA1 = 1 / (l.input_quant_multipler * l.weights_quant_multipler);
        //float ALPHA2 = 0;
        //printf(" ALPHA1 = %f \n", ALPHA1);


        //   x          w        y and z   bias     alpha1/alpha2
        // X_INT8    X_INT8    X_INT8    X_FLOAT     X_FLOAT

        // y = act ( alpha1 * conv(x) + alpha2 * z + bias )
        cudnnStatus_t cudnnstat =
            cudnnConvolutionBiasActivationForward(cudnn_handle(),
                &ALPHA1,    // ALPHA
                l.srcTensorDesc,
                state.input,
                l.weightDesc,
                l.weights_int8_int8x4_gpu, //l.weights_gpu,
                l.convDesc,
                l.fw_algo,
                state.workspace,
                l.workspace_size,
                &zero,    // ALPHA2
                l.dstTensorDesc,
                l.output_gpu,
                l.biasTensorDesc,
                l.biases_gpu,
                l.activationDesc,
                l.dstTensorDesc,
                l.output_gpu);




        /*
        // cuDNN >= v5.1
        cudnnStatus_t cudnnstat =
        cudnnConvolutionForward(cudnn_handle(),
        &ALPHA1,//&one,
        l.srcTensorDesc,
        state.input, //state.input_int8, // state.input,
        l.weightDesc,
        l.weights_int8_int8x4_gpu, //l.weights_int8_gpu, //l.weights_gpu,
        l.convDesc,
        l.fw_algo,
        state.workspace,
        l.workspace_size,
        &zero,
        l.dstTensorDesc,
        l.output_gpu);

        */


        //printf("  l.w = %d, l.h = %d, l.c = %d, l.n = %d \n", l.w, l.h, l.c, l.n);
        if (cudnnstat != CUDNN_STATUS_SUCCESS) {
            if (cudnnstat == CUDNN_STATUS_ARCH_MISMATCH) {
                printf("\n Error: CUDNN_STATUS_ARCH_MISMATCH - This GPU doesn't support DP4A (INT8 weights and input) \n");
            }
            else if (cudnnstat == CUDNN_STATUS_NOT_SUPPORTED) {
                printf("\n Error: CUDNN_STATUS_NOT_SUPPORTED (INT8 weights and input) \n");
            }
            else if (cudnnstat == CUDNN_STATUS_BAD_PARAM) {
                printf("\n Error: CUDNN_STATUS_BAD_PARAM (INT8 weights and input) \n");
            }
            printf("\n cudnnstat = %d \n", cudnnstat);
            getchar();
        }
        else {
            //printf("\n cudnnstat == CUDNN_STATUS_SUCCESS \n");
        }



        //status = cudaMemcpy(l.output, l.output_gpu, sizeof(float)*l.outputs*l.batch, cudaMemcpyDeviceToHost);
        //for (i = 0; i < l.outputs && i < 100; ++i) printf(" %f, ", l.output[i]);
        //draw_distribution(l.output, l.outputs*l.batch, "Output");


        //add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);

    }
#else // INT8CONV

    float one = 1;
    float zero = 0;
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
        &zero,
        l.dstTensorDesc,
        l.output_gpu);


    if (l.batch_normalize) {
        // blas_kernels.cu
        //normalize_gpu(l.output_gpu, l.rolling_mean_gpu, l.rolling_variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
        //scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
    }

    // blas_kernels.cu
    add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
#endif // INT8CONV


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


// upsample_layer.c
void forward_upsample_layer_cuda(const layer l, network_state state)
{
    fill_ongpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    //printf(" l.reverse = %d \n", l.reverse);
    if (l.reverse) {
        upsample_gpu(l.output_gpu, l.out_w, l.out_h, l.c, l.batch, l.stride, 0, l.scale, state.input);
    }
    else {
        upsample_gpu(state.input, l.w, l.h, l.c, l.batch, l.stride, 1, l.scale, l.output_gpu);
    }
}

// shortcut_layer.c
void forward_shortcut_layer_cuda(const layer l, network_state state)
{
    copy_ongpu(l.outputs*l.batch, state.input, 1, l.output_gpu, 1);
    shortcut_gpu(l.batch, l.w, l.h, l.c, state.net.layers[l.index].output_gpu, l.out_w, l.out_h, l.out_c, l.output_gpu);
    activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

// region layer
void forward_region_layer_gpu_cuda(const layer l, network_state state)
{
    // blas_kernels.cu
    flatten_ongpu(state.input, l.h*l.w, l.n*(l.coords + l.classes + 1), l.batch, 1, l.output_gpu);
    if (l.softmax_tree) {    // Yolo 9000
        int i;
        int count = 5;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            // blas_kernels.cu
            softmax_gpu(l.output_gpu + count, group_size, l.classes + 5, l.w*l.h*l.n*l.batch, 1, l.output_gpu + count);
            count += group_size;
        }
    }
    else if (l.softmax) {    // Yolo v2
                            // blas_kernels.cu
        softmax_gpu(l.output_gpu + 5, l.classes, l.classes + 5, l.w*l.h*l.n*l.batch, 1, l.output_gpu + 5);
    }

    float *in_cpu = (float *)calloc(l.batch*l.inputs, sizeof(float));
    float *truth_cpu = 0;
    if (state.truth) {
        int num_truth = l.batch*l.truths;
        truth_cpu = (float *)calloc(num_truth, sizeof(float));
        cudaError_t status = cudaMemcpy(state.truth, truth_cpu, num_truth * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaError_t status = cudaMemcpy(in_cpu, l.output_gpu, l.batch*l.inputs * sizeof(float), cudaMemcpyDeviceToHost);
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
            l.output[index + 4] = 1.0F / (1.0F + expf(-x));    // logistic_activate_cpu(l.output[index + 4]);
        }
    }

    free(cpu_state.input);
}

// yolo_layer.c Yolo v3
void forward_yolo_layer_cuda(const layer l, network_state state)
{
    copy_ongpu(l.batch*l.inputs, state.input, 1, l.output_gpu, 1);
    int b, n;
    for (b = 0; b < l.batch; ++b) {
        for (n = 0; n < l.n; ++n) {
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array_ongpu(l.output_gpu + index, 2 * l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array_ongpu(l.output_gpu + index, (1 + l.classes)*l.w*l.h, LOGISTIC);
        }
    }

    cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
    //return;
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
        else if (l.type == UPSAMPLE) {
            forward_upsample_layer_cuda(l, state);
            //printf("\n UPSAMPLE \n");
        }
        else if (l.type == SHORTCUT) {
            forward_shortcut_layer_cuda(l, state);
            //printf("\n SHORTCUT \n");
        }
        else if (l.type == YOLO) {
            forward_yolo_layer_cuda(l, state);
            //printf("\n YOLO \n");
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


void forward_network_gpu_cudnn_quantized(network net, network_state state)
{
    state.workspace = net.workspace;
    int i;
    for (i = 0; i < net.n; ++i) {
        state.index = i;
        layer l = net.layers[i];

        if (l.type == CONVOLUTIONAL) {

            //printf("\n %d - CONVOLUTIONAL \t\t l.size = %d  \n", i, l.size);
            //if (l.quantized && i != 80 && i != 92 && i != 104) forward_convolutional_layer_gpu_cudnn_quantized(l, state); // mAP = 0, very strange
            if (l.quantized) forward_convolutional_layer_gpu_cudnn_quantized(l, state);
            else forward_convolutional_layer_gpu_cudnn(l, state);
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
        else if (l.type == UPSAMPLE) {
            forward_upsample_layer_cuda(l, state);
            //printf("\n UPSAMPLE \n");
        }
        else if (l.type == SHORTCUT) {
            forward_shortcut_layer_cuda(l, state);
            //printf("\n SHORTCUT \n");
        }
        else if (l.type == YOLO) {
            forward_yolo_layer_cuda(l, state);
            //printf("\n YOLO \n");
        }
        else if (l.type == REGION) {
            forward_region_layer_gpu_cuda(l, state);
            //printf("\n REGION \n");
        }
        else {
            printf("\n layer: %d \n", l.type);
        }
        state.input = l.output_gpu;
        state.input_int8 = l.output_gpu_int8;
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
    //status = cudaMalloc((void **)&(state.input), sizeof(float)*size);
    state.input = net.input_state_gpu;
    status = cudaMemcpy(state.input, input, sizeof(float)*size, cudaMemcpyHostToDevice);
    state.truth = 0;
    state.train = 0;
    state.delta = 0;

    forward_network_gpu_cudnn(net, state); // network on GPU
    //status = cudaFree(state.input);
    //status = cudaFree(state.input_int8);
    //float *out = get_network_output_gpu(net);
    int i;
    for (i = net.n - 1; i > 0; --i) if (net.layers[i].type != COST) break;
    layer l = net.layers[i];
    if (l.type != REGION && l.type != YOLO) status = cudaMemcpy(l.output, l.output_gpu, l.outputs*l.batch * sizeof(float), cudaMemcpyDeviceToHost);
    return l.output;
}


// detect on GPU
float *network_predict_gpu_cudnn_quantized(network net, float *input)
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

    forward_network_gpu_cudnn_quantized(net, state); // network on GPU
    status = cudaFree(state.input);
    //status = cudaFree(state.input_int8);
    //float *out = get_network_output_gpu(net);
    int i;
    for (i = net.n - 1; i > 0; --i) if (net.layers[i].type != COST) break;
    layer l = net.layers[i];
    if (l.type != REGION && l.type != YOLO) status = cudaMemcpy(l.output, l.output_gpu, l.outputs*l.batch * sizeof(float), cudaMemcpyDeviceToHost);
    return l.output;
}

// init weights and cuDNN for quantized IINT8x4
void init_gpu_int8x4(network net)
{
    cudaError_t status = cudaSetDevice(net.gpu_index);

    int k;
    for (k = 0; k < net.n; ++k) {
        layer &l = net.layers[k];
        if (l.type == CONVOLUTIONAL && k > 0) {
            if (l.weights_int8_gpu == NULL) {
                size_t const weights_size = l.size*l.size*l.c*l.n;
                status = cudaMalloc((void **)&(l.weights_int8_gpu), sizeof(int8_t)*weights_size);
                status = cudaMalloc((void **)&(l.weights_int8_int8x4_gpu), sizeof(int8_t)*weights_size);
                status = cudaMemcpy(l.weights_int8_gpu, l.weights_int8, sizeof(int8_t)*weights_size, cudaMemcpyHostToDevice);

                // convert weights CUDNN_TENSOR_NCHW -> CUDNN_TENSOR_NCHW_VECT_C
                cudnnTensorDescriptor_t src_weights_desc;
                cudnnCreateTensorDescriptor(&src_weights_desc);
                cudnnSetTensor4dDescriptor(src_weights_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_INT8, l.n, l.c, l.size, l.size);

                cudnnTensorDescriptor_t dst_weights_desc;
                cudnnCreateTensorDescriptor(&dst_weights_desc);
                cudnnSetTensor4dDescriptor(dst_weights_desc, CUDNN_TENSOR_NCHW_VECT_C, CUDNN_DATA_INT8x4, l.n, l.c, l.size, l.size);

                float one = 1;
                float zero = 0;
                cudnnStatus_t transform_status;
                transform_status =
                    cudnnTransformTensor(
                        cudnn_handle(),
                        &one,
                        src_weights_desc,
                        l.weights_int8_gpu,
                        &zero,
                        dst_weights_desc,
                        l.weights_int8_int8x4_gpu);

                checkCUDNN(transform_status);

                cudnnDestroyTensorDescriptor(src_weights_desc);
                cudnnDestroyTensorDescriptor(dst_weights_desc);

                status = cudaMalloc((void **)&(l.biases_quant_gpu), sizeof(float)*l.n);
                status = cudaMemcpy(l.biases_quant_gpu, l.biases_quant, sizeof(float)*l.n, cudaMemcpyHostToDevice);
            }
        }
    }
}


