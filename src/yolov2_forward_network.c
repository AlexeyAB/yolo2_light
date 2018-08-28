#include "additionally.h"    // some definitions from: im2col.h, blas.h, list.h, utils.h, activations.h, tree.h, layer.h, network.h
// softmax_layer.h, reorg_layer.h, route_layer.h, region_layer.h, maxpool_layer.h, convolutional_layer.h

#define GEMMCONV

/*
// from: box.h
typedef struct {
    float x, y, w, h;
} box;
*/


// binary transpose
size_t binary_transpose_align_input(int k, int n, float *b, char **t_bit_input, size_t ldb_align, int bit_align)
{
    size_t new_ldb = k + (ldb_align - k%ldb_align); // (k / 8 + 1) * 8;
    size_t t_intput_size = new_ldb * n;
    size_t t_bit_input_size = t_intput_size / 8;// +1;
    *t_bit_input = calloc(t_bit_input_size, sizeof(char));

    //printf("\n t_bit_input_size = %d, k = %d, n = %d, new_ldb = %d \n", t_bit_input_size, k, n, new_ldb);
    int src_size = k * bit_align;
    transpose_bin(b, *t_bit_input, k, n, bit_align, new_ldb, 8);

    return t_intput_size;
}

// 4 layers in 1: convolution, batch-normalization, BIAS and activation
void forward_convolutional_layer_cpu(layer l, network_state state)
{

    int out_h = (l.h + 2 * l.pad - l.size) / l.stride + 1;    // output_height=input_height for stride=1 and pad=1
    int out_w = (l.w + 2 * l.pad - l.size) / l.stride + 1;    // output_width=input_width for stride=1 and pad=1
    int i, f, j;

    // fill zero (ALPHA)
    for (i = 0; i < l.outputs; ++i) l.output[i] = 0;

    if (l.xnor) {
        if (!l.align_bit_weights)
        {
            binarize_weights(l.weights, l.n, l.c*l.size*l.size, l.binary_weights);
            //printf("\n binarize_weights l.align_bit_weights = %p \n", l.align_bit_weights);
        }
        binarize_cpu(state.input, l.c*l.h*l.w*l.batch, l.binary_input);

        l.weights = l.binary_weights;
        state.input = l.binary_input;
    }

    // l.n - number of filters on this layer
    // l.c - channels of input-array
    // l.h - height of input-array
    // l.w - width of input-array
    // l.size - width and height of filters (the same size for all filters)


    // 1. Convolution !!!
#ifndef GEMMCONV
    int fil;
    // filter index
#pragma omp parallel for      // "omp parallel for" - automatic parallelization of loop by using OpenMP
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
                    float sum = 0;

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

                            sum += state.input[input_index] * l.weights[weights_index];
                        }
                    }
                    // l.output[filters][width][height] +=
                    //        state.input[channels][width][height] *
                    //        l.weights[filters][channels][filter_width][filter_height];
                    l.output[output_index] += sum;
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
        //im2col_cpu(state.input, l.c, l.h, l.w, l.size, l.stride, l.pad, b);    // im2col.c
        //im2col_cpu_custom(state.input, l.c, l.h, l.w, l.size, l.stride, l.pad, b);    // AVX2

        // XNOR-net - bit-1: weights, input, calculation
        if (l.xnor && (l.stride == 1 && l.pad == 1)) {
            memset(b, 0, l.bit_align*l.size*l.size*l.c * sizeof(float));
            //im2col_cpu_custom_align(state.input, l.c, l.h, l.w, l.size, l.stride, l.pad, b, l.bit_align);
            im2col_cpu_custom_bin(state.input, l.c, l.h, l.w, l.size, l.stride, l.pad, b, l.bit_align);

            int ldb_align = l.lda_align;
            size_t new_ldb = k + (ldb_align - k%ldb_align);
            char *t_bit_input = NULL;
            size_t t_intput_size = binary_transpose_align_input(k, n, b, &t_bit_input, ldb_align, l.bit_align);

            // 5x times faster than gemm()-float32
            gemm_nn_custom_bin_mean_transposed(m, n, k, 1, l.align_bit_weights, new_ldb, t_bit_input, new_ldb, c, n, l.mean_arr);

            //gemm_nn_custom_bin_mean_transposed(m, n, k, 1, bit_weights, k, t_bit_input, new_ldb, c, n, mean_arr);

            //free(t_input);
            free(t_bit_input);
        }
        else {
            im2col_cpu_custom(state.input, l.c, l.h, l.w, l.size, l.stride, l.pad, b);    // AVX2
            int t;
            #pragma omp parallel for
            for (t = 0; t < m; ++t) {
                gemm_nn(1, n, k, 1, a + t*k, k, b, n, c + t*n, n);
            }
        }
        c += n*m;
        state.input += l.c*l.h*l.w;
    }

#endif

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
    //if (l.activation == LEAKY) {
    //    for (i = 0; i < l.n*out_size; ++i) {
    //        l.output[i] = leaky_activate(l.output[i]);
    //    }
    //}
    activate_array_cpu_custom(l.output, l.n*out_size, l.activation);

}



// MAX pooling layer
void forward_maxpool_layer_cpu(const layer l, network_state state)
{
    if (!state.train) {
        forward_maxpool_layer_avx(state.input, l.output, l.indexes, l.size, l.w, l.h, l.out_w, l.out_h, l.c, l.pad, l.stride, l.batch);
        return;
    }

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
                            max_i = (val > max) ? index : max_i;    // get max index
                            max = (val > max) ? val : max;            // get max value
                        }
                    }
                    l.output[out_index] = max;        // store max value
                    l.indexes[out_index] = max_i;    // store max index
                }
            }
        }
    }
}


// Route layer - just copy 1 or more layers into the current layer
void forward_route_layer_cpu(const layer l, network_state state)
{
    int i, j;
    int offset = 0;
    // number of merged layers
    for (i = 0; i < l.n; ++i) {
        int index = l.input_layers[i];                    // source layer index
        float *input = state.net.layers[index].output;    // source layer output ptr
        int input_size = l.input_sizes[i];                // source layer size
                                                        // batch index
        for (j = 0; j < l.batch; ++j) {
            memcpy(l.output + offset + j*l.outputs, input + j*input_size, input_size * sizeof(float));
        }
        offset += input_size;
    }
}


// Reorg layer - just change dimension sizes of the previous layer (some dimension sizes are increased by decreasing other)
void forward_reorg_layer_cpu(const layer l, network_state state)
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

    //printf("\n out_c = %d, out_w = %d, out_h = %d, stride = %d, forward = %d \n", out_c, out_w, out_h, stride, forward);
    //printf("  in_c = %d,  in_w = %d,  in_h = %d \n", in_c, out_w*stride, out_h*stride);

    // batch
    for (b = 0; b < batch; ++b) {
        // channel
        for (k = 0; k < out_c; ++k) {
            // y
            for (j = 0; j < out_h; ++j) {
                // x
                for (i = 0; i < out_w; ++i) {
                    int in_index = i + out_w*(j + out_h*(k + out_c*b));
                    int c2 = k % in_c;
                    int offset = k / in_c;
                    int w2 = i*stride + offset % stride;
                    int h2 = j*stride + offset / stride;
                    int out_index = w2 + out_w*stride*(h2 + out_h*stride*(c2 + in_c*b));
                    out[in_index] = x[out_index];
                }
            }
        }
    }
}



// ---- upsample layer ----

// upsample_layer.c
void upsample_cpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{
    int i, j, k, b;
    for (b = 0; b < batch; ++b) {
        for (k = 0; k < c; ++k) {
            for (j = 0; j < h*stride; ++j) {
                for (i = 0; i < w*stride; ++i) {
                    int in_index = b*w*h*c + k*w*h + (j / stride)*w + i / stride;
                    int out_index = b*w*h*c*stride*stride + k*w*h*stride*stride + j*w*stride + i;
                    if (forward) out[out_index] = scale*in[in_index];
                    else in[in_index] += scale*out[out_index];
                }
            }
        }
    }
}

// upsample_layer.c
void forward_upsample_layer_cpu(const layer l, network_state net)
{
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    if (l.reverse) {
        upsample_cpu(l.output, l.out_w, l.out_h, l.c, l.batch, l.stride, 0, l.scale, net.input);
    }
    else {
        upsample_cpu(net.input, l.w, l.h, l.c, l.batch, l.stride, 1, l.scale, l.output);
    }
}

// blas.c (shortcut_layer)
void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float *out)
{
    int stride = w1 / w2;
    int sample = w2 / w1;
    assert(stride == h1 / h2);
    assert(sample == h2 / h1);
    if (stride < 1) stride = 1;
    if (sample < 1) sample = 1;
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int i, j, k, b;
    for (b = 0; b < batch; ++b) {
        for (k = 0; k < minc; ++k) {
            for (j = 0; j < minh; ++j) {
                for (i = 0; i < minw; ++i) {
                    int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
                    int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
                    out[out_index] += add[add_index];
                }
            }
        }
    }
}

// blas.c
void copy_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for (i = 0; i < N; ++i) Y[i*INCY] = X[i*INCX];
}

// shortcut_layer.c
void forward_shortcut_layer_cpu(const layer l, network_state state)
{
    copy_cpu(l.outputs*l.batch, state.input, 1, l.output, 1);
    shortcut_cpu(l.batch, l.w, l.h, l.c, state.net.layers[l.index].output, l.out_w, l.out_h, l.out_c, l.output);
    activate_array(l.output, l.outputs*l.batch, l.activation);
}

// ---- yolo layer ----

void forward_yolo_layer_cpu(const layer l, network_state state)
{
    int i, j, b, t, n;
    memcpy(l.output, state.input, l.outputs*l.batch * sizeof(float));

#ifndef GPU
    for (b = 0; b < l.batch; ++b) {
        for (n = 0; n < l.n; ++n) {
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array(l.output + index, 2 * l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array(l.output + index, (1 + l.classes)*l.w*l.h, LOGISTIC);
        }
    }
#endif


    //memset(l.delta, 0, l.outputs * l.batch * sizeof(float));

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
void forward_region_layer_cpu(const layer l, network_state state)
{
    int i, b;
    int size = l.coords + l.classes + 1;    // 4 Coords(x,y,w,h) + Classes + 1 Probability-t0
    memcpy(l.output, state.input, l.outputs*l.batch * sizeof(float));

    //flatten(l.output, l.w*l.h, size*l.n, l.batch, 1);
    // convert many channels to the one channel (depth=1)
    // (each grid cell will have a number of float-variables equal = to the initial number of channels)
    {
        float *x = l.output;
        int layer_size = l.w*l.h;    // W x H - size of layer
        int layers = size*l.n;        // number of channels (where l.n = number of anchors)
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
            l.output[index + 4] = 1.0F / (1.0F + expf(-x));    // logistic_activate_cpu(l.output[index + 4]);
        }
    }


    if (l.softmax_tree) {    // Yolo 9000
        for (b = 0; b < l.batch; ++b) {
            for (i = 0; i < l.h*l.w*l.n; ++i) {
                int index = size*i + b*l.outputs;
                softmax_tree(l.output + index + 5, 1, 0, 1, l.softmax_tree, l.output + index + 5);
            }
        }
    }
    else if (l.softmax) {    // Yolo v2
        // softmax activation only for Classes probability
        for (b = 0; b < l.batch; ++b) {
            // for each item (x, y, anchor-index)
            for (i = 0; i < l.h*l.w*l.n; ++i) {
                int index = size*i + b*l.outputs;
                softmax_cpu(l.output + index + 5, l.classes, 1, l.output + index + 5);
            }
        }
    }

}





void yolov2_forward_network_cpu(network net, network_state state)
{
    state.workspace = net.workspace;
    int i;
    for (i = 0; i < net.n; ++i) {
        state.index = i;
        layer l = net.layers[i];

        if (l.type == CONVOLUTIONAL) {
            forward_convolutional_layer_cpu(l, state);
            //printf("\n CONVOLUTIONAL \t\t l.size = %d  \n", l.size);
        }
        else if (l.type == MAXPOOL) {
            forward_maxpool_layer_cpu(l, state);
            //printf("\n MAXPOOL \t\t l.size = %d  \n", l.size);
        }
        else if (l.type == ROUTE) {
            forward_route_layer_cpu(l, state);
            //printf("\n ROUTE \t\t\t l.n = %d  \n", l.n);
        }
        else if (l.type == REORG) {
            forward_reorg_layer_cpu(l, state);
            //printf("\n REORG \n");
        }
        else if (l.type == UPSAMPLE) {
            forward_upsample_layer_cpu(l, state);
            //printf("\n UPSAMPLE \n");
        }
        else if (l.type == SHORTCUT) {
            forward_shortcut_layer_cpu(l, state);
            //printf("\n SHORTCUT \n");
        }
        else if (l.type == YOLO) {
            forward_yolo_layer_cpu(l, state);
            //printf("\n YOLO \n");
        }
        else if (l.type == REGION) {
            forward_region_layer_cpu(l, state);
            //printf("\n REGION \n");
        }
        else {
            printf("\n layer: %d \n", l.type);
        }


        state.input = l.output;
    }
}


// detect on CPU
float *network_predict_cpu(network net, float *input)
{
    network_state state;
    state.net = net;
    state.index = 0;
    state.input = input;
    state.truth = 0;
    state.train = 0;
    state.delta = 0;
    yolov2_forward_network_cpu(net, state);    // network on CPU
                                            //float *out = get_network_output(net);
    int i;
    for (i = net.n - 1; i > 0; --i) if (net.layers[i].type != COST) break;
    return net.layers[i].output;
}


// --------------------
// x - last conv-layer output
// biases - anchors from cfg-file
// n - number of anchors from cfg-file
box get_region_box_cpu(float *x, float *biases, int n, int index, int i, int j, int w, int h)
{
    box b;
    b.x = (i + logistic_activate(x[index + 0])) / w;    // (col + 1./(1. + exp(-x))) / width_last_layer
    b.y = (j + logistic_activate(x[index + 1])) / h;    // (row + 1./(1. + exp(-x))) / height_last_layer
    b.w = expf(x[index + 2]) * biases[2 * n] / w;        // exp(x) * anchor_w / width_last_layer
    b.h = expf(x[index + 3]) * biases[2 * n + 1] / h;    // exp(x) * anchor_h / height_last_layer
    return b;
}

// get prediction boxes
void get_region_boxes_cpu(layer l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness, int *map)
{
    int i, j, n;
    float *predictions = l.output;
    // grid index
    for (i = 0; i < l.w*l.h; ++i) {
        int row = i / l.w;
        int col = i % l.w;
        // anchor index
        for (n = 0; n < l.n; ++n) {
            int index = i*l.n + n;    // index for each grid-cell & anchor
            int p_index = index * (l.classes + 5) + 4;
            float scale = predictions[p_index];                // scale = t0 = Probability * IoU(box, object)
            if (l.classfix == -1 && scale < .5) scale = 0;    // if(t0 < 0.5) t0 = 0;
            int box_index = index * (l.classes + 5);
            boxes[index] = get_region_box_cpu(predictions, l.biases, n, box_index, col, row, l.w, l.h);
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
                    float prob = scale*predictions[class_index + j];    // prob = IoU(box, object) = t0 * class-probability
                    probs[index][j] = (prob > thresh) ? prob : 0;        // if (IoU < threshold) IoU = 0;
                }
            }
            if (only_objectness) {
                probs[index][0] = scale;
            }
        }
    }
}

// ------ Calibration --------

// detect on CPU
float *network_calibrate_cpu(network net, float *input)
{
    network_state state;
    state.net = net;
    state.index = 0;
    state.input = input;
    state.truth = 0;
    state.train = 0;
    state.delta = 0;
    //yolov2_forward_network_cpu(net, state);    // network on CPU

    // input calibration - for quantinization
    static int max_num = 100;
    static int counter = 0;
    static float *input_mult_array = NULL;
    if (net.do_input_calibration > 0) {    // calibration for quantinization
        max_num = net.do_input_calibration;
        if (input_mult_array == NULL) {
            input_mult_array = (float *)calloc(net.n * max_num, sizeof(float));
        }
        ++counter;
        // save calibration coefficients
        if (counter > max_num) {
            printf("\n\n Saving coefficients to the input_calibration.txt file... \n\n");
            FILE* fw = fopen("input_calibration.txt", "wb");
            char buff[1024];
            //printf("\n float input_mult[] = { ");
            char *str1 = "input_calibration = ";
            printf("%s", str1);
            fwrite(str1, sizeof(char), strlen(str1), fw);
            int i;
            for (i = 0; i < net.n; ++i)
                if (net.layers[i].type == CONVOLUTIONAL) {
                    printf("%g, ", input_mult_array[0 + i*max_num]);
                    sprintf(buff, "%g, ", input_mult_array[0 + i*max_num]);
                    fwrite(buff, sizeof(char), strlen(buff), fw);
                }
            char *str2 = "16";
            printf("%s \n ---------------------------", str2);
            fwrite(str2, sizeof(char), strlen(str2), fw);
            fclose(fw);
            getchar();
            exit(0);
        }
    }

    state.workspace = net.workspace;
    int i;
    for (i = 0; i < net.n; ++i) {
        state.index = i;
        layer l = net.layers[i];

        if (l.type == CONVOLUTIONAL) {
            if (net.do_input_calibration) {    // calibration for quantinization
                                            //float multiplier = entropy_calibration(state.input, l.inputs, 1.0 / 8192, 2048);
                float multiplier = entropy_calibration(state.input, l.inputs, 1.0 / 16, 4096);
                //float multiplier = entropy_calibration(state.input, l.inputs, 1.0 / 4, 2*4096);
                printf(" multiplier = %f, l.inputs = %d \n\n", multiplier, l.inputs);
                input_mult_array[counter + i*max_num] = multiplier;
                if (counter >= max_num) {
                    int j;
                    float res_mult = 0;
                    for (j = 0; j < max_num; ++j)
                        res_mult += input_mult_array[j + i*max_num];
                    res_mult = res_mult / max_num;
                    input_mult_array[0 + i*max_num] = res_mult;
                    printf(" res_mult = %f, max_num = %d \n", res_mult, max_num);
                }
            }

            forward_convolutional_layer_cpu(l, state);
            //printf("\n CONVOLUTIONAL \t\t l.size = %d  \n", l.size);
        }
        else if (l.type == MAXPOOL) {
            forward_maxpool_layer_cpu(l, state);
            //printf("\n MAXPOOL \t\t l.size = %d  \n", l.size);
        }
        else if (l.type == ROUTE) {
            forward_route_layer_cpu(l, state);
            //printf("\n ROUTE \t\t\t l.n = %d  \n", l.n);
        }
        else if (l.type == REORG) {
            forward_reorg_layer_cpu(l, state);
            //printf("\n REORG \n");
        }
        else if (l.type == REGION) {
            forward_region_layer_cpu(l, state);
            //printf("\n REGION \n");
        }
        else {
            printf("\n layer: %d \n", l.type);
        }


        state.input = l.output;
    }

    //int i;
    for (i = net.n - 1; i > 0; --i) if (net.layers[i].type != COST) break;
    return net.layers[i].output;
}
