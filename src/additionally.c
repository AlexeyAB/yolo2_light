#include "additionally.h"
#include "gpu.h"

#ifdef CUDNN
#pragma comment(lib, "cudnn.lib")  
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
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col) 
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

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

// -------------- blas.c --------------

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
				printf("%ld\n", size);
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
	for (i = 0; i < n; i += sub) sum += pow(a[i] - b[i], 2);
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
    if(n <= 0) return -1;
    int i, max_i = 0;
    float max = a[0];
    for(i = 1; i < n; ++i){
        if(a[i] > max){
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
    for(i = 0; i < n; ++i) free(ptrs[i]);
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
    if(!file) file_error(filename);
    list *lines = make_list();
    while((path=fgetl(file))){
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
    while(n){
        a[count++] = n->val;
        n = n->next;
    }
    return a;
}

// list.c
void free_node(node *n)
{
	node *next;
	while(n) {
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
    network net = {0};
    net.n = n;
    net.layers = calloc(net.n, sizeof(layer));
    net.seen = calloc(1, sizeof(int));
    #ifdef GPU
    net.input_gpu = calloc(1, sizeof(float *));
    net.truth_gpu = calloc(1, sizeof(float *));
    #endif
    return net;
}


// network.c
#ifdef GPU
#ifdef CUDNN
void cudnn_convolutional_setup(layer *l)
{
    cudnnSetTensor4dDescriptor(l->dsrcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->ddstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 
    cudnnSetFilter4dDescriptor(l->dweightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c, l->size, l->size); 

    cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 
    cudnnSetFilter4dDescriptor(l->weightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c, l->size, l->size); 
#if(CUDNN_MAJOR >= 6)
	cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);	// cudnn 6.0
#else
	cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION);	// cudnn 5.1
#endif
    cudnnGetConvolutionForwardAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->weightDesc,
            l->convDesc,
            l->dstTensorDesc,
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            0,
            &l->fw_algo);
    cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle(),
            l->weightDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dsrcTensorDesc,
            CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
            0,
            &l->bd_algo);
    cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dweightDesc,
            CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
            0,
            &l->bf_algo);
}
#endif
#endif



// network.c
void set_batch_network(network *net, int b)
{
    net->batch = b;
    int i;
    for(i = 0; i < net->n; ++i){
        net->layers[i].batch = b;
#ifdef CUDNN
        if(net->layers[i].type == CONVOLUTIONAL){
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
	if (l.bias_updates)       free(l.bias_updates);
	if (l.scales)             free(l.scales);
	if (l.scale_updates)      free(l.scale_updates);
	if (l.weights)            free(l.weights);
	if (l.weight_updates)     free(l.weight_updates);
	if (l.delta)              free(l.delta);
	if (l.output)             free(l.output);
	if (l.squared)            free(l.squared);
	if (l.norms)              free(l.norms);
	if (l.spatial_mean)       free(l.spatial_mean);
	if (l.mean)               free(l.mean);
	if (l.variance)           free(l.variance);
	if (l.mean_delta)         free(l.mean_delta);
	if (l.variance_delta)     free(l.variance_delta);
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
	if (l.weight_updates_gpu)      cuda_free(l.weight_updates_gpu);
	if (l.biases_gpu)              cuda_free(l.biases_gpu);
	if (l.bias_updates_gpu)        cuda_free(l.bias_updates_gpu);
	if (l.scales_gpu)              cuda_free(l.scales_gpu);
	if (l.scale_updates_gpu)       cuda_free(l.scale_updates_gpu);
	if (l.output_gpu)              cuda_free(l.output_gpu);
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
	l.delta = calloc(inputs*batch, sizeof(float));

	// commented only for this custom version of Yolo v2
	//l.forward = forward_softmax_layer;
	//l.backward = backward_softmax_layer;
#ifdef GPU
	// commented only for this custom version of Yolo v2
	//l.forward_gpu = forward_softmax_layer_gpu;
	//l.backward_gpu = backward_softmax_layer_gpu;

	l.output_gpu = cuda_make_array(l.output, inputs*batch);
	l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
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
	l.delta = calloc(output_size, sizeof(float));

	// commented only for this custom version of Yolo v2
	//l.forward = forward_reorg_layer;
	//l.backward = backward_reorg_layer;
#ifdef GPU
	// commented only for this custom version of Yolo v2
	//l.forward_gpu = forward_reorg_layer_gpu;
	//l.backward_gpu = backward_reorg_layer_gpu;

	l.output_gpu = cuda_make_array(l.output, output_size);
	l.delta_gpu = cuda_make_array(l.delta, output_size);
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
	l.delta = calloc(outputs*batch, sizeof(float));
	l.output = calloc(outputs*batch, sizeof(float));;

	// commented only for this custom version of Yolo v2
	//l.forward = forward_route_layer;
	//l.backward = backward_route_layer;
#ifdef GPU
	// commented only for this custom version of Yolo v2
	//l.forward_gpu = forward_route_layer_gpu;
	//l.backward_gpu = backward_route_layer_gpu;

	l.delta_gpu = cuda_make_array(l.delta, outputs*batch);
	l.output_gpu = cuda_make_array(l.output, outputs*batch);
#endif
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
	l.bias_updates = calloc(n * 2, sizeof(float));
	l.outputs = h*w*n*(classes + coords + 1);
	l.inputs = l.outputs;
	l.truths = 30 * (5);
	l.delta = calloc(batch*l.outputs, sizeof(float));
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
	l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
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
	l.out_w = (w + 2 * padding) / stride;
	l.out_h = (h + 2 * padding) / stride;
	l.out_c = c;
	l.outputs = l.out_h * l.out_w * l.out_c;
	l.inputs = h*w*c;
	l.size = size;
	l.stride = stride;
	int output_size = l.out_h * l.out_w * l.out_c * batch;
	l.indexes = calloc(output_size, sizeof(int));
	l.output = calloc(output_size, sizeof(float));
	l.delta = calloc(output_size, sizeof(float));
	// commented only for this custom version of Yolo v2
	//l.forward = forward_maxpool_layer;
	//l.backward = backward_maxpool_layer;
#ifdef GPU
	// commented only for this custom version of Yolo v2
	//l.forward_gpu = forward_maxpool_layer_gpu;
	//l.backward_gpu = backward_maxpool_layer_gpu;
	l.indexes_gpu = cuda_make_int_array(output_size);
	l.output_gpu = cuda_make_array(l.output, output_size);
	l.delta_gpu = cuda_make_array(l.delta, output_size);
#endif
	fprintf(stderr, "max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
	return l;
}


// -------------- convolutional_layer.c --------------


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
			&s);
		if (s > most) most = s;
		return most;
	}
#endif
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
convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam)
{
	int i;
	convolutional_layer l = { 0 };
	l.type = CONVOLUTIONAL;

	l.h = h;
	l.w = w;
	l.c = c;
	l.n = n;
	l.binary = binary;
	l.xnor = xnor;
	l.batch = batch;
	l.stride = stride;
	l.size = size;
	l.pad = padding;
	l.batch_normalize = batch_normalize;

	l.weights = calloc(c*n*size*size, sizeof(float));
	l.weight_updates = calloc(c*n*size*size, sizeof(float));

	l.biases = calloc(n, sizeof(float));
	l.bias_updates = calloc(n, sizeof(float));

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
	l.delta = calloc(l.batch*l.outputs, sizeof(float));

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
	}

	if (batch_normalize) {
		l.scales = calloc(n, sizeof(float));
		l.scale_updates = calloc(n, sizeof(float));
		for (i = 0; i < n; ++i) {
			l.scales[i] = 1;
		}

		l.mean = calloc(n, sizeof(float));
		l.variance = calloc(n, sizeof(float));

		l.mean_delta = calloc(n, sizeof(float));
		l.variance_delta = calloc(n, sizeof(float));

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
		if (adam) {
			l.m_gpu = cuda_make_array(l.m, c*n*size*size);
			l.v_gpu = cuda_make_array(l.v, c*n*size*size);
		}

		l.weights_gpu = cuda_make_array(l.weights, c*n*size*size);
		l.weight_updates_gpu = cuda_make_array(l.weight_updates, c*n*size*size);

		l.biases_gpu = cuda_make_array(l.biases, n);
		l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

		l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*n);
		l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);

		if (binary) {
			l.binary_weights_gpu = cuda_make_array(l.weights, c*n*size*size);
		}
		if (xnor) {
			l.binary_weights_gpu = cuda_make_array(l.weights, c*n*size*size);
			l.binary_input_gpu = cuda_make_array(0, l.inputs*l.batch);
		}

		if (batch_normalize) {
			l.mean_gpu = cuda_make_array(l.mean, n);
			l.variance_gpu = cuda_make_array(l.variance, n);

			l.rolling_mean_gpu = cuda_make_array(l.mean, n);
			l.rolling_variance_gpu = cuda_make_array(l.variance, n);

			l.mean_delta_gpu = cuda_make_array(l.mean, n);
			l.variance_delta_gpu = cuda_make_array(l.variance, n);

			l.scales_gpu = cuda_make_array(l.scales, n);
			l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);

			l.x_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
			l.x_norm_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
		}
#ifdef CUDNN
		cudnnCreateTensorDescriptor(&l.srcTensorDesc);
		cudnnCreateTensorDescriptor(&l.dstTensorDesc);
		cudnnCreateFilterDescriptor(&l.weightDesc);
		cudnnCreateTensorDescriptor(&l.dsrcTensorDesc);
		cudnnCreateTensorDescriptor(&l.ddstTensorDesc);
		cudnnCreateFilterDescriptor(&l.dweightDesc);
		cudnnCreateConvolutionDescriptor(&l.convDesc);
		cudnn_convolutional_setup(&l);
#endif
	}
#endif
	l.workspace_size = get_workspace_size(l);
	l.activation = activation;

	fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);

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
#endif	// OPENCV

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

	CvSize size;
	size.width = disp->width, size.height = disp->height;
	static CvVideoWriter* output_video = NULL;    // cv::VideoWriter output_video;
	if (output_video == NULL)
	{
		const char* output_name = "test_dnn_out.avi";
		output_video = cvCreateVideoWriter(output_name, CV_FOURCC('D', 'I', 'V', 'X'), 25, size, 1);
	}
	cvWriteFrame(output_video, disp);	// comment this line to improve FPS !!!

	cvReleaseImage(&disp);
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
/*	if (l.adam) {
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
	fread(net->seen, sizeof(int), 1, fp);
	int transpose = (major > 1000) || (minor > 1000);

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

	convolutional_layer layer = make_convolutional_layer(batch, h, w, c, n, size, stride, padding, activation, batch_normalize, binary, xnor, params.net.adam);
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
	int padding = option_find_int_quiet(options, "padding", (size - 1) / 2);

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
	if (strcmp(type, "[region]") == 0) return REGION;
	if (strcmp(type, "[conv]") == 0
		|| strcmp(type, "[convolutional]") == 0) return CONVOLUTIONAL;
	if (strcmp(type, "[net]") == 0
		|| strcmp(type, "[network]") == 0) return NETWORK;
	if (strcmp(type, "[max]") == 0
		|| strcmp(type, "[maxpool]") == 0) return MAXPOOL;
	if (strcmp(type, "[reorg]") == 0) return REORG;
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
network parse_network_cfg(char *filename)
{
	list *sections = read_cfg(filename);
	node *n = sections->front;
	if (!n) error("Config file has no sections");
	network net = make_network(sections->size - 1);
	net.gpu_index = gpu_index;
	size_params params;

	section *s = (section *)n->val;
	list *options = s->options;
	if (strcmp(s->type, "[net]") == 0 && strcmp(s->type, "[network]") == 0)
		error("First section must be [net] or [network]");
	parse_net_options(options, &net);

	params.h = net.h;
	params.w = net.w;
	params.c = net.c;
	params.inputs = net.inputs;
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
			l = parse_convolutional(options, params);
		}
		else if (lt == REGION) {
			l = parse_region(options, params);
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
		}
		else {
			net.workspace = calloc(1, workspace_size);
		}
#else
		net.workspace = calloc(1, workspace_size);
#endif
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




#endif