#include "additionally.h"
#include "gpu.h"

#ifdef CUDNN
#pragma comment(lib, "cudnn.lib")  
#endif


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
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION);
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