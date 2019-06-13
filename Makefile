GPU=1
OPENCV=0
OPENMP=0
AVX=0
SSE=0

# set GPU=1 to speedup on GPU with cuDNN
# set AVX=1, SSE=1 and OPENMP=1 to speedup on CPU (if error occurs then set AVX=0)

DEBUG=0

ifeq ($(GPU), 1)
CUDNN=1
endif

ARCH= -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52] \
      -gencode arch=compute_61,code=[sm_61,compute_61]

# GeForce RTX 2080 Ti, RTX 2080, RTX 2070, Quadro RTX 8000, Quadro RTX 6000, Quadro RTX 5000, Tesla T4, XNOR Tensor Cores
# ARCH= -gencode arch=compute_75,code=[sm_75,compute_75]

# Tesla V100
# ARCH= -gencode arch=compute_70,code=[sm_70,compute_70]

# GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030, Titan Xp, Tesla P40, Tesla P4
# ARCH= -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=compute_61

# GP100/Tesla P100 – DGX-1
# ARCH= -gencode arch=compute_60,code=sm_60

# For Jetson Tx1 uncomment:
# ARCH= -gencode arch=compute_51,code=[sm_51,compute_51]

# For Jetson Tx2 or Drive-PX2 uncomment:
# ARCH= -gencode arch=compute_62,code=[sm_62,compute_62]


VPATH=./src/
EXEC=./bin/darknet
OBJDIR=./obj/

CC=gcc
CPP=g++
NVCC=nvcc 
OPTS=-Ofast
LDFLAGS= -lm -pthread 
COMMON= 
CFLAGS=-Wall -Wfatal-errors

ifeq ($(DEBUG), 1) 
OPTS= -Og -g
endif

ifeq ($(AVX), 1) 
CFLAGS+= -ffp-contract=fast -msse3 -msse4.1 -msse4.2 -msse4a -mavx -mavx2 -mfma -DAVX
endif

ifeq ($(SSE), 1) 
CFLAGS+= -ffp-contract=fast -msse4.1 -msse4a -DSSE41
endif

CFLAGS+=$(OPTS)

ifeq ($(OPENCV), 1) 
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
LDFLAGS+= `pkg-config --libs opencv` 
COMMON+= `pkg-config --cflags opencv` 
endif

ifeq ($(OPENMP), 1)
CFLAGS+= -fopenmp
LDFLAGS+= -lgomp
endif

ifeq ($(GPU), 1) 
COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
endif

ifeq ($(CUDNN), 1) 
COMMON+= -DCUDNN 
CFLAGS+= -DCUDNN -I/usr/local/cudnn/include
LDFLAGS+= -L/usr/local/cudnn/lib64 -lcudnn
endif

OBJ=main.o additionally.o box.o yolov2_forward_network.o yolov2_forward_network_quantized.o
ifeq ($(GPU), 1) 
LDFLAGS+= -lstdc++ 
OBJ+=gpu.o yolov2_forward_network_gpu.o 
endif

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile

all: obj bash results $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

obj:
	mkdir -p obj
bash:
	find . -name "*.sh" -exec chmod +x {} \;
results:
	mkdir -p results

.PHONY: clean

clean:
	rm -rf $(OBJS) $(EXEC)

