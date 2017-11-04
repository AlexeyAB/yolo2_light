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


#ifndef OCLWRAPPER_H_
#define OCLWRAPPER_H_

#include <chrono>
#include <algorithm>
#include <string>
#include <iostream>
#include "CL/cl.h"
#include "cl_wrapper.hpp"
#include "clblast.h"
//#include "clblast_half.h"

#define PINNED_MEM_OUTPUT

#define PROFILE_KERNELS		0
#define BLOCK_KERNEL_EXEC	0


using namespace std;;

string ExePath();

/*
typedef struct {

	void			*m_PinnedMemory;
	OCLBuffer		*m_OCLBuffer;

}StructPinnedOCLBuffer;
*/


float sec(clock_t clocks);

enum {
	NN_KERNEL_IDX_IM2COL3X3,
	NN_KERNEL_IDX_IM2COL1X1,
	NN_KERNEL_IDX_ADDBIAS,
	NN_KERNEL_IDX_NORMSCALEADDBIAS,
	NN_KERNEL_IDX_LEAKY_ACTIVATE,
	NN_KERNEL_IDX_FLATARR,
	NN_KERNEL_IDX_SOFTMAX,
	NN_KERNEL_IDX_MAXPOOL,
	NN_KERNEL_IDX_REORG,
	NN_KERNEL_IDX_RESETARR,
	NN_MAX_KERNEL_COUNT
};

static const char* NN_KERNEL_NAMES[NN_MAX_KERNEL_COUNT] = {

	"image2columarray3x3",
	"image2columarray1x1",
	"addbias",
	"normscaleaddbias",
	"leakyactivatearray",
	"flattenarray",
	"softmax",
	"maxpool",
	"reorg",
	"resetarray"
};

class OCLManager {

public:

	OCLManager();
	~OCLManager();
	int Initialize();
	int Finalize();
	void ReleaseLock();
	void SetLock();

	const char *GetDeviceName() { return m_DeviceName; };

//private:

	Program*			m_OpenCLProgram;
	void				*m_RefObject;
	int					m_Status;
	int					m_LockStatus;
	int					m_CallerId;
	//HANDLE				m_LockMutex;
	CLSetup				m_OpenCLSetup;
	KernelLauncher*		m_OpenCLKernels[NN_MAX_KERNEL_COUNT];
	char				m_DeviceName[256];
};



#endif /* OCLWRAPPER_H_ */

