
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

#include "include/OCLManager.h"

#define BLOCK 8

#ifdef WIN32

#include <Windows.h>
string ExePath() {

	char buffer[MAX_PATH];
	GetModuleFileName(NULL, buffer, MAX_PATH);
	string::size_type pos = string(buffer).find_last_of("\\/");
	return string(buffer).substr(0, pos);
}

#elif __linux__

string ExePath() {

	return "";
}

#endif

float sec(clock_t clocks) {

	return (float)clocks / CLOCKS_PER_SEC;
}

typedef struct {

	int m_NumX;
	int m_NumY;
	int m_NumZ;
}StructOCLDims;

OCLManager::OCLManager() {

	m_OpenCLProgram = NULL;

	for(int i = 0; i < NN_MAX_KERNEL_COUNT; i++)
		m_OpenCLKernels[i] = 0;

	//m_LockMutex = NULL;
	//cl_int err = clblasSetup();

	m_Status = OCL_STATUS_INITIALIZED;
}

OCLManager::~OCLManager() {

	//clblasTeardown();
	for( int i = 0; i < NN_MAX_KERNEL_COUNT; i++ )
		delete m_OpenCLKernels[i];

	delete m_OpenCLProgram;

	
	m_Status = OCL_STATUS_FINALIZED;
}

int OCLManager::Initialize() {

	std::string file;
	
#ifdef WIN32
	file = ExePath() + "\\DeepNNFP32.cl";
#elif __linux__
	file = "DeepNNFP32.cl";
#endif

	std::vector<std::string> kernelFiles;
	kernelFiles.push_back(file);

	m_OpenCLSetup.init(m_DeviceName);
	m_OpenCLProgram = m_OpenCLSetup.createProgram(kernelFiles);
	m_OpenCLProgram->buildProgram();

	for( int i = 0; i < NN_MAX_KERNEL_COUNT; i++ )
		m_OpenCLKernels[i] = m_OpenCLProgram->createKernelLauncher(NN_KERNEL_NAMES[i]);

	//m_LockMutex = CreateMutex(NULL, FALSE, NULL);

	/*if( pthread_mutex_init(&m_LockMutex, NULL) != 0 ) {
	    printf("ERROR - OCLWrapper::Initialize() Mutex initialization error \n");
	    m_Status = OCL_STATUS_MUTEX_ERROR;
	    return m_Status;
	}*/


	m_LockStatus = OCL_LOCK_RELEASE;
	m_Status = OCL_STATUS_READY;
	return m_Status;
}

int OCLManager::Finalize() {

	//if(m_LockMutex != NULL)
	//if (m_Status != OCL_STATUS_MUTEX_ERROR) {
	
		//pthread_mutex_destroy(&m_LockMutex);
		//CloseHandle(m_LockMutex);
	//}

	m_Status = OCL_STATUS_FINALIZED;
	return m_Status;
}

void OCLManager::SetLock() {

	//pthread_mutex_lock(&m_LockMutex);
	//WaitForSingleObject(m_LockMutex, INFINITE);
}

void OCLManager::ReleaseLock() {

	//pthread_mutex_unlock(&m_LockMutex);
	//ReleaseMutex(m_LockMutex);
}

/*
StructPinnedOCLBuffer* OCLManager::InitializePinnedFloatArray(size_t numItems) {

	StructPinnedOCLBuffer *pinnedMemBuffer = new StructPinnedOCLBuffer;

	size_t totalSize = sizeof(float) * numItems;
	pinnedMemBuffer->m_OCLBuffer = m_OpenCLSetup.createBuffer(totalSize, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, NULL);
	pinnedMemBuffer->m_PinnedMemory = pinnedMemBuffer->m_OCLBuffer->map(CL_MAP_READ, totalSize, 0, CL_TRUE);

	return pinnedMemBuffer;
}

void OCLManager::FinalizePinnedFloatArray(StructPinnedOCLBuffer *oclPinnedBuffer) {

	oclPinnedBuffer->m_OCLBuffer->unmap(oclPinnedBuffer->m_PinnedMemory);
	delete oclPinnedBuffer;
}
*/

float OCLManager::ResetArray(int N, OCLBuffer *inArray, OCLBuffer *biasArray, int filtSize) {

	int globalDimX = N / BLOCK;
	if (globalDimX % BLOCK != 0)
		globalDimX = ((globalDimX + BLOCK) / BLOCK) * BLOCK;

	float execTime = 0.0f;

	m_OpenCLKernels[NN_KERNEL_IDX_RESETARR]->pGlobal(globalDimX)->pLocal(BLOCK);
	m_OpenCLKernels[NN_KERNEL_IDX_RESETARR]->arg(0, inArray->getMem());
	m_OpenCLKernels[NN_KERNEL_IDX_RESETARR]->arg(1, filtSize);
	return m_OpenCLKernels[NN_KERNEL_IDX_RESETARR]->run(PROFILE_KERNELS, BLOCK_KERNEL_EXEC);
}



float OCLManager::SoftMax(OCLBuffer *input, int n, int offset, int groups, float temp, OCLBuffer *output, int base) {

	int inputs = n;
	int batch = groups;
	int LOCAL_BLOCK = 2;// BLOCK;

	int globalDimX = batch / LOCAL_BLOCK;

	if (globalDimX % LOCAL_BLOCK != 0)
		globalDimX = ((globalDimX + LOCAL_BLOCK) / LOCAL_BLOCK) * LOCAL_BLOCK;
	
	m_OpenCLKernels[NN_KERNEL_IDX_SOFTMAX]->pGlobal(globalDimX)->pLocal(LOCAL_BLOCK);
	m_OpenCLKernels[NN_KERNEL_IDX_SOFTMAX]->arg(0, inputs);
	m_OpenCLKernels[NN_KERNEL_IDX_SOFTMAX]->arg(1, offset);
	m_OpenCLKernels[NN_KERNEL_IDX_SOFTMAX]->arg(2, batch);
	m_OpenCLKernels[NN_KERNEL_IDX_SOFTMAX]->arg(3, input->getMem());
	m_OpenCLKernels[NN_KERNEL_IDX_SOFTMAX]->arg(4, temp);
	m_OpenCLKernels[NN_KERNEL_IDX_SOFTMAX]->arg(5, output->getMem());
	m_OpenCLKernels[NN_KERNEL_IDX_SOFTMAX]->arg(6, base);
	return m_OpenCLKernels[NN_KERNEL_IDX_SOFTMAX]->run(PROFILE_KERNELS, BLOCK_KERNEL_EXEC);
}
