//#include <windows.h>
#include "include/buffer.h"


OCLBuffer::OCLBuffer(cl_mem tmp, cl_command_queue *queue)
{
    this->_memory = tmp;
    this->_pQueue = queue;
}

cl_mem OCLBuffer::getMem()
{
    return _memory;
}

void OCLBuffer::read(void *hostMem, const size_t size, const size_t offset, const cl_bool blocking)
{
	//DWORD readVal = ::GetTickCount();
    cl_int status = 0;
	cl_event execEvent = NULL;
    //status = clEnqueueReadBuffer(*_pQueue, _memory, blocking, offset, size, hostMem, 0, NULL, NULL);
	status = clEnqueueReadBuffer(*_pQueue, _memory, blocking, offset, size, hostMem, 0, NULL, NULL);// &execEvent);
	if (status == CL_SUCCESS) {


		//clWaitForEvents(1, &execEvent);
		clFinish(*_pQueue);
		//::Sleep(100);
		//printf("Array read time (Ticks) was {%d} msecs\n", ::GetTickCount() - readVal);
		
		
		//long long start, end;
		//double total;

		//status = clGetEventProfilingInfo(execEvent, CL_PROFILING_COMMAND_START,
		//	sizeof(start), &start, NULL);
		//status = clGetEventProfilingInfo(execEvent, CL_PROFILING_COMMAND_END,
		//	sizeof(end), &end, NULL);

		//clReleaseEvent(execEvent);

		//total = (double)(end - start) / 1e6; /* Convert nanoseconds to msecs */
		//printf("Array read time was {%5.2f} msecs\n", total);
	}

	

	//clFinish(*_pQueue);
    //DEBUG_CL(status);
}

void OCLBuffer::write(const void *hostMem, const size_t size, const size_t offset, const cl_bool blocking)
{
    cl_int status = 0;
    status = clEnqueueWriteBuffer(*_pQueue, _memory, blocking, offset, size, hostMem, 0, NULL, NULL);
    DEBUG_CL(status);
    if(status != CL_SUCCESS)
    	printf("clWrite buffer error : %s", getCLErrorString(status));
}

void *OCLBuffer::map(const cl_map_flags flags, const size_t size, const size_t offset, const cl_bool blocking)
{
    cl_int status = 0;
    void *data = clEnqueueMapBuffer(*_pQueue, _memory, blocking, flags, offset, size, 0, NULL, NULL, &status);
    DEBUG_CL(status);
    if(status != CL_SUCCESS)
    	printf("clEnqueueMapBuffer error : %s", getCLErrorString(status));
    return data;
}

void OCLBuffer::unmap(void *mappedPtr)
{
	cl_int status = 0;
	cl_event event;

	if((status = clEnqueueUnmapMemObject(*_pQueue, _memory, mappedPtr, 0, NULL, &event)) != CL_SUCCESS) {
		if(status != CL_SUCCESS)
			printf("clEnqueueUnmapMemObject error : %s", getCLErrorString(status));
	}
	else{
		clWaitForEvents(1, &event);
		clReleaseEvent(event);
	}
}

void OCLBuffer::FillBuffer(const void *hostMem, const size_t size, const size_t offset) {

	cl_int status = 0;

	if((status = clEnqueueFillBuffer(*_pQueue, _memory, &hostMem, sizeof(cl_uint), 0, size, 0, NULL, NULL) != CL_SUCCESS))
	{
		if (status != CL_SUCCESS)
			printf("FillBuffer error : %s", getCLErrorString(status));
	}
}
