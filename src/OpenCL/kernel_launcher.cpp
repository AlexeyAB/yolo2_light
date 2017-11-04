#include "include/kernel_launcher.h"

KernelLauncher::KernelLauncher(cl_device_id device_id, cl_kernel *kernel, cl_command_queue *queue, std::string kernelName)
{
    cl_int status;
    this->_pKernel = kernel;
    this->_pQueue = queue;
    this->_dimensions = -1;
	_device_id = device_id;
    _globalWorkSize[0] = _globalWorkSize[1] = _globalWorkSize[2] =
            _localWorkSize[0] = _localWorkSize[1] = _localWorkSize[2] = NULL;

    //Finding number of arguments in given kernel and making
    //an bool array to track its data content
    status = clGetKernelInfo(*_pKernel, CL_KERNEL_NUM_ARGS, sizeof(cl_int), &_numArgs, NULL);
    //DEBUG_CL(status);
    printf("Number of kernel Arguments : %d %s \n",_numArgs, kernelName.c_str());
    this->_argListData = (cl_bool*) malloc(_numArgs*sizeof(cl_bool));//new cl_bool[numArgs];
    for(int i=0; i<_numArgs; i++)
        this->_argListData[i] = false;

	cl_int err = clGetKernelWorkGroupInfo(*_pKernel, _device_id,
		CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_optimal_local_workgroup_size, NULL);
	//err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);

	size_t compileWorkGroupSize[3];
	err = clGetKernelWorkGroupInfo(*_pKernel, _device_id,
		CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
		sizeof(size_t) * 3,
		compileWorkGroupSize,
		NULL);

	_kernel_name = kernelName;

}

int KernelLauncher::countArgs()
{
    int ret=0;

    for(int i=0; i<_numArgs; i++)
        if(_argListData[i])
            ret++;

    return ret;
}

float KernelLauncher::run(bool profile, bool block)
{
	double total = 0;
	cl_int status = CL_SUCCESS;
	cl_event execEvent = NULL;

	status = clEnqueueNDRangeKernel(*_pQueue, *_pKernel, _dimensions,
			NULL, _globalWorkSize, _localWorkSize, 0,
			NULL, (profile? &execEvent: NULL));

	if (status == CL_SUCCESS && block)
		clFinish(*_pQueue);

	if (profile && status == CL_SUCCESS) {

		clWaitForEvents(1, &execEvent);
		long long start, end;
		status = clGetEventProfilingInfo(execEvent, CL_PROFILING_COMMAND_START,
			sizeof(start), &start, NULL);
		status = clGetEventProfilingInfo(execEvent, CL_PROFILING_COMMAND_END,
			sizeof(end), &end, NULL);

		total = (double)(end - start) / 1e6; /* Convert nanoseconds to msecs */
		printf("Total kernel time was {%5.3f} msecs - %s \n", total, _kernel_name.c_str());
		
		clReleaseEvent(execEvent);
	}

	if (status != CL_SUCCESS) {

		char * kernelName = (char*)_kernel_name.c_str();
		int z = 0;
		z++;
	}

	//DEBUG_CL(status);
	return (float)total;
}
KernelLauncher& KernelLauncher::global(const int g) {
    if (_dimensions == -1) _dimensions = 1;
    else if (_dimensions != 1) {
        std::cerr << "Work group dimension incoherence" << std::endl;
    }
    _globalWorkSize[0] = g;
    return *this;
}

KernelLauncher& KernelLauncher::global(const int gx, const int gy) {
    if (_dimensions == -1) _dimensions = 2;
    else if (_dimensions != 2) {
        std:: cerr << "Work group dimension incoherence" << std::endl;
    }
    _globalWorkSize[0] = gx;
    _globalWorkSize[1] = gy;
    return *this;
}

KernelLauncher& KernelLauncher::global(const int gx, const int gy, const int gz) {
    if (_dimensions == -1) _dimensions = 3;
    else if (_dimensions != 3) {
        std::cerr << "Work group dimension incoherence" << std::endl;
    }
    _globalWorkSize[0] = gx;
    _globalWorkSize[1] = gy;
    _globalWorkSize[2] = gz;
    return *this;
}

KernelLauncher& KernelLauncher::local(const int l) {
    if (_dimensions == -1) _dimensions = 1;
    else if (_dimensions != 1) {
        std::cerr << "Work group dimension incoherence" << std::endl;
    }
    _localWorkSize[0] = l;
    return *this;
}

KernelLauncher& KernelLauncher::local(const int lx, const int ly) {
    if (_dimensions == -1) _dimensions = 2;
    else if (_dimensions != 2) {
        std::cerr << "Work group dimension incoherence" << std::endl;
    }
    _localWorkSize[0] = lx;
    _localWorkSize[1] = ly;
    return *this;
}

KernelLauncher& KernelLauncher::local(const int lx, const int ly, const int lz) {
    if (_dimensions == -1) _dimensions = 3;
    else if (_dimensions != 3) {
        std::cerr << "Work group dimension incoherence" << std::endl;
    }
    _localWorkSize[0] = lx;
    _localWorkSize[1] = ly;
    _localWorkSize[2] = lz;
    return *this;
}

//////////////////////////
KernelLauncher* KernelLauncher::pGlobal(const int g) {
    if (_dimensions == -1) _dimensions = 1;
    else if (_dimensions != 1) {
        std::cerr << "Work group dimension incoherence" << std::endl;
    }
    _globalWorkSize[0] = g;
    return this;
}

KernelLauncher* KernelLauncher::pGlobal(const int gx, const int gy) {
    if (_dimensions == -1) _dimensions = 2;
    else if (_dimensions != 2) {
        std::cerr << "Work group dimension incoherence" << std::endl;
    }
    _globalWorkSize[0] = gx;
    _globalWorkSize[1] = gy;
    return this;
}

KernelLauncher* KernelLauncher::pGlobal(const int gx, const int gy, const int gz) {
    if (_dimensions == -1) _dimensions = 3;
    else if (_dimensions != 3) {
        std::cerr << "Work group dimension incoherence" << std::endl;
    }
    _globalWorkSize[0] = gx;
    _globalWorkSize[1] = gy;
    _globalWorkSize[2] = gz;
    return this;
}

KernelLauncher* KernelLauncher::pLocal(const int l) {
    if (_dimensions == -1) _dimensions = 1;
    else if (_dimensions != 1) {
        std::cerr << "Work group dimension incoherence" << std::endl;
    }
    _localWorkSize[0] = l;
    return this;
}

KernelLauncher* KernelLauncher::pLocal(const int lx, const int ly) {
    if (_dimensions == -1) _dimensions = 2;
    else if (_dimensions != 2) {
        std::cerr << "Work group dimension incoherence" << std::endl;
    }
    _localWorkSize[0] = lx;
    _localWorkSize[1] = ly;
    return this;
}

KernelLauncher* KernelLauncher::pLocal(const int lx, const int ly, const int lz) {
    if (_dimensions == -1) _dimensions = 3;
    else if (_dimensions != 3) {
        std::cerr << "Work group dimension incoherence" << std::endl;
    }
    _localWorkSize[0] = lx;
    _localWorkSize[1] = ly;
    _localWorkSize[2] = lz;
    return this;
}

