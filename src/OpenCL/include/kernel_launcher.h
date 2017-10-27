

#ifndef KERNELLAUNCHER_H
#define KERNELLAUNCHER_H

#include "iv_common.h"



class KernelLauncher
{
public:
    KernelLauncher(cl_device_id _device_id, cl_kernel* kernel, cl_command_queue* queue, std::string kernelNAme);

    ///For a continuous aggignment using an object
    KernelLauncher& global(const int g);
    KernelLauncher& global(const int gx, const int gy);
    KernelLauncher& global(const int gx, const int gy, const int gz);
    KernelLauncher& local(const int l);
    KernelLauncher& local(const int lx, const int ly);
    KernelLauncher& local(const int lx, const int ly, const int lz);
    ///For a continuous aggignment using an pointer object
    KernelLauncher* pGlobal(const int g);
    KernelLauncher* pGlobal(const int gx, const int gy);
    KernelLauncher* pGlobal(const int gx, const int gy, const int gz);
    KernelLauncher* pLocal(const int l);
    KernelLauncher* pLocal(const int lx, const int ly);
    KernelLauncher* pLocal(const int lx, const int ly, const int lz);

    int countArgs();

    ///For a continuous aggignment using an object
    template<class T>
    KernelLauncher& arg(const int index, T x) {
        if (index >= _numArgs || index < 0) {
            std::cout << "Error: argument index out of range" << std::endl;
            exit(-1);///!TODO: Custom exit code
        }
        cl_int status = clSetKernelArg(*_pKernel, index, sizeof(x), &x);
        DEBUG_CL(status);
        _argListData[index] = true;
        return *this;
    }
    ///For a continuous aggignment using an pointer object
    template<class T>
    KernelLauncher& arg(T x) {
        int nArgs = countArgs();
        if (nArgs >= _numArgs) {
            std::cout << "Error trying to enqueue too much arguments" << std::endl;
            std::cout << "Expected " << _numArgs << ", got " << nArgs << std::endl;
            exit(-1);///!TODO: Custom exit code
        }
        for(int i=0; i<_numArgs; i++)
            if(!_argListData[i])
                return arg(i, x);
        return *this;
    }

    ///For a continuous aggignment using an object
    template<class T>
    KernelLauncher* pArg(const int index, T &x) {
        if (index >= _numArgs || index < 0) {
            std::cout << "Error: argument index out of range" << std::endl;
            exit(-1);///!TODO: Custom exit code
        }
        cl_int status = clSetKernelArg(*_pKernel, index, sizeof(x), &x);
        DEBUG_VALUE("Setting Kernel Argument: ", index);
        DEBUG_VALUE("Value/Address: ", x);
        DEBUG_VALUE("Size : ", sizeof(x));
        DEBUG_CL(status);
        _argListData[index] = true;
        return this;
    }
    ///For a continuous aggignment using an pointer object
    template<class T>
    KernelLauncher* pArg(T &x) {
        int nArgs = countArgs();
        if (nArgs > _numArgs) {
            std::cout << "Error trying to enqueue too much arguments" << std::endl;
            std::cout << "Expected " << _numArgs << ", got " << nArgs << std::endl;
            exit(-1);///!TODO: Custom exit code
        }
        for(int i=0; i<_numArgs; i++)
            if(!_argListData[i])
                return pArg(i, x);
        return this;
    }
    float run(bool profile, bool block);

    ~KernelLauncher()
    {
        clReleaseKernel(*_pKernel);
    }

	size_t GetOptimalLWGSize() {

		return _optimal_local_workgroup_size;
	}

protected:
private:
    cl_kernel*          _pKernel;
    cl_command_queue*   _pQueue;
    cl_int              _numArgs;
    size_t              _globalWorkSize[3];
    size_t              _localWorkSize[3];
    cl_int              _dimensions;
	std::string			_kernel_name;
    cl_bool*            _argListData;
	cl_device_id        _device_id;
	size_t              _optimal_local_workgroup_size;
};


#endif // KERNELLAUNCHER_H
