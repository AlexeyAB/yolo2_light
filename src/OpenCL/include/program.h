#include <CL/cl.h>
#include "iv_common.h"
#include "kernel_launcher.h"



#ifndef PROGRAM_H
#define PROGRAM_H

class Program
{
public:
    //Program(std::string &kernelFilePath, cl_context* context, cl_command_queue* queue, cl_device_id* device);
    Program(std::vector<std::string> &kernelFilePath, cl_context* context, cl_command_queue* queue, cl_device_id* device);
    void createProgram(std::string filePath);
    void buildProgram();
    KernelLauncher* createKernelLauncher(std::string kernelName);
    ~Program()
    {
        clReleaseProgram(_program);
    }

protected:

private:
    cl_program _program;
    cl_kernel _kernel;
    cl_int    _numKernels;
    std::map<std::string, cl_kernel> _kernels;
    std::string _filesPath;

    cl_context* _pContext;
    cl_command_queue* _pQueue;
    cl_device_id* _pDeviceID;

    cl_int _status;
    cl_bool _buildState;
};

#endif // PROGRAM_H
