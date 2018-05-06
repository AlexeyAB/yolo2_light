#include "include/cl_wrapper.hpp"

//////////////////////////////////////////////////////////////////////////////
//! Print info about the device
//!
//! @param iLogMode       enum LOGBOTH, LOGCONSOLE, LOGFILE
//! @param device         OpenCL id of the device
//////////////////////////////////////////////////////////////////////////////
void oclPrintDevInfo(cl_device_id device)
{
    char device_string[1024];
    bool nv_device_attibute_query = false;

    // CL_DEVICE_NAME
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
    std::cout << "CL_DEVICE_NAME:: " << device_string << std::endl;

    // CL_DEVICE_VENDOR
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(device_string), &device_string, NULL);
    std::cout << "CL_DEVICE_VENDOR:: " << device_string << std::endl;

    // CL_DRIVER_VERSION
    clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(device_string), &device_string, NULL);
    std::cout << "CL_DRIVER_VERSION:: " << device_string << std::endl;

    // CL_DEVICE_VERSION
    clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(device_string), &device_string, NULL);
    std::cout << "CL_DEVICE_VERSION:: " << device_string << std::endl;


#if !defined(__APPLE__) && !defined(__MACOSX)
    // CL_DEVICE_OPENCL_C_VERSION (if CL_DEVICE_VERSION version > 1.0)
    if (strncmp("OpenCL 1.0", device_string, 10) != 0)
    {
        // This code is unused for devices reporting OpenCL 1.0, but a def is needed anyway to allow compilation using v 1.0 headers 
        // This constant isn't #defined in 1.0
#ifndef CL_DEVICE_OPENCL_C_VERSION
#define CL_DEVICE_OPENCL_C_VERSION 0x103D   
#endif

        clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, sizeof(device_string), &device_string, NULL);
        std::cout << "CL_DEVICE_OPENCL_C_VERSION:: " << device_string << std::endl;
    }
#endif

    // CL_DEVICE_TYPE
    cl_device_type type;
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
    if (type & CL_DEVICE_TYPE_CPU)
        std::cout << "CL_DEVICE_TYPE::CL_DEVICE_TYPE_CPU" << std::endl;
    if (type & CL_DEVICE_TYPE_GPU)
        std::cout << "CL_DEVICE_TYPE::CL_DEVICE_TYPE_GPU" << std::endl;
    if (type & CL_DEVICE_TYPE_ACCELERATOR)
        std::cout << "CL_DEVICE_TYPE::CL_DEVICE_TYPE_ACCELERATOR" << std::endl;
    if (type & CL_DEVICE_TYPE_DEFAULT)
        std::cout << "CL_DEVICE_TYPE::CL_DEVICE_TYPE_DEFAULT" << std::endl;

    // CL_DEVICE_MAX_COMPUTE_UNITS
    cl_uint compute_units;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
    std::cout << "  CL_DEVICE_MAX_COMPUTE_UNITS: " << compute_units << std::endl;

    // CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
    size_t workitem_dims;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(workitem_dims), &workitem_dims, NULL);
    std::cout << "  CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: " << workitem_dims << std::endl;
}

void CLSetup::init(char* deviceName)
{
    getPlatformID();
    getDeviceID(deviceName);
    getContextnQueue();
}

void CLSetup::getPlatformID()
{

    cl_uint num_of_platforms = 0;
    // get total number of available platforms:
    cl_int err = CL_SUCCESS;
    err = clGetPlatformIDs(0, 0, &num_of_platforms);
    
    cl_platform_id* platforms = new cl_platform_id[num_of_platforms];
    // get IDs for all platforms:
    err = clGetPlatformIDs(num_of_platforms, platforms, 0);

    cl_uint selected_platform_index = num_of_platforms;

    for (cl_uint i = 0; i < num_of_platforms; ++i)
    {
        // Get the length for the i-th platform name
        size_t platform_name_length = 0;
        err = clGetPlatformInfo(
            platforms[i],
            CL_PLATFORM_NAME,
            0,
            0,
            &platform_name_length
        );

        // Get the name itself for the i-th platform
        char* platform_name = new char[platform_name_length];
        err = clGetPlatformInfo(
            platforms[i],
            CL_PLATFORM_NAME,
            platform_name_length,
            platform_name,
            0
        );

        // decide if this i-th platform is what we are looking for
        // we select the first one matched skipping the next one if any
        if (strstr(platform_name, /*"AMD"*/ "NVIDIA" /*"Intel(R) OpenCL"*/) &&
            selected_platform_index == num_of_platforms)
        {
            selected_platform_index = i;
            _platformID = platforms[i];
            // do not stop here, just see all available platforms
        }

        delete[] platform_name;
    }


    /// !TODO: Multiple Platforms
    //    cl_platform_id* _platformID;
    //    _status = clGetPlatformIDs(NUMBER_OF_PLATFORMS, NULL, &_numPlatforms);
    //    DEBUG_CL(_status);
    //    _platformID = (cl_platform_id *)malloc(sizeof(cl_platform_id) * _numPlatforms);
    //    _status =clGetPlatformIDs(_numPlatforms, _platformID, NULL);
    //    DEBUG_CL(_status);
    //    _platformIDsVector.assign(_platformID[0], _platformID[_numPlatforms]);
}

void CLSetup::getDeviceID(char *devName)
{
    /// !TODO: For Multiple Devices
    _status = clGetDeviceIDs(_platformID,CL_DEVICE_TYPE_GPU, 1, NULL, &_numDevices);
    //DEBUG_CL(_status);
    std::cout<<"CL_COMPUTE DEVICES: "<<_numDevices<<std::endl;
    _status = clGetDeviceIDs(_platformID, CL_DEVICE_TYPE_GPU, 1, &_deviceID, NULL);
    //DEBUG_CL(_status);
    std::cout<<"CL_DEVICE_ID: "<<_deviceID<<std::endl;

    char device_string[1024];
    clGetDeviceInfo(_deviceID, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
    strcpy(devName, device_string);

    // Getting some information about the device
    // Getting some information about the device

    oclPrintDevInfo(_deviceID);
    clGetDeviceInfo(_deviceID, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &_maxComputeUnits, NULL);
    std::cout<<"CL_DEVICE_MAX_COMPUTE_UNITS: "<<_maxComputeUnits<<std::endl;
    clGetDeviceInfo(_deviceID, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &_maxWorkGroupSize, NULL);
    clGetDeviceInfo(_deviceID, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &_maxMemAllocSize, NULL);
    clGetDeviceInfo(_deviceID, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &_globalMemSize, NULL);
    clGetDeviceInfo(_deviceID, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(cl_ulong), &_constMemSize, NULL);
    clGetDeviceInfo(_deviceID, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &_localMemSize, NULL);
    ///!TODO:Add get KernelInfo APIs

}

void CLSetup::getContextnQueue()
{
    cl_command_queue_properties queueProps = NULL;// CL_QUEUE_PROFILING_ENABLE;
    _context = clCreateContext(NULL, 1, &_deviceID, NULL, NULL, &_status);
    //DEBUG_CL(_status);
    _queue = clCreateCommandQueue(_context, _deviceID, queueProps, &_status);
    //DEBUG_CL(_status);
}

///
/// \brief CLSetup::createProgram
/// \param kernelFilePath
/// \return
///
Program *CLSetup::createProgram(std::vector<std::string> kernelFilePath)
{
    ///!TODO: Add support for char** along with string
    Program* tmp = new Program(kernelFilePath, &_context, &_queue,
                               &_deviceID);
    return tmp;
}


















