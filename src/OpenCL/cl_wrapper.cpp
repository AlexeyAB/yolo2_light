#include "include/cl_wrapper.hpp"
#include "include/oclUtils.h"

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
    DEBUG_CL(_status);
    std::cout<<"CL_COMPUTE DEVICES: "<<_numDevices<<std::endl;
    _status = clGetDeviceIDs(_platformID, CL_DEVICE_TYPE_GPU, 1, &_deviceID, NULL);
    DEBUG_CL(_status);
    std::cout<<"CL_DEVICE_ID: "<<_deviceID<<std::endl;

	char device_string[1024];
	clGetDeviceInfo(_deviceID, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
	strcpy(devName, device_string);

    // Getting some information about the device
    // Getting some information about the device

    oclPrintDevInfo(LOGCONSOLE, _deviceID);
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
    DEBUG_CL(_status);
    _queue = clCreateCommandQueue(_context, _deviceID, queueProps, &_status);
    DEBUG_CL(_status);
}

///
/// \brief CLSetup::createProgram
/// \param kernelFilePath
/// \return
///
Program *CLSetup::createProgram(std::vector<std::string> kernelFilePath)
//Program *CLSetup::createProgram(std::string& kernelFilePath)
{
    ///!TODO: Add support for char** along with string
    Program* tmp = new Program(kernelFilePath, &_context, &_queue,
                               &_deviceID);
    return tmp;
}

OCLBuffer* CLSetup::createBuffer(const size_t size, const cl_mem_flags flags,
                              void *hostMem)
{
    cl_mem buff = clCreateBuffer(_context,flags, size, hostMem ,&_status);
    if(_status == CL_SUCCESS)
    {
        OCLBuffer* ret = new OCLBuffer(buff, &_queue);
        return ret;
    }
    DEBUG_CL(_status);
    if(_status != CL_SUCCESS)
    	printf("createBuffer error : %s", getCLErrorString(_status));
    return NULL; //TODO: Return custom status value
}



















