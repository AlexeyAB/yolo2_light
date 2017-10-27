#ifndef CL_WRAPPER_H
#define CL_WRAPPER_H

/**
 * Enable CL_INFO_PRINT macro to see OpenCL Device and Kernel Info
 *
 */
#include "iv_common.h"
#include "CL/cl.h"
//#include "CL/cl.hpp"

#include "program.h"
#include "buffer.h"
#include "image2D.h"
#include "sampler.h"


#define NUMBER_OF_PLATFORMS     5
#define NUMBER_OF_DEVICES       5
class CLSetup
{
public:
    void init(char* deviceName);
    void getPlatformID();
    void getDeviceID(char* deviceName);
    void getContextnQueue();
    //Program* createProgram(std::string &kernelFilePath);
    Program* createProgram(std::vector<std::string> kernelFilePath);
    OCLBuffer* createBuffer(const size_t size,
                         const cl_mem_flags flags,
                         void* hostMem);
    Image2D* createImage2D(const size_t width,
                           const size_t height,
                           const cl_image_format* format,
                           const cl_mem_flags flags = 0,
                           const size_t rowPitch = 0,
                           void* hostMem = NULL);
    Sampler* createSampler(cl_bool normalizedCoords,
                           cl_addressing_mode addrMode,
                           cl_filter_mode filterMode);
	
	cl_command_queue* getQueue() {

		return &_queue;
	}

	cl_context* getContext() {

		return &_context;
	}

    //>>>>>>>>>>>>>>>>>Get Info
    int getNumberOfPlatforms()
    {
        return _numPlatforms;
    }
    void getDeviceName()
    {
        char*   infoName;
        size_t  infoCLSize;
        clGetPlatformInfo(_platformID,CL_PLATFORM_NAME, NULL, NULL, &infoCLSize);
        infoName = (char*)malloc(sizeof(char)*infoCLSize);
        clGetPlatformInfo(_platformID,CL_PLATFORM_NAME, sizeof(char) * infoCLSize, infoName, NULL );
        std::cout<<"CL_PLATFORM_NAME          : "<<infoName<<std::endl;
        free(infoName);
    }
    int getNumberOfDevices()
    {
        return _numDevices;
    }
    ///
    /// \brief Compute units are equivalent of MultiProcessor
    ///        (which can have either 8, 32, 48 or even 192 cores),
    ///        and these are designed to be able to simultanesouly
    ///        run up to 8 work groups (blocks in CUDA) each.
    ///        Eg: GeForce GT 640 : 2 * 196 = 384 cores
    ///        Eg: Vivante GC2000 : 4 *
    /// \return Maximum Compute Units available in your GPU in size_t

    cl_uint getMaxComputeUnits()
    {
        return _maxComputeUnits;
    }
    size_t getPerferredWorkGroupSize()
    {
        return _perferredWrkGrpSize; ///!TODO:Need to get after from kernel init
    }
    ///
    /// \brief getMaxWorkGroupSize
    ///        Returns maximum number of work items in a workgroup
    ///        Eg: GeForce GT 640 : 48KBytes
    ///        Eg: Vivante GC2000 : 1KBytes
    /// \return
    ///
    size_t getMaxWorkGroupSize()
    {
        return _maxWorkGroupSize;
    }
    ///
    /// \brief Returns the local memory size of device kernel groups
    ///
    /// \return
    ///
    cl_ulong getLocalMemSize()
    {
        return _localMemSize;
    }
    size_t getPrefferedWorkGroupSize()
    {
        return _preferredWorkGrpSize;
    }

    void getSupportedExtensions()
    {
        size_t infoCLSize;
        char* infoExtensions;
        clGetPlatformInfo(_platformID,CL_PLATFORM_EXTENSIONS, NULL, NULL, &infoCLSize);
        infoExtensions = (char*)malloc(sizeof(char)*infoCLSize);
        clGetPlatformInfo(_platformID,CL_PLATFORM_EXTENSIONS, sizeof(char) * infoCLSize, infoExtensions, NULL );
        std::cout<<"CL_PLATFORM_EXTENSIONS          : "<<infoExtensions<<std::endl;
        free(infoExtensions);
    }

    ~CLSetup()
    {
        clReleaseCommandQueue(_queue);
        clReleaseContext(_context);
    }

protected:
private:
    //>>>>>>>>>>>>>>>>>Info Members
    cl_int          _status;
    cl_long         _infoValue;

    //>>>>>>>>>>>>>>>>>Platform Members
    cl_uint         _numPlatforms;
    cl_platform_id  _platformID;
    /**
     * @brief _platformIDsVector
     *        _platformIDsVector[interestedPlatformNum]
     */
    std::vector<cl_platform_id> _platformIDsVector; //!TODO:

    //>>>>>>>>>>>>>>>>>Device Members
    cl_uint         _numDevices;
    cl_device_id    _deviceID;

    cl_uint         _maxComputeUnits;
    size_t          _maxWorkGroupSize;
    cl_ulong        _maxMemAllocSize;
    cl_ulong        _globalMemSize;
    cl_ulong        _constMemSize;
    cl_ulong        _localMemSize;
    size_t          _preferredWorkGrpSize;



    //>>>>>>>>>>>>>>>>>Context Members
    cl_context      _context;

    //>>>>>>>>>>>>>>>>>Queue Members
    cl_command_queue      _queue;

    //>>>>>>>>>>>>>>>>>Program Members
    cl_program      _program;

    //>>>>>>>>>>>>>>>>>Kernel Members
    cl_kernel   _kernel;
    size_t      _compileWrkGrpSize;
    size_t      _wrkGrpSize;
    size_t      _perferredWrkGrpSize;
    cl_ulong    _localMem;
    cl_ulong    _privateMem;

    //>>>>>>>>>>>>>>>>>Image2D Members
    cl_mem im2d;


};

//const char * get_error_string(cl_int err);


#endif // CL_WRAPPER_H
