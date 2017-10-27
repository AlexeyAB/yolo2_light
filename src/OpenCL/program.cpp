#include "include/program.h"
#include "include/oclUtils.h"


int ReadSourceFromFile(const char* fileName, char** source, size_t* sourceSize)
{
	int errorCode = CL_SUCCESS;

	FILE* fp = fopen(fileName, "rb");
	if (fp == NULL)
	{
		printf("Error: Couldn't find program source file '%s'.\n", fileName);
		errorCode = CL_INVALID_VALUE;
	}
	else {
		fseek(fp, 0, SEEK_END);
		*sourceSize = ftell(fp);
		fseek(fp, 0, SEEK_SET);

		*source = new char[*sourceSize];
		if (*source == NULL)
		{
			printf("Error: Couldn't allocate %d bytes for program source from file '%s'.\n", (int)(*sourceSize), fileName);
			errorCode = CL_OUT_OF_HOST_MEMORY;
		}
		else {
			fread(*source, 1, *sourceSize, fp);
		}
	}
	return errorCode;
}


void checkErr( cl_int err,int line, const char *n,  bool verbosity=false ) {
  if( err != CL_SUCCESS ) {
	  std::cerr << n << "\r\t\t\t\t\t\tline:" << line<<" "<<oclErrorString(err) << std::endl;
      //assert(0);
  }
  else if( n != NULL ) {
      if( verbosity) std::cerr << n << "\r\t\t\t\t\t\t" << "OK" <<std::endl;

  }
}

Program::Program(std::vector<std::string> &kernelFilePath, cl_context *context, cl_command_queue *queue, cl_device_id *device)
//Program::Program(std::string &kernelFilePath, cl_context *context, cl_command_queue *queue, cl_device_id *device)
{
    this->_pContext     = context;
    this->_pQueue       = queue;
    this->_pDeviceID    = device;

    std::ifstream programFile(kernelFilePath[0].c_str());
    //std::ifstream programFile(kernelFilePath.c_str());
    std::string programBuffer(std::istreambuf_iterator<char>(programFile),
                              (std::istreambuf_iterator<char>()));
    if(programBuffer.empty())
    {
        std::cout<<"Kernel File Not Found in specified location!"<<std::endl;
    }
    size_t   programSize = programBuffer.size();


	char* source = NULL;
	size_t src_size = 0;
	cl_int err = CL_SUCCESS;
	err = ReadSourceFromFile(kernelFilePath[0].c_str(), &source, &src_size);

    //_program = clCreateProgramWithSource((*_pContext), 1,(const char **)&programBuffer, &programSize, &_status);
	_program = clCreateProgramWithSource((*_pContext), 1, (const char **)&source, &src_size, &_status);
    //DEBUG_CL(_status);
    checkErr(_status, __LINE__,"clCreateProgramWithSource");
    if( _status != CL_SUCCESS )
    	printf("clCreateProgramWithSource ERROR - %s\n", oclErrorString(_status));
      else
    	  printf("clCreateProgramWithSource success\n");



    _buildState = false;
}


void Program::buildProgram()
{
    char *programLog;
    size_t programLogSize;
    //const char options[] = "-cl-std=CL1.0 -cl-mad-enable -Werror";
	//const char options[] = "-cl-mad-enable -Werror";
    const char options[] = "-cl-mad-enable -cl-unsafe-math-optimizations -cl-fast-relaxed-math -cl-single-precision-constant -cl-no-signed-zeros";
    _status= clBuildProgram(_program, 1, _pDeviceID, options, NULL, NULL);
    if(_status<0)
    {
        clGetProgramBuildInfo(_program, *_pDeviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &programLogSize );
        programLog = (char*)malloc(sizeof(char)*programLogSize+1);
        clGetProgramBuildInfo(_program, *_pDeviceID, CL_PROGRAM_BUILD_LOG, programLogSize+1, programLog, NULL);
        printf("\nBuild Log :%s\n",programLog);
        free(programLog);
//        exit(0); ///!TODO: Custom Code
    }
    DEBUG_CL(_status);
    if( _status != CL_SUCCESS )
        	printf("clGetProgramBuildInfo() ERROR - %s\n", oclErrorString(_status));
          else
        	  printf("clGetProgramBuildInfo() success\n");

    //_kernel = clCreateKernel(_program, kernelName.c_str(), &_status);
    //DEBUG_CL(_status);
    // Creates the kernels
    // Needs to verify if the file compiled is actually a kernel
    _status = clCreateKernelsInProgram(_program, 0, NULL, (cl_uint*)&(_numKernels));
    cl_kernel* k = new cl_kernel[_numKernels];
    _status = clCreateKernelsInProgram(_program, _numKernels, k, NULL);
    DEBUG_CL(_status);

    // Creates the hash with the kernels
    for (int i = 0; i < _numKernels; i++) {
        char name[256];
        _status = clGetKernelInfo(k[i], CL_KERNEL_FUNCTION_NAME, sizeof(char)*256, (void*) name, NULL);
        //DEBUG_CL(_status);
        if( _status != CL_SUCCESS )
                	printf("buildProgram ERROR - %s\n", oclErrorString(_status));
                  else
                	  printf("buildProgram kernels success\n");
        _kernels[name] = k[i];
        printf("Kernel No: %d, name - %s\n", i+1, name);
        //DEBUG_VALUE("Kernel Name: ", name);
    }

    _buildState = true;

}

KernelLauncher* Program::createKernelLauncher(std::string kernelName)
{
    if(!_buildState)
        ERROR_PRINT_STRING("You forgot to build the kernel");

    /// @TIPS: Always use a pointer to an variable that needs to be returned
    KernelLauncher *kl = new KernelLauncher(*this->_pDeviceID, &_kernels[kernelName], _pQueue, kernelName);
    return kl;
}

