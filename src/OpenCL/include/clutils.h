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

#ifndef __CL_UTILS__H__
#define __CL_UTILS__H__

#include <CL/opencl.h>


//#include <Windows.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include "CL/cl.h"

///!TODO: Use bits as enum value
typedef enum {
    IV_CHAR_FILE_OPEN_FAIL = 0,
    IV_CHAR_FILE_STATUS_FAIL
}IV_ERRORS_INFO;

const char* getCLErrorString(int err);

#define ERROR_PRINT_STRING(dbgMsg)    \
    {   \
    std::cout<<"\n=====> Line Number: "<<__LINE__<<" Function Name: "<<__func__\
    <<"\n "<<dbgMsg<<std::endl; \
    exit(0); \
    };


//General defines
#define MAX_STR_LEN          256
#define HALF_STR_LEN         MAX_STR_LEN/2
#define Q_STR_LEN             HALF_STR_LEN/2

//OpenCL related
#define OCL_STATUS_READY          0
#define OCL_STATUS_INITIALIZED    1
#define OCL_STATUS_PROGRAM_ERROR  2
#define OCL_STATUS_KERNEL_ERROR   3
#define OCL_STATUS_MUTEX_ERROR    4
#define OCL_STATUS_FINALIZED      5

#define OCL_LOCK_SET              1
#define OCL_LOCK_RELEASE          2


char* utils_cl_enum_to_string           (cl_int value);

char* utils_get_ocl_error               (cl_int err_code);

int   utils_get_platform_and_device     (cl_device_type dev_type,
                                         cl_platform_id *platform,
                                         cl_device_id *device_id,
                                         int just_print);

void  utils_print_device_info           (cl_device_id dev_id);

void  utils_print_platform_info         (cl_platform_id dev_id);

char* utils_read_file                   (const char* filename);

cl_ulong utils_get_event_time           (cl_event event,
                                         cl_profiling_info param);

cl_ulong utils_get_event_execution_time (cl_event event);

#define CHECK_OCL_ERROR(op_name, err_code) \
  if (err_code != CL_SUCCESS) \
  {\
    printf ("%s:%d: " #op_name " failed! %s\n", __FILE__, __LINE__, utils_get_ocl_error (err_code));\
  }

#endif
