
#ifndef __CL_UTILS__H__
#define __CL_UTILS__H__

#include <CL/opencl.h>

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
