#include "include/clutils.h"
#include <stdlib.h>
#include <stdio.h>
#include <CL/opencl.h>

char* utils_cl_enum_to_string (cl_int value)
{
  switch (value)
  {
    /* cl_channel_order */
    case 0x10B0: return "CL_R";
    case 0x10B1: return "CL_A";
    case 0x10B2: return "CL_RG";
    case 0x10B3: return "CL_RA";
    case 0x10B4: return "CL_RGB";
    case 0x10B5: return "CL_RGBA";
    case 0x10B6: return "CL_BGRA";
    case 0x10B7: return "CL_ARGB";
    case 0x10B8: return "CL_INTENSITY";
    case 0x10B9: return "CL_LUMINANCE";
    case 0x10BA: return "CL_Rx";
    case 0x10BB: return "CL_RGx";
    case 0x10BC: return "CL_RGBx";

    /* cl_channel_type */
    case 0x10D0: return "CL_SNORM_INT8";
    case 0x10D1: return "CL_SNORM_INT16";
    case 0x10D2: return "CL_UNORM_INT8";
    case 0x10D3: return "CL_UNORM_INT16";
    case 0x10D4: return "CL_UNORM_SHORT_565";
    case 0x10D5: return "CL_UNORM_SHORT_555";
    case 0x10D6: return "CL_UNORM_INT_101010";
    case 0x10D7: return "CL_SIGNED_INT8";
    case 0x10D8: return "CL_SIGNED_INT16";
    case 0x10D9: return "CL_SIGNED_INT32";
    case 0x10DA: return "CL_UNSIGNED_INT8";
    case 0x10DB: return "CL_UNSIGNED_INT16";
    case 0x10DC: return "CL_UNSIGNED_INT32";
    case 0x10DD: return "CL_HALF_FLOAT";
    case 0x10DE: return "CL_FLOAT";
    default: return "Unknown value!";
  }
}

char* utils_get_ocl_error (cl_int err_code)
{
  switch (err_code)
  {
    case CL_SUCCESS:                          return "Success!";
    case CL_DEVICE_NOT_FOUND:                 return "Device not found.";
    case CL_DEVICE_NOT_AVAILABLE:             return "Device not available";
    case CL_COMPILER_NOT_AVAILABLE:           return "Compiler not available";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:    return "Memory object allocation failure";
    case CL_OUT_OF_RESOURCES:                 return "Out of resources";
    case CL_OUT_OF_HOST_MEMORY:               return "Out of host memory";
    case CL_PROFILING_INFO_NOT_AVAILABLE:     return "Profiling information not available";
    case CL_MEM_COPY_OVERLAP:                 return "Memory copy overlap";
    case CL_IMAGE_FORMAT_MISMATCH:            return "Image format mismatch";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:       return "Image format not supported";
    case CL_BUILD_PROGRAM_FAILURE:            return "Program build failure";
    case CL_MAP_FAILURE:                      return "Map failure";
    case CL_INVALID_VALUE:                    return "Invalid value";
    case CL_INVALID_DEVICE_TYPE:              return "Invalid device type";
    case CL_INVALID_PLATFORM:                 return "Invalid platform";
    case CL_INVALID_DEVICE:                   return "Invalid device";
    case CL_INVALID_CONTEXT:                  return "Invalid context";
    case CL_INVALID_QUEUE_PROPERTIES:         return "Invalid queue properties";
    case CL_INVALID_COMMAND_QUEUE:            return "Invalid command queue";
    case CL_INVALID_HOST_PTR:                 return "Invalid host pointer";
    case CL_INVALID_MEM_OBJECT:               return "Invalid memory object";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  return "Invalid image format descriptor";
    case CL_INVALID_IMAGE_SIZE:               return "Invalid image size";
    case CL_INVALID_SAMPLER:                  return "Invalid sampler";
    case CL_INVALID_BINARY:                   return "Invalid binary";
    case CL_INVALID_BUILD_OPTIONS:            return "Invalid build options";
    case CL_INVALID_PROGRAM:                  return "Invalid program";
    case CL_INVALID_PROGRAM_EXECUTABLE:       return "Invalid program executable";
    case CL_INVALID_KERNEL_NAME:              return "Invalid kernel name";
    case CL_INVALID_KERNEL_DEFINITION:        return "Invalid kernel definition";
    case CL_INVALID_KERNEL:                   return "Invalid kernel";
    case CL_INVALID_ARG_INDEX:                return "Invalid argument index";
    case CL_INVALID_ARG_VALUE:                return "Invalid argument value";
    case CL_INVALID_ARG_SIZE:                 return "Invalid argument size";
    case CL_INVALID_KERNEL_ARGS:              return "Invalid kernel arguments";
    case CL_INVALID_WORK_DIMENSION:           return "Invalid work dimension";
    case CL_INVALID_WORK_GROUP_SIZE:          return "Invalid work group size";
    case CL_INVALID_WORK_ITEM_SIZE:           return "Invalid work item size";
    case CL_INVALID_GLOBAL_OFFSET:            return "Invalid global offset";
    case CL_INVALID_EVENT_WAIT_LIST:          return "Invalid event wait list";
    case CL_INVALID_EVENT:                    return "Invalid event";
    case CL_INVALID_OPERATION:                return "Invalid operation";
    case CL_INVALID_GL_OBJECT:                return "Invalid OpenGL object";
    case CL_INVALID_BUFFER_SIZE:              return "Invalid buffer size";
    case CL_INVALID_MIP_LEVEL:                return "Invalid mip-map level";
    default:                                  return "Unknown error";
  }
} 

char* utils_read_file (const char* filename)
{
  // locals 
  FILE* f = NULL;
  size_t file_size;

  f = fopen (filename, "r");
  if (f == NULL)
  {
    return NULL;
  }

  // get the length
  fseek (f, 0, SEEK_END);
  file_size = ftell (f);
  fseek (f, 0, SEEK_SET);

  // allocate a buffer and fill it
  char* contents = (char *)malloc (file_size + 1);
  if (fread (contents, file_size, sizeof (char), f) != 1)
  {
    fclose (f);
    free (contents);
    return NULL;
  }

  fclose (f);
  contents[file_size] = '\0';

  return contents;
}

int utils_get_platform_and_device (cl_device_type dev_type,
                                   cl_platform_id *platform,
                                   cl_device_id *device_id,
                                   int just_print)
{
  cl_int err_code;
  cl_uint num_plat;
  cl_int p = clGetPlatformIDs (0, NULL, &num_plat);

  if (just_print)
  {
    if (num_plat == 1) printf ("There is 1 platform available\n");
    else printf ("There are %u platforms available\n", num_plat);
  }
  if (num_plat <= 0) return 0;

  cl_platform_id *ids = (cl_platform_id *)malloc (sizeof (cl_platform_id) * num_plat);
  clGetPlatformIDs (num_plat, ids, NULL);

  for (unsigned int i=0; i<num_plat; i++)
  {
    if (just_print) utils_print_platform_info (ids[i]);

    cl_uint num_devs;
    err_code = clGetDeviceIDs (ids[i], dev_type, 0, NULL, &num_devs);
    if (err_code != CL_SUCCESS || num_devs <= 0) continue;

    cl_device_id *dev_ids = (cl_device_id *)malloc (sizeof (cl_device_id) * num_devs);
    clGetDeviceIDs (ids[i], dev_type, num_devs, dev_ids, NULL);
    for (unsigned int j=0; j<num_devs; j++)
    {
      if (just_print) utils_print_device_info (dev_ids[j]);
    }

    if (!just_print)
    {
      *platform = ids[i];
      *device_id = dev_ids[0];
      return 1;
    }
  }

  return 0;
}

void utils_print_platform_info (cl_platform_id platform)
{
  char buffer[8*1024];

  clGetPlatformInfo (platform, CL_PLATFORM_NAME, sizeof (buffer), buffer, NULL);
  printf ("PLATFORM_NAME: %s\n", buffer);
  clGetPlatformInfo (platform, CL_PLATFORM_VERSION, sizeof (buffer), buffer, NULL);
  printf ("  VERSION: %s\n", buffer);
  clGetPlatformInfo (platform, CL_PLATFORM_VENDOR, sizeof (buffer), buffer, NULL);
  printf ("  VENDOR: %s\n", buffer);
  clGetPlatformInfo (platform, CL_PLATFORM_PROFILE, sizeof (buffer), buffer, NULL);
  printf ("  PROFILE: %s\n", buffer);
}

void utils_print_device_info (cl_device_id dev_id)
{
  cl_ulong ul_prop;
  cl_uint ui_prop;
  size_t size_prop;
  char buffer[8*1024];

  clGetDeviceInfo (dev_id, CL_DEVICE_NAME, sizeof (buffer), buffer, NULL);
  printf ("    DEVICE: %s\n", buffer);
  clGetDeviceInfo (dev_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof (ui_prop), &ui_prop, NULL);
  printf ("    MAX COMPUTE UNITS: %u\n", ui_prop);
  clGetDeviceInfo (dev_id, CL_DEVICE_VERSION, sizeof (buffer), buffer, NULL);
  printf ("    DEVICE VERSION: %s\n", buffer);
  clGetDeviceInfo (dev_id, CL_DRIVER_VERSION, sizeof (buffer), buffer, NULL);
  printf ("    DRIVER VERSION: %s\n", buffer);

  clGetDeviceInfo (dev_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof (ul_prop), &ul_prop, NULL);
  printf ("    GLOBAL MEM SIZE: %lu\n", (unsigned long)ul_prop);
  clGetDeviceInfo (dev_id, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof (ul_prop), &ul_prop, NULL);
  printf ("    GLOBAL MEM CACHE SIZE: %lu\n", (unsigned long)ul_prop);
  clGetDeviceInfo (dev_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof (ul_prop), &ul_prop, NULL);
  printf ("    LOCAL MEM SIZE: %lu\n", (unsigned long)ul_prop);
  clGetDeviceInfo (dev_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof (ui_prop), &ui_prop, NULL);
  printf ("    MAX CLOCK FREQUENCY: %u\n", ui_prop);
  clGetDeviceInfo (dev_id, CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof (size_prop), &size_prop, NULL);
  printf ("    TIMER RESOLUTION: %lu\n", (unsigned long)size_prop);
}

cl_ulong utils_get_event_time (cl_event event, cl_profiling_info param)
{
  cl_ulong result;
  cl_int err_code;

  err_code = clGetEventProfilingInfo (event, param,
                                      sizeof (result), &result, NULL);
  CHECK_OCL_ERROR (clGetEventProfilingInfo, err_code);

  return result;
}

cl_ulong utils_get_event_execution_time (cl_event event)
{
  return utils_get_event_time (event, CL_PROFILING_COMMAND_END) -
    utils_get_event_time (event, CL_PROFILING_COMMAND_START);
}

