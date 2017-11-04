#pragma once
#ifndef OCL_H
#define OCL_H

#ifdef OPENCL
#include "CL/cl.h"


#ifdef __cplusplus
extern "C" {
#endif

	bool ocl_initialize();
	void ocl_push_array(cl_mem x_gpu, float *x, size_t n);
	cl_mem ocl_make_array(float *x, size_t n);
	cl_mem ocl_make_int_array(size_t n);
	void ocl_push_convolutional_layer(convolutional_layer layer);


#ifdef __cplusplus
}
#endif

#endif	// OPENCL

#endif	// OCL_H