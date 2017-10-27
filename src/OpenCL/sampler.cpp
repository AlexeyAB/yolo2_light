#include "include/sampler.h"
#include "include/iv_common.h"

Sampler::Sampler(cl_context* context, cl_bool normalizedCoords, cl_addressing_mode addrMode, cl_filter_mode filterMode)
{
//    clCreateSampler(cl_context          /* context */,
//                    cl_bool             /* normalized_coords */,
//                    cl_addressing_mode  /* addressing_mode */,
//                    cl_filter_mode      /* filter_mode */,
//                    cl_int *            /* errcode_ret */)
    // Create the image sampler
    cl_int status;
    _sampler = clCreateSampler(*context, normalizedCoords,
                                         addrMode, filterMode, &status);
    DEBUG_CL(status);
}

cl_sampler& Sampler::getSampler()
{
    return _sampler;
}
