#ifndef SAMPLER_H
#define SAMPLER_H
#include <CL/cl.h>

class Sampler
{
public:
    Sampler(cl_context*          /* context */,
            cl_bool             /* normalized_coords */,
            cl_addressing_mode  /* addressing_mode */,
            cl_filter_mode      /* filter_mode */);
    cl_sampler& getSampler();
protected:
private:
    cl_context* _pContext;
    cl_sampler _sampler;
};

#endif // SAMPLER_H
