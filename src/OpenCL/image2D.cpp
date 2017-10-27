#include "include/image2D.h"

/**
 * @brief
 *
 * @param mem
 * @param rowStep Width/Column * Num of channels
 * @param queue
 */
Image2D::Image2D(cl_mem mem, cl_command_queue *queue, int rowStep)
    :OCLBuffer(mem, queue)
{
    //Needed to copy the right amount of data back to the memory
    DEBUG_VALUE("Image2D::Image2D Constructor: ", mem);
    this->_rowPitch = rowStep;
}

void Image2D::read(void *hostMem, const size_t origin[], const size_t region[], cl_bool blocking)
{
	cl_int err = 0;
    DEBUG_STRING("Image2D::read");
    DEBUG_VALUE("memory :",_memory);

    DEBUG_STRING("Image Region:");
    DEBUG_VALUE("Width :" , region[0]);
    DEBUG_VALUE("Height :", region[1]);
    DEBUG_VALUE("Depth :" , region[2]);
    DEBUG_VALUE("Row pitch is: ", _rowPitch);

    DEBUG_STRING("Image Origin:");
    DEBUG_VALUE("Width :" , origin[0]);
    DEBUG_VALUE("Height :", origin[1]);
    DEBUG_VALUE("Depth :" , origin[2]);

    if(blocking)
        DEBUG_STRING("BLOCKING READ");

    err = clEnqueueReadImage(*_pQueue, _memory, blocking,
                             origin, region, _rowPitch, 0,
                             hostMem, 0, NULL, NULL);

    DEBUG_CL(err);
}


/**
 * @brief
 *
 * @param hostMem
 * @param size[]  region
 * @param offset[] origin
 * @param blocking
 */
void Image2D:: write(void *hostMem, const size_t origin[], const size_t region[], cl_bool blocking)
{
    cl_int err = 0;
    DEBUG_STRING("Image2D:: write");
    DEBUG_VALUE("Image2D _memory write:",_memory);
    err = clEnqueueWriteImage(*_pQueue, _memory, blocking, origin,
                              region, _rowPitch, 0,
                              hostMem, 0, NULL, NULL);
    DEBUG_CL(err);
}


void *Image2D::map(cl_map_flags flags, const size_t size[], const size_t offset[], size_t &_rowPitch, cl_bool blocking)
{
    size_t slicePitch;
    cl_int err = 0;
    void* ret = clEnqueueMapImage(*_pQueue, _memory, blocking, flags, offset, size, &_rowPitch, &slicePitch, 0, NULL, NULL, &err);
    DEBUG_CL(err);
    return ret;
}


void Image2D::copyToBuffer(OCLBuffer &dst, const size_t size[], const size_t srcOffset[], const size_t dstOffset)
{
    cl_int err = 0;
    err = clEnqueueCopyImageToBuffer(*_pQueue, _memory, dst.getMem(), srcOffset, size, dstOffset, 0, NULL, NULL);
    DEBUG_CL(err);
}


void *Image2D::getInfo(const cl_image_info paramName)
{
    cl_int err = 0;
    size_t size;
    err = clGetImageInfo (_memory, paramName, 0, NULL, &size);
    DEBUG_CL(err);

    if(size > 0) {
        void* info = malloc(size);
        err = clGetImageInfo (_memory, paramName, size, info, &size);
        DEBUG_CL(err);
        return info;
    }
    else return NULL;
}
