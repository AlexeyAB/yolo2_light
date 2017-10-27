
#ifndef IMAGE_2D_H
#define IMAGE_2D_H

#include "CL/cl.h"
#include "buffer.h"

class Image2D : public OCLBuffer
{
public:
    /**
 * @brief
 *
 * @param mem
 * @param queue
 * @param rowPitch
 */
    Image2D(cl_mem mem, cl_command_queue* queue, int rowPitch = 0);
    /**
         * @brief
         *
         * @param hostMem
         * @param size[]
         * @param offset[]
         * @param blocking
         */
    void read(void* hostMem, const size_t size[2],
    const size_t offset[2], cl_bool blocking = CL_TRUE);
    /**
         * @brief
         *
         * @param hostMem
         * @param size[]
         * @param offset[]
         * @param blocking
         */
    void write(void* hostMem, const size_t size[2],
    const size_t offset[2], cl_bool blocking = CL_TRUE);
    /**
         * @brief
         *
         * @param flags
         * @param size[]
         * @param offset[]
         * @param rowPitch
         * @param blocking
         */
    void* map(cl_map_flags flags, const size_t size[2],
    const size_t offset[2], size_t& rowPitch, cl_bool blocking = CL_TRUE);
    /**
         * @brief
         *
         * @param dst
         * @param size[]
         * @param srcOffset[]
         * @param dstOffset
         */
    void copyToBuffer(OCLBuffer& dst, const size_t size[2],
    const size_t srcOffset[2], const size_t dstOffset = 0);
    /**
         * @brief
         *
         * @param paramName
         */
    void* getInfo(const cl_image_info paramName);
    /**
         * @brief
         *
         */
    ~Image2D()
    {

    }

protected:
    size_t _rowPitch;

private:
};



#endif //IMAGE_2D_H
