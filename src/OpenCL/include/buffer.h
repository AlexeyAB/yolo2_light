/*
****************************************************************************
BSD License
Copyright (c) 2014, i-Vizon
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. All advertising materials mentioning features or use of this software
   must display the following acknowledgement:
   This product includes software developed by the i-Vizon.
4. Neither the name of the i-Vizon nor the
   names of its contributors may be used to endorse or promote products
   derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Mageswaran.D ''AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL Mageswaran.D BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

****************************************************************************
*/
/*
* =============================================================================
*
*
*   FILENAME            : Buffer.h
*
*   DESCRIPTION         : A wrapper library for OpenCL and its native counter part
*                         intialization. With boost thread support.
*
*   AUTHOR              : Mageswaran D
*
*
*   CHANGE HISTORY      :
*
*   DATE                : 17th Mar 2014
*
*   File Created.
*
* =============================================================================
*/
#ifndef BUFFER_H
#define BUFFER_H

#include "iv_common.h"
#include "CL/cl.h"

class Image2D;
/**
 * @brief
 *
 */
class OCLBuffer
{
public:
    /**
     * @brief
     *
     * @param tmp
     * @param queue
     */
    OCLBuffer(cl_mem tmp, cl_command_queue *queue);

    /**
     * @brief
     *
     * @return cl_mem
     */
    cl_mem getMem();

    /**
     * @brief
     *
     * @param hostMem
     * @param size
     * @param offset
     * @param blocking
     */
    void read(void* hostMem, const size_t size, const size_t offset=0, const cl_bool blocking=CL_TRUE);

    /**
     * @brief
     *
     * @param hostMem
     * @param size
     * @param offset
     * @param blocking
     */
    void write(const void* hostMem, const size_t size, const size_t offset=0, const cl_bool blocking=CL_TRUE);

    /**
     * @brief
     *
     * @param dst
     * @param size
     * @param srcOffset
     * @param dstOffset
     */
    void copy(OCLBuffer& dst, const size_t size, const size_t srcOffset=0, const size_t dstOffset=0);


    /**
         * @brief
         *
         * @param flags
         * @param size
         * @param offset
         * @param blocking
         */
    void* map(const cl_map_flags flags, const size_t size, const size_t offset, const cl_bool blocking=CL_TRUE);

	void FillBuffer(const void *hostMem, const size_t size, const size_t offset);

    /**
         * @brief
         *
         * @param mappedPtr
         */
    void unmap(void* mappedPtr);

    /**
         * @brief
         *
         * @param dst
         * @param size[]
         * @param srcOffset
         * @param dstOffset[]
         */
    void copyToImage2D(Image2D& dst, const size_t size[2], const size_t srcOffset, const size_t dstOffset[2]);

    /**
         * @brief
         *
         * @param dst
         * @param size[]
         * @param srcOffset
         * @param dstOffset[]
         */
    void copyToImage3D(Image2D& dst, const size_t size[3], const size_t srcOffset, const size_t dstOffset[3]);

    /**
         * @brief
         *
         * @param paramName
         */
    void* getMemInfo(const cl_mem_info paramName);

    virtual ~OCLBuffer()
    {
        clReleaseMemObject(_memory);
        //DEBUG_STRING("Releasing GPU Memory Buffers");
    }

protected:
    cl_mem _memory; /**< TODO */
    cl_command_queue* _pQueue; /**< TODO */
private:

};

#endif // BUFFER_H
