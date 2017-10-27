

#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <iostream>
#include <vector>

#include "CL/cl.h"

///!TODO: Use bits as enum value
typedef enum {
    IV_CHAR_FILE_OPEN_FAIL = 0,
    IV_CHAR_FILE_STATUS_FAIL
}IV_ERRORS_INFO;

const char* getCLErrorString(int err);
const char* getCustomErrorString(int err, IV_ERRORS_INFO info);
std::string getImgType(int imgTypeInt);


/*#define DEBUG_CL(err) \
    if(err< 0) { \
    std::cout<<"FILE: "<<__FILE__<<std::cout<<" Line Number: "<<__LINE__\
    <<" Function Name : "<<__func__<<"Error Name:" \
    <<getCLErrorString(err) \
    <<std::endl; \
    exit(err); }*/

#define DEBUG_IV(err, info) \
    if(err< 0) { \
    std::cout<<"Line Number: "<<__LINE__<<" Function Name : "<<__func__<<"Error Name:" \
    <<getCustomErrorString(err, info) \
    <<std::endl; \
    exit(err); }

//#define DEBUG_VALUE(dbgMsg, value)    \
//    std::cout<<"=====>"<<"   " <<dbgMsg<<"......."<<value<<std::endl;

//#define DEBUG_STRING(dbgMsg)    \
    //std::cout<<">>>>>>"<<dbgMsg<<std::endl;

#define ERROR_PRINT_VALUE(dbgMsg, value)    \
    { \
    std::cout<<"\n=====> Line Number: "<<__LINE__<<" Function Name :"<<__func__\
    <<"\n "<<dbgMsg<<" "<<value<<std::endl; \
    exit(0); \
    };

#define ERROR_PRINT_STRING(dbgMsg)    \
    {   \
    std::cout<<"\n=====> Line Number: "<<__LINE__<<" Function Name: "<<__func__\
    <<"\n "<<dbgMsg<<std::endl; \
    exit(0); \
    };

#ifdef IVIZON_DEBUG
    #define F_LOG LogBlock _l(__func__)
        struct LogBlock {
            const char *mLine;
            LogBlock(const char *line) : mLine(line) {
                std::cout<<mLine <<"  ----->#### Enter \n";
            }
            ~LogBlock() {
                std::cout<<mLine <<" <-----#### Leave \n";
            }
        };
#else
    #define F_LOG {}
    #define DEBUG_CL(err) {}
    #define DEBUG_STRING(dbgMsg) {}
    #define DEBUG_VALUE(dbgMsg, value) {}
#endif



// CV Defines
#define CV_8U   0
#define CV_8S   1
#define CV_16U  2
#define CV_16S  3
#define CV_32S  4
#define CV_32F  5
#define CV_64F  6

#define CV_CN_MAX     512

#define CV_MAT_DEPTH_MASK       (CV_DEPTH_MAX - 1)

#define CV_CN_SHIFT   3
#define CV_MAT_DEPTH(flags)     ((flags) & CV_MAT_DEPTH_MASK)
#define CV_DEPTH_MAX  (1 << CV_CN_SHIFT)

#define CV_MAKETYPE(depth,cn) (CV_MAT_DEPTH(depth) + (((cn)-1) << CV_CN_SHIFT))
#define CV_MAKE_TYPE CV_MAKETYPE

#define CV_8UC1 CV_MAKETYPE(CV_8U,1)
#define CV_8UC2 CV_MAKETYPE(CV_8U,2)
#define CV_8UC3 CV_MAKETYPE(CV_8U,3)
#define CV_8UC4 CV_MAKETYPE(CV_8U,4)
#define CV_8UC(n) CV_MAKETYPE(CV_8U,(n))

#define CV_8SC1 CV_MAKETYPE(CV_8S,1)
#define CV_8SC2 CV_MAKETYPE(CV_8S,2)
#define CV_8SC3 CV_MAKETYPE(CV_8S,3)
#define CV_8SC4 CV_MAKETYPE(CV_8S,4)
#define CV_8SC(n) CV_MAKETYPE(CV_8S,(n))

#define CV_16UC1 CV_MAKETYPE(CV_16U,1)
#define CV_16UC2 CV_MAKETYPE(CV_16U,2)
#define CV_16UC3 CV_MAKETYPE(CV_16U,3)
#define CV_16UC4 CV_MAKETYPE(CV_16U,4)
#define CV_16UC(n) CV_MAKETYPE(CV_16U,(n))

#define CV_16SC1 CV_MAKETYPE(CV_16S,1)
#define CV_16SC2 CV_MAKETYPE(CV_16S,2)
#define CV_16SC3 CV_MAKETYPE(CV_16S,3)
#define CV_16SC4 CV_MAKETYPE(CV_16S,4)
#define CV_16SC(n) CV_MAKETYPE(CV_16S,(n))

#define CV_32SC1 CV_MAKETYPE(CV_32S,1)
#define CV_32SC2 CV_MAKETYPE(CV_32S,2)
#define CV_32SC3 CV_MAKETYPE(CV_32S,3)
#define CV_32SC4 CV_MAKETYPE(CV_32S,4)
#define CV_32SC(n) CV_MAKETYPE(CV_32S,(n))

#define CV_32FC1 CV_MAKETYPE(CV_32F,1)
#define CV_32FC2 CV_MAKETYPE(CV_32F,2)
#define CV_32FC3 CV_MAKETYPE(CV_32F,3)
#define CV_32FC4 CV_MAKETYPE(CV_32F,4)
#define CV_32FC(n) CV_MAKETYPE(CV_32F,(n))

#define CV_64FC1 CV_MAKETYPE(CV_64F,1)
#define CV_64FC2 CV_MAKETYPE(CV_64F,2)
#define CV_64FC3 CV_MAKETYPE(CV_64F,3)
#define CV_64FC4 CV_MAKETYPE(CV_64F,4)
#define CV_64FC(n) CV_MAKETYPE(CV_64F,(n))


/// IV Data Types
//This is done to have more control on memory and range on numbers

typedef unsigned char   IV_8U;
typedef char            IV_8S;
typedef unsigned        IV_16U;
typedef signed          IV_16S;
typedef int             IV_32S;
typedef unsigned int    IV_32U;
typedef float           IV_32F;
typedef double          IV_64F;
#endif
