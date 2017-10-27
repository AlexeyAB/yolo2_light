
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


#ifndef INCLUDES_H_
#define INCLUDES_H_


//#include <Windows.h>
#include <string.h>
#include <stdlib.h>
#include "CL/cl.h"

//General defines
#define MAX_STR_LEN          256
#define HALF_STR_LEN         MAX_STR_LEN/2
#define Q_STR_LEN   	 	 HALF_STR_LEN/2

//OpenCL related
#define OCL_STATUS_READY          0
#define OCL_STATUS_INITIALIZED    1
#define OCL_STATUS_PROGRAM_ERROR  2
#define OCL_STATUS_KERNEL_ERROR   3
#define OCL_STATUS_MUTEX_ERROR    4
#define OCL_STATUS_FINALIZED      5

#define OCL_LOCK_SET              1
#define OCL_LOCK_RELEASE          2


/*
//Deep NN related
#define NN_MAX_KERNEL_COUNT				9	// 13
#define NN_KERNEL_IDX_IM2COL3X3			0
#define NN_KERNEL_IDX_IM2COL1X1			1
#define NN_KERNEL_IDX_NORMARR			2
#define NN_KERNEL_IDX_SCALEBIAS			3
#define NN_KERNEL_IDX_ADDBIAS			4
#define NN_KERNEL_IDX_SCALEADDBIAS		5		
#define NN_KERNEL_IDX_NORMSCALEADDBIAS	6
#define NN_KERNEL_IDX_LEAKY_ACTIVATE	7
#define NN_KERNEL_IDX_LINEAR_ACTIVATE	8
#define NN_KERNEL_IDX_FLATARR			9
#define NN_KERNEL_IDX_SOFTMAX			10
#define NN_KERNEL_IDX_MAXPOOL			11
#define NN_KERNEL_IDX_RESETARR			12
*/



#endif /* INCLUDES_H_ */


