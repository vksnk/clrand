#include <stdio.h>
#include <stdlib.h>

#include <CL/cl.h>

#include "clrand.h"

const char *source = "__kernel void increment(__global int *input, __global int *output) {	int tid = get_global_id(0);	output[tid] = input[tid] + 1;}";

const int BUF_SIZE = 1024 * 1024;

int main() {
	printf("Pseudo-random number generator using OpenCL example\n");

	cl_int error;
	cl_int arg;

	cl_platform_id platform;
	cl_device_id device_id;
	cl_context context;
	cl_command_queue queue;

	cl_mem output;

	clrand_context rnd;
	
	error = clGetPlatformIDs(1, &platform, NULL);
	check_for_error(error, "Can not get platforms id");

	error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	check_for_error(error, "Can not get device id");

	context = clCreateContext(0, 1, &device_id, NULL, NULL, &error);
	check_for_error(error, "Can not create context");

	queue = clCreateCommandQueue(context, device_id, 0, &error);
	check_for_error(error, "Can not create command queue");

	clrand_init(&rnd, context, queue, 256);
	

	output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * BUF_SIZE, NULL, NULL);

	int *data = (int *)malloc(sizeof(int) * BUF_SIZE);

	int global_num = BUF_SIZE;
	int local_num = 64;

	int i;
	clrand_uniform(&rnd, output);
	//clEnqueueNDRangeKernel(queue, increment_kernel, 1, NULL, &global_num, &local_num, 0, NULL, NULL);
	clFinish(queue);
	
	clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(int) * BUF_SIZE, data, 0, NULL, NULL);  
	
	int correct;
	for(i = 0; i < 384; i++) { 
		correct = correct && (data[i] == i + 1); 
		printf("%d ", data[i]);
	}
	
	if(correct) {
		printf("Correct!\n");
	} else {
		printf("Wrong!\n");
	}
	
	free(data);

	//releasing clrand
	clrand_release(&rnd);

	//releasing OpenCL objects

	clReleaseMemObject(output);

	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	return 0;
}
