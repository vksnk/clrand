#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <CL/cl.h>

#include "clrand.h"

const char *source = "__kernel void increment(__global int *input, __global int *output) {	int tid = get_global_id(0);	output[tid] = input[tid] + 1;}";

const int BUF_SIZE = 32 * 1024*1024;
const int HIST_SIZE = 256;

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

	clrand_init(&rnd, context, queue, 2048);
	//clrand_set_seed(&rnd, time(NULL));

	output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * BUF_SIZE, NULL, NULL);

	clrand_uniform(&rnd, output, BUF_SIZE);
	
	clFinish(queue);
	int *data = (int *)malloc(sizeof(int) * BUF_SIZE);
	clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(int) * BUF_SIZE, data, 0, NULL, NULL);  
	
	int i;
	
	//build and output histogram to check that distribution is really uniform
	int hist[HIST_SIZE];
	for(i = 0; i < HIST_SIZE; i++) hist[i] = 0;
	for(i = 0; i < BUF_SIZE; i++) { 
		//printf("%d ", data[i]);
		int val = (int)((float)data[i] / RAND_MAX * HIST_SIZE);
		hist[val]++;
	}
	
	for(i = 0; i < HIST_SIZE; i++) printf("%d ", hist[i]);

	free(data);

	//releasing clrand
	clrand_release(&rnd);

	//releasing OpenCL objects

	clReleaseMemObject(output);

	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	return 0;
}
