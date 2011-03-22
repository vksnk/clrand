#include <stdio.h>
#include <stdlib.h>

#include <CL/cl.h>

#include "clrand.h"

const char *source = "__kernel void increment(__global int *input, __global int *output) {	int tid = get_global_id(0);	output[tid] = input[tid] + 1;}";

const int BUF_SIZE = 1024;

void check_for_error(cl_int error, char *error_message) {
	if(error == CL_SUCCESS) return;
	printf("Error::%s\n", error_message);
	exit(1);
}

int main() {
	printf("Pseudo-random number generator using OpenCL example\n");

	cl_int error;
	cl_int arg;

	cl_platform_id platform;
	cl_device_id device_id;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_kernel increment_kernel;

	cl_mem input;
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

	program = clCreateProgramWithSource(context, 1, &source, NULL, &error);
	check_for_error(error, "Can not create program from sources");

	error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	check_for_error(error, "Can not build program");

	clrand_init(&rnd, context, queue);
	increment_kernel = clCreateKernel(program, "increment", &error);
	check_for_error(error, "Can not create kernel");

	input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * BUF_SIZE, NULL, NULL);
	output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * BUF_SIZE, NULL, NULL);

	int *data = (int *)malloc(sizeof(int) * BUF_SIZE);
	int i = 0;
	for(; i < BUF_SIZE; i++) { 
		data[i] = i; 
	}

	error = clEnqueueWriteBuffer(queue, input, CL_TRUE, 0, sizeof(int) * BUF_SIZE, data, 0, NULL, NULL);
	check_for_error(error, "Can not write buffer");
	
	error = 0;
	arg = 0;
	error |= clSetKernelArg(increment_kernel, arg++, sizeof(cl_mem), &input);
	error |= clSetKernelArg(increment_kernel, arg++, sizeof(cl_mem), &output);
	
	check_for_error(error, "Can not set kernel arguments");

	int global_num = BUF_SIZE;
	int local_num = 64;

	clEnqueueNDRangeKernel(queue, increment_kernel, 1, NULL, &global_num, &local_num, 0, NULL, NULL);
	clFinish(queue);

	clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(int) * BUF_SIZE, data, 0, NULL, NULL);  

	int correct;
	for(i = 0; i < BUF_SIZE; i++) { 
		correct = correct && (data[i] == i + 1); 
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
	clReleaseMemObject(input);
	clReleaseMemObject(output);
	clReleaseProgram(program);
	clReleaseKernel(increment_kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	return 0;
}
