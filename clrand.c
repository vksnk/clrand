#include <stdio.h>
#include <stdlib.h>

#include "clrand.h"

static char* XORSHIFT_FILE_NAME = "xorshift.cl";

//actually, this function should not be here
void check_for_error(cl_int error, char *error_message) {
	if(error == CL_SUCCESS) return;
	printf("Error::%s\n", error_message);
	exit(1);
}

size_t get_file_size(char* fn) {
	FILE* fp = fopen(fn, "rt");
	fseek(fp, 0L, SEEK_END);
	size_t size = ftell(fp);
	fclose(fp);

	return size;
}

void load_kernels(char* content, size_t size) {
	FILE* fp = fopen(XORSHIFT_FILE_NAME, "r");
	fread(content, 1, size, fp);
	content[size] = '\0';
	fclose(fp);
}

void show_build_log(clrand_context* ctx) {
	cl_build_status build_status;
	cl_int error;
	error = clGetProgramBuildInfo(ctx->program, ctx->device, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &build_status, NULL);

	size_t ret_val_size;
	error = clGetProgramBuildInfo(ctx->program, ctx->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);

	char *build_log = (char* )malloc(ret_val_size+1);
	error = clGetProgramBuildInfo(ctx->program, ctx->device, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
	build_log[ret_val_size] = '\0';
	printf("BUILD LOG: \n%s", build_log);
	free(build_log);
}

int clrand_update_seed(clrand_context* ctx) {
	int i = 0;
	cl_int error;
	
	//context per thread is four int vector, so size of 'seed' is thread_num * 4
	int buffer_size = ctx->parallel_thread_num * 4;
	int* buffer = (int* )malloc(sizeof(int) * buffer_size);
	//fill seed using CPU-side rand(). I hope it's ok :)
	for(i = 0; i < buffer_size; i++) buffer[i] = rand();
	
	if(ctx->dev_seed) clReleaseMemObject(ctx->dev_seed);
	//upload generated seed to GPU device
	ctx->dev_seed = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY, sizeof(cl_uint) * buffer_size, NULL, NULL);
	error = clEnqueueWriteBuffer(ctx->queue, ctx->dev_seed, CL_TRUE, 0, sizeof(cl_uint) * buffer_size, buffer, 0, NULL, NULL);
	check_for_error(error, "Can not upload new seed");
	free(buffer);

	return 0;
}


int clrand_init(clrand_context* ctx, cl_device_id device, cl_context cl_ctx, cl_command_queue queue, int parallel_thread_num) {
	cl_int error;

	ctx->device = device;
	ctx->dev_seed = NULL;
	ctx->context = cl_ctx;
	ctx->queue = queue;
	ctx->parallel_thread_num = parallel_thread_num;

	clRetainContext(cl_ctx);
	clRetainCommandQueue(queue);

	size_t size = get_file_size(XORSHIFT_FILE_NAME);
	char* content = (char* )malloc(sizeof(char) * (size + 1));
	load_kernels(content, size);
	
	//printf("%s", content);

	ctx->program = clCreateProgramWithSource(ctx->context, 1, (const char**)&content, NULL, &error);
	check_for_error(error, "clrand::Can not create program from sources");

	error = clBuildProgram(ctx->program, 0, NULL, NULL, NULL, NULL);
	show_build_log(ctx);
	check_for_error(error, "clrand::Can not build program");

	ctx->uniform = clCreateKernel(ctx->program, "uniform_int", &error);
	check_for_error(error, "Can not create kernel uniform_int");
	ctx->uniform_float = clCreateKernel(ctx->program, "uniform_float", &error);
	check_for_error(error, "Can not create kernel uniform_float");
	ctx->normal_float = clCreateKernel(ctx->program, "normal_float", &error);
	check_for_error(error, "Can not create kernel normal_float");

	free(content);

	clrand_update_seed(ctx);

	return 0;
}

int clrand_release(clrand_context* ctx) {
	if(ctx->dev_seed) clReleaseMemObject(ctx->dev_seed);
	clReleaseKernel(ctx->uniform);
	clReleaseKernel(ctx->uniform_float);
	clReleaseKernel(ctx->normal_float);
	clReleaseProgram(ctx->program);
	clReleaseContext(ctx->context);
	clReleaseCommandQueue(ctx->queue);

	return 0;
}

int clrand_set_seed(clrand_context* ctx, int seed) {
	srand(seed);
	clrand_update_seed(ctx);
	return 0;
}

int clrand_uniform(clrand_context* ctx, cl_mem buffer, cl_int buf_size) {
	cl_kernel kernel = ctx->uniform;
	
	cl_int error = 0;
	int arg = 0;
	
	error |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &ctx->dev_seed);
	error |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &buffer);
	error != clSetKernelArg(kernel, arg++, sizeof(cl_mem), &buf_size);
	check_for_error(error, "Can not set kernel arguments for uniform float");

	error = clEnqueueNDRangeKernel(ctx->queue, kernel, 1, NULL, &ctx->parallel_thread_num, NULL, 0, NULL, NULL);
	check_for_error(error, "Can not call uniform kernel");

	return 0;
}

int clrand_uniform_float(clrand_context* ctx, cl_mem buffer, cl_int buf_size) {
	cl_kernel kernel = ctx->uniform_float;
	
	cl_int error = 0;
	int arg = 0;
	
	error |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &ctx->dev_seed);
	error |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &buffer);
	error != clSetKernelArg(kernel, arg++, sizeof(cl_mem), &buf_size);
	check_for_error(error, "Can not set kernel arguments for uniform float");

	error = clEnqueueNDRangeKernel(ctx->queue, kernel, 1, NULL, &ctx->parallel_thread_num, NULL, 0, NULL, NULL);
	check_for_error(error, "Can not call uniform float kernel");

	return 0;
}

int clrand_normal_float(clrand_context* ctx, cl_mem buffer, cl_int buf_size) {
	cl_kernel kernel = ctx->normal_float;
	
	cl_int error = 0;
	int arg = 0;
	
	error |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &ctx->dev_seed);
	error |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &buffer);
	error != clSetKernelArg(kernel, arg++, sizeof(cl_mem), &buf_size);
	check_for_error(error, "Can not set kernel arguments for normal float");

	error = clEnqueueNDRangeKernel(ctx->queue, kernel, 1, NULL, &ctx->parallel_thread_num, NULL, 0, NULL, NULL);
	check_for_error(error, "Can not call normal float kernel");

	return 0;
}

