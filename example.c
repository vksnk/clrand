#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <string.h>

#include <CL/cl.h>

#include "clrand.h"

const int BUF_SIZE = 1024 * 1024;
const int HIST_SIZE = 16;
const int THREAD_NUM = 256;
const int HIST_LINE_LEN = 20;

#define DATA_TYPE float

void xorshift_CPU(unsigned int* data) {
	int i = 0;
	
	unsigned int x = 123456789;
	unsigned int y = 362436069;
	unsigned int z = 521288629;
	unsigned int w = 88675123;
	unsigned int t;

	for(i = 0; i < BUF_SIZE; i++) {
		t = x ^ (x << 11);
		x = y; y = z; z = w;
		w = w ^ (w >> 19) ^ (t ^ (t >> 8));
		data[i] = w;
	}
}

//build and output histogram to check that distribution is really uniform	
void print_hist(DATA_TYPE* data) {
	int i;
	int val;
	int hist[HIST_SIZE];
	for(i = 0; i < HIST_SIZE; i++) hist[i] = 0;
	printf("Buffer size: %d\n", BUF_SIZE * 4);
	for(i = 0; i < BUF_SIZE; i++) { 
		//printf("%f ", data[i]);
		//int val = (double)(data[i]) / UINT_MAX * HIST_SIZE;
		int val = data[i] * HIST_SIZE;
		if((val < 0) || (val >= HIST_SIZE)) continue;
		hist[val]++;
	}
	printf("Histogram:: \n");
	for(i = 0; i < HIST_SIZE; i++) printf("%d ", hist[i]);

	printf("\n");
}

void pretty_print_hist(DATA_TYPE* data) {
	int i = 0;
	DATA_TYPE min_val = 10000000, max_val = 0;
	for(i = 0; i < BUF_SIZE; i++) {
		if(data[i] > max_val) max_val = data[i];
		if(data[i] < min_val) min_val = data[i];
	}
	int hist[HIST_SIZE];
	//printf("\n%f %f\n", min_val, max_val);
	memset(hist, 0, HIST_SIZE*sizeof(int));
	for(i = 0; i < BUF_SIZE; i++) {
		int val = ((float)data[i] - min_val) / (max_val - min_val) * HIST_SIZE;
		//printf("%f %d\n", data[i], val);
		hist[val]++;
	}
	
	int max_at_hist = 0;
	for(i = 0; i < HIST_SIZE; i++) if(max_at_hist < hist[i]) max_at_hist = hist[i];

	for(i = 0; i < HIST_SIZE; i++) {
		printf("\n%8d :", hist[i]);
		int j = 0;
		int len = (float)(hist[i] - 1) / max_at_hist * HIST_LINE_LEN + 1;
		for(j = 0; j < len; j++) printf("*");
	}
	printf("\n");
}

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

	clock_t gpu_start = clock();
	clrand_init(&rnd, device_id, context, queue, THREAD_NUM);
	clrand_set_seed(&rnd, time(NULL));

	output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(DATA_TYPE) * BUF_SIZE, NULL, NULL);


	clrand_normal_float(&rnd, output, BUF_SIZE);
	clFinish(queue);
	clock_t gpu_end = clock();
	
	DATA_TYPE *data = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * BUF_SIZE);
	clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(DATA_TYPE) * BUF_SIZE, data, 0, NULL, NULL);  
	
	pretty_print_hist(data);
	
	free(data);

	printf("GPU:: %.3fs\n", (float)(gpu_end - gpu_start) / CLOCKS_PER_SEC);
	//printf("CPU:: %.3fs\n", (float)(cpu_end - cpu_start) / CLOCKS_PER_SEC);

	//releasing clrand
	clrand_release(&rnd);

	//releasing OpenCL objects
	clReleaseMemObject(output);

	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	return 0;
}
