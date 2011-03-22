#include <stdio.h>
#include <stdlib.h>

#include "clrand.h"

static char* XORSHIFT_FILE_NAME = "xorshift.cl";

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

int clrand_create_rng_context(clrand_context* ctx) {

}


int clrand_init(clrand_context* ctx, cl_context cl_ctx, cl_command_queue queue) {
	clRetainContext(cl_ctx);
	clRetainCommandQueue(queue);

	ctx->context = cl_ctx;
	ctx->queue = queue;

	size_t size = get_file_size(XORSHIFT_FILE_NAME);
	char* content = (char* )malloc(sizeof(char) * (size + 1));
	load_kernels(content, size);
	
	printf("%s", content);

	free(content);

	clrand_create_rng_context(ctx);
}

int clrand_release(clrand_context* ctx) {
	clReleaseContext(ctx->context);
	clReleaseCommandQueue(ctx->queue);
}

int clrand_set_seed(clrand_context* ctx, int seed) {
	srand(seed);
	clrand_create_rng_context(ctx);
}

int clrand_rand_uniform(clrand_context* ctx, cl_mem buffer) {

}

int clrand_rand_normal(clrand_context* ctx, cl_mem buffer) {

}
