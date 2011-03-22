#include <CL/cl.h>


struct clrand_context {
	cl_context context;
	cl_command_queue queue;
	cl_mem dev_seed;
	cl_program program;
	cl_kernel uniform;
	cl_kernel normal;
};

typedef struct clrand_context clrand_context;

int clrand_init(clrand_context* ctx, cl_context cl_ctx, cl_command_queue queue);
int clrand_release(clrand_context* ctx);
