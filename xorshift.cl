uint xorshift_int(uint4* ctx) {
	uint t = ctx->x ^ (ctx->x << 11);
	*ctx = ctx->yzww;
	ctx->w = ctx->w ^ (ctx->w >> 19) ^ (t ^ (t >> 8));

	return ctx->w;
}

float xorshift_float(uint4* ctx) {
	return xorshift_int(ctx) * 2.3283064e-10;
}

#define M_MY_PI 3.14159265f
float2 normal_box_muller(uint4* ctx) {
	float u1 = xorshift_float(ctx);
	float u2 = xorshift_float(ctx);

	float ln_u1 = sqrt(-2.0f * log(u1));

	float p2_u2 = 2.0f * M_MY_PI * u2;
	
	float2 res;
	res.x = ln_u1 * cos(p2_u2);
	res.y = ln_u1 * sin(p2_u2);
	return res;
}

__kernel void uniform_int(__global uint4* context, __global uint* output, int len)
{
	int num_per_thread = len / get_global_size(0);
	int tid = get_global_id(0);
	
	//read context from global memory
	uint4 ctx = context[tid];
	int base_offset = tid * num_per_thread;

	for(int i = 0; i < num_per_thread; i++) {
		output[base_offset + i] = xorshift_int(&ctx);
	}

	//save context back to global memory
	context[tid] = ctx;
}

__kernel void uniform_float(__global uint4* context, __global float* output, int len)
{
	int num_per_thread = len / get_global_size(0);
	int tid = get_global_id(0);
	
	//read context from global memory
	uint4 ctx = context[tid];
	int base_offset = tid * num_per_thread;

	for(int i = 0; i < num_per_thread; i++) {
		output[base_offset + i] = xorshift_float(&ctx);
	}

	//save context back to global memory
	context[tid] = ctx;
}

__kernel void normal_float(__global uint4* context, __global float2* output, int len)
{
	int num_per_thread = len / get_global_size(0) / 2;
	int tid = get_global_id(0);
	
	//read context from global memory
	uint4 ctx = context[tid];
	int base_offset = tid * num_per_thread;

	for(int i = 0; i < num_per_thread; i++) {
		output[base_offset + i] = normal_box_muller(&ctx);
	}

	//save context back to global memory
	context[tid] = ctx;
}
