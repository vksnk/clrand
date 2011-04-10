uint xorshift_int(uint4* ctx) {
	uint t = ctx->x ^ (ctx->x << 11);
	ctx->x = ctx->y;
	ctx->y = ctx->z;
	ctx->z = ctx->w;
		
	ctx->w = ctx->w ^ (ctx->w >> 19) ^ (t ^ (t >> 8));

	return ctx->w;
}

float xorshift_float(uint4* ctx) {
	return xorshift_int(ctx) * 2.3283064e-10;
}

__kernel void uniform_rng(__global uint4* context, __global uint* output, int len)
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
