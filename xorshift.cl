__kernel void uniform_rng(__global int4* context, __global int* output, int num_per_thread)
{
	int tid = get_global_id(0);
	
	//read context from global memory
	int4 ctx = context[tid];
	uint4 mask = (uint4)(1, 2, 3, 0);
	int base_offset = tid * num_per_thread;

	for(int i = 0; i < num_per_thread; i++) {
		int t = ctx.x ^ (ctx.x << 11);
		//ctx = shuffle(ctx, mask);
		ctx.x = ctx.y;
		ctx.y = ctx.z;
		ctx.z = ctx.w;
		
		ctx.w = ctx.w ^ (ctx.w >> 19) ^ (t ^ (t >> 8));
		output[base_offset + i] = ctx.w;
	}

	//save context back to global memory
	context[tid] = ctx;
}
