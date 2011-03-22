__kernel void uniform_rng(__global int4* context, __global int* output)
{
	int tid = get_global_id(0);
	output[tid] = 666;
}
