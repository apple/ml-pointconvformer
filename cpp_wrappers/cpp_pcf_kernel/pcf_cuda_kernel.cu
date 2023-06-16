//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2022-2023 Apple Inc. All Rights Reserved.
//
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <vector>

// First goal is to get forward to work to speed up the inference
// Second goal is to get backward to work to save GPU memory so that we can deal with 2cm better and have larger batch sizes

template <typename scalar_t>
__global__ void pcf_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> __restrict__ input,
    const torch::PackedTensorAccessor32<long,3,torch::RestrictPtrTraits> __restrict__ neighbor_inds,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ guidance,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ weights,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> __restrict__ output)
{
/* input: B x N x C_in tensor, B = batch size, N = number of points, C_in = number of channels, input features
   neighbor_inds: B x N x K tensor, K = neighborhood size, indices of the neighborhood of each point
   guidance: B x N x K x num_heads tensor, guidance weight for each point in each neighborhood
   weights: B x N x K x C_mid, C_mid = number of middle channels
   output: B x N x (C_mid*C_in), final output of the PCF layer
   
   This implements a fused layer of:
  sum_{i} input[b,n][neighbor_inds[i]][j] * guidance[b,n,head[j],i] * weights[b,n,k,i]
   It outputs a tensor of shape B x N x (C_mid * C_in)
   It avoids serializing the input hence faster than naive pyTorch implementation
  */
  	int i,k,ii,jj,kk, iter0;
	const int B = input.size(0);
	const int N = input.size(1);
	const int Nout = neighbor_inds.size(1);
	const int C_in = input.size(2);
	const int K = neighbor_inds.size(2);
	const int C_mid = weights.size(3);
	const int increment = blockDim.x / C_mid;
	const int num_heads = guidance.size(3);
  	/* parallelize ii and i on blocks */
  	// Supposedly blockIdx.x should go up to B * N
  	for (iter0 = blockIdx.x; iter0< B * Nout; iter0+= gridDim.x)
  	{
  		// ii is the index on the batch dimension
  	  	ii = iter0 / Nout;
  	  	// i is the index on the point dimension
  	  	i = iter0 % Nout;
		// Suppose each point is a block, then split it into threads
		// output channels is at least 8, C_mid is usually at least 16, so we are safe dividing on this dimension
                // C_mid is at most 16, so for sure each C_mid gets its own thread, and maybe we have many threads for the same C_mid
                jj = threadIdx.x / increment;
		// Throw out the excessive threads
		if (jj >= C_mid)
			continue;
		#pragma unroll
		for(kk=threadIdx.x % increment;kk<C_in;kk+=increment)
		{
			scalar_t partial_sum = 0.0;
			long cur_head = kk % num_heads;
			#pragma unroll
			for (k=0;k<K;k++)
			{
                        	scalar_t real_weight = weights[ii][i][k][jj] * guidance[ii][i][k][cur_head];
				partial_sum += input[ii][neighbor_inds[ii][i][k]][kk] * real_weight; 
			}
			output[ii][i][jj + kk*C_mid] = partial_sum;
		}
         }
}

template <typename scalar_t>
__global__ void pconv_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> __restrict__ input,
    const torch::PackedTensorAccessor32<long,3,torch::RestrictPtrTraits> __restrict__ neighbor_inds,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ weights,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ additional_features,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> __restrict__ output)
{
/* input: B x M x C_in tensor, B = batch size, M = number of points in the original point cloud, C_in = number of channels, input features
   neighbor_inds: B x N x K tensor, K = neighborhood size, indices of the neighborhood of each point
   weights: B x N x K x C_mid, C_mid = number of middle channels
   additional_features: B x N x K x C_add, additional features that do not require indexing
   output: B x N x (C_mid*C_in), final output of the PCF layer
   
   This implements a fused layer of:
  concat(sum_{i} input[b,n][neighbor_inds[i]][j] * weights[b,n,k,i], sum_{i} additional_features[b,n][i][j] * weights[b,n,k,i]
   It outputs a tensor of shape B x N x (C_mid * (C_in + C_add)
   It avoids serializing the input hence faster than naive pyTorch implementation
  */
        int i,k,ii,jj,kk, iter0;
        const int B = input.size(0);
        const int N = input.size(1);
        const int Nout = neighbor_inds.size(1);
        const int C_in = input.size(2);
        const int K = neighbor_inds.size(2);
        const int C_mid = weights.size(3);
	const int C_add = additional_features.size(3);
        const int increment = blockDim.x / C_mid;
        /* parallelize ii and i on blocks */
        // Supposedly blockIdx.x should go up to B * N
        for (iter0 = blockIdx.x; iter0< B * Nout; iter0+= gridDim.x)
        {
                // ii is the index on the batch dimension
                ii = iter0 / Nout;
                // i is the index on the point dimension
                i = iter0 % Nout;
                // Suppose each point is a block, then split it into threads
                // output channels is at least 8, C_mid is usually at least 16, so we are safe dividing on this dimension
                // C_mid is at most 16, so for sure each C_mid gets its own thread, and maybe we have many threads for the same C_mid
                jj = threadIdx.x / increment;
                // Throw out the excessive threads
                if (jj >= C_mid)
                        continue;
// This is our main non-contiguous memory access because we need to sparse gather the input
// But maybe we coalesce memory across all the threads in the block?
                #pragma unroll
                for(kk=threadIdx.x % increment;kk<C_in;kk+=increment)
                {
                        scalar_t partial_sum = 0.0;
                        #pragma unroll
                        for (k=0;k<K;k++)
                                partial_sum += input[ii][neighbor_inds[ii][i][k]][kk] * weights[ii][i][k][jj];
                        output[ii][i][jj + kk*C_mid] = partial_sum;
                }
                if (C_add > 0)
                {
		    #pragma unroll
		    for(kk=threadIdx.x % increment;kk<C_add;kk+=increment)
		    {
			scalar_t partial_sum = 0.0;
			#pragma unroll
			for (k=0;k<K;k++)
				partial_sum += additional_features[ii][i][k][kk] * weights[ii][i][k][jj];
			output[ii][i][jj + (kk + C_in)*C_mid] = partial_sum;
		     }
                }
                // At this point we would have fully populated the interm array of C_mid * C_in
	}
}

template <typename scalar_t>
__global__ void pcf_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> __restrict__ grad_output,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> __restrict__ input,
    const torch::PackedTensorAccessor32<long,3,torch::RestrictPtrTraits> __restrict__ neighbor_inds,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ guidance,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ weights,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> __restrict__ grad_input,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ grad_guidance,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ grad_weights)
/* 
   grad_output: B x N x (C_mid * C_in), the gradient derived from above
   input: B x N x C_in tensor, B = batch size, N = number of points, C_in = number of channels, input features
   neighbor_inds: B x N x K tensor, K = neighborhood size, indices of the neighborhood of each point
   guidance: B x N x K x num_heads tensor, guidance weight for each point in each neighborhood
   weights: B x N x K x C_mid , C_mid = number of middle channels
   grad_input: same shape as input, gradient of input
   grad_guidance: same shape as guidance, gradient of guidance
   grad_weights: same shape as weights, gradient of weights

   Forward is:
    sum_{i} input[b,n][neighbor_inds[i]][j] * guidance[b,n,head[j],i] * weights[b,n,k,i]
*/

{
        int i,j,k, ii, kk,iter0;
        const int B = input.size(0);
        const int N = input.size(1);
        const int Nout = neighbor_inds.size(1);
        const int C_in = input.size(2);
        const int K = neighbor_inds.size(2);
        const int C_mid = weights.size(3);
        const int increment = blockDim.x / C_mid;
        const int num_heads = guidance.size(3);
	const int cur_mid = threadIdx.x / increment;
	if (cur_mid >= C_mid)
		return;
        // Supposedly blockIdx.x should go up to B * N
        for (iter0 = blockIdx.x; iter0< B * Nout; iter0+= gridDim.x)
        {
                // ii is the index on the batch dimension
                ii = iter0 / Nout;
                // i is the index on the point dimension
                i = iter0 % Nout;
		k = threadIdx.x % K;
		scalar_t weight_grad_temp = 0.0;
		// Max number of heads
		scalar_t guidance_grad_temp[32];
		scalar_t cur_compute;
		for (kk=0;kk<num_heads;kk++)
			guidance_grad_temp[kk] = 0.0;
		#pragma unroll
		for (kk=0;kk<C_in;kk++)
		{
			long cur_channel = cur_mid * C_in + kk;
			long cur_head = kk % num_heads;
			cur_compute = grad_output[ii][i][cur_channel] * weights[ii][i][k][cur_mid];
       			guidance_grad_temp[cur_head] += cur_compute * input[ii][neighbor_inds[ii][i][k]][kk];
			weight_grad_temp += grad_output[ii][i][cur_channel] * guidance[ii][i][k][cur_head] * input[ii][neighbor_inds[ii][i][k]][kk];
		// It would be quite expensive to store this in shared memory so use this naive approach for now, using atomicAdd to avoid racing conditions
			atomicAdd(&grad_input[ii][neighbor_inds[ii][i][k]][kk],  guidance[ii][i][k][cur_head] * cur_compute);
		}
		grad_weights[ii][i][k][cur_mid] = weight_grad_temp;
		#pragma unroll
		for (kk=0;kk<num_heads;kk++)
			atomicAdd(&grad_guidance[ii][i][k][kk],guidance_grad_temp[kk]);
		__syncthreads();
	}
}

template <typename scalar_t>
__global__ void pconv_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> __restrict__ grad_output,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> __restrict__ input,
    const torch::PackedTensorAccessor32<long,3,torch::RestrictPtrTraits> __restrict__ neighbor_inds,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ weights,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ additional_features,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> __restrict__ grad_input,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ grad_weights,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ grad_additional)
/* 
   grad_output: B x N x (C_mid * C_in), the gradient derived from above
   input: B x N x C_in tensor, B = batch size, N = number of points, C_in = number of channels, input features
   neighbor_inds: B x N x K tensor, K = neighborhood size, indices of the neighborhood of each point
   weights: B x N x K x C_mid, C_mid = number of middle channels
   additional_features: B x N x K x C_add, additional features that do not need gather
   grad_input: same shape as input, gradient of input
   grad_weights: same shape as weights, gradient of weights

   Forward is:
    sum_{i} input[b,n][neighbor_inds[i]][j] * weights[b,n,k,i]
*/

{
        int i,j,k, ii, kk,iter0;
        const int B = input.size(0);
        const int N = input.size(1);
        const int Nout = neighbor_inds.size(1);
        const int C_in = input.size(2);
        const int K = neighbor_inds.size(2);
        const int C_mid = weights.size(3);
	const int C_add = additional_features.size(3);
        const int increment = blockDim.x / C_mid;
        const int cur_mid = threadIdx.x / increment;
        if (cur_mid >= C_mid)
                return;
        // Supposedly blockIdx.x should go up to B * N
        for (iter0 = blockIdx.x; iter0< B * Nout; iter0+= gridDim.x)
        {
                // ii is the index on the batch dimension
                ii = iter0 / Nout;
                // i is the index on the point dimension
                i = iter0 % Nout;
                k = threadIdx.x % K;
                scalar_t weight_grad_temp = 0.0;
                scalar_t cur_compute;
                #pragma unroll
                for (kk=0;kk<C_in;kk++)
                {
                        long cur_channel = cur_mid * (C_in + C_add) + kk;
                        cur_compute = grad_output[ii][i][cur_channel] * weights[ii][i][k][cur_mid];
                        weight_grad_temp += grad_output[ii][i][cur_channel] * input[ii][neighbor_inds[ii][i][k]][kk];
                // It would be quite expensive to store this in shared memory so use this naive approach for now, using atomicAdd to avoid racing conditions
                        atomicAdd(&grad_input[ii][neighbor_inds[ii][i][k]][kk], cur_compute);
                }
		for (kk=0;kk<C_add;kk++)
		{
			long cur_channel = cur_mid * (C_in + C_add) + kk + C_in;
			cur_compute = grad_output[ii][i][cur_channel] * weights[ii][i][k][cur_mid];
			weight_grad_temp += grad_output[ii][i][cur_channel] * additional_features[ii][i][k][kk];
			grad_additional[ii][i][k][kk] = cur_compute;
		}
                grad_weights[ii][i][k][cur_mid] = weight_grad_temp;
	}
}

torch::Tensor pcf_cuda_forward(
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor guidance,
    torch::Tensor weights
)
{
	const int B = input.size(0);
	const int N = input.size(1);
	const int Nout = neighbor_inds.size(1);
	const int C_in = input.size(2);
	const int C_mid =  weights.size(3);
	const int numBlocks = B * Nout;
	const int numThreads = C_mid * C_in > 256 ? 256 : C_mid * C_in;
        auto output = torch::zeros({B,Nout,C_mid*C_in}, input.type());
	AT_DISPATCH_FLOATING_TYPES(output.type(), "pcf_cuda_forward_kernel", ([&] {
	pcf_cuda_forward_kernel<scalar_t><<<numBlocks, numThreads>>>(
		input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
		neighbor_inds.packed_accessor32<long,3,torch::RestrictPtrTraits>(),
		guidance.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
		weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
	}));
	return output;
}

std::vector<torch::Tensor> pcf_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor guidance,
    torch::Tensor weights)
// gradient should be same shape as input
// grad_output: B x N x (C_mid * C_in)
{
    const int B = input.size(0);
    const int N = input.size(1);
    const int Nout = neighbor_inds.size(1);
    const int C_in = input.size(2);
    const int C_mid =  weights.size(3);
    const int K = neighbor_inds.size(2);
    const int numBlocks = B * Nout;
    const int numThreads = C_mid * K;
    auto grad_input = torch::zeros_like(input);
    auto grad_guidance = torch::zeros_like(guidance);
    auto grad_weights = torch::zeros_like(weights);
    AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "pcf_cuda_backward_kernel", ([&] {
	pcf_cuda_backward_kernel<scalar_t><<<numBlocks,numThreads>>>(
        grad_output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        neighbor_inds.packed_accessor32<long,3,torch::RestrictPtrTraits>(),
        guidance.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        grad_input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        grad_guidance.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        grad_weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
	}));
    return {grad_input,grad_guidance,grad_weights};
}

torch::Tensor pconv_cuda_forward(
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features
)
{
        const int B = input.size(0);
        const int N = input.size(1);
        const int Nout = neighbor_inds.size(1);
        const int C_in = input.size(2);
	const int C_add = additional_features.size(3);
        const int C_mid =  weights.size(3);
        const int numBlocks = B * Nout;
        const int numThreads = C_mid * C_in > 256 ? 256 : C_mid * C_in;
        auto output = torch::zeros({B,Nout,C_mid*(C_in+C_add)}, input.type());
        AT_DISPATCH_FLOATING_TYPES(output.type(), "pconv_cuda_forward_kernel", ([&] {
        pconv_cuda_forward_kernel<scalar_t><<<numBlocks, numThreads>>>(
                input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                neighbor_inds.packed_accessor32<long,3,torch::RestrictPtrTraits>(),
                weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
		additional_features.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
        }));
        return output;
}

std::vector<torch::Tensor> pconv_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features)
// gradient should be same shape as input
// grad_output: B x N x (C_mid * C_in)
{
    const int B = input.size(0);
    const int N = input.size(1);
    const int Nout = neighbor_inds.size(1);
    const int C_in = input.size(2);
    const int C_mid =  weights.size(3);
    const int K = neighbor_inds.size(2);
    const int numBlocks = B * Nout;
    const int numThreads = C_mid * K;
    auto grad_input = torch::zeros_like(input);
    auto grad_weights = torch::zeros_like(weights);
    auto grad_additional = torch::zeros_like(additional_features);
    AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "pconv_cuda_backward_kernel", ([&] {
        pconv_cuda_backward_kernel<scalar_t><<<numBlocks,numThreads>>>(
        grad_output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        neighbor_inds.packed_accessor32<long,3,torch::RestrictPtrTraits>(),
        weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
	additional_features.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        grad_input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        grad_weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
	grad_additional.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
        }));
    return {grad_input,grad_weights,grad_additional};
}

