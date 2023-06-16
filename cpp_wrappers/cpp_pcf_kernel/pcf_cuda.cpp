//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2022-2023 Apple Inc. All Rights Reserved.
//
#include <torch/extension.h>

#include <vector>

// TODO: For now, not attempting to fuse the next downsampling linear layer because hand-written matmul 
//       cannot compare with optimized gemm kernels. May explore using CUTLASS for that at a later time.
torch::Tensor pcf_cuda_forward(
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor guidance,
    torch::Tensor weights
    );

// Need a version for PointConv too
torch::Tensor pconv_cuda_forward(
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features);

std::vector<torch::Tensor> pconv_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features);
    
std::vector<torch::Tensor> pcf_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor guidance,
    torch::Tensor weights);
    
#define CHECK_CUDA(x) {TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")}
#define CHECK_CONTIGUOUS(x) {TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")}
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor pconv_forward(
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features
)
{   
    CHECK_INPUT(input);
    CHECK_INPUT(neighbor_inds);
    CHECK_INPUT(weights);
    CHECK_INPUT(additional_features);

    return pconv_cuda_forward(input, neighbor_inds, weights, additional_features);
}

std::vector<torch::Tensor> pconv_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features)
{
    CHECK_INPUT(grad_output);
    CHECK_INPUT(neighbor_inds);
    CHECK_INPUT(weights);
    CHECK_INPUT(additional_features);
    return pconv_cuda_backward(grad_output,input, neighbor_inds, weights,additional_features);
}


torch::Tensor pcf_forward(
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor guidance,
    torch::Tensor weights
) 
{
    CHECK_INPUT(input);
    CHECK_INPUT(neighbor_inds);
    CHECK_INPUT(guidance);
    CHECK_INPUT(weights);
    return pcf_cuda_forward(input, neighbor_inds, guidance, weights);
}

std::vector<torch::Tensor> pcf_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor guidance,
    torch::Tensor weights)
{
    CHECK_INPUT(grad_output);
    CHECK_INPUT(neighbor_inds);
    CHECK_INPUT(guidance);
    CHECK_INPUT(weights);
    return pcf_cuda_backward(grad_output,input, neighbor_inds, guidance, weights);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pcf_forward", &pcf_forward, "PCF forward (CUDA)");
  m.def("pcf_backward", &pcf_backward, "PCF backward (CUDA)");
  m.def("pconv_forward", &pconv_forward, "PointConv forward (CUDA)");
  m.def("pconv_backward", &pconv_backward, "PointConv backward (CUDA)");
}
