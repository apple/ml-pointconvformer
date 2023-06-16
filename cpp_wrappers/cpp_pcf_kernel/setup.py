from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='PCFcuda',
    version='1.1',
    author='Fuxin Li',
    author_email='fli26@apple.com',
    description='PointConvFormer CUDA Kernel',
    ext_modules=[
        CUDAExtension('pcf_cuda', [
            'pcf_cuda.cpp',
            'pcf_cuda_kernel.cu',
        ],extra_compile_args={'nvcc': ['-L/usr/local/cuda/lib64 -lcudadevrt -lcudart']})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
