export TensorRT_Root=/workspace/alex1_slam/tensorrt/
export TensorRT_Lib=$TensorRT_Root/lib
export TensorRT_Inc=$TensorRT_Root/include
export TensorRT_Bin=$TensorRT_Root/bin

export CUDA_HOME=/usr/local/cuda
export CUDA_Lib=/usr/local/cuda/lib64
export CUDA_Inc=/usr/local/cuda/include
export CUDA_Bin=/usr/local/cuda/bin


export CUDNN_Lib=/usr/local/cuda/lib64
export SPCONV_CUDA_VERSION=11.4

export PATH=$TensorRT_Bin:$CUDA_Bin:$PATH
export LD_LIBRARY_PATH=$TensorRT_Lib:$CUDA_Lib:$CUDNN_Lib:$BuildDirectory:$LD_LIBRARY_PATH
export PYTHONPATH=$BuildDirectory:$PYTHONPATH

export CUDASM=86 #only for 4070, fo A100, CUDASM=80
