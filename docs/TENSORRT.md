## TensorRT


### Setup

1. Install TensorRT

    ```
    $ pip install 'pycuda<2021.1'
    $ pip install nvidia-pyindex
    $ pip install nvidia-tensorrt==8.0.3.4
    ``` 

1. Verify installation

    ```
    >>> import tensorrt
    >>> import pycuda.driver
    ``` 

 
## Usage

* The following example shows how to initialize a classification TensorRT inferer and make an inference:

    ```
    >>> import torch
    >>> from aibox_vision.lib.task.classification.tensorrt_inferer import TensorRTInferer
  
    >>> path_to_checkpoint = '/path/to/checkpoint.pth'
    >>> inferer = TensorRTInferer(path_to_checkpoint, device_id=0, input_shape=(3, 224, 224))  # or use_fp16=True
  
    >>> image = torch.randn(3, 224, 224, dtype=torch.float32)  # CUDA tensor is available as well
    >>> output_data = inferer.infer(image, lower_prob_thresh=0.6, upper_prob_thresh=1.0)
    >>> print(output_data)
  
    >>> inferer.dispose()
    ```
  
    > `model.trt.engine` will be created under the same directory of `checkpoint.pth`
    > while first-time initializing a TensorRT inferer.
    > Note that the `model.trt.engine` file should not be migrated between different hardware devices
    > since it was optimized according to current environment. 
 