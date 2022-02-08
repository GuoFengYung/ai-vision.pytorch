from dataclasses import dataclass

import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import torch
from pycuda.tools import clear_context_caches
from torch import Tensor


class TensorRTInferer:

    @dataclass
    class Inference:
        final_pred_class: Tensor
        final_pred_prob: Tensor

    def __init__(self, path_to_tensorrt_engine: str, device_id: int):
        super().__init__()
        self._device_id = device_id
        self._cuda_device = torch.device('cuda', self._device_id)

        cuda.init()
        self._cuda_context = cuda.Device(self._device_id).retain_primary_context()
        self._cuda_context.push()

        trt_logger = trt.Logger()
        with open(path_to_tensorrt_engine, 'rb') as f, trt.Runtime(trt_logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            self._execution_context = engine.create_execution_context()

    def warmup(self, image: Tensor):
        self.infer(image, lower_prob_thresh=0.5, upper_prob_thresh=1.0)

    def infer(self, image: Tensor,
              lower_prob_thresh: float, upper_prob_thresh: float) -> Inference:
        if image.device != self._cuda_device:
            image = image.to(self._cuda_device)

        h_pred_prob = cuda.pagelocked_empty(trt.volume(self._execution_context.get_binding_shape(1)), dtype=np.float32)
        h_pred_class = cuda.pagelocked_empty(trt.volume(self._execution_context.get_binding_shape(2)), dtype=np.int32)

        d_pred_prob = cuda.mem_alloc(h_pred_prob.nbytes)
        d_pred_class = cuda.mem_alloc(h_pred_class.nbytes)

        self._execution_context.execute(bindings=[image.data_ptr(), int(d_pred_prob), int(d_pred_class)])

        cuda.memcpy_dtoh(h_pred_prob, d_pred_prob)
        cuda.memcpy_dtoh(h_pred_class, d_pred_class)

        pred_prob = torch.from_numpy(h_pred_prob)
        pred_class = torch.from_numpy(h_pred_class)

        if (pred_prob >= lower_prob_thresh) & (pred_prob <= upper_prob_thresh):
            final_pred_class = pred_class
        else:
            final_pred_class = torch.tensor(0).to(pred_class)

        final_pred_prob = pred_prob

        inference = TensorRTInferer.Inference(final_pred_class, final_pred_prob)
        return inference

    def dispose(self):
        self._cuda_context.pop()
        self._cuda_context = None
        clear_context_caches()
