import argparse
import os
import tempfile
import time
from ast import literal_eval
from typing import Tuple, Optional

import tensorrt as trt
import torch
from PIL import Image
from tqdm import tqdm

from calibrator import EntropyCalibrator
from aibox_vision.lib.preprocessor import Preprocessor
from aibox_vision.lib.task.classification.checkpoint import Checkpoint
from aibox_vision.lib.task.classification.model import Model
from aibox_vision.lib.task.classification.tensorrt_inferer import TensorRTInferer


def profile_pytorch_model(model: Model, device: torch.device, input_shape: Tuple[int, int, int]):
    time_checkpoint = time.time()
    for _ in tqdm(range(100)):
        image = torch.randn(*input_shape, dtype=torch.float, device=device)
        model.forward(image.unsqueeze(dim=0))
    elapsed = time.time() - time_checkpoint

    inference_fps = 100 / elapsed
    print(f'Inference FPS = {inference_fps:.2f}')


def profile_tensorrt_engine(tensorrt_inferer: TensorRTInferer, input_shape: Tuple[int, int, int]):
    time_checkpoint = time.time()
    for _ in tqdm(range(100)):
        image = torch.randn(*input_shape, dtype=torch.float)
        tensorrt_inferer.infer(image, lower_prob_thresh=0.5, upper_prob_thresh=1.0)
    elapsed = time.time() - time_checkpoint

    inference_fps = 100 / elapsed
    print(f'Inference FPS = {inference_fps:.2f}')


def compare_pytorch_model_and_tensorrt_engine(model: Model, tensorrt_inferer: TensorRTInferer, device: torch.device,
                                              path_to_images_dir: str, preprocessor: Preprocessor):
    assert os.path.isdir(path_to_images_dir)
    image_paths = [os.path.join(path_to_images_dir, filename) for filename in os.listdir(path_to_images_dir)]

    pytorch_pred_class_list = []
    tensorrt_pred_class_list = []
    for image_path in tqdm(image_paths):
        image = Image.open(image_path)
        image, _ = preprocessor.process(image, is_train_or_eval=False)
        pytorch_pred_class = model.forward(image.unsqueeze(dim=0).to(device))[1][0].item()
        tensorrt_pred_class = tensorrt_inferer.infer(image, lower_prob_thresh=0.5, upper_prob_thresh=1.0).final_pred_class.item()

        pytorch_pred_class_list.append(pytorch_pred_class)
        tensorrt_pred_class_list.append(tensorrt_pred_class)

    matches = []
    for pytorch_pred_class, tensorrt_pred_class in zip(pytorch_pred_class_list, tensorrt_pred_class_list):
        match = pytorch_pred_class == tensorrt_pred_class
        matches.append(match)

    print(f'Matched {sum(matches)} of total {len(matches)}, rate = {sum(matches) / len(matches):.2%}')


def build(path_to_checkpoint: str, quantization_precision: int, device_id: int, input_shape: Tuple[int, int, int],
          path_to_calibration_images_dir: Optional[str], path_to_comparison_images_dir: Optional[str]):
    assert os.path.isfile(path_to_checkpoint)

    if quantization_precision == 32:
        tensorrt_engine_filename = 'model.trt.engine'
    elif quantization_precision == 16:
        tensorrt_engine_filename = 'model.fp16.trt.engine'
    elif quantization_precision == 8:
        tensorrt_engine_filename = 'model.int8.trt.engine'
    else:
        raise ValueError(f'Unexpected quantization precision: {quantization_precision}')

    if quantization_precision == 8:
        assert path_to_calibration_images_dir is not None
        assert os.path.isdir(path_to_calibration_images_dir)

    path_to_tensorrt_engine = os.path.join(os.path.dirname(path_to_checkpoint), tensorrt_engine_filename)

    device = torch.device('cuda', device_id)
    checkpoint = Checkpoint.load(path_to_checkpoint, device)
    model: Model = checkpoint.model
    model.eval()
    model.algorithm.to_onnx_compatible()

    print('Profiling PyTorch model...')
    profile_pytorch_model(model, device, input_shape)

    with tempfile.NamedTemporaryFile() as f:
        example_inputs = torch.randn(input_shape, dtype=torch.float, device=device).unsqueeze(dim=0)
        torch.onnx.export(model, example_inputs, f,
                          input_names=['input'], output_names=['output'],
                          do_constant_folding=True)

        f.seek(0)
        onnx_model = f.read()

    trt_logger = trt.Logger()
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(trt_logger) as builder, \
            builder.create_network(explicit_batch) as network, \
            trt.OnnxParser(network, trt_logger) as parser:
        if not parser.parse(onnx_model):
            message = '\n'.join([str(parser.get_error(i)) for i in range(parser.num_errors)])
            raise RuntimeError(message)

        with builder.create_builder_config() as config:
            # NOTE: This determines the amount of memory available to the builder
            #       when building an optimized engine and should generally be set as high as possible
            config.max_workspace_size = 1 << 30  # 1 GB

            if quantization_precision == 16:
                config.set_flag(trt.BuilderFlag.FP16)
            elif quantization_precision == 8:
                config.set_flag(trt.BuilderFlag.INT8)

                calibrator = EntropyCalibrator(path_to_calibration_images_dir, batch_size=8, preprocessor=model.preprocessor)
                config.int8_calibrator = calibrator

            with builder.build_serialized_network(network, config) as engine, open(path_to_tensorrt_engine, 'wb') as f:
                f.write(engine)

    print(f'TensorRT engine has built at {path_to_tensorrt_engine}')

    tensorrt_inferer = TensorRTInferer(path_to_tensorrt_engine, device_id)

    print('Profiling TensorRT engine...')
    profile_tensorrt_engine(tensorrt_inferer, input_shape)

    if path_to_comparison_images_dir is not None:
        print('Comparing Pytorch model and TensorRT engine')
        compare_pytorch_model_and_tensorrt_engine(model, tensorrt_inferer, device,
                                                  path_to_comparison_images_dir, model.preprocessor)

    tensorrt_inferer.dispose()


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to checkpoint')
        parser.add_argument('--quantization_precision', type=int, choices=[32, 16, 8], required=True)
        parser.add_argument('--device_id', type=int, required=True)
        parser.add_argument('--input_shape', type=str, required=True)
        parser.add_argument('--calibration_images_dir', type=str)
        parser.add_argument('--comparison_images_dir', type=str)
        args = parser.parse_args()

        path_to_checkpoint = args.checkpoint
        quantization_precision = args.quantization_precision
        device_id = args.device_id
        input_shape = literal_eval(args.input_shape)
        path_to_calibration_images_dir = args.calibration_images_dir
        path_to_comparison_images_dir = args.comparison_images_dir

        build(path_to_checkpoint, quantization_precision, device_id, input_shape,
              path_to_calibration_images_dir, path_to_comparison_images_dir)

    main()
