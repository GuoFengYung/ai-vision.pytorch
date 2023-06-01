import glob
import os
import random
from typing import List, Optional

import numpy as np
# noinspection PyUnresolvedReferences
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from PIL import Image

from aibox_vision.lib.preprocessor import Preprocessor


def load_data(path_to_images_dir: str, preprocessor: Preprocessor):
    image_paths = []
    for ext in ['.jpg', 'JPEG', 'png', 'PNG']:
        image_paths += list(sorted(glob.glob(os.path.join(path_to_images_dir, f'*{ext}'))))
    image_paths = random.choices(image_paths, k=1000)

    num_image_paths = len(image_paths)
    print(f'Found {num_image_paths} images')
    assert len(image_paths) > 0

    images = []
    for image_path in image_paths:
        image = Image.open(image_path)
        image, _ = preprocessor.process(image, is_train_or_eval=False)
        image = np.asarray(image)
        images.append(image)
    images = np.stack(images, axis=0)
    return images


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):

    def __init__(self, path_to_calibration_images_dir: str, batch_size: int, preprocessor: Preprocessor):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.data = load_data(path_to_calibration_images_dir, preprocessor)
        self.batch_size = batch_size
        self.current_index = 0
        self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names: List[str]) -> Optional[List]:
        if self.current_index + self.batch_size > self.data.shape[0]:
            return None

        current_batch = int(self.current_index / self.batch_size)
        if current_batch % 10 == 0:
            print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))

        batch = self.data[self.current_index:self.current_index + self.batch_size].ravel()
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size

        return [self.device_input]

    def read_calibration_cache(self):
        pass

    def write_calibration_cache(self, cache):
        pass
