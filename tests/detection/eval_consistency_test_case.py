import os
import shutil
import uuid
from dataclasses import dataclass
from threading import Event
from typing import Optional, List
from unittest import TestCase

from aibox_vision.api.eval import _eval
from aibox_vision.api.train import _train
from aibox_vision.lib.task.detection.algorithm import Algorithm
from aibox_vision.lib.task.detection.backbone import Backbone
from aibox_vision.lib.task.detection.config import Config
from aibox_vision.lib.task import Task


class EvalConsistencyTestCase(TestCase):

    @dataclass
    class Argument:
        data_dir: str
        finetune_checkpoint: Optional[str]
        algorithm: Algorithm.Name
        backbone: Backbone.Name
        anchor_sizes: List[int]
        backbone_pretrained: bool
        learning_rate: float
        momentum: float
        weight_decay: float
        num_epochs_to_finish: int

    def setUp(self):
        super().setUp()
        self.path_to_outputs_dir = os.path.join('outputs', 'tests', 'tmp')

    def tearDown(self):
        if os.path.isdir(self.path_to_outputs_dir):
            shutil.rmtree(self.path_to_outputs_dir)

    def test_consistency(self):
        argument = self.Argument(
            data_dir=os.path.join('examples', 'CatDog'),
            finetune_checkpoint=None,
            algorithm=Algorithm.Name.FASTER_RCNN,
            backbone=Backbone.Name.RESNET50,
            anchor_sizes=[128, 256, 512],
            backbone_pretrained=True,
            learning_rate=0.001,
            momentum=0.9,
            weight_decay=0.0005,
            num_epochs_to_finish=5
        )
        checkpoint_id = self._train(argument)
        path_to_checkpoint = os.path.join(self.path_to_outputs_dir, f'checkpoints-{checkpoint_id}',
                                          f'epoch-{5:06d}', 'checkpoint.pth')

        mean_aps = [self._eval(path_to_checkpoint, argument) for i in range(5)]
        self.assertEqual(len(set(mean_aps)),  1)

    def _train(self, argument: Argument) -> str:
        checkpoint_id = str(uuid.uuid4()).split('-')[0]
        path_to_checkpoints_dir = os.path.join(self.path_to_outputs_dir, f'checkpoints-{checkpoint_id}')
        os.makedirs(path_to_checkpoints_dir)

        config_dict = Config.parse_config_dict(
            task_name=Task.Name.DETECTION.value,
            path_to_checkpoints_dir=path_to_checkpoints_dir,
            path_to_data_dir=argument.data_dir,
            path_to_finetuning_checkpoint=argument.finetune_checkpoint,
            algorithm_name=argument.algorithm.value,
            backbone_name=argument.backbone.value,
            anchor_sizes=str(argument.anchor_sizes),
            backbone_pretrained=str(argument.backbone_pretrained),
            learning_rate=str(argument.learning_rate),
            momentum=str(argument.momentum),
            weight_decay=str(argument.weight_decay),
            num_epochs_to_validate=str(argument.num_epochs_to_finish),
            num_epochs_to_finish=str(argument.num_epochs_to_finish)
        )
        config = Config(**config_dict)

        terminator = Event()
        _train(config, terminator)

        return checkpoint_id

    def _eval(self, path_to_checkpoint: str, argument: Argument) -> float:
        return _eval(
            task_name=Task.Name.DETECTION,
            mode='val',
            path_to_checkpoint=path_to_checkpoint,
            path_to_data_dir=argument.data_dir,
            num_workers=0
        )
