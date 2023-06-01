import os
import shutil
import uuid
from copy import copy
from dataclasses import dataclass
from threading import Event
from typing import Optional, List
from unittest import TestCase

from aibox_vision.api.train import _train
from aibox_vision.lib.task.detection.algorithm import Algorithm
from aibox_vision.lib.task.detection.backbone import Backbone
from aibox_vision.lib.task.detection.config import Config
from aibox_vision.lib.task import Task


class TrainFinetuneTestCase(TestCase):

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

    def test_finetune_checkpoint(self):
        origin_argument = self.Argument(
            data_dir=os.path.join('examples', 'CatDog'),
            finetune_checkpoint=None,
            algorithm=Algorithm.Name.FASTER_RCNN,
            backbone=Backbone.Name.RESNET50,
            anchor_sizes=[128, 256, 512],
            backbone_pretrained=True,
            learning_rate=0.001,
            momentum=0.9,
            weight_decay=0.0005,
            num_epochs_to_finish=2
        )
        checkpoint_id = self._train(origin_argument)
        path_to_finetuning_checkpoint = os.path.join(self.path_to_outputs_dir, f'checkpoints-{checkpoint_id}',
                                                     f'epoch-{2:06d}', 'checkpoint.pth')

        self._finetune(origin_argument, path_to_finetuning_checkpoint)

        self._finetune_with_overwriting_data_dir(origin_argument, path_to_finetuning_checkpoint)
        self._finetune_with_overwriting_learning_rate(origin_argument, path_to_finetuning_checkpoint)
        self._finetune_with_overwriting_momentum(origin_argument, path_to_finetuning_checkpoint)
        self._finetune_with_overwriting_weight_decay(origin_argument, path_to_finetuning_checkpoint)

        self.assertRaises(RuntimeError, self._finetune_with_overwriting_algorithm, origin_argument, path_to_finetuning_checkpoint)
        self.assertRaises(RuntimeError, self._finetune_with_overwriting_backbone, origin_argument, path_to_finetuning_checkpoint)
        self.assertRaises(AssertionError, self._finetune_with_overwriting_anchor_sizes, origin_argument, path_to_finetuning_checkpoint)
        self.assertRaises(AssertionError, self._finetune_with_overwriting_backbone_pretrained, origin_argument, path_to_finetuning_checkpoint)

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

    def _finetune(self, origin_argument: Argument, path_to_finetuning_checkpoint):
        finetuning_argument = copy(origin_argument)
        finetuning_argument.finetune_checkpoint = path_to_finetuning_checkpoint
        finetuning_argument.num_epochs_to_finish = 5
        self._train(finetuning_argument)

    def _finetune_with_overwriting_data_dir(self, origin_argument: Argument, path_to_finetuning_checkpoint):
        finetuning_argument = copy(origin_argument)
        finetuning_argument.data_dir = os.path.join('examples', 'Person')
        finetuning_argument.finetune_checkpoint = path_to_finetuning_checkpoint
        finetuning_argument.num_epochs_to_finish = 5
        self._train(finetuning_argument)

    def _finetune_with_overwriting_learning_rate(self, origin_argument: Argument, path_to_finetuning_checkpoint):
        finetuning_argument = copy(origin_argument)
        finetuning_argument.finetune_checkpoint = path_to_finetuning_checkpoint
        finetuning_argument.learning_rate = 0.0001
        finetuning_argument.num_epochs_to_finish = 5
        self._train(finetuning_argument)

    def _finetune_with_overwriting_momentum(self, origin_argument: Argument, path_to_finetuning_checkpoint):
        finetuning_argument = copy(origin_argument)
        finetuning_argument.finetune_checkpoint = path_to_finetuning_checkpoint
        finetuning_argument.momentum = 0.8
        finetuning_argument.num_epochs_to_finish = 5
        self._train(finetuning_argument)

    def _finetune_with_overwriting_weight_decay(self, origin_argument: Argument, path_to_finetuning_checkpoint):
        finetuning_argument = copy(origin_argument)
        finetuning_argument.finetune_checkpoint = path_to_finetuning_checkpoint
        finetuning_argument.weight_decay = 0.0001
        finetuning_argument.num_epochs_to_finish = 5
        self._train(finetuning_argument)

    def _finetune_with_overwriting_algorithm(self, origin_argument: Argument, path_to_finetuning_checkpoint):
        finetuning_argument = copy(origin_argument)
        finetuning_argument.finetune_checkpoint = path_to_finetuning_checkpoint
        finetuning_argument.algorithm = Algorithm.Name.FPN
        finetuning_argument.num_epochs_to_finish = 5
        self._train(finetuning_argument)

    def _finetune_with_overwriting_backbone(self, origin_argument: Argument, path_to_finetuning_checkpoint):
        finetuning_argument = copy(origin_argument)
        finetuning_argument.finetune_checkpoint = path_to_finetuning_checkpoint
        finetuning_argument.backbone = Backbone.Name.RESNET18
        finetuning_argument.num_epochs_to_finish = 5
        self._train(finetuning_argument)

    def _finetune_with_overwriting_anchor_sizes(self, origin_argument: Argument, path_to_finetuning_checkpoint):
        finetuning_argument = copy(origin_argument)
        finetuning_argument.finetune_checkpoint = path_to_finetuning_checkpoint
        finetuning_argument.anchor_sizes = [8, 16, 32]
        finetuning_argument.num_epochs_to_finish = 5
        self._train(finetuning_argument)

    def _finetune_with_overwriting_backbone_pretrained(self, origin_argument: Argument, path_to_finetuning_checkpoint):
        finetuning_argument = copy(origin_argument)
        finetuning_argument.finetune_checkpoint = path_to_finetuning_checkpoint
        finetuning_argument.backbone_pretrained = False
        finetuning_argument.num_epochs_to_finish = 5
        self._train(finetuning_argument)
