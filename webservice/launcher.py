import base64
import glob
import json
import multiprocessing
import os
import platform
import shutil
import tempfile
import time
import uuid
from ast import literal_eval
from concurrent.futures.process import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime
from distutils.util import strtobool
from io import BytesIO
from multiprocessing import Event
from multiprocessing import Process
from typing import Dict, Tuple, Optional

import nvidia_smi
import seaborn as sns
import torch
from PIL import Image, ImageDraw
from flask import Flask, request, Response
from flask_wtf import CSRFProtect
from thop import profile, clever_format
from torchvision.transforms.functional import to_pil_image

import aibox_vision
from aibox_vision.api.infer import _infer
from aibox_vision.api.train import _train
from aibox_vision.api.upgrade import _upgrade
from aibox_vision.lib.augmenter import Augmenter
from aibox_vision.lib.checkpoint import Checkpoint
from aibox_vision.lib.config import Config
from aibox_vision.lib.db import DB
from aibox_vision.lib.preprocessor import Preprocessor
from aibox_vision.lib.task import Task
from aibox_vision.lib.task.instance_segmentation.palette import Palette


class VerificationError(Exception):
    def __init__(self, response: Response):
        self.response = response


version = aibox_vision.__version__
path_to_outputs_dir = './outputs'
summary_db_filename = 'summary.db'

job_id_to_process_and_terminator_dict: Dict[str, Tuple[Process, Event]] = {}
job_id_and_epoch_and_target_device_id_to_deployed_checkpoint_dict: Dict[Tuple[str, int, Optional[int]], Checkpoint] = {}

multiprocessing.set_start_method('spawn', force=True)  # set `force` for `ProcessPoolExecutor`
app = Flask(__name__)
CSRFProtect(app)
app.config['WTF_CSRF_ENABLED'] = False


@app.route(f'/api', methods=['GET'])
def hello() -> str:
    return f'Welcome to Mirle Vision API {version}'


@app.route(f'/api/version', methods=['GET'])
def query_version() -> str:
    return version


@app.route(f'/api/{version}/environment', methods=['GET'])
def query_environment() -> Response:
    devices = []
    if torch.cuda.is_available():
        for device_id in range(torch.cuda.device_count()):
            device_property = torch.cuda.get_device_properties(device_id)
            devices.append({
                'id': device_id,
                'name': device_property.name,
                'memory_in_mb': device_property.total_memory // 1024 // 1024
            })

    response = {
        'result': 'OK',
        'environment': {
            'hardware': {
                'devices': devices
            },
            'software': {
                'python_version': platform.python_version(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'n/a',
                'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() and torch.backends.cudnn.enabled else 'n/a'
            }
        }
    }
    return Response(response=json.dumps(response), status=200, mimetype='application/json')


@app.route(f'/api/{version}/devices', methods=['GET'])
def query_devices() -> Response:
    devices = []
    if torch.cuda.is_available():
        nvidia_smi.nvmlInit()
        for device_id in range(torch.cuda.device_count()):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device_id)
            memory_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            devices.append({
                'name': str(nvidia_smi.nvmlDeviceGetName(handle), encoding='utf-8'),
                'utilization_in_percent': nvidia_smi.nvmlDeviceGetUtilizationRates(handle).gpu,
                'memory_total_in_mb': memory_info.total // 1024 // 1024,
                'memory_used_in_mb': memory_info.used // 1024 // 1024,
                'memory_free_in_mb': memory_info.free // 1024 // 1024,
                'power_limit_in_watt': nvidia_smi.nvmlDeviceGetPowerManagementLimit(handle) // 1000,
                'power_usage_in_watt': nvidia_smi.nvmlDeviceGetPowerUsage(handle) // 1000
            })

    response = {
        'result': 'OK',
        'devices': devices
    }
    return Response(response=json.dumps(response), status=200, mimetype='application/json')


@app.route(f'/api/{version}/train', methods=['POST'])
def start_train() -> Response:
    job_id = '{:s}-{:s}'.format(time.strftime('%Y%m%d%H%M%S'), str(uuid.uuid4()))
    path_to_checkpoints_dir = os.path.join(path_to_outputs_dir, job_id)
    os.makedirs(path_to_checkpoints_dir)

    task_name = Task.Name(request.values['task'])
    path_to_data_dir = request.values['data_dir']
    path_to_extra_data_dirs = None

    path_to_resuming_checkpoint = None
    if 'resume_job_id_and_epoch' in request.values:
        origin_job_id, epoch = literal_eval(request.values['resume_job_id_and_epoch'])
        path_to_resuming_checkpoint = os.path.join(path_to_outputs_dir, origin_job_id, f'epoch-{epoch:06d}', 'checkpoint.pth')

    path_to_finetuning_checkpoint = None
    if 'finetune_job_id_and_epoch' in request.values:
        origin_job_id, epoch = literal_eval(request.values['finetune_job_id_and_epoch'])
        path_to_finetuning_checkpoint = os.path.join(path_to_outputs_dir, origin_job_id, f'epoch-{epoch:06d}', 'checkpoint.pth')

    path_to_loading_checkpoint = None
    if 'load_job_id_and_epoch' in request.values:
        origin_job_id, epoch = literal_eval(request.values['load_job_id_and_epoch'])
        path_to_loading_checkpoint = os.path.join(path_to_outputs_dir, origin_job_id, f'epoch-{epoch:06d}', 'checkpoint.pth')

    common_args = (
        str(task_name.value),
        path_to_checkpoints_dir,
        path_to_data_dir,
        path_to_extra_data_dirs,
        path_to_resuming_checkpoint,
        path_to_finetuning_checkpoint,
        path_to_loading_checkpoint,
        request.values['num_workers'] if 'num_workers' in request.values else None,
        request.values['visible_devices'] if 'visible_devices' in request.values else None,
        request.values['needs_freeze_bn'] if 'needs_freeze_bn' in request.values else None,
        request.values['image_resized_width'] if 'image_resized_width' in request.values else None,
        request.values['image_resized_height'] if 'image_resized_height' in request.values else None,
        request.values['image_min_side'] if 'image_min_side' in request.values else None,
        request.values['image_max_side'] if 'image_max_side' in request.values else None,
        request.values['image_side_divisor'] if 'image_side_divisor' in request.values else None,
        request.values['aug_strategy'] if 'aug_strategy' in request.values else None,
        request.values['aug_hflip_prob'] if 'aug_hflip_prob' in request.values else None,
        request.values['aug_vflip_prob'] if 'aug_vflip_prob' in request.values else None,
        request.values['aug_rotate90_prob'] if 'aug_rotate90_prob' in request.values else None,
        request.values['aug_crop_prob_and_min_max'] if 'aug_crop_prob_and_min_max' in request.values else None,
        request.values['aug_zoom_prob_and_min_max'] if 'aug_zoom_prob_and_min_max' in request.values else None,
        request.values['aug_scale_prob_and_min_max'] if 'aug_scale_prob_and_min_max' in request.values else None,
        request.values['aug_translate_prob_and_min_max'] if 'aug_translate_prob_and_min_max' in request.values else None,
        request.values['aug_rotate_prob_and_min_max'] if 'aug_rotate_prob_and_min_max' in request.values else None,
        request.values['aug_shear_prob_and_min_max'] if 'aug_shear_prob_and_min_max' in request.values else None,
        request.values['aug_blur_prob_and_min_max'] if 'aug_blur_prob_and_min_max' in request.values else None,
        request.values['aug_sharpen_prob_and_min_max'] if 'aug_sharpen_prob_and_min_max' in request.values else None,
        request.values['aug_color_prob_and_min_max'] if 'aug_color_prob_and_min_max' in request.values else None,
        request.values['aug_brightness_prob_and_min_max'] if 'aug_brightness_prob_and_min_max' in request.values else None,
        request.values['aug_grayscale_prob_and_min_max'] if 'aug_grayscale_prob_and_min_max' in request.values else None,
        request.values['aug_contrast_prob_and_min_max'] if 'aug_contrast_prob_and_min_max' in request.values else None,
        request.values['aug_noise_prob_and_min_max'] if 'aug_noise_prob_and_min_max' in request.values else None,
        request.values['aug_resized_crop_prob_and_width_height'] if 'aug_resized_crop_prob_and_width_height' in request.values else None,
        request.values['batch_size'] if 'batch_size' in request.values else None,
        request.values['learning_rate'] if 'learning_rate' in request.values else None,
        request.values['momentum'] if 'momentum' in request.values else None,
        request.values['weight_decay'] if 'weight_decay' in request.values else None,
        request.values['clip_grad_base_and_max'] if 'clip_grad_base_and_max' in request.values else None,
        request.values['step_lr_sizes'] if 'step_lr_sizes' in request.values else None,
        request.values['step_lr_gamma'] if 'step_lr_gamma' in request.values else None,
        request.values['warm_up_factor'] if 'warm_up_factor' in request.values else None,
        request.values['warm_up_num_iters'] if 'warm_up_num_iters' in request.values else None,
        request.values['num_batches_to_display'] if 'num_batches_to_display' in request.values else None,
        request.values['num_epochs_to_validate'] if 'num_epochs_to_validate' in request.values else None,
        request.values['num_epochs_to_finish'] if 'num_epochs_to_finish' in request.values else None,
        request.values['max_num_checkpoint'] if 'max_num_checkpoint' in request.values else None
    )

    if task_name == Task.Name.CLASSIFICATION:
        from aibox_vision.lib.task.classification.config import Config as ClassificationConfig
        config_dict = ClassificationConfig.parse_config_dict(
            *common_args,
            request.values['algorithm'] if 'algorithm' in request.values else None,
            request.values['pretrained'] if 'pretrained' in request.values else None,
            request.values['num_frozen_levels'] if 'num_frozen_levels' in request.values else None,
            request.values['eval_center_crop_ratio'] if 'eval_center_crop_ratio' in request.values else None
        )
        config = ClassificationConfig(**config_dict)
    elif task_name == Task.Name.DETECTION:
        from aibox_vision.lib.task.detection.config import Config as DetectionConfig
        config_dict = DetectionConfig.parse_config_dict(
            *common_args,
            request.values['algorithm'] if 'algorithm' in request.values else None,
            request.values['backbone'] if 'backbone' in request.values else None,
            request.values['anchor_ratios'] if 'anchor_ratios' in request.values else None,
            request.values['anchor_sizes'] if 'anchor_sizes' in request.values else None,
            request.values['backbone_pretrained'] if 'backbone_pretrained' in request.values else None,
            request.values['backbone_num_frozen_levels'] if 'backbone_num_frozen_levels' in request.values else None,
            request.values['train_rpn_pre_nms_top_n'] if 'train_rpn_pre_nms_top_n' in request.values else None,
            request.values['train_rpn_post_nms_top_n'] if 'train_rpn_post_nms_top_n' in request.values else None,
            request.values['eval_rpn_pre_nms_top_n'] if 'eval_rpn_pre_nms_top_n' in request.values else None,
            request.values['eval_rpn_post_nms_top_n'] if 'eval_rpn_post_nms_top_n' in request.values else None,
            request.values['num_anchor_samples_per_batch'] if 'num_anchor_samples_per_batch' in request.values else None,
            request.values['num_proposal_samples_per_batch'] if 'num_proposal_samples_per_batch' in request.values else None,
            request.values['num_detections_per_image'] if 'num_detections_per_image' in request.values else None,
            request.values['anchor_smooth_l1_loss_beta'] if 'anchor_smooth_l1_loss_beta' in request.values else None,
            request.values['proposal_smooth_l1_loss_beta'] if 'proposal_smooth_l1_loss_beta' in request.values else None,
            request.values['proposal_nms_threshold'] if 'proposal_nms_threshold' in request.values else None,
            request.values['detection_nms_threshold'] if 'detection_nms_threshold' in request.values else None
        )
        config = DetectionConfig(**config_dict)
    elif task_name == Task.Name.INSTANCE_SEGMENTATION:
        from aibox_vision.lib.task.instance_segmentation.config import Config as InstanceSegmentationConfig
        config_dict = InstanceSegmentationConfig.parse_config_dict(
            *common_args,
            request.values['algorithm'] if 'algorithm' in request.values else None
        )
        config = InstanceSegmentationConfig(**config_dict)
    else:
        raise ValueError

    terminator = Event()
    process = Process(target=_train,
                      args=(config, terminator))
    process.start()
    job_id_to_process_and_terminator_dict[job_id] = (process, terminator)

    response = {
        'result': 'OK',
        'job_id': job_id
    }
    return Response(response=json.dumps(response), status=200, mimetype='application/json')


@app.route(f'/api/{version}/train/<job_id>', methods=['PATCH'])
def stop_train(job_id: str) -> Response:
    try:
        _verify_job_is_tracking(job_id)

        process, terminator = job_id_to_process_and_terminator_dict[job_id]
        terminator.set()

        response = {
            'result': 'OK'
        }
        return Response(response=json.dumps(response), status=200, mimetype='application/json')
    except VerificationError as e:
        return e.response


@app.route(f'/api/{version}/train/<job_id>', methods=['PUT'])
def restore_train(job_id: str) -> Response:
    try:
        _verify_job_exists(job_id)
        _verify_job_is_stopped(job_id)

        if job_id in job_id_to_process_and_terminator_dict:
            del job_id_to_process_and_terminator_dict[job_id]

        path_to_config_pickle_file = os.path.join(path_to_outputs_dir, job_id, 'config.pkl')
        config = Config.deserialize(path_to_config_pickle_file)

        path_to_db = os.path.join(path_to_outputs_dir, job_id, summary_db_filename)
        db = DB(path_to_db)
        checkpoint = db.select_checkpoint_table_latest()

        if checkpoint is not None:
            latest_epoch = checkpoint.epoch
            path_to_resuming_checkpoint = os.path.join(config.path_to_checkpoints_dir,
                                                       f'epoch-{latest_epoch:06d}', 'checkpoint.pth')
            config.path_to_resuming_checkpoint = path_to_resuming_checkpoint

        terminator = Event()
        process = Process(target=_train,
                          args=(config, terminator))
        process.start()
        job_id_to_process_and_terminator_dict[job_id] = (process, terminator)

        response = {
            'result': 'OK'
        }
        return Response(response=json.dumps(response), status=200, mimetype='application/json')
    except VerificationError as e:
        return e.response


@app.route(f'/api/{version}/train/<job_id>', methods=['DELETE'])
def clear_train(job_id: str) -> Response:
    try:
        _verify_job_exists(job_id)
        _verify_job_is_stopped(job_id)

        if job_id in job_id_to_process_and_terminator_dict:
            del [job_id]

        path_to_checkpoints_dir = os.path.join(path_to_outputs_dir, job_id)
        shutil.rmtree(path_to_checkpoints_dir)

        response = {
            'result': 'OK'
        }
        return Response(response=json.dumps(response), status=200, mimetype='application/json')
    except VerificationError as e:
        return e.response


@app.route(f'/api/{version}/train/<job_id>', methods=['GET'])
def monitor_train(job_id: str) -> Response:
    try:
        _verify_job_exists(job_id)

        path_to_db = os.path.join(path_to_outputs_dir, job_id, summary_db_filename)
        db = DB(path_to_db)
        log = db.select_log_table_latest()

        response = {
            'result': 'OK',
            'status': log.status.value,
            'datetime': datetime.fromtimestamp(log.datetime).strftime('%Y-%m-%d %H:%M:%S'),
            'epoch': log.epoch,
            'total_epoch': log.total_epoch,
            'batch': log.batch,
            'total_batch': log.total_batch,
            'avg_loss': log.avg_loss,
            'learning_rate': log.learning_rate,
            'samples_per_sec': log.samples_per_sec,
            'eta_hr': log.eta_hrs,
            'exception': log.exception if log.exception is None else asdict(log.exception)
        }
        return Response(response=json.dumps(response), status=200, mimetype='application/json')
    except VerificationError as e:
        return e.response


@app.route(f'/api/{version}/train/<job_id>/hyper-parameters', methods=['GET'])
def query_hyper_parameters(job_id: str) -> Response:
    try:
        _verify_job_exists(job_id)

        path_to_config_pkl = os.path.join(path_to_outputs_dir, job_id, 'config.pkl')
        config = Config.deserialize(path_to_config_pkl)

        hyper_parameter_dict = {
            'image_resized_width': config.image_resized_width,
            'image_resized_height': config.image_resized_height,
            'image_min_side': config.image_min_side,
            'image_max_side': config.image_max_side,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'momentum': config.momentum,
            'weight_decay': config.weight_decay,
            'clip_grad_base_and_max': config.clip_grad_base_and_max,
            'step_lr_sizes': config.step_lr_sizes,
            'step_lr_gamma': config.step_lr_gamma,
            'num_epochs_to_validate': config.num_epochs_to_validate,
            'num_epochs_to_finish': config.num_epochs_to_finish
        }

        if config.task_name == Task.Name.CLASSIFICATION:
            from aibox_vision.lib.task.classification.config import Config as ClassificationConfig
            config: ClassificationConfig
            hyper_parameter_dict.update({
                'pretrained': config.pretrained,
                'num_frozen_levels': config.num_frozen_levels,
                'eval_center_crop_ratio': config.eval_center_crop_ratio
            })
        elif config.task_name == Task.Name.DETECTION:
            from aibox_vision.lib.task.detection.config import Config as DetectionConfig
            config: DetectionConfig
            hyper_parameter_dict.update({
                'anchor_sizes': config.anchor_sizes,
                'backbone_pretrained': config.backbone_pretrained,
                'backbone_num_frozen_levels': config.backbone_num_frozen_levels,
                'proposal_nms_threshold': config.proposal_nms_threshold,
                'detection_nms_threshold': config.detection_nms_threshold
            })
        elif config.task_name == Task.Name.INSTANCE_SEGMENTATION:
            from aibox_vision.lib.task.instance_segmentation.config import Config as InstanceSegmentationConfig
            config: InstanceSegmentationConfig
            hyper_parameter_dict.update({
            })
        else:
            raise ValueError

        response = {
            'result': 'OK',
            'hyper-parameters': hyper_parameter_dict
        }

        return Response(response=json.dumps(response), status=200, mimetype='application/json')
    except VerificationError as e:
        return e.response


@app.route(f'/api/{version}/train/<job_id>/losses', methods=['GET'])
def query_losses(job_id: str) -> Response:
    try:
        _verify_job_exists(job_id)

        path_to_db = os.path.join(path_to_outputs_dir, job_id, summary_db_filename)
        db = DB(path_to_db)
        losses = [log.avg_loss for log in db.select_log_table()]

        response = {
            'result': 'OK',
            'losses': losses
        }
        return Response(response=json.dumps(response), status=200, mimetype='application/json')
    except VerificationError as e:
        return e.response


@app.route(f'/api/{version}/train/<job_id>/checkpoints/epoch/<int:epoch>', methods=['GET'])
def retrieve_checkpoint(job_id: str, epoch: int) -> Response:
    try:
        _verify_job_exists(job_id)

        path_to_db = os.path.join(path_to_outputs_dir, job_id, summary_db_filename)
        db = DB(path_to_db)
        checkpoint = db.select_checkpoint_table_for_epoch(epoch)
        path_to_epoch_dir = os.path.join(path_to_outputs_dir, job_id, f'epoch-{epoch:06d}')

        if checkpoint.task_name == Task.Name.CLASSIFICATION:
            path_to_image = os.path.join(path_to_epoch_dir, 'metric-auc.png')
        elif checkpoint.task_name == Task.Name.DETECTION:
            from aibox_vision.lib.task.detection.evaluator import Evaluator
            quality = Evaluator.Evaluation.Quality.STANDARD
            size = Evaluator.Evaluation.Size.ALL
            path_to_plot_dir = os.path.join(path_to_epoch_dir,
                                            'quality-{:s}'.format(quality.value),
                                            'size-{:s}'.format(size.value))
            path_to_image = os.path.join(path_to_plot_dir, 'metric-ap.png')
        elif checkpoint.task_name == Task.Name.INSTANCE_SEGMENTATION:
            from aibox_vision.lib.task.instance_segmentation.evaluator import Evaluator
            quality = Evaluator.Evaluation.Quality.STANDARD
            size = Evaluator.Evaluation.Size.ALL
            path_to_plot_dir = os.path.join(path_to_epoch_dir,
                                            'quality-{:s}'.format(quality.value),
                                            'size-{:s}'.format(size.value))
            path_to_image = os.path.join(path_to_plot_dir, 'metric-ap.png')
        else:
            raise ValueError

        image = Image.open(path_to_image).convert('RGB')
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        base64_metric_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')

        response = {
            'result': 'OK',
            'checkpoint': {
                'epoch': checkpoint.epoch,
                'avg_loss': checkpoint.avg_loss,
                'metrics': asdict(checkpoint.metrics),
                'base64_metric_plot': base64_metric_plot,
                'is_best': checkpoint.is_best,
                'is_available': checkpoint.is_available
            }
        }
        return Response(response=json.dumps(response), status=200, mimetype='application/json')
    except VerificationError as e:
        return e.response


@app.route(f'/api/{version}/train/<job_id>/checkpoints', methods=['GET'])
def retrieve_checkpoints(job_id: str) -> Response:
    try:
        _verify_job_exists(job_id)

        path_to_db = os.path.join(path_to_outputs_dir, job_id, summary_db_filename)
        db = DB(path_to_db)
        checkpoints = []

        for checkpoint in db.select_checkpoint_table():
            if checkpoint.task_name == Task.Name.CLASSIFICATION:
                checkpoints.append({
                    'epoch': checkpoint.epoch,
                    'avg_loss': checkpoint.avg_loss,
                    'accuracy': checkpoint.metrics.overall['accuracy'],
                    'is_best': checkpoint.is_best,
                    'is_available': checkpoint.is_available
                })
            elif checkpoint.task_name == Task.Name.DETECTION:
                checkpoints.append({
                    'epoch': checkpoint.epoch,
                    'avg_loss': checkpoint.avg_loss,
                    'mean_ap': checkpoint.metrics.specific['aps'][0],
                    'is_best': checkpoint.is_best,
                    'is_available': checkpoint.is_available
                })
            elif checkpoint.task_name == Task.Name.INSTANCE_SEGMENTATION:
                checkpoints.append({
                    'epoch': checkpoint.epoch,
                    'avg_loss': checkpoint.avg_loss,
                    'mean_ap': checkpoint.metrics.specific['aps'][0],
                    'is_best': checkpoint.is_best,
                    'is_available': checkpoint.is_available
                })
            else:
                raise ValueError

        response = {
            'result': 'OK',
            'checkpoints': checkpoints
        }
        return Response(response=json.dumps(response), status=200, mimetype='application/json')
    except VerificationError as e:
        return e.response


@app.route(f'/api/{version}/train/<job_id>/plots/model-graph', methods=['GET'])
def obtain_model_graph_plot(job_id: str) -> Response:
    try:
        _verify_job_exists(job_id)

        path_to_image = os.path.join(path_to_outputs_dir, job_id, 'model-graph.png')
        image = Image.open(path_to_image).convert('RGB')

        buffer = BytesIO()
        image.save(buffer, format='PNG')
        base64_model_graph_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')

        response = {
            'result': 'OK',
            'base64_model_graph_plot': base64_model_graph_plot,
        }
        return Response(response=json.dumps(response), status=200, mimetype='application/json')
    except VerificationError as e:
        return e.response


@app.route(f'/api/{version}/train/<job_id>/plots/loss', methods=['GET'])
def obtain_loss_plot(job_id: str) -> Response:
    try:
        _verify_job_exists(job_id)

        path_to_image = os.path.join(path_to_outputs_dir, job_id, 'loss.png')
        image = Image.open(path_to_image).convert('RGB')

        buffer = BytesIO()
        image.save(buffer, format='PNG')
        base64_loss_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')

        response = {
            'result': 'OK',
            'base64_loss_plot': base64_loss_plot,
        }
        return Response(response=json.dumps(response), status=200, mimetype='application/json')
    except VerificationError as e:
        return e.response


@app.route(f'/api/{version}/train/<job_id>/plots/confusion-matrix/epoch/<int:epoch>', methods=['GET'])
def obtain_confusion_matrix_plot(job_id: str, epoch: int) -> Response:
    try:
        _verify_job_exists(job_id)

        path_to_db = os.path.join(path_to_outputs_dir, job_id, summary_db_filename)
        db = DB(path_to_db)
        checkpoint = db.select_checkpoint_table_for_epoch(epoch)

        if checkpoint.task_name == Task.Name.CLASSIFICATION:
            path_to_plot_dir = os.path.join(path_to_outputs_dir, job_id, f'epoch-{epoch:06d}')
        elif checkpoint.task_name == Task.Name.DETECTION:
            raise ValueError('API `obtain_confusion_matrix_plot` is not supported for detection task yet.')
        elif checkpoint.task_name == Task.Name.INSTANCE_SEGMENTATION:
            path_to_plot_dir = os.path.join(path_to_outputs_dir, job_id, f'epoch-{epoch:06d}')
        else:
            raise ValueError

        path_to_image = os.path.join(path_to_plot_dir, 'confusion-matrix.png')
        image = Image.open(path_to_image).convert('RGB')

        buffer = BytesIO()
        image.save(buffer, format='PNG')
        base64_confusion_matrix_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')

        response = {
            'result': 'OK',
            'base64_confusion_matrix_plot': base64_confusion_matrix_plot,
        }
        return Response(response=json.dumps(response), status=200, mimetype='application/json')
    except VerificationError as e:
        return e.response


@app.route(f'/api/{version}/train/<job_id>/plots/threshold/epoch/<int:epoch>/class/<int:cls>', methods=['GET'])
def obtain_threshold_plot(job_id: str, epoch: int, cls: int) -> Response:
    try:
        _verify_job_exists(job_id)

        path_to_db = os.path.join(path_to_outputs_dir, job_id, summary_db_filename)
        db = DB(path_to_db)
        checkpoint = db.select_checkpoint_table_for_epoch(epoch)

        if checkpoint.task_name == Task.Name.CLASSIFICATION:
            raise ValueError('API `obtain_threshold_plot` is not supported for classification task yet.')
        elif checkpoint.task_name == Task.Name.DETECTION:
            from aibox_vision.lib.task.detection.evaluator import Evaluator
            quality = Evaluator.Evaluation.Quality.STANDARD
            size = Evaluator.Evaluation.Size.ALL
            path_to_plot_dir = os.path.join(path_to_outputs_dir, job_id, f'epoch-{epoch:06d}',
                                            'quality-{:s}'.format(quality.value),
                                            'size-{:s}'.format(size.value))
        elif checkpoint.task_name == Task.Name.INSTANCE_SEGMENTATION:
            raise ValueError('API `obtain_threshold_plot` is not supported for instance segmentation task yet.')
        else:
            raise ValueError

        path_to_image = os.path.join(path_to_plot_dir, 'thresh-{:s}.png'.format(str(cls) if cls > 0 else 'mean'))
        image = Image.open(path_to_image).convert('RGB')

        buffer = BytesIO()
        image.save(buffer, format='PNG')
        base64_threshold_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')

        response = {
            'result': 'OK',
            'base64_threshold_plot': base64_threshold_plot,
        }
        return Response(response=json.dumps(response), status=200, mimetype='application/json')
    except VerificationError as e:
        return e.response


@app.route(f'/api/{version}/infer/<job_id>', methods=['POST'])
def start_infer(job_id: str) -> Response:
    try:
        _verify_job_exists(job_id)

        task_name = Task.Name(request.values['task'])
        epoch = int(request.values['epoch'])
        lower_prob_thresh = float(request.values['lower_prob_thresh']) if 'lower_prob_thresh' in request.values else 0.7
        upper_prob_thresh = float(request.values['upper_prob_thresh']) if 'upper_prob_thresh' in request.values else 1.0
        device_ids = literal_eval(request.values['device_ids']) if 'device_ids' in request.values else None
        returns_images = bool(strtobool(request.values['returns_images'])) if 'returns_images' in request.values else True
        path_to_image = request.values['path_to_image'] if 'path_to_image' in request.values else None
        base64_image = request.values['base64_image'] if 'base64_image' in request.values else None

        assert path_to_image is not None or base64_image is not None
        assert not (path_to_image is not None and base64_image is not None)

        if path_to_image is None and base64_image is not None:
            path_to_image = os.path.join(tempfile.gettempdir(), '{:s}.png'.format(str(uuid.uuid4()).split('-')[0]))
            Image.open(BytesIO(base64.b64decode(base64_image))).save(path_to_image)

        path_to_checkpoint_or_checkpoint = None

        # NOTE: Check deployed checkpoint only if exactly one device ID was specified
        if device_ids is not None and len(device_ids) <= 1:
            target_device_id = -1 if len(device_ids) == 0 else device_ids[0]
            if (job_id, epoch, target_device_id) in job_id_and_epoch_and_target_device_id_to_deployed_checkpoint_dict:
                checkpoint = job_id_and_epoch_and_target_device_id_to_deployed_checkpoint_dict[(job_id, epoch, target_device_id)]
                path_to_checkpoint_or_checkpoint = checkpoint

        if path_to_checkpoint_or_checkpoint is None:
            path_to_checkpoint = os.path.join(path_to_outputs_dir, job_id, f'epoch-{epoch:06d}', 'checkpoint.pth')
            path_to_checkpoint_or_checkpoint = path_to_checkpoint

        if task_name == Task.Name.CLASSIFICATION:
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    _infer,
                    task_name,
                    path_to_checkpoint_or_checkpoint,
                    lower_prob_thresh, upper_prob_thresh,
                    device_ids,
                    path_to_image_list=[path_to_image],
                    path_to_results_dir=None
                )

            path_to_image_to_base64_images_dict, \
                path_to_image_to_final_pred_category_dict, \
                path_to_image_to_final_pred_prob_dict = future.result()
            response = {
                'result': 'OK',
                'base64_images': path_to_image_to_base64_images_dict[path_to_image],
                'classification': {
                    'category': path_to_image_to_final_pred_category_dict[path_to_image],
                    'prob': path_to_image_to_final_pred_prob_dict[path_to_image]
                }
            }
        elif task_name == Task.Name.DETECTION:
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    _infer,
                    task_name,
                    path_to_checkpoint_or_checkpoint,
                    lower_prob_thresh, upper_prob_thresh,
                    device_ids,
                    path_to_image_list=[path_to_image],
                    path_to_results_dir=None
                )

            path_to_image_to_base64_images_dict, \
                path_to_image_to_final_detection_bboxes_dict, \
                path_to_image_to_final_detection_categories_dict, \
                path_to_image_to_final_detection_probs_dict = future.result()
            response = {
                'result': 'OK',
                'base64_images': path_to_image_to_base64_images_dict[path_to_image],
                'detections': [{
                    'bbox': bbox,
                    'category': category,
                    'prob': prob
                } for bbox, category, prob in zip(path_to_image_to_final_detection_bboxes_dict[path_to_image],
                                                  path_to_image_to_final_detection_categories_dict[path_to_image],
                                                  path_to_image_to_final_detection_probs_dict[path_to_image])]
            }
        elif task_name == Task.Name.INSTANCE_SEGMENTATION:
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    _infer,
                    task_name,
                    path_to_checkpoint_or_checkpoint,
                    lower_prob_thresh, upper_prob_thresh,
                    device_ids,
                    path_to_image_list=[path_to_image],
                    path_to_results_dir=None
                )

            path_to_image_to_base64_images_dict, \
                path_to_image_to_final_detection_bboxes_dict, \
                path_to_image_to_final_detection_categories_dict, \
                path_to_image_to_final_detection_probs_dict, \
                path_to_image_to_final_detection_colors_dict, \
                path_to_image_to_final_detection_areas_dict, \
                path_to_image_to_final_detection_polygon_groups_dict, \
                path_to_image_to_final_detection_base64_mask_image_dict = future.result()
            response = {
                'result': 'OK',
                'base64_images': path_to_image_to_base64_images_dict[path_to_image],
                'instance_segmentations': [{
                    'bbox': bbox,
                    'category': category,
                    'prob': prob,
                    'color': color,
                    'area': area,
                    'polygons': [{
                        'x': [point[0] for point in polygon],
                        'y': [point[1] for point in polygon]
                    } for polygon in polygon_group],
                } for bbox, category, prob, color, area, polygon_group in
                    zip(path_to_image_to_final_detection_bboxes_dict[path_to_image],
                        path_to_image_to_final_detection_categories_dict[path_to_image],
                        path_to_image_to_final_detection_probs_dict[path_to_image],
                        path_to_image_to_final_detection_colors_dict[path_to_image],
                        path_to_image_to_final_detection_areas_dict[path_to_image],
                        path_to_image_to_final_detection_polygon_groups_dict[path_to_image])],
                'instance_segmentation_base64_map': path_to_image_to_final_detection_base64_mask_image_dict[path_to_image]
            }
        else:
            raise ValueError

        if not returns_images:
            del response['base64_images']

        return Response(response=json.dumps(response), status=200, mimetype='application/json')
    except VerificationError as e:
        return e.response


@app.route(f'/api/{version}/infer_multiple/<job_id>', methods=['POST'])
def start_infer_multiple(job_id: str) -> Response:
    try:
        _verify_job_exists(job_id)

        task_name = Task.Name(request.values['task'])
        epoch = int(request.values['epoch'])
        lower_prob_thresh = float(request.values['lower_prob_thresh']) if 'lower_prob_thresh' in request.values else 0.7
        upper_prob_thresh = float(request.values['upper_prob_thresh']) if 'upper_prob_thresh' in request.values else 1.0
        device_ids = literal_eval(request.values['device_ids']) if 'device_ids' in request.values else None
        path_to_image_pattern_list = literal_eval(request.values['path_to_image_pattern_list'])

        path_to_checkpoint = os.path.join(path_to_outputs_dir, job_id, f'epoch-{epoch:06d}', 'checkpoint.pth')

        path_to_image_list = []
        for path_to_image_pattern in path_to_image_pattern_list:
            path_to_image_list += glob.glob(path_to_image_pattern)
        path_to_image_list = sorted(path_to_image_list)

        if task_name == Task.Name.CLASSIFICATION:
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    _infer,
                    task_name,
                    path_to_checkpoint,
                    lower_prob_thresh, upper_prob_thresh,
                    device_ids,
                    path_to_image_list=path_to_image_list,
                    path_to_results_dir=None
                )

            _, \
                path_to_image_to_final_pred_category_dict, \
                path_to_image_to_final_pred_prob_dict = future.result()
            response = {
                'result': 'OK',
                'images': [{
                    'path': path_to_image,
                    'classification': {
                        'category': path_to_image_to_final_pred_category_dict[path_to_image],
                        'prob': path_to_image_to_final_pred_prob_dict[path_to_image]
                    }
                } for path_to_image in path_to_image_list]
            }
        elif task_name == Task.Name.DETECTION:
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    _infer,
                    task_name,
                    path_to_checkpoint,
                    lower_prob_thresh, upper_prob_thresh,
                    device_ids,
                    path_to_image_list=path_to_image_list,
                    path_to_results_dir=None
                )

            path_to_image_to_base64_images_dict, \
                path_to_image_to_final_detection_bboxes_dict, \
                path_to_image_to_final_detection_categories_dict, \
                path_to_image_to_final_detection_probs_dict = future.result()
            response = {
                'result': 'OK',
                'images': [{
                    'path': path_to_image,
                    'detections': [{
                        'bbox': bbox,
                        'category': category,
                        'prob': prob
                    } for bbox, category, prob in zip(path_to_image_to_final_detection_bboxes_dict[path_to_image],
                                                      path_to_image_to_final_detection_categories_dict[path_to_image],
                                                      path_to_image_to_final_detection_probs_dict[path_to_image])]
                } for path_to_image in path_to_image_list]
            }
        elif task_name == Task.Name.INSTANCE_SEGMENTATION:
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    _infer,
                    task_name,
                    path_to_checkpoint,
                    lower_prob_thresh, upper_prob_thresh,
                    device_ids,
                    path_to_image_list=path_to_image_list,
                    path_to_results_dir=None
                )

            path_to_image_to_base64_images_dict, \
                path_to_image_to_final_detection_bboxes_dict, \
                path_to_image_to_final_detection_categories_dict, \
                path_to_image_to_final_detection_probs_dict, \
                path_to_image_to_final_detection_colors_dict, \
                path_to_image_to_final_detection_areas_dict, \
                path_to_image_to_final_detection_polygon_groups_dict, \
                path_to_image_to_final_detection_base64_mask_image_dict = future.result()
            response = {
                'result': 'OK',
                'images': [{
                    'path': path_to_image,
                    'instance_segmentations': [{
                        'bbox': bbox,
                        'category': category,
                        'prob': prob,
                        'color': color,
                        'area': area,
                        'polygons': [{
                            'x': [point[0] for point in polygon],
                            'y': [point[1] for point in polygon]
                        } for polygon in polygon_group],
                    } for bbox, category, prob, color, area, polygon_group in
                        zip(path_to_image_to_final_detection_bboxes_dict[path_to_image],
                            path_to_image_to_final_detection_categories_dict[path_to_image],
                            path_to_image_to_final_detection_probs_dict[path_to_image],
                            path_to_image_to_final_detection_colors_dict[path_to_image],
                            path_to_image_to_final_detection_areas_dict[path_to_image],
                            path_to_image_to_final_detection_polygon_groups_dict[path_to_image])],
                    'instance_segmentation_base64_map': path_to_image_to_final_detection_base64_mask_image_dict[path_to_image]
                } for path_to_image in path_to_image_list]
            }
        else:
            raise ValueError

        return Response(response=json.dumps(response), status=200, mimetype='application/json')
    except VerificationError as e:
        return e.response


@app.route(f'/api/{version}/deploy/list', methods=['GET'])
def deploy_list() -> Response:
    job_id_and_epoch_and_target_device_id_list = [
        job_id_and_epoch_and_target_device_id
        for job_id_and_epoch_and_target_device_id, _
        in job_id_and_epoch_and_target_device_id_to_deployed_checkpoint_dict.items()
    ]
    response = {
        'result': 'OK',
        'deployments': [{
            'job_id': job_id,
            'epoch': epoch,
            'target_device_id': target_device_id
        } for job_id, epoch, target_device_id in job_id_and_epoch_and_target_device_id_list]
    }
    return Response(response=json.dumps(response), status=200, mimetype='application/json')


@app.route(f'/api/{version}/deploy/<job_id>', methods=['POST'])
def deploy_checkpoint(job_id: str) -> Response:
    try:
        _verify_job_exists(job_id)

        task_name = Task.Name(request.values['task'])
        epoch = int(request.values['epoch'])
        target_device_id = int(request.values['target_device_id'])

        assert target_device_id == -1 or (
                torch.cuda.is_available() and 0 <= target_device_id < torch.cuda.device_count())

        if task_name == Task.Name.CLASSIFICATION:
            from aibox_vision.lib.task.classification.checkpoint import Checkpoint
        elif task_name == Task.Name.DETECTION:
            from aibox_vision.lib.task.detection.checkpoint import Checkpoint
        elif task_name == Task.Name.INSTANCE_SEGMENTATION:
            from aibox_vision.lib.task.instance_segmentation.checkpoint import Checkpoint
        else:
            raise ValueError

        if (job_id, epoch, target_device_id) not in job_id_and_epoch_and_target_device_id_to_deployed_checkpoint_dict:
            path_to_checkpoint = os.path.join(path_to_outputs_dir, job_id, f'epoch-{epoch:06d}', 'checkpoint.pth')
            device = torch.device('cpu') if target_device_id == -1 else torch.device('cuda', target_device_id)
            checkpoint = Checkpoint.load(path_to_checkpoint, device)
            job_id_and_epoch_and_target_device_id_to_deployed_checkpoint_dict[(job_id, epoch, target_device_id)] = \
                checkpoint

        response = {
            'result': 'OK'
        }
        return Response(response=json.dumps(response), status=200, mimetype='application/json')
    except VerificationError as e:
        return e.response


@app.route(f'/api/{version}/deploy/<job_id>', methods=['DELETE'])
def undeploy_checkpoint(job_id: str) -> Response:
    try:
        _verify_job_exists(job_id)

        epoch = int(request.values['epoch'])
        target_device_id = int(request.values['target_device_id'])

        if (job_id, epoch, target_device_id) in job_id_and_epoch_and_target_device_id_to_deployed_checkpoint_dict:
            del job_id_and_epoch_and_target_device_id_to_deployed_checkpoint_dict[(job_id, epoch, target_device_id)]

        response = {
            'result': 'OK'
        }
        return Response(response=json.dumps(response), status=200, mimetype='application/json')
    except VerificationError as e:
        return e.response


@app.route(f'/api/{version}/upgrade/<job_id>', methods=['POST'])
def start_upgrade(job_id: str) -> Response:
    try:
        _verify_job_exists(job_id)

        try:
            task_name = Task.Name(request.values['task'])
            path_to_checkpoints_dir = os.path.join(path_to_outputs_dir, job_id)
            _upgrade(task_name, path_to_checkpoints_dir)
            upgraded = True
            exception = None
        except Exception as e:
            upgraded = False
            exception = {
                'type': e.__class__.__name__,
                'message': str(e)
            }

        response = {
            'result': 'OK',
            'upgraded': upgraded,
            'exception': exception
        }
        return Response(response=json.dumps(response), status=200, mimetype='application/json')
    except VerificationError as e:
        return e.response


@app.route(f'/api/{version}/aug', methods=['POST'])
def preview_aug() -> Response:
    task_name = Task.Name(request.values['task'])
    path_to_checkpoins_dir = ''
    path_to_data_dir = request.values['data_dir']

    config_dict = Config.parse_config_dict(task_name.value, path_to_checkpoins_dir, path_to_data_dir)
    config = Config(**config_dict)

    index = int(request.values['index'])
    aug_strategy = Augmenter.Strategy(request.values['aug_strategy']) if 'aug_strategy' in request.values else config.aug_strategy
    aug_hflip_prob = float(request.values['aug_hflip_prob']) if 'aug_hflip_prob' in request.values else config.aug_hflip_prob
    aug_vflip_prob = float(request.values['aug_vflip_prob']) if 'aug_vflip_prob' in request.values else config.aug_vflip_prob
    aug_rotate90_prob = float(request.values['aug_rotate90_prob']) if 'aug_rotate90_prob' in request.values else config.aug_rotate90_prob
    aug_crop_prob_and_min_max = literal_eval(request.values['aug_crop_prob_and_min_max']) if 'aug_crop_prob_and_min_max' in request.values else config.aug_crop_prob_and_min_max
    aug_zoom_prob_and_min_max = literal_eval(request.values['aug_zoom_prob_and_min_max']) if 'aug_zoom_prob_and_min_max' in request.values else config.aug_zoom_prob_and_min_max
    aug_scale_prob_and_min_max = literal_eval(request.values['aug_scale_prob_and_min_max']) if 'aug_scale_prob_and_min_max' in request.values else config.aug_scale_prob_and_min_max
    aug_translate_prob_and_min_max = literal_eval(request.values['aug_translate_prob_and_min_max']) if 'aug_translate_prob_and_min_max' in request.values else config.aug_translate_prob_and_min_max
    aug_rotate_prob_and_min_max = literal_eval(request.values['aug_rotate_prob_and_min_max']) if 'aug_rotate_prob_and_min_max' in request.values else config.aug_rotate_prob_and_min_max
    aug_shear_prob_and_min_max = literal_eval(request.values['aug_shear_prob_and_min_max']) if 'aug_shear_prob_and_min_max' in request.values else config.aug_shear_prob_and_min_max
    aug_blur_prob_and_min_max = literal_eval(request.values['aug_blur_prob_and_min_max']) if 'aug_blur_prob_and_min_max' in request.values else config.aug_blur_prob_and_min_max
    aug_sharpen_prob_and_min_max = literal_eval(request.values['aug_sharpen_prob_and_min_max']) if 'aug_sharpen_prob_and_min_max' in request.values else config.aug_sharpen_prob_and_min_max
    aug_color_prob_and_min_max = literal_eval(request.values['aug_color_prob_and_min_max']) if 'aug_color_prob_and_min_max' in request.values else config.aug_color_prob_and_min_max
    aug_brightness_prob_and_min_max = literal_eval(request.values['aug_brightness_prob_and_min_max']) if 'aug_brightness_prob_and_min_max' in request.values else config.aug_brightness_prob_and_min_max
    aug_grayscale_prob_and_min_max = literal_eval(request.values['aug_grayscale_prob_and_min_max']) if 'aug_grayscale_prob_and_min_max' in request.values else config.aug_grayscale_prob_and_min_max
    aug_contrast_prob_and_min_max = literal_eval(request.values['aug_contrast_prob_and_min_max']) if 'aug_contrast_prob_and_min_max' in request.values else config.aug_contrast_prob_and_min_max
    aug_noise_prob_and_min_max = literal_eval(request.values['aug_noise_prob_and_min_max']) if 'aug_noise_prob_and_min_max' in request.values else config.aug_noise_prob_and_min_max
    aug_resized_crop_prob_and_width_height = literal_eval(request.values['aug_resized_crop_prob_and_width_height']) if 'aug_resized_crop_prob_and_width_height' in request.values else config.aug_resized_crop_prob_and_width_height

    preprocessor = Preprocessor.build_noop()
    augmenter = Augmenter(aug_strategy,
                          aug_hflip_prob, aug_vflip_prob, aug_rotate90_prob,
                          aug_crop_prob_and_min_max,
                          aug_zoom_prob_and_min_max, aug_scale_prob_and_min_max,
                          aug_translate_prob_and_min_max, aug_rotate_prob_and_min_max,
                          aug_shear_prob_and_min_max,
                          aug_blur_prob_and_min_max, aug_sharpen_prob_and_min_max,
                          aug_color_prob_and_min_max, aug_brightness_prob_and_min_max,
                          aug_grayscale_prob_and_min_max, aug_contrast_prob_and_min_max,
                          aug_noise_prob_and_min_max,
                          aug_resized_crop_prob_and_width_height)

    if task_name == Task.Name.CLASSIFICATION:
        from aibox_vision.lib.task.classification.dataset import Dataset
        dataset = Dataset(path_to_data_dir, Dataset.Mode.UNION, preprocessor, augmenter)

        item = dataset[index]
        image = to_pil_image(item.image)
    elif task_name == Task.Name.DETECTION:
        from aibox_vision.lib.task.detection.dataset import Dataset
        dataset = Dataset(path_to_data_dir, Dataset.Mode.UNION, preprocessor, augmenter, exclude_difficulty=False)

        item = dataset[index]
        image = to_pil_image(item.image)
        bboxes = item.bboxes

        draw = ImageDraw.Draw(image)
        for bbox in bboxes:
            draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline='green', width=2)
    elif task_name == Task.Name.INSTANCE_SEGMENTATION:
        from aibox_vision.lib.task.instance_segmentation.dataset import Dataset
        dataset = Dataset(path_to_data_dir, Dataset.Mode.UNION, preprocessor, augmenter, exclude_difficulty=False)

        item = dataset[index]
        image = to_pil_image(item.image)
        bboxes = item.bboxes
        masks = item.masks

        draw = ImageDraw.Draw(image)
        flatten_palette = Palette.get_flatten_palette()

        for bbox in bboxes:
            draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline='green', width=2)

        for color, mask in enumerate(masks, start=1):
            mask_image = to_pil_image(mask * color)
            mask_image.putpalette(flatten_palette)
            blended_image = Image.blend(image.convert('RGBA'), mask_image.convert('RGBA'), alpha=0.5).convert('RGB')
            image = Image.composite(blended_image, image, mask=to_pil_image(mask * 255).convert('1'))
    else:
        raise ValueError

    buffer = BytesIO()
    image.save(buffer, format='JPEG')
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    response = {
        'result': 'OK',
        'base64_image': base64_image
    }
    return Response(response=json.dumps(response), status=200, mimetype='application/json')


@app.route(f'/api/{version}/summarize/<job_id>', methods=['POST'])
def summarize_checkpoint(job_id: str) -> Response:
    try:
        _verify_job_exists(job_id)

        task_name = Task.Name(request.values['task'])
        epoch = int(request.values['epoch'])

        if task_name == Task.Name.CLASSIFICATION:
            from aibox_vision.lib.task.classification.checkpoint import Checkpoint
        elif task_name == Task.Name.DETECTION:
            raise ValueError('API `summarize_checkpoint` is not supported for detection task yet.')
        elif task_name == Task.Name.INSTANCE_SEGMENTATION:
            raise ValueError('API `summarize_checkpoint` is not supported for instance segmentation task yet.')
        else:
            raise ValueError

        path_to_checkpoint = os.path.join(path_to_outputs_dir, job_id, f'epoch-{epoch:06d}', 'checkpoint.pth')
        device = torch.device('cpu')
        checkpoint = Checkpoint.load(path_to_checkpoint, device)
        model = checkpoint.model

        fake_input = torch.randn(1, 3, 224, 224)
        macs, params = profile(model, inputs=(fake_input,))
        flops = macs * 2
        flops, params = clever_format([flops, params], "%.3f")

        param_keys = list(dict(model.named_parameters()).keys())

        response = {
            'result': 'OK',
            'flops': flops,
            'params': params,
            'param_keys': param_keys
        }
        return Response(response=json.dumps(response), status=200, mimetype='application/json')
    except VerificationError as e:
        return e.response


@app.route(f'/api/{version}/visualize/<job_id>', methods=['POST'])
def visualize_parameter(job_id: str) -> Response:
    try:
        _verify_job_exists(job_id)

        task_name = Task.Name(request.values['task'])
        epoch = int(request.values['epoch'])
        param_key = request.values['param_key']

        if task_name == Task.Name.CLASSIFICATION:
            from aibox_vision.lib.task.classification.checkpoint import Checkpoint
        elif task_name == Task.Name.DETECTION:
            from aibox_vision.lib.task.detection.checkpoint import Checkpoint
        elif task_name == Task.Name.INSTANCE_SEGMENTATION:
            from aibox_vision.lib.task.instance_segmentation.checkpoint import Checkpoint
        else:
            raise ValueError

        path_to_checkpoint = os.path.join(path_to_outputs_dir, job_id, f'epoch-{epoch:06d}', 'checkpoint.pth')
        device = torch.device('cpu')
        checkpoint = Checkpoint.load(path_to_checkpoint, device)
        model = checkpoint.model

        param_dict = dict(model.named_parameters())
        param = param_dict[param_key]
        param = param.flatten().detach().numpy()
        ax = sns.distplot(param)
        fig = ax.get_figure()

        buffer = BytesIO()
        fig.savefig(buffer, format='PNG')
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        response = {
            'result': 'OK',
            'base64_image': base64_image
        }
        return Response(response=json.dumps(response), status=200, mimetype='application/json')
    except KeyError as e:
        response = {
            'result': 'Error',
            'message': f'Key {str(e)} does not exist'
        }
        return Response(response=json.dumps(response), status=400, mimetype='application/json')
    except VerificationError as e:
        return e.response


def _verify_job_exists(job_id: str):
    path_to_checkpoints_dir = os.path.join(path_to_outputs_dir, job_id)
    if not os.path.exists(path_to_checkpoints_dir):
        response = {
            'result': 'Error',
            'message': 'Job not found.'
        }
        raise VerificationError(Response(response=json.dumps(response), status=400, mimetype='application/json'))


def _verify_job_is_tracking(job_id: str):
    if job_id not in job_id_to_process_and_terminator_dict:
        response = {
            'result': 'Error',
            'message': 'Job is not under tracking.'
        }
        raise VerificationError(Response(response=json.dumps(response), status=400, mimetype='application/json'))


def _verify_job_is_stopped(job_id: str):
    # NOTE: Assume that all running jobs must be under tracking
    if job_id in job_id_to_process_and_terminator_dict:
        process, _ = job_id_to_process_and_terminator_dict[job_id]

        if process.is_alive():
            response = {
                'result': 'Error',
                'message': 'Job is not stopped.'
            }
            raise VerificationError(Response(response=json.dumps(response), status=400, mimetype='application/json'))
