import time
from json import dumps
from unittest import TestCase

import requests


class WebAPITestCase(TestCase):

    def setUp(self):
        super().setUp()
        self.server_address = '127.0.0.1:5000'

    def test_all_web_apis(self):
        # version
        print('version')
        response = requests.get(f'http://{self.server_address}/api/version')
        self.assertEqual(200, response.status_code)

        version = response.text
        self.assertIsInstance(version, str)

        # environment
        print('environment')
        response = requests.get(f'http://{self.server_address}/api/{version}/environment')
        self.assertEqual(200, response.status_code)
        json = response.json()
        print(dumps(json, indent=2))
        self.assertEqual('OK', json['result'])

        # devices
        print('devices')
        response = requests.get(f'http://{self.server_address}/api/{version}/devices')
        self.assertEqual(200, response.status_code)
        json = response.json()
        print(dumps(json, indent=2))
        self.assertEqual('OK', json['result'])

        # train
        print('train')
        data = {
            'task': 'instance_segmentation',
            'data_dir': './examples/CatDog',
            'algorithm': 'mask_rcnn',
            'visible_devices': '[0]'
        }
        response = requests.post(f'http://{self.server_address}/api/{version}/train', data)
        self.assertEqual(200, response.status_code)
        json = response.json()
        print(dumps(json, indent=2))
        self.assertEqual('OK', json['result'])
        job_id = json['job_id']
        self.assertIsInstance(job_id, str)

        time.sleep(10)

        # stop
        print('stop')
        response = requests.patch(f'http://{self.server_address}/api/{version}/train/{job_id}')
        self.assertEqual(200, response.status_code)
        json = response.json()
        print(dumps(json, indent=2))
        self.assertEqual('OK', json['result'])

        time.sleep(20)

        # restore
        print('restore')
        response = requests.put(f'http://{self.server_address}/api/{version}/train/{job_id}')
        self.assertEqual(200, response.status_code)
        json = response.json()
        print(dumps(json, indent=2))
        self.assertEqual('OK', json['result'])

        time.sleep(10)

        # stop
        print('stop')
        response = requests.patch(f'http://{self.server_address}/api/{version}/train/{job_id}')
        self.assertEqual(200, response.status_code)
        json = response.json()
        print(dumps(json, indent=2))
        self.assertEqual('OK', json['result'])

        time.sleep(10)

        # monitor
        print('monitor')
        response = requests.get(f'http://{self.server_address}/api/{version}/train/{job_id}')
        self.assertEqual(200, response.status_code)
        json = response.json()
        print(dumps(json, indent=2))
        self.assertEqual('OK', json['result'])
        self.assertIsNone(json['exception'])

        # hyper-parameters
        print('hyper-parameters')
        response = requests.get(f'http://{self.server_address}/api/{version}/train/{job_id}/hyper-parameters')
        self.assertEqual(200, response.status_code)
        json = response.json()
        print(dumps(json, indent=2))
        self.assertEqual('OK', json['result'])

        # losses
        print('losses')
        response = requests.get(f'http://{self.server_address}/api/{version}/train/{job_id}/losses')
        self.assertEqual(200, response.status_code)
        json = response.json()
        print(dumps(json, indent=2))
        self.assertEqual('OK', json['result'])

        # checkpoints
        print('checkpoints')
        response = requests.get(f'http://{self.server_address}/api/{version}/train/{job_id}/checkpoints')
        self.assertEqual(200, response.status_code)
        json = response.json()
        print(dumps(json, indent=2))
        self.assertEqual('OK', json['result'])

        epoch = json['checkpoints'][-1]['epoch']

        # checkpoint
        print('checkpoint')
        response = requests.get(f'http://{self.server_address}/api/{version}/train/{job_id}/checkpoints/epoch/{epoch}')
        self.assertEqual(200, response.status_code)
        json = response.json()
        json['checkpoint']['base64_metric_plot'] = json['checkpoint']['base64_metric_plot'][:10] + '...'
        print(dumps(json, indent=2))
        self.assertEqual('OK', json['result'])

        # loss plot
        print('loss plot')
        response = requests.get(f'http://{self.server_address}/api/{version}/train/{job_id}/plots/loss')
        self.assertEqual(200, response.status_code)
        json = response.json()
        json['base64_loss_plot'] = json['base64_loss_plot'][:10] + '...'
        print(dumps(json, indent=2))
        self.assertEqual('OK', json['result'])

        # infer
        print('infer')
        data = {
            'task': 'instance_segmentation',
            'epoch': epoch,
            'path_to_image': './examples/CatDog/images/000037.jpg'
        }
        response = requests.post(f'http://{self.server_address}/api/{version}/infer/{job_id}', data)
        self.assertEqual(200, response.status_code)
        json = response.json()
        for i in range(len(json['base64_images'])):
            json['base64_images'][i] = json['base64_images'][i][:10] + '...'
        json['instance_segmentation_base64_map'] = json['instance_segmentation_base64_map'][:10] + '...'
        print(dumps(json, indent=2))
        self.assertEqual('OK', json['result'])

        # infer multiple
        print('infer multiple')
        data = {
            'task': 'instance_segmentation',
            'epoch': epoch,
            'path_to_image_pattern_list': '["./examples/CatDog/images/*.jpg", "./examples/Person/images/000001.jpg"]'
        }
        response = requests.post(f'http://{self.server_address}/api/{version}/infer_multiple/{job_id}', data)
        self.assertEqual(200, response.status_code)
        json = response.json()
        for i in range(len(json['images'])):
            json['images'][i]['instance_segmentation_base64_map'] = json['images'][i]['instance_segmentation_base64_map'][:10] + '...'
        print(dumps(json, indent=2))
        self.assertEqual('OK', json['result'])

        # preview aug
        print('preview aug')
        data = {
            'task': 'instance_segmentation',
            'data_dir': './examples/CatDog',
            'index': 0,
            'aug_strategy': 'all',
            'aug_hflip_prob': 0.5,
            'aug_vflip_prob': 0.5,
            'aug_rotate_prob_and_min_max': '(0.5, (0, 1))',
            'aug_noise_prob_and_min_max': '(0.5, (0, 1))'
        }
        response = requests.post(f'http://{self.server_address}/api/{version}/aug', data)
        self.assertEqual(200, response.status_code)
        json = response.json()
        json['base64_image'] = json['base64_image'][:10] + '...'
        print(dumps(json, indent=2))
        self.assertEqual('OK', json['result'])

        # clear
        print('clear')
        response = requests.delete(f'http://{self.server_address}/api/{version}/train/{job_id}')
        self.assertEqual(200, response.status_code)
        json = response.json()
        print(dumps(json, indent=2))
        self.assertEqual('OK', json['result'])
