## Web API

* API Version: v1.5.0


### Table of contents

* [Query API version](#query-api-version)
* [Query environment information](#query-environment-information)
* [Query devices usage](#query-devices-usage)
* [Start a training job](#start-a-training-job)
* [Stop a training job](#stop-a-training-job)
* [Restore a training job](#restore-a-training-job)
* [Clear a training job](#clear-a-training-job)
* [Monitor a training job](#monitor-a-training-job)
* [Query hyper-parameters from a training job](#query-hyper-parameters-from-a-job)
* [Query loss history from a training job](#query-loss-history-from-a-training-job)
* [Retrieve a checkpoint from a training job](#retrieve-a-checkpoint-from-a-training-job)
* [Retrieve all checkpoints from a training job](#retrieve-all-checkpoints-from-a-training-job)
* [Obtain the plot of model graph](#obtain-the-plot-of-model-graph)
* [Obtain the plot of loss curve](#obtain-the-plot-of-loss-curve)
* [Obtain the plot of confusion matrix](#obtain-the-plot-of-confusion-matrix)
* [Obtain the plot of threshold versus PR](#obtain-the-plot-of-threshold-versus-pr)
* [Infer an image](#infer-an-image)
* [Infer multiple images](#infer-multiple-images)
* [List all deployed checkpoint](#list-all-deployed-checkpoints)
* [Deploy a checkpoint](#deploy-a-checkpoint)
* [Undeploy a checkpoint](#undeploy-a-checkpoint)
* [Upgrade a job](#upgrade-a-job)
* [Preview data augmentation](#preview-data-augmentation)
* [Summarize a checkpoint](#summarize-a-checkpoint)
* [Visualize a parameter from a checkpoint](#visualize-a-parameter-from-a-checkpoint)


##### Query API version

* Endpoint: http://$SERVER_ADDRESS/api/version

* Method: GET

* Arguments: application/x-www-form-urlencoded
        
    * None

* Response: TEXT
    
    ```
    v1.5.0
    ```
    
* Test:

    ```
    $ curl -X GET http://$SERVER_ADDRESS/api/version
    ```


##### Query environment information

* Endpoint: http://$SERVER_ADDRESS/api/$VERSION/environment

* Method: GET

* Arguments: application/x-www-form-urlencoded
        
    * None

* Response: JSON
    
    ```json
    {
      "result": "OK",
      "environment": {
        "hardware": {
          "devices": [
            {
              "id": 0,
              "name": "Tesla V100-SXM2-16GB",
              "memory_in_mb": 16130
            },
            {
              "id": 1,
              "name": "Tesla V100-SXM2-16GB",
              "memory_in_mb": 16130
            }
          ]
        },
        "software": {
          "python_version": "3.7.3",
          "cuda_version": "10.0.130",
          "cudnn_version": "7501"
        }
      }
    }
    ```
    
* Test:

    ```
    $ curl -X GET http://$SERVER_ADDRESS/api/$VERSION/environment
    ```


##### Query devices usage

* Endpoint: http://$SERVER_ADDRESS/api/$VERSION/devices

* Method: GET

* Arguments: application/x-www-form-urlencoded
        
    * None

* Response: JSON
    
    ```json
    {
      "result": "OK",
      "devices": [
        {
          "id": 0,
          "name": "Tesla V100-SXM2-16GB",
          "utilization_in_percent": 60,
          "memory_total_in_mb": 16130,
          "memory_used_in_mb": 793,
          "memory_free_in_mb": 15337,
          "power_limit_in_watt": 250,
          "power_usage_in_watt": 63
        },
        {
          "id": 1,
          "name": "Tesla V100-SXM2-16GB",
          "utilization_in_percent": 60,
          "memory_total_in_mb": 16130,
          "memory_used_in_mb": 793,
          "memory_free_in_mb": 15337,
          "power_limit_in_watt": 250,
          "power_usage_in_watt": 63
        }
      ]
    }
    ```
    
* Test:

    ```
    $ curl -X GET http://$SERVER_ADDRESS/api/$VERSION/devices
    ```

##### Start a training job

* Endpoint: http://$SERVER_ADDRESS/api/$VERSION/train

* Method: POST

* Arguments: application/x-www-form-urlencoded

    <table>
        <thead>
            <tr>
                <th align="center"><sup><sub>Keyword</sub></sup></th>
                <th align="center"><sup><sub>Example / Default</sub></sup></th>
                <th align="center"><sup><sub>Required</sub></sup></th>
                <th align="center"><sup><sub>
                    Overwritable <br>
                    Resume | Fine-Tune
                </sub></sup></th>
                <th align="center"><sup><sub>Type</sub></sup></th>
                <th align="center"><sup><sub>Constraints</sub></sup></th>
                <th align="center"><sup><sub>Note</sub></sup></th>
        </tr>
    </thead>
    <tbody>
        <tr>
        	<td align="center"><sup><sub>task</sub></sup></td>
        	<td align="center"><sup><sub>detection</sub></sup></td>
        	<td align="center"><sup><sub>True</sub></sup></td>
        	<td align="center"><sup><sub>False | False</sub></sup></td>
        	<td align="center"><sup><sub>str</sub></sup></td>
        	<td align="center"><sup><sub>
        	    Value: {classification, detection, instance_segmentation}
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>data_dir</sub></sup></td>
        	<td align="center"><sup><sub>/path/to/dataset</sub></sup></td>
        	<td align="center"><sup><sub>True</sub></sup></td>
        	<td align="center"><sup><sub>False | True</sub></sup></td>
        	<td align="center"><sup><sub>str</sub></sup></td>
        	<td align="center"><sup><sub>
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>resume_job_id_and_epoch</sub></sup></td>
        	<td align="center"><sup><sub>None</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | False</sub></sup></td>
        	<td align="center"><sup><sub>str</sub></sup></td>
        	<td align="center"><sup><sub>
        	    Value: {resume_job_id_and_epoch, finetune_job_id_and_epoch} is None
            </sub> </sup></td>
        	<td align="center"><sup><sub>
        	    e.g., ("20200106173410-49c73c67-8e0e-4061-a782-39b8b1cc8d8a", 2)
        	</sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>finetune_job_id_and_epoch</sub></sup></td>
        	<td align="center"><sup><sub>None</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>False | True</sub></sup></td>
        	<td align="center"><sup><sub>str</sub></sup></td>
        	<td align="center"><sup><sub>
        	    Value: {resume_job_id_and_epoch, finetune_job_id_and_epoch} is None
        	</sub></sup></td>
        	<td align="center"><sup><sub>
        	    e.g., ("20200106173410-49c73c67-8e0e-4061-a782-39b8b1cc8d8a", 2)
        	</sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>num_workers</sub></sup></td>
        	<td align="center"><sup><sub>2</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>int</sub></sup></td>
        	<td align="center"><sup><sub>
        	    Value: [0, Inf]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>visible_devices</sub></sup></td>
        	<td align="center"><sup><sub>None</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>Optional[List[int]</sub></sup></td>
        	<td align="center"><sup><sub>
        	    #Elements: [0, #Devices] <br>
        	    Value: [0, #Devices)
        	</sub></sup></td>
        	<td align="center"><sup><sub>
        	    Leave empty for no devices (i.e., use CPU); `None` for all devices if any devices are available, otherwise CPU is used 
        	</sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>needs_freeze_bn</sub></sup></td>
        	<td align="center"><sup><sub>
        	    task? <br>
        	    classification: False <br>
        	    detection: True <br>
        	    instance_segmentation: True
            </sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>bool</sub></sup></td>
        	<td align="center"><sup><sub>
        	    Value: {True, False}
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>image_resized_width</sub></sup></td>
        	<td align="center"><sup><sub>
        	    task? <br>
        	    classification: 224 <br>
        	    detection: -1 <br>
        	    instance_segmentation: -1
            </sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>int</sub></sup></td>
        	<td align="center"><sup><sub>
        	    Value: {(0, Inf], -1}
        	</sub></sup></td>
        	<td align="center"><sup><sub>-1 to keep original width</sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>image_resized_height</sub></sup></td>
        	<td align="center"><sup><sub>
        	    task? <br>
        	    classification: 224 <br>
        	    detection: -1 <br>
        	    instance_segmentation: -1
            </sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>int</sub></sup></td>
        	<td align="center"><sup><sub>
        	    Value: {(0, Inf], -1}
        	</sub></sup></td>
        	<td align="center"><sup><sub>-1 to keep original height</sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>image_min_side</sub></sup></td>
        	<td align="center"><sup><sub>
        	    task? <br>
        	    classification: -1 <br>
        	    detection: 600 <br>
        	    instance_segmentation: 800
            </sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>int</sub></sup></td>
        	<td align="center"><sup><sub>
        	    Value: {(0, image_max_side], -1}
        	</sub></sup></td>
        	<td align="center"><sup><sub>-1 for unconstrained minimal side</sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>image_max_side</sub></sup></td>
        	<td align="center"><sup><sub>
        	    task? <br>
        	    classification: -1 <br>
        	    detection: 1000 <br>
        	    instance_segmentation: 1333
            </sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>int</sub></sup></td>
        	<td align="center"><sup><sub>
        	    Value: {[image_min_side, Inf], -1}
            </sub></sup></td>
        	<td align="center"><sup><sub>-1 for unconstrained maximal side</sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>aug_strategy</sub></sup></td>
        	<td align="center"><sup><sub>all</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>str</sub></sup></td>
        	<td align="center"><sup><sub>
        	    Value: {all, one, some}
            </sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>aug_hflip_prob</sub></sup></td>
        	<td align="center"><sup><sub>0.5</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>float</sub></sup></td>
        	<td align="center"><sup><sub>
        	    prob Value: [0, 1]
        	</sub></sup></td>
        	<td align="center"><sup><sub> </sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>aug_vflip_prob</sub></sup></td>
        	<td align="center"><sup><sub>0</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>float</sub></sup></td>
        	<td align="center"><sup><sub>
        	    prob Value: [0, 1]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>aug_rotate90_prob</sub></sup></td>
        	<td align="center"><sup><sub>0</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>float</sub></sup></td>
        	<td align="center"><sup><sub>
        	    prob Value: [0, 1]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>aug_crop_prob_and_min_max</sub></sup></td>
        	<td align="center"><sup><sub>(0.0, (0.0, 1.0))</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>Tuple[float, Tuple[float, float]]</sub></sup></td>
        	<td align="center"><sup><sub>
        	    prob Value: [0, 1] <br>
        	    min_max Value: [0, 1]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>aug_zoom_prob_and_min_max</sub></sup></td>
        	<td align="center"><sup><sub>(0.0, (-1.0, 1.0))</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>Tuple[float, Tuple[float, float]]</sub></sup></td>
        	<td align="center"><sup><sub>
        	    prob Value: [0, 1] <br>
        	    min_max Value: [-1, 1]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>aug_scale_prob_and_min_max</sub></sup></td>
        	<td align="center"><sup><sub>(0.0, (-1.0, 1.0))</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>Tuple[float, Tuple[float, float]]</sub></sup></td>
        	<td align="center"><sup><sub>
        	    prob Value: [0, 1] <br>
        	    min_max Value: [-1, 1]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>aug_translate_prob_and_min_max</sub></sup></td>
        	<td align="center"><sup><sub>(0.0, (-1.0, 1.0))</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>Tuple[float, Tuple[float, float]]</sub></sup></td>
        	<td align="center"><sup><sub>
        	    prob Value: [0, 1] <br>
        	    min_max Value: [-1, 1]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>aug_rotate_prob_and_min_max</sub></sup></td>
        	<td align="center"><sup><sub>(0.0, (-1.0, 1.0))</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>Tuple[float, Tuple[float, float]]</sub></sup></td>
        	<td align="center"><sup><sub>
        	    prob Value: [0, 1] <br>
        	    min_max Value: [-1, 1]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>aug_shear_prob_and_min_max</sub></sup></td>
        	<td align="center"><sup><sub>(0.0, (-1.0, 1.0))</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>Tuple[float, Tuple[float, float]]</sub></sup></td>
        	<td align="center"><sup><sub>
        	    prob Value: [0, 1] <br>
        	    min_max Value: [-1, 1]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>aug_blur_prob_and_min_max</sub></sup></td>
        	<td align="center"><sup><sub>(0.0, (0.0, 1.0))</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>Tuple[float, Tuple[float, float]]</sub></sup></td>
        	<td align="center"><sup><sub>
        	    prob Value: [0, 1] <br>
        	    min_max Value: [0, 1]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>aug_sharpen_prob_and_min_max</sub></sup></td>
        	<td align="center"><sup><sub>(0.0, (0.0, 1.0))</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>Tuple[float, Tuple[float, float]]</sub></sup></td>
        	<td align="center"><sup><sub>
        	    prob Value: [0, 1] <br>
        	    min_max Value: [0, 1]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>aug_color_prob_and_min_max</sub></sup></td>
        	<td align="center"><sup><sub>(0.0, (-1.0, 1.0))</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>Tuple[float, Tuple[float, float]]</sub></sup></td>
        	<td align="center"><sup><sub>
        	    prob Value: [0, 1] <br>
        	    min_max Value: [-1, 1]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>aug_brightness_prob_and_min_max</sub></sup></td>
        	<td align="center"><sup><sub>(0.0, (-1.0, 1.0))</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>Tuple[float, Tuple[float, float]]</sub></sup></td>
        	<td align="center"><sup><sub>
        	    prob Value: [0, 1] <br>
        	    min_max Value: [-1, 1]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>aug_grayscale_prob_and_min_max</sub></sup></td>
        	<td align="center"><sup><sub>(0.0, (0.0, 1.0))</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>Tuple[float, Tuple[float, float]]</sub></sup></td>
        	<td align="center"><sup><sub>
        	    prob Value: [0, 1] <br>
        	    min_max Value: [0, 1]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>aug_contrast_prob_and_min_max</sub></sup></td>
        	<td align="center"><sup><sub>(0.0, (-1.0, 1.0))</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>Tuple[float, Tuple[float, float]]</sub></sup></td>
        	<td align="center"><sup><sub>
        	    prob Value: [0, 1] <br>
        	    min_max Value: [-1, 1]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>aug_noise_prob_and_min_max</sub></sup></td>
        	<td align="center"><sup><sub>(0.0, (0.0, 1.0))</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>Tuple[float, Tuple[float, float]]</sub></sup></td>
        	<td align="center"><sup><sub>
        	    prob Value: [0, 1] <br>
        	    min_max Value: [0, 1]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>aug_resized_crop_prob_and_width_height</sub></sup></td>
        	<td align="center"><sup><sub>(0.0, (224, 224))</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>Tuple[float, Tuple[int, int]]</sub></sup></td>
        	<td align="center"><sup><sub>
        	    prob Value: [0, 1] <br>
        	    min_max Value: (0, Inf]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>batch_size</sub></sup></td>
        	<td align="center"><sup><sub>
        	    task? <br>
        	    classification: 8 <br>
        	    detection: 2 <br>
        	    instance_segmentation: 1
            </sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>int</sub></sup></td>
        	<td align="center"><sup><sub>
        	    Value: [1, Inf] 
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>learning_rate</sub></sup></td>
        	<td align="center"><sup><sub>
        	    task? <br>
        	    classification: 0.008 <br>
        	    detection: 0.002 <br>
        	    instance_segmentation: 0.001
            </sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>False | True</sub></sup></td>
        	<td align="center"><sup><sub>float</sub></sup></td>
        	<td align="center"><sup><sub>
        	    Value: (0, 1]
        	</sub></sup></td>
        	<td align="center"><sup><sub>
        	    It is recommended to linearly scale the learning rate with the batch size
        	</sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>momentum</sub></sup></td>
        	<td align="center"><sup><sub>0.9</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>False | True</sub></sup></td>
        	<td align="center"><sup><sub>float</sub></sup></td>
        	<td align="center"><sup><sub>
        	    Value: (0, 1]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>weight_decay</sub></sup></td>
        	<td align="center"><sup><sub>0.0005</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>False | True</sub></sup></td>
        	<td align="center"><sup><sub>float</sub></sup></td>
        	<td align="center"><sup><sub>
        	    Value: (0, 1]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>clip_grad_base_and_max</sub></sup></td>
        	<td align="center"><sup><sub>None</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>Tuple[str, float]</sub></sup></td>
        	<td align="center"><sup><sub>
        	    base Value: {value, norm} <br>
        	    max Value: (0, Inf]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>step_lr_sizes</sub></sup></td>
        	<td align="center"><sup><sub>
        	    task? <br>
        	    classification: [6, 8] <br>
        	    detection: [10, 14] <br>
        	    instance_segmentation: [20, 28]
            </sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>List[int]</sub></sup></td>
        	<td align="center"><sup><sub>
        	    #Elements: [0, 5] <br>
                Value: [0, num_epochs_to_finish]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>step_lr_gamma</sub></sup></td>
        	<td align="center"><sup><sub>0.1</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>float</sub></sup></td>
        	<td align="center"><sup><sub>
        	    Value: [0, 1]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>num_batches_to_display</sub></sup></td>
        	<td align="center"><sup><sub>20</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>int</sub></sup></td>
        	<td align="center"><sup><sub>
        	    Value: [1, Inf]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>num_epochs_to_validate</sub></sup></td>
        	<td align="center"><sup><sub>1</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>int</sub></sup></td>
        	<td align="center"><sup><sub>
        	    Value: [1, num_epochs_to_finish]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>num_epochs_to_finish</sub></sup></td>
        	<td align="center"><sup><sub>
        	    task? <br>
        	    classification: 10 <br>
        	    detection: 16 <br>
        	    instance_segmentation: 32
            </sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>int</sub></sup></td>
        	<td align="center"><sup><sub>
        	    Value: [1, Inf]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>max_num_checkpoints</sub></sup></td>
        	<td align="center"><sup><sub>5</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>int</sub></sup></td>
        	<td align="center"><sup><sub>
        	    Value: [1, Inf]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
            <td align="center" colspan="100%"><sup><sub>For classification task</sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>algorithm</sub></sup></td>
        	<td align="center"><sup><sub>resnet50</sub></sup></td>
        	<td align="center"><sup><sub>True</sub></sup></td>
        	<td align="center"><sup><sub>False | False</sub></sup></td>
        	<td align="center"><sup><sub>str</sub></sup></td>
        	<td align="center"><sup><sub>
        	    Value: {mobilenet_v2, googlenet, inception_v3, resnet18, resnet34, resnet50, resnet101, efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7, resnest50, resnest101, resnest200, resnest269}
            </sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>num_frozen_levels</sub></sup></td>
        	<td align="center"><sup><sub>2</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>int</sub></sup></td>
        	<td align="center"><sup><sub>
                Value: [0, 5]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>eval_center_crop_ratio</sub></sup></td>
        	<td align="center"><sup><sub>1</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>float</sub></sup></td>
        	<td align="center"><sup><sub>
                Value: (0, 1]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
            <td align="center" colspan="100%"><sup><sub>For detection task</sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>algorithm</sub></sup></td>
        	<td align="center"><sup><sub>faster_rcnn</sub></sup></td>
        	<td align="center"><sup><sub>True</sub></sup></td>
        	<td align="center"><sup><sub>False | False</sub></sup></td>
        	<td align="center"><sup><sub>str</sub></sup></td>
        	<td align="center"><sup><sub>
        	    Value: {faster_rcnn, fpn}
            </sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>backbone</sub></sup></td>
        	<td align="center"><sup><sub>resnet18</sub></sup></td>
        	<td align="center"><sup><sub>True</sub></sup></td>
        	<td align="center"><sup><sub>False | False</sub></sup></td>
        	<td align="center"><sup><sub>str</sub></sup></td>
        	<td align="center"><sup><sub>
        	    Value: {resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2, senet154, se_resnext50_32x4d, se_resnext101_32x4d, nasnet_a_large, pnasnet_5_large, resnest50, resnest101, resnest200, resnest269}
            </sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>anchor_sizes</sub></sup></td>
        	<td align="center"><sup><sub>
        	    algorithm? <br>
        	    faster_rcnn: [128, 256, 512] <br>
        	    fpn: [128]
            </sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>False | False</sub></sup></td>
        	<td align="center"><sup><sub>List[int]</sub></sup></td>
        	<td align="center"><sup><sub>
                #Elements: [1, 5] <br>
                Value: [2, image_max_side]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>backbone_pretrained</sub></sup></td>
        	<td align="center"><sup><sub>True</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>False | False</sub></sup></td>
        	<td align="center"><sup><sub>bool</sub></sup></td>
        	<td align="center"><sup><sub>
                Value: {True, False}
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>backbone_num_frozen_levels</sub></sup></td>
        	<td align="center"><sup><sub>2</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>int</sub></sup></td>
        	<td align="center"><sup><sub>
                Value: [0, 5]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>train_rpn_pre_nms_top_n</sub></sup></td>
        	<td align="center"><sup><sub>
        	    algorithm? <br>
        	    faster_rcnn: 12000 <br>
        	    fpn: 2000
            </sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>int</sub></sup></td>
        	<td align="center"><sup><sub>
        	    Value: [1, Inf]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>train_rpn_post_nms_top_n</sub></sup></td>
        	<td align="center"><sup><sub>
        	    algorithm? <br>
        	    faster_rcnn: 2000 <br>
        	    fpn: 2000
            </sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>int</sub></sup></td>
        	<td align="center"><sup><sub>
        	    Value: [1, Inf]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>eval_rpn_pre_nms_top_n</sub></sup></td>
        	<td align="center"><sup><sub>
        	    algorithm? <br>
        	    faster_rcnn: 6000 <br>
        	    fpn: 1000
            </sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>int</sub></sup></td>
        	<td align="center"><sup><sub>
        	    Value: [1, Inf]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>eval_rpn_post_nms_top_n</sub></sup></td>
        	<td align="center"><sup><sub>
        	    algorithm? <br>
        	    faster_rcnn: 1000 <br>
        	    fpn: 1000
            </sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>int</sub></sup></td>
        	<td align="center"><sup><sub>
        	    Value: [1, Inf]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>num_anchor_samples_per_batch</sub></sup></td>
        	<td align="center"><sup><sub>256</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>int</sub></sup></td>
        	<td align="center"><sup><sub>
        	    Value: [1, Inf]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>num_proposal_samples_per_batch</sub></sup></td>
        	<td align="center"><sup><sub>128</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>int</sub></sup></td>
        	<td align="center"><sup><sub>
        	    Value: [1, Inf]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>num_detections_per_image</sub></sup></td>
        	<td align="center"><sup><sub>100</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>int</sub></sup></td>
        	<td align="center"><sup><sub>
        	    Value: [1, Inf]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>proposal_nms_threshold</sub></sup></td>
        	<td align="center"><sup><sub>0.7</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>float</sub></sup></td>
        	<td align="center"><sup><sub>
        	    Value: [0, 1]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>detection_nms_threshold</sub></sup></td>
        	<td align="center"><sup><sub>0.5</sub></sup></td>
        	<td align="center"><sup><sub>False</sub></sup></td>
        	<td align="center"><sup><sub>True | True</sub></sup></td>
        	<td align="center"><sup><sub>float</sub></sup></td>
        	<td align="center"><sup><sub>
        	    Value: [0, 1]
        	</sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
        <tr>
            <td align="center" colspan="100%"><sup><sub>For instance segmentation task</sub></sup></td>
        </tr>
        <tr>
        	<td align="center"><sup><sub>algorithm</sub></sup></td>
        	<td align="center"><sup><sub>mask_rcnn</sub></sup></td>
        	<td align="center"><sup><sub>True</sub></sup></td>
        	<td align="center"><sup><sub>False | False</sub></sup></td>
        	<td align="center"><sup><sub>str</sub></sup></td>
        	<td align="center"><sup><sub>
        	    Value: {mask_rcnn}
            </sub></sup></td>
        	<td align="center"><sup><sub></sub></sup></td>
        </tr>
    </tbody>
    </table>

* Response: JSON
    
    ```json
    {
      "result": "OK",
      "job_id": "00000001"
    }
    ```
    
* Test:

    ```
    $ curl -X POST -d task=classification -d data_dir=./examples/CatDog -d algorithm=resnet50 http://$SERVER_ADDRESS/api/$VERSION/train
    $ curl -X POST -d task=detection -d data_dir=./examples/CatDog -d algorithm=faster_rcnn -d backbone=resnet50 http://$SERVER_ADDRESS/api/$VERSION/train
    $ curl -X POST -d task=instance_segmentation -d data_dir=./examples/CatDog -d algorithm=mask_rcnn http://$SERVER_ADDRESS/api/$VERSION/train
    ```
        
##### Stop a training job

* Endpoint: http://$SERVER_ADDRESS/api/$VERSION/train/$job_id

* Method: PATCH

* Arguments: application/x-www-form-urlencoded
        
    * None

* Response: JSON
    
    ```json
    {
      "result": "OK"
    }
    ```
    
    or 
    
    ```json
    {
      "result": "Error",
      "message": "Job is not under tracking."
    }
    ```
    
* Test:

    ```
    $ curl -X PATCH http://$SERVER_ADDRESS/api/$VERSION/train/$job_id
    ```
        
##### Restore a training job

* Endpoint: http://$SERVER_ADDRESS/api/$VERSION/train/$job_id

* Method: PUT

* Arguments: application/x-www-form-urlencoded
        
    * None

* Response: JSON
    
    ```json
    {
      "result": "OK"
    }
    ```
    
    or 
    
    ```json
    {
      "result": "Error",
      "message": "Job not found."
    }
    ```
    
    or 
    
    ```json
    {
      "result": "Error",
      "message": "Job is not stopped."
    }
    ```
    
* Test:

    ```
    $ curl -X PUT http://$SERVER_ADDRESS/api/$VERSION/train/$job_id
    ```
        
##### Clear a training job

* Endpoint: http://$SERVER_ADDRESS/api/$VERSION/train/$job_id

* Method: DELETE

* Arguments: application/x-www-form-urlencoded
        
    * None

* Response: JSON
    
    ```json
    {
      "result": "OK"
    }
    ```
    
    or 
    
    ```json
    {
      "result": "Error",
      "message": "Job not found."
    }
    ```
    
    or 
    
    ```json
    {
      "result": "Error",
      "message": "Job is not stopped."
    }
    ```
    
* Test:

    ```
    $ curl -X DELETE http://$SERVER_ADDRESS/api/$VERSION/train/$job_id
    ```

##### Monitor a training job

* Endpoint: http://$SERVER_ADDRESS/api/$VERSION/train/$job_id

* Method: GET

* Arguments: application/x-www-form-urlencoded
        
    * None
    
* Response: JSON
    
    ```json
    {
      "result": "OK",
      "status": "initializing",
      "datetime": "2019-08-13 11:44:56",
      "epoch": 0,
      "total_epoch": 16,
      "batch": 0,
      "total_batch": 0,
      "avg_loss": -1,
      "learning_rate": -1,
      "samples_per_sec": -1,
      "eta_hr": -1,
      "exception": null
    }
    ```
    
    or
    
    ```json
    {
      "result": "OK",
      "status": "initialized",
      "datetime": "2019-08-13 11:44:56",
      "epoch": 0,
      "total_epoch": 16,
      "batch": 0,
      "total_batch": 5011,
      "avg_loss": -1,
      "learning_rate": -1,
      "samples_per_sec": -1,
      "eta_hr": -1,
      "exception": null
    }
    ```
    
    or
    
    ```json
    {
      "result": "OK",
      "status": "running",
      "datetime": "2019-08-13 11:44:56",
      "epoch": 1,
      "total_epoch": 16,
      "batch": 78,
      "total_batch": 5011,
      "avg_loss": 2.300551319733644,
      "learning_rate": 0.0004373052,
      "samples_per_sec": 11.606803074998755,
      "eta_hr": 1.4372318164507547,
      "exception": null
    }
    ```
    
    or
    
    ```json
    {
      "result": "OK",
      "status": "stopped",
      "datetime": "2019-08-13 11:44:56",
      "epoch": 2,
      "total_epoch": 16,
      "batch": 2500,
      "total_batch": 5011,
      "avg_loss": 1.255213953957331,
      "learning_rate": 0.001,
      "samples_per_sec": 11.606803074998755,
      "eta_hr": 1.2956763967138165,
      "exception": null
    }
    ```
    
    or
    
    ```json
    {
      "result": "OK",
      "status": "finished",
      "datetime": "2019-08-13 11:44:56",
      "epoch": 16,
      "total_epoch": 16,
      "batch": 5011,
      "total_batch": 5011,
      "avg_loss": 1.255213953957331,
      "learning_rate": 0.001,
      "samples_per_sec": 11.606803074998755,
      "eta_hr": 0,
      "exception": null
    }
    ```
    
    or
    
    ```json
    {
      "result": "OK",
      "status": "exception",
      "datetime": "2019-08-13 11:44:56",
      "epoch": 0,
      "total_epoch": 16,
      "batch": 0,
      "total_batch": 0,
      "avg_loss": -1,
      "learning_rate": -1,
      "samples_per_sec": -1,
      "eta_hr": -1,
      "exception": {
        "code": "E03",
        "type": "ValueError",
        "message": "Invalid backbone name",
        "traceback": "Traceback (most recent call last):\n  File..."
      }
    }
    ```
    
    or
    
    ```json
    {
      "result": "OK",
      "status": "exception",
      "datetime": "2019-08-13 11:44:56",
      "epoch": 2,
      "total_epoch": 16,
      "batch": 2500,
      "total_batch": 5011,
      "avg_loss": 1.255213953957331,
      "learning_rate": 0.001,
      "samples_per_sec": 11.606803074998755,
      "eta_hr": 1.2956763967138165,
      "exception": {
        "code": "E01",
        "type": "RuntimeError",
        "message": "Out of memory",
        "traceback": "Traceback (most recent call last):\n  File..."
      }
    }
    ```
    
    >   `status` can be either `initializing`, `initialized`, `running`, `stopped`, `finished` or `exception`
    
    >   The definition of `code` in `exception` can be found [here](ERROR_CODE.md)
    
      or
    
      ```json
      {
        "result": "Error",
        "message": "Job not found."
      }
      ```
    
* Test:

    ```
    $ curl -X GET http://$SERVER_ADDRESS/api/$VERSION/train/$job_id
    ```

##### Query hyper-parameters from a job

* Endpoint: http://$SERVER_ADDRESS/api/$VERSION/train/$job_id/hyper-parameters

* Method: GET

* Arguments: application/x-www-form-urlencoded
        
    * None
    
* Response: JSON

    <table>
        <thead>
            <tr>
                <th>For classification task</th>
                <th>For detection task</th>
                <th>For instance segmentation task</th>
            </tr>
        </thead>
        <tbody>
        </tbody>
            <tr>
                <td>
                    <pre><code class="json">
    {
      "result": "OK",
      "hyper-parameters": {
        "image_resized_width": 224,
        "image_resized_height": 224,
        "image_min_side": -1,
        "image_max_side": -1,
        "batch_size": 1,
        "learning_rate": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "step_lr_sizes": [20, 26],
        "step_lr_gamma": 0.1,
        "num_epochs_to_validate": 1,
        "num_epochs_to_finish": 30,
        "pretrained": true,
        "num_frozen_levels": 2,
        "eval_center_crop_ratio": 1
      }
    }
                    </code></pre>
                </td>
                <td>
                    <pre><code class="json">
    {
      "result": "OK",
      "hyper-parameters": {
        "image_resized_width": -1,
        "image_resized_height": -1,
        "image_min_side": 600,
        "image_max_side": 1000,
        "batch_size": 1,
        "learning_rate": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "step_lr_sizes": [20, 26],
        "step_lr_gamma": 0.1,
        "num_epochs_to_validate": 1,
        "num_epochs_to_finish": 30,
        "anchor_sizes": [128, 256, 512],
        "backbone_pretrained": true,
        "backbone_num_frozen_levels": 2,
        "proposal_nms_threshold": 0.7,
        "detection_nms_threshold": 0.5
      }
    }
                    </code></pre>
                </td>
                <td>
                    <pre><code class="json">
    {
      "result": "OK",
      "hyper-parameters": {
        "image_resized_width": -1,
        "image_resized_height": -1,
        "image_min_side": 600,
        "image_max_side": 1000,
        "batch_size": 1,
        "learning_rate": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "step_lr_sizes": [20, 26],
        "step_lr_gamma": 0.1,
        "num_epochs_to_validate": 1,
        "num_epochs_to_finish": 30
      }
    }
                    </code></pre>
                </td>
            </tr>
    </table>

    or 
    
    ```json
    {
      "result": "Error",
      "message": "Job not found."
    }
    ```
    
* Test:

    ```
    $ curl -X GET http://$SERVER_ADDRESS/api/$VERSION/train/$job_id/hyper-parameters
    ```

##### Query loss history from a training job

* Endpoint: http://$SERVER_ADDRESS/api/$VERSION/train/$job_id/losses

* Method: GET

* Arguments: application/x-www-form-urlencoded
        
    * None
    
* Response: JSON
    
    ```json
    {
      "result": "OK",
      "losses": [
        2.618614435195923,
        2.5296010971069336,
        2.402136961619059,
        2.271590903401375,
        2.208071458339691,
        2.133672147989273
      ]
    }
    ```
    
    or 
    
    ```json
    {
      "result": "Error",
      "message": "Job not found."
    }
    ```
    
* Test:

    ```
    $ curl -X GET http://$SERVER_ADDRESS/api/$VERSION/train/$job_id/losses
    ```

##### Retrieve a checkpoint from a training job

* Endpoint: http://$SERVER_ADDRESS/api/$VERSION/train/$job_id/checkpoints/epoch/$epoch

* Method: GET

* Arguments: application/x-www-form-urlencoded
        
    * None
    
* Response: JSON
    
    <table>
        <thead>
            <tr>
                <th>For classification task</th>
                <th>For detection task</th>
                <th>For instance segmentation task</th>
            </tr>
        </thead>
        <tbody>
        </tbody>
            <tr>
                <td>
                    <pre><code class="json">
    {
      "result": "OK",
      "checkpoint": {
        "epoch": 1,
        "avg_loss": 1.0336986339092256,
        "metrics": {
          "overall": {
            "accuracy": 0.2301
            "avg_recall": 0.1732,
            "avg_precision": 0.2256,
            "avg_f1_score": 0.1989,
          },
          "specific": {
            "categories": ["mean", "cat", "dog"],
            "aucs": [0.1345, 0.1521, 0.1163],
            "sensitivities": [0.1345, 0.1521, 0.1163],
            "specificities": [0.1345, 0.1521, 0.1163]
          }
        },
        "base64_metric_plot": "/9j/4AAQSk...",
        "is_best": 0,
        "is_available": 1
      }
    }
                    </code></pre>
                </td>
                <td>
                    <pre><code class="json">
    {
      "result": "OK",
      "checkpoint": {
        "epoch": 1,
        "avg_loss": 1.0336986339092256,
        "metrics": {
          "overall": {},
          "specific": {
            "categories": ["mean", "cat", "dog"],
            "aps": [0.1345, 0.1521, 0.1163],
            "f1_scores": [0.1345, 0.1521, 0.1163],
            "precisions": [0.1345, 0.1521, 0.1163],
            "recalls": [0.1345, 0.1521, 0.1163],
            "accuracies": [0.1345, 0.1521, 0.1163]
          }
        },
        "base64_metric_plot": "/9j/4AAQSk...",
        "is_best": 0,
        "is_available": 1
      }
    }
                    </code></pre>
                </td>
                <td>
                    <pre><code class="json">
    {
      "result": "OK",
      "checkpoint": {
        "epoch": 1,
        "avg_loss": 1.0336986339092256,
        "metrics": {
          "overall": {},
          "specific": {
            "categories": ["mean", "cat", "dog"],
            "aps": [0.1345, 0.1521, 0.1163]
          }
        },
        "base64_metric_plot": "/9j/4AAQSk...",
        "is_best": 0,
        "is_available": 1
      }
    }
                    </code></pre>
                </td>
            </tr>
    </table>
    
    or 
    
    ```json
    {
      "result": "Error",
      "message": "Job not found."
    }
    ```
    
* Test:

    ```
    $ curl -X GET http://$SERVER_ADDRESS/api/$VERSION/train/$job_id/checkpoints/epoch/$epoch
    ```

##### Retrieve all checkpoints from a training job

* Endpoint: http://$SERVER_ADDRESS/api/$VERSION/train/$job_id/checkpoints

* Method: GET

* Arguments: application/x-www-form-urlencoded
        
    * None
    
* Response: JSON

    <table>
        <thead>
            <tr>
                <th>For classification task</th>
                <th>For detection task</th>
                <th>For instance segmentation task</th>
            </tr>
        </thead>
        <tbody>
        </tbody>
            <tr>
                <td>
                    <pre><code class="json">
    {
      "result": "OK",
      "checkpoints": [
        {
          "epoch": 1,
          "avg_loss": 1.0336986339092256,
          "accuracy": 0.2301,
          "is_best": 0,
          "is_available": 0
        },
        {
          "epoch": 2,
          "avg_loss": 0.892367762029171,
          "accuracy": 0.3392,
          "is_best": 0,
          "is_available": 1
        },
        {
          "epoch": 3,
          "avg_loss": 0.8016865515708923,
          "accuracy": 0.4308,
          "is_best": 1,
          "is_available": 1
        }
      ]
    }
                    </code></pre>
                </td>
                <td>
                    <pre><code class="json">
    {
      "result": "OK",
      "checkpoints": [
        {
          "epoch": 1,
          "avg_loss": 1.0336986339092256,
          "mean_ap": 0.1345,
          "is_best": 0,
          "is_available": 0
        },
        {
          "epoch": 2,
          "avg_loss": 0.892367762029171,
          "mean_ap": 0.3392,
          "is_best": 0,
          "is_available": 1
        },
        {
          "epoch": 3,
          "avg_loss": 0.8016865515708923,
          "mean_ap": 0.4308,
          "is_best": 1,
          "is_available": 1
        }
      ]
    }
                    </code></pre>
                </td>
                <td>
                    <pre><code class="json">
    {
      "result": "OK",
      "checkpoints": [
        {
          "epoch": 1,
          "avg_loss": 1.0336986339092256,
          "mean_ap": 0.1345,
          "is_best": 0,
          "is_available": 0
        },
        {
          "epoch": 2,
          "avg_loss": 0.892367762029171,
          "mean_ap": 0.3392,
          "is_best": 0,
          "is_available": 1
        },
        {
          "epoch": 3,
          "avg_loss": 0.8016865515708923,
          "mean_ap": 0.4308,
          "is_best": 1,
          "is_available": 1
        }
      ]
    }
                    </code></pre>
                </td>
            </tr>
    </table>
    
    or 
    
    ```json
    {
      "result": "Error",
      "message": "Job not found."
    }
    ```
    
* Test:

    ```
    $ curl -X GET http://$SERVER_ADDRESS/api/$VERSION/train/$job_id/checkpoints
    ```

##### Obtain the plot of model graph

* Endpoint: http://$SERVER_ADDRESS/api/$VERSION/train/$job_id/plots/model-graph

* Method: GET

* Arguments: application/x-www-form-urlencoded
        
    * None
    
* Response: JSON
    
    ```json
    {
      "result": "OK",
      "base64_model_graph_plot": "/9j/4AAQSk..."
    }
    ```
    
    or 
    
    ```json
    {
      "result": "Error",
      "message": "Job not found."
    }
    ```
    
* Test:

    ```
    $ curl -X GET http://$SERVER_ADDRESS/api/$VERSION/train/$job_id/plots/model-graph
    ```

##### Obtain the plot of loss curve

* Endpoint: http://$SERVER_ADDRESS/api/$VERSION/train/$job_id/plots/loss

* Method: GET

* Arguments: application/x-www-form-urlencoded
        
    * None
    
* Response: JSON
    
    ```json
    {
      "result": "OK",
      "base64_loss_plot": "/9j/4AAQSk..."
    }
    ```
    
    or 
    
    ```json
    {
      "result": "Error",
      "message": "Job not found."
    }
    ```
    
* Test:

    ```
    $ curl -X GET http://$SERVER_ADDRESS/api/$VERSION/train/$job_id/plots/loss
    ```

##### Obtain the plot of confusion matrix

* Endpoint: http://$SERVER_ADDRESS/api/$VERSION/train/$job_id/plots/confusion-matrix/epoch/$epoch

* Method: GET

* Arguments: application/x-www-form-urlencoded
        
    * None
    
* Response: JSON
    
    ```json
    {
      "result": "OK",
      "base64_confusion_matrix_plot": "/9j/4AAQSk..."
    }
    ```
    
    or 
    
    ```json
    {
      "result": "Error",
      "message": "Job not found."
    }
    ```
    
* Test:

    ```
    $ curl -X GET http://$SERVER_ADDRESS/api/$VERSION/train/$job_id/plots/confusion-matrix/epoch/$epoch
    ```

##### Obtain the plot of threshold versus PR

* Endpoint: http://$SERVER_ADDRESS/api/$VERSION/train/$job_id/plots/threshold/epoch/$epoch/class/$class

    > $class should be in range of \[0, NUM_CLASSES\], where 0 stands for `mean`

* Method: GET

* Arguments: application/x-www-form-urlencoded
        
    * None
    
* Response: JSON
    
    ```json
    {
      "result": "OK",
      "base64_threshold_plot": "/9j/4AAQSk..."
    }
    ```
    
    or 
    
    ```json
    {
      "result": "Error",
      "message": "Job not found."
    }
    ```
    
* Test:

    ```
    $ curl -X GET http://$SERVER_ADDRESS/api/$VERSION/train/$job_id/plots/threshold/epoch/$epoch/class/$class
    ```

##### Infer an image

* Endpoint: http://$SERVER_ADDRESS/api/$VERSION/infer/$job_id

* Method: POST

* Arguments: application/x-www-form-urlencoded
        
    | Keyword                        | Example / Default   | Required  |
    |:-----------------------------: |:-------------------:|:---------:|
    | task                           | detection           | True      |
    | epoch                          | 30                  | True      |
    | lower_prob_thresh              | 0.7                 | False     |
    | upper_prob_thresh              | 1.0                 | False     |
    | device_ids                     | None                | False     |
    | returns_images                 | True                | False     |
    | path_to_image                  | /path/to/image      | False     |
    | base64_image                   | /9j/4AAQSk...       | False     |
    
    > `path_to_image` and `base64_image` must have exactly one value  

* Response: JSON
    
    <table>
        <thead>
            <tr>
                <th>For classification task</th>
                <th>For detection task</th>
                <th>For instance segmentation task</th>
            </tr>
        </thead>
        <tbody>
        </tbody>
            <tr>
                <td>
                    If `returns_images` is True:
                    <pre><code class="json">
    {
      "result": "OK",
      "base64_images": [
        "/9j/4AAQSk...",
        "/9j/4AAQSk..."
      ],
      "classification": {
        "category": "cat",
        "prob": 0.910342812538147
      }
    }
                    </code></pre>
                    If `returns_images` is False:
                    <pre><code class="json">
    {
      "result": "OK",
      "classification": {
        "category": "cat",
        "prob": 0.910342812538147
      }
    }
                    </code></pre>
                </td>
                <td>
                    If `returns_images` is True:
                    <pre><code class="json">
    {
      "result": "OK",
      "base64_images": [
        "/9j/4AAQSk...",
        "/9j/4AAQSk...",
        "/9j/4AAQSk...",
        "/9j/4AAQSk..."
      ],
      "detections": [
        {
          "bbox": [
            1.8327522277832031,
            102.76934814453125,
            288.8274230957031,
            257.7099304199219
          ],
          "category": "cat",
          "prob": 0.910342812538147
        },
        {
          "bbox": [
            76.39014434814453,
            82.31609344482422,
            358.691650390625,
            232.7666473388672
          ],
          "category": "cat",
          "prob": 0.6415448784828186
        }
      ]
    }
                    </code></pre>
                    If `returns_images` is False:
                    <pre><code class="json">
    {
      "result": "OK",
      "detections": [
        {
          "bbox": [
            1.8327522277832031,
            102.76934814453125,
            288.8274230957031,
            257.7099304199219
          ],
          "category": "cat",
          "prob": 0.910342812538147
        },
        {
          "bbox": [
            76.39014434814453,
            82.31609344482422,
            358.691650390625,
            232.7666473388672
          ],
          "category": "cat",
          "prob": 0.6415448784828186
        }
      ]
    }
                    </code></pre>
                </td>
                <td>
                    If `returns_images` is True:
                    <pre><code class="json">
    {
      "result": "OK",
      "base64_images": [
        "/9j/4AAQSk...",
        "/9j/4AAQSk...",
        "/9j/4AAQSk..."
      ],
      "instance_segmentations": [
        {
          "bbox": [
            1.8327522277832031,
            102.76934814453125,
            288.8274230957031,
            257.7099304199219
          ],
          "category": "cat",
          "prob": 0.910342812538147,
          "color": 1,
          "area": 184,
          "polygons": [
            {
              "x": [228, 306, 270, 272, 294],
              "y": [97, 129, 158, 169, 169]
            }
          ]
        },
        {
          "bbox": [
            76.39014434814453,
            82.31609344482422,
            358.691650390625,
            232.7666473388672
          ],
          "category": "cat",
          "prob": 0.6415448784828186,
          "color": 2,
          "area": 184,
          "polygons": [
            {
              "x": [228, 306, 270, 272, 294],
              "y": [97, 129, 158, 169, 169]
            }
          ]
        }
      ],
      "instance_segmentation_base64_map": "/9j/4AAQSk..."
    }
                    </code></pre>
                    If `returns_images` is False:
                    <pre><code class="json">
    {
      "result": "OK",
      "instance_segmentations": [
        {
          "bbox": [
            1.8327522277832031,
            102.76934814453125,
            288.8274230957031,
            257.7099304199219
          ],
          "category": "cat",
          "prob": 0.910342812538147,
          "color": 1,
          "area": 184,
          "polygons": [
            {
              "x": [228, 306, 270, 272, 294],
              "y": [97, 129, 158, 169, 169]
            }
          ]
        },
        {
          "bbox": [
            76.39014434814453,
            82.31609344482422,
            358.691650390625,
            232.7666473388672
          ],
          "category": "cat",
          "prob": 0.6415448784828186,
          "color": 2,
          "area": 184,
          "polygons": [
            {
              "x": [228, 306, 270, 272, 294],
              "y": [97, 129, 158, 169, 169]
            },
            {
              "x": [228, 306, 270, 272, 294],
              "y": [97, 129, 158, 169, 169]
            }
          ]
        }
      ],
      "instance_segmentation_base64_map": "/9j/4AAQSk..."
    }
                    </code></pre>
                </td>
            </tr>
    </table>
    
    or 
    
    ```json
    {
      "result": "Error",
      "message": "Job not found."
    }
    ```

    > `bbox` has format [left, top, right, bottom]
   
* Test:

    ```
    $ curl -X POST -d task=detection -d epoch=$epoch -d path_to_image=./examples/CatDog/images/000037.jpg http://$SERVER_ADDRESS/api/$VERSION/infer/$job_id
    $ curl -X POST -d task=detection -d epoch=$epoch --data-urlencode base64_image="/9j/4AAQSk..." http://$SERVER_ADDRESS/api/$VERSION/infer/$job_id
    ```


##### Infer multiple images

* Endpoint: http://$SERVER_ADDRESS/api/$VERSION/infer_multiple/$job_id

* Method: POST

* Arguments: application/x-www-form-urlencoded
        
    | Keyword                        | Example / Default                        | Required  |
    |:-----------------------------: |:----------------------------------------:|:---------:|
    | task                           | detection                                | True      |
    | epoch                          | 30                                       | True      |
    | lower_prob_thresh              | 0.7                                      | False     |
    | upper_prob_thresh              | 1.0                                      | False     |
    | device_ids                     | None                                     | False     |
    | path_to_image_pattern_list     | ['/path/to/A/*.jpg', '/path/to/B/1.png'] | True      |

* Response: JSON

    <table>
        <thead>
            <tr>
                <th>For classification task</th>
                <th>For detection task</th>
                <th>For instance segmentation task</th>
            </tr>
        </thead>
        <tbody>
        </tbody>
            <tr>
                <td>
                    <pre><code class="json">
    {
      "result": "OK",
      "images": [
        {
          "path": "/path/to/A/1.jpg",
          "classification": {
            "category": "dog",
            "prob": 0.6503969430923462
          }
        },
        {
          "path": "/path/to/B/1.jpg",
          "classification": {
            "category": "cat",
            "prob": 0.8297173976898193
          }
        }
      ]
    }
                    </code></pre>
                </td>
                <td>
                    <pre><code class="json">
    {
      "result": "OK",
      "images": [
        {
          "path": "/path/to/A/1.jpg",
          "detections": [
            {
              "bbox": [
                31.642398834228516,
                226.52566528320312,
                212.12637329101562,
                425.2940673828125
              ],
              "category": "dog",
              "prob": 0.6503969430923462
            }
          ]
        },
        {
          "path": "/path/to/B/1.jpg",
          "detections": [
            {
              "bbox": [
                8.502140045166016,
                58.40647888183594,
                421.7301940917969,
                266.6134338378906
              ],
              "category": "cat",
              "prob": 0.8297173976898193
            },
            {
              "bbox": [
                243.823974609375,
                30.574464797973633,
                499.5306701660156,
                264.3441162109375
              ],
              "category": "cat",
              "prob": 0.6324196457862854
            }
          ]
        }
      ]
    }
                    </code></pre>
                </td>
                <td>
                    <pre><code class="json">
    {
      "result": "OK",
      "images": [
        {
          "path": "/path/to/A/1.jpg",
          "instance_segmentations": [
            {
              "bbox": [
                31.642398834228516,
                226.52566528320312,
                212.12637329101562,
                425.2940673828125
              ],
              "category": "dog",
              "prob": 0.6503969430923462,
              "color": 1,
              "area": 184,
              "polygons": [
                {
                  "x": [228, 306, 270, 272, 294],
                  "y": [97, 129, 158, 169, 169]
                }
              ]
            }
          ],
          "instance_segmentation_base64_map": "/9j/4AAQSk..."
        },
        {
          "path": "/path/to/B/1.jpg",
          "instance_segmentations": [
            {
              "bbox": [
                8.502140045166016,
                58.40647888183594,
                421.7301940917969,
                266.6134338378906
              ],
              "category": "cat",
              "prob": 0.8297173976898193,
              "color": 1,
              "area": 184,
              "polygons": [
                {
                  "x": [228, 306, 270, 272, 294],
                  "y": [97, 129, 158, 169, 169]
                }
              ]
            },
            {
              "bbox": [
                243.823974609375,
                30.574464797973633,
                499.5306701660156,
                264.3441162109375
              ],
              "category": "cat",
              "prob": 0.6324196457862854,
              "color": 2,
              "area": 184,
              "polygons": [
                {
                  "x": [228, 306, 270, 272, 294],
                  "y": [97, 129, 158, 169, 169]
                },
                {
                  "x": [228, 306, 270, 272, 294],
                  "y": [97, 129, 158, 169, 169]
                }
              ]
            }
          ],
          "instance_segmentation_base64_map": "/9j/4AAQSk..."
        }
      ]
    }
                    </code></pre>
                </td>
            </tr>
    </table>
    
    or 
    
    ```json
    {
      "result": "Error",
      "message": "Job not found."
    }
    ```

    > `bbox` has format [left, top, right, bottom]
   
* Test:

    ```
    $ curl -X POST -d task=detection -d epoch=$epoch -d path_to_image_pattern_list="['./examples/CatDog/images/*.jpg', './examples/Person/images/000001.jpg']" http://$SERVER_ADDRESS/api/$VERSION/infer_multiple/$job_id
    ```


##### List all deployed checkpoints

* Endpoint: http://$SERVER_ADDRESS/api/$VERSION/deploy/list

* Method: GET

* Arguments: application/x-www-form-urlencoded
        
    * None
    
* Response: JSON
    
    ```json
    {
      "result": "OK",
      "deployments": [
        {
          "job_id": "00000001",
          "epoch": 3,
          "target_device_id": 0      
        },
        {
          "job_id": "00000002",
          "epoch": 10,
          "target_device_id": 1      
        }
      ]
    }
    ```
    
* Test:

    ```
    $ curl -X GET http://$SERVER_ADDRESS/api/$VERSION/deploy/list
    ```


##### Deploy a checkpoint

* Endpoint: http://$SERVER_ADDRESS/api/$VERSION/deploy/$job_id

* Method: POST

* Arguments: application/x-www-form-urlencoded
        
    | Keyword                        | Example / Default   | Required  |
    |:-----------------------------: |:-------------------:|:---------:|
    | task                           | detection           | True      |
    | epoch                          | 30                  | True      |
    | target_device_id               | 1                   | True      |
    
    > `target_device_id` can be either -1 or in range of \[0, GPU_COUNT\), where -1 indicates CPU    
    
* Response: JSON
    
    ```json
    {
      "result": "OK"
    }
    ```
    
    or 
    
    ```json
    {
      "result": "Error",
      "message": "Job not found."
    }
    ```
    
* Test:

    ```
    $ curl -X POST -d task=detection -d epoch=$epoch -d target_device_id=0 http://$SERVER_ADDRESS/api/$VERSION/deploy/$job_id
    ```


##### Undeploy a checkpoint

* Endpoint: http://$SERVER_ADDRESS/api/$VERSION/deploy/$job_id

* Method: DELETE

* Arguments: application/x-www-form-urlencoded
        
    | Keyword                        | Example / Default   | Required  |
    |:-----------------------------: |:-------------------:|:---------:|
    | epoch                          | 30                  | True      |
    | target_device_id               | 1                   | True      |
    
    > `target_device_id` can be either -1 or in range of \[0, GPU_COUNT\), where -1 indicates CPU    
    
* Response: JSON
    
    ```json
    {
      "result": "OK"
    }
    ```
    
    or 
    
    ```json
    {
      "result": "Error",
      "message": "Job not found."
    }
    ```
    
* Test:

    ```
    $ curl -X DELETE -d task=detection -d epoch=$epoch -d target_device_id=0 http://$SERVER_ADDRESS/api/$VERSION/deploy/$job_id
    ```


##### Upgrade a job

* Endpoint: http://$SERVER_ADDRESS/api/$VERSION/upgrade/$job_id

* Method: POST

* Arguments: application/x-www-form-urlencoded
        
    | Keyword                        | Example / Default                        | Required  |
    |:-----------------------------: |:----------------------------------------:|:---------:|
    | task                           | detection                                | True      |

* Response: JSON
    
    ```json
    {
      "result": "OK",
      "upgraded": true,
      "exception": null
    }
    ```
  
    or
    
    ```json
    {
      "result": "OK",
      "upgraded": false,
      "exception": {
        "type": "ValueError",
        "message": "The version under 1.1.0 is not upgradable."
      }
    }
    ```
    
    or 
    
    ```json
    {
      "result": "Error",
      "message": "Job not found."
    }
    ```
   
* Test:

    ```
    $ curl -X POST -d task=detection http://$SERVER_ADDRESS/api/$VERSION/upgrade/$job_id
    ```


#### Preview data augmentation

* Endpoint: http://$SERVER_ADDRESS/api/$VERSION/aug

* Method: POST

* Arguments: application/x-www-form-urlencoded
        
    | Keyword                                     | Example / Default   | Required  |
    |:-------------------------------------------:|:-------------------:|:---------:|
    | task                                        | detection           | True      |
    | data_dir                                    | /path/to/dataset    | True      |
    | index                                       | 0                   | True      |
    | aug_strategy                                | all                 | False     |
    | aug_hflip_prob                              | 0.5                 | False     |
    | aug_vflip_prob                              | 0                   | False     |
    | aug_rotate90_prob                           | 0                   | False     |
    | aug_crop_prob_and_min_max                   | (0, (0, 1))         | False     |
    | aug_zoom_prob_and_min_max                   | (0, (-1, 1))        | False     |
    | aug_scale_prob_and_min_max                  | (0, (-1, 1))        | False     |
    | aug_translate_prob_and_min_max              | (0, (-1, 1))        | False     |
    | aug_rotate_prob_and_min_max                 | (0, (-1, 1))        | False     |
    | aug_shear_prob_and_min_max                  | (0, (-1, 1))        | False     |
    | aug_blur_prob_and_min_max                   | (0, (0, 1))         | False     |
    | aug_sharpen_prob_and_min_max                | (0, (0, 1))         | False     |
    | aug_color_prob_and_min_max                  | (0, (-1, 1))        | False     |
    | aug_brightness_prob_and_min_max             | (0, (-1, 1))        | False     |
    | aug_grayscale_prob_and_min_max              | (0, (0, 1))         | False     |
    | aug_contrast_prob_and_min_max               | (0, (-1, 1))        | False     |
    | aug_noise_prob_and_min_max                  | (0, (0, 1))         | False     |
    | aug_resized_crop_prob_and_width_height      | (0, (224, 224))     | False     |

* Response: JSON
    
    ```json
    {
      "result": "OK",
      "base64_aug_image": "/9j/4AAQSk..."
    }
    ```
    
* Test:

    ```
    $ curl -X POST -d task=detection -d data_dir=./examples/CatDog -d index=0 -d aug_strategy=all -d aug_hflip_prob=0.5 -d aug_vflip_prob=0.5 -d aug_rotate_prob_and_min_max="(0.5, (0, 1))" -d aug_noise_prob_and_min_max="(0.5, (0, 1))" http://$SERVER_ADDRESS/api/$VERSION/aug
    ```


##### Summarize a checkpoint

* Endpoint: http://$SERVER_ADDRESS/api/$VERSION/summarize/$job_id

* Method: POST

* Arguments: application/x-www-form-urlencoded
        
    | Keyword                        | Example / Default   | Required  |
    |:-----------------------------: |:-------------------:|:---------:|
    | task                           | classification      | True      |
    | epoch                          | 2                   | True      |
    
* Response: JSON
    
    ```json
    {
      "result": "OK",
      "flops": "8.219G",
      "params": "23.514M",
      "param_keys": [
          "algorithm.net.conv1.weight",
          "algorithm.net.bn1.weight",
          "algorithm.net.bn1.bias",
          "algorithm.net.layer1.0.conv1.weight"
      ]
    }
    ```
    
    or 
    
    ```json
    {
      "result": "Error",
      "message": "Job not found."
    }
    ```
    
* Test:

    ```
    $ curl -X POST -d task=classification -d epoch=$epoch http://$SERVER_ADDRESS/api/$VERSION/summarize/$job_id
    ```


##### Visualize a parameter from a checkpoint

* Endpoint: http://$SERVER_ADDRESS/api/$VERSION/visualize/$job_id

* Method: POST

* Arguments: application/x-www-form-urlencoded
        
    | Keyword                        | Example / Default          | Required  |
    |:-----------------------------: |:--------------------------:|:---------:|
    | task                           | classification             | True      |
    | epoch                          | 2                          | True      |
    | param_key                      | algorithm.net.conv1.weight | True      |
    
* Response: JSON
    
    ```json
    {
      "result": "OK",
      "base64_image": "/9j/4AAQSk..."
    }
    ```
    
    or 
    
    ```json
    {
      "result": "Error",
      "message": "Key 'algorithm.net.conv1.weight1' does not exist"
    }
    ```
    
    or 
    
    ```json
    {
      "result": "Error",
      "message": "Job not found."
    }
    ```
    
* Test:

    ```
    $ curl -X POST -d task=classification -d epoch=$epoch -d param_key=algorithm.net.conv1.weight http://$SERVER_ADDRESS/api/$VERSION/visualize/$job_id
    ```
