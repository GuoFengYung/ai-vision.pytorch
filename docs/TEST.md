## Test


### Setup

> Be sure that web service has launched

1. Generate CIFAR-10 dataset by following the instruction of `From CIFAR-10 dataset` in [Dataset Converter](../tools/dataset-converter/README.md)

    > Or download converted CIFAR-10 dataset from [here](https://drive.google.com/open?id=1Ht51oY-9vNqF7mKrt4qPONiS0FTurzEZ) and extract it

1. Generate CIFAR-100 dataset by following the instruction of `From CIFAR-100 dataset` in [Dataset Converter](../tools/dataset-converter/README.md)

1. Generate VOC2007 dataset by following the instruction of `From VOC2007 dataset` in [Dataset Converter](../tools/dataset-converter/README.md)

    > Or download converted VOC2007 dataset from [here](https://drive.google.com/open?id=1Lfhwb2dEdcK6kyxY-tNm5zTRKjDOR8R4) and extract it

1. Generate COCO2017 dataset by following the instruction of `From COCO2017 dataset` in [Dataset Converter](../tools/dataset-converter/README.md)

1. Put above data under `./data`

    ```
    - mirle-vision.pytorch
        - data
            - CIFAR-10
            - CIFAR-100
            - VOC2007
            - COCO2017
    ```


### Unit Test

* For classification task

    * Test training with resuming checkpoint
    
        ```
        $ python -m unittest tests.classification.train_resume_test_case.TrainResumeTestCase
        ```
    
    * Test training with fine-tuning checkpoint
    
        ```
        $ python -m unittest tests.classification.train_finetune_test_case.TrainFinetuneTestCase
        ```
      
    * Test Web APIs
    
        * Manually
    
            ```
            $ SERVER_ADDRESS=127.0.0.1:5000
            $ curl -X GET http://$SERVER_ADDRESS/api/version
            $ VERSION=v1.5.0
            $ curl -X GET http://$SERVER_ADDRESS/api/$VERSION/environment
            $ curl -X GET http://$SERVER_ADDRESS/api/$VERSION/devices
            $ curl -X POST -d task=classification -d data_dir=./examples/CatDog -d visible_devices="[0]" -d algorithm=resnet50 http://$SERVER_ADDRESS/api/$VERSION/train
            $ job_id=xxx
            $ curl -X PATCH http://$SERVER_ADDRESS/api/$VERSION/train/$job_id
            $ curl -X PUT http://$SERVER_ADDRESS/api/$VERSION/train/$job_id
            $ curl -X PATCH http://$SERVER_ADDRESS/api/$VERSION/train/$job_id
            $ curl -X GET http://$SERVER_ADDRESS/api/$VERSION/train/$job_id
            $ curl -X GET http://$SERVER_ADDRESS/api/$VERSION/train/$job_id/hyper-parameters
            $ curl -X GET http://$SERVER_ADDRESS/api/$VERSION/train/$job_id/losses
            $ curl -X GET http://$SERVER_ADDRESS/api/$VERSION/train/$job_id/checkpoints
            $ epoch=xxx
            $ curl -X GET http://$SERVER_ADDRESS/api/$VERSION/train/$job_id/checkpoints/epoch/$epoch
            $ curl -X GET http://$SERVER_ADDRESS/api/$VERSION/train/$job_id/plots/loss
            $ curl -X POST -d task=classification -d epoch=$epoch -d path_to_image=./examples/CatDog/images/000037.jpg http://$SERVER_ADDRESS/api/$VERSION/infer/$job_id
            $ curl -X POST -d task=classification -d epoch=$epoch -d path_to_image_pattern_list="['./examples/CatDog/images/*.jpg', './examples/Person/images/000001.jpg']" http://$SERVER_ADDRESS/api/$VERSION/infer_multiple/$job_id
            $ curl -X POST -d task=classification -d data_dir=./examples/CatDog -d index=0 -d aug_strategy=all -d aug_hflip_prob=0.5 -d aug_vflip_prob=0.5 -d aug_rotate_prob_and_min_max="(0.5, (0, 1))" -d aug_noise_prob_and_min_max="(0.5, (0, 1))" http://$SERVER_ADDRESS/api/$VERSION/aug
            $ curl -X DELETE http://$SERVER_ADDRESS/api/$VERSION/train/$job_id
            ```
    
        * Automatically
        
            ```
            $ python -m unittest tests.classification.web_api_test_case.WebAPITestCase
            ```

* For detection task

    * Test training with resuming checkpoint
    
        ```
        $ python -m unittest tests.detection.train_resume_test_case.TrainResumeTestCase
        ```
    
    * Test training with fine-tuning checkpoint
    
        ```
        $ python -m unittest tests.detection.train_finetune_test_case.TrainFinetuneTestCase
        ```
      
    * Test evaluation consistency
    
        ```
        $ python -m unittest tests.detection.eval_consistency_test_case.EvalConsistencyTestCase
        ```
      
    * Test Web APIs
    
        * Manually
    
            ```
            $ SERVER_ADDRESS=127.0.0.1:5000
            $ curl -X GET http://$SERVER_ADDRESS/api/version
            $ VERSION=v1.5.0
            $ curl -X GET http://$SERVER_ADDRESS/api/$VERSION/environment
            $ curl -X GET http://$SERVER_ADDRESS/api/$VERSION/devices
            $ curl -X POST -d task=detection -d data_dir=./examples/CatDog -d visible_devices="[0]" -d algorithm=faster_rcnn -d backbone=resnet18 http://$SERVER_ADDRESS/api/$VERSION/train
            $ job_id=xxx
            $ curl -X PATCH http://$SERVER_ADDRESS/api/$VERSION/train/$job_id
            $ curl -X PUT http://$SERVER_ADDRESS/api/$VERSION/train/$job_id
            $ curl -X PATCH http://$SERVER_ADDRESS/api/$VERSION/train/$job_id
            $ curl -X GET http://$SERVER_ADDRESS/api/$VERSION/train/$job_id
            $ curl -X GET http://$SERVER_ADDRESS/api/$VERSION/train/$job_id/hyper-parameters
            $ curl -X GET http://$SERVER_ADDRESS/api/$VERSION/train/$job_id/losses
            $ curl -X GET http://$SERVER_ADDRESS/api/$VERSION/train/$job_id/checkpoints
            $ epoch=xxx
            $ curl -X GET http://$SERVER_ADDRESS/api/$VERSION/train/$job_id/checkpoints/epoch/$epoch
            $ curl -X GET http://$SERVER_ADDRESS/api/$VERSION/train/$job_id/plots/loss
            $ curl -X POST -d task=detection -d epoch=$epoch -d path_to_image=./examples/CatDog/images/000037.jpg http://$SERVER_ADDRESS/api/$VERSION/infer/$job_id
            $ curl -X POST -d task=detection -d epoch=$epoch -d path_to_image_pattern_list="['./examples/CatDog/images/*.jpg', './examples/Person/images/000001.jpg']" http://$SERVER_ADDRESS/api/$VERSION/infer_multiple/$job_id
            $ curl -X POST -d task=detection -d data_dir=./examples/CatDog -d index=0 -d aug_strategy=all -d aug_hflip_prob=0.5 -d aug_vflip_prob=0.5 -d aug_rotate_prob_and_min_max="(0.5, (0, 1))" -d aug_noise_prob_and_min_max="(0.5, (0, 1))" http://$SERVER_ADDRESS/api/$VERSION/aug
            $ curl -X DELETE http://$SERVER_ADDRESS/api/$VERSION/train/$job_id
            ```
        
        * Automatically
        
            ```
            $ python -m unittest tests.detection.web_api_test_case.WebAPITestCase
            ```

* For instance segmentation task

    * Test training with resuming checkpoint
    
        ```
        $ python -m unittest tests.instance_segmentation.train_resume_test_case.TrainResumeTestCase
        ```
    
    * Test training with fine-tuning checkpoint
    
        ```
        $ python -m unittest tests.instance_segmentation.train_finetune_test_case.TrainFinetuneTestCase
        ```
      
    * Test Web APIs
    
        * Manually
    
            ```
            $ SERVER_ADDRESS=127.0.0.1:5000
            $ curl -X GET http://$SERVER_ADDRESS/api/version
            $ VERSION=v1.5.0
            $ curl -X GET http://$SERVER_ADDRESS/api/$VERSION/environment
            $ curl -X GET http://$SERVER_ADDRESS/api/$VERSION/devices
            $ curl -X POST -d task=instance_segmentation -d data_dir=./examples/CatDog -d visible_devices="[0]" -d algorithm=mask_rcnn http://$SERVER_ADDRESS/api/$VERSION/train
            $ job_id=xxx
            $ curl -X PATCH http://$SERVER_ADDRESS/api/$VERSION/train/$job_id
            $ curl -X PUT http://$SERVER_ADDRESS/api/$VERSION/train/$job_id
            $ curl -X PATCH http://$SERVER_ADDRESS/api/$VERSION/train/$job_id
            $ curl -X GET http://$SERVER_ADDRESS/api/$VERSION/train/$job_id
            $ curl -X GET http://$SERVER_ADDRESS/api/$VERSION/train/$job_id/hyper-parameters
            $ curl -X GET http://$SERVER_ADDRESS/api/$VERSION/train/$job_id/losses
            $ curl -X GET http://$SERVER_ADDRESS/api/$VERSION/train/$job_id/checkpoints
            $ epoch=xxx
            $ curl -X GET http://$SERVER_ADDRESS/api/$VERSION/train/$job_id/checkpoints/epoch/$epoch
            $ curl -X GET http://$SERVER_ADDRESS/api/$VERSION/train/$job_id/plots/loss
            $ curl -X POST -d task=instance_segmentation -d epoch=$epoch -d path_to_image=./examples/CatDog/images/000037.jpg http://$SERVER_ADDRESS/api/$VERSION/infer/$job_id
            $ curl -X POST -d task=instance_segmentation -d epoch=$epoch -d path_to_image_pattern_list="['./examples/CatDog/images/*.jpg', './examples/Person/images/000001.jpg']" http://$SERVER_ADDRESS/api/$VERSION/infer_multiple/$job_id
            $ curl -X POST -d task=instance_segmentation -d data_dir=./examples/CatDog -d index=0 -d aug_strategy=all -d aug_hflip_prob=0.5 -d aug_vflip_prob=0.5 -d aug_rotate_prob_and_min_max="(0.5, (0, 1))" -d aug_noise_prob_and_min_max="(0.5, (0, 1))" http://$SERVER_ADDRESS/api/$VERSION/aug
            $ curl -X DELETE http://$SERVER_ADDRESS/api/$VERSION/train/$job_id
            ```
        
        * Automatically
        
            ```
            $ python -m unittest tests.instance_segmentation.web_api_test_case.WebAPITestCase
            ```
    

### Integration Test

* For classification task

    * Test training a model with ResNet-50 algorithm over CIFAR-10 dataset
    
        ```
        $ bash ./tests/classification/python_resnet50_cifar10_test.sh
        ```

    * Test training a model with EfficientNet-B7 algorithm over CIFAR-10 dataset
    
        ```
        $ bash ./tests/classification/python_efficientnet_b7_cifar10_test.sh
        ```

* For detection task

    * Test training a model with Faster RCNN algorithm and ResNet-50 backbone over VOC2007 dataset
    
        ```
        $ bash ./tests/detection/python_faster_rcnn_voc2007_test.sh
        ```
    
    * Test training a model with FPN algorithm and ResNet-50 backbone over VOC2007 dataset
    
        ```
        $ bash ./tests/detection/python_fpn_voc2007_test.sh
        ```
            
* For instance segmentation task
    
    * Test training a model with Mask RCNN algorithm over VOC2007 dataset
    
        ```
        $ bash ./tests/instance_segmentation/python_mask_rcnn_voc2007_test.sh
        ```


### Continuous Integration Test using TeamCity

> Assume that 2 GPUs with ~11 GB memory were ready

> Note that `Version Control Settings` has to be set up

1. Daily Build

    1. Examples Test:
    
        ```
        nvidia-smi
        %env.PYTHON_HOME%/python ./tests/classification/ci_python_examples_test.py
        nvidia-smi
        %env.PYTHON_HOME%/python ./tests/detection/ci_python_faster_rcnn_examples_test.py
        nvidia-smi
        %env.PYTHON_HOME%/python ./tests/detection/ci_python_fpn_examples_test.py
        nvidia-smi
        %env.PYTHON_HOME%/python ./tests/instance_segmentation/ci_python_mask_rcnn_examples_test.py
        ```
       
    1. CIFAR-10 and VOC2007 Test:
    
        ```
        nvidia-smi
        %env.PYTHON_HOME%/python ./tests/classification/ci_python_resnet50_cifar10_test.py %env.PROJECT_ROOT%/data/CIFAR-10
        nvidia-smi
        %env.PYTHON_HOME%/python ./tests/classification/ci_python_efficientnet_b7_cifar10_test.py %env.PROJECT_ROOT%/data/CIFAR-10
        nvidia-smi
        %env.PYTHON_HOME%/python ./tests/detection/ci_python_faster_rcnn_voc2007_test.py %env.PROJECT_ROOT%/data/VOC2007
        nvidia-smi
        %env.PYTHON_HOME%/python ./tests/detection/ci_python_fpn_voc2007_test.py %env.PROJECT_ROOT%/data/VOC2007
        nvidia-smi
        %env.PYTHON_HOME%/python ./tests/instance_segmentation/ci_python_mask_rcnn_voc2007_test.py %env.PROJECT_ROOT%/data/VOC2007
        ```
       
    1. Parse Log to Report
    
        ```
        sleep 60  # wait log generated for 1 minute
        curl -u "%system.teamcity.auth.userId%:%system.teamcity.auth.password%" "%teamcity.serverUrl%/httpAuth/downloadBuildLog.html?buildId=%teamcity.build.id%" > "/tmp/teamcity_build_%teamcity.build.id%.log"
        %env.PYTHON_HOME%/python ./tools/dev-ops/parse_teamcity_build_log.py "/tmp/teamcity_build_%teamcity.build.id%.log"
        ```
        
        > The latest build log can be found in `/tmp/teamcity_build_xxx.log`

1. Weekly Build

    1. CIFAR-100 and COCO2017 Test:
    
        ```
        nvidia-smi
        %env.PYTHON_HOME%/python ./tests/classification/ci_python_resnet101_cifar100_test.py %env.PROJECT_ROOT%/data/CIFAR-100
        nvidia-smi
        %env.PYTHON_HOME%/python ./tests/detection/ci_python_fpn_coco2017_test.py %env.PROJECT_ROOT%/data/COCO2017
        nvidia-smi
        %env.PYTHON_HOME%/python ./tests/instance_segmentation/ci_python_mask_rcnn_coco2017_test.py %env.PROJECT_ROOT%/data/COCO2017
        ```
       
    1. Parse Log to Report
    
        ```
        sleep 60  # wait log generated for 1 minute
        curl -u "%system.teamcity.auth.userId%:%system.teamcity.auth.password%" "%teamcity.serverUrl%/httpAuth/downloadBuildLog.html?buildId=%teamcity.build.id%" > "/tmp/teamcity_build_%teamcity.build.id%.log"
        %env.PYTHON_HOME%/python ./tools/dev-ops/parse_teamcity_build_log.py "/tmp/teamcity_build_%teamcity.build.id%.log"
        ```
       
        > The latest build log can be found in `/tmp/teamcity_build_xxx.log`
