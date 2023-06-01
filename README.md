# ai-vision.pytorch

> The following instructions are made for **Development Mode**, to see **Production Mode** visit [DEPLOYMENT.md](docs/DEPLOYMENT.md)

## Requirements

* System Libraries

    * CUDA 11.1
    
    * cuDNN 8.0.3
    
    * Python 3.7 with conda environment
    
        ```
        $ conda create -n mvision python=3.7
        ```
      
    * Graphviz
    
        * Ubuntu 18.04 with x84_64 Architecture
        
            ```
            $ sudo apt install graphviz
            ```
    
        * Ubuntu 18.04 with ppc64le Architecture
        
            ```
            $ sudo apt install graphviz
            ```
    
        * CentOS 7 with x86_64 Architecture
        
            ```
            $ sudo yum install graphviz
            ```

* Python Libraries

    * Ubuntu 18.04 with x84_64 Architecture
    
        ```
        $ CUDA_VERSION=11.1
        $ conda install pytorch=1.10.0 torchvision=0.11.1 cudatoolkit=$CUDA_VERSION numpy pillow -c pytorch -c nvidia
        $ pip install matplotlib opencv-python opencv-contrib-python requests tqdm flask flask-wtf nvidia-ml-py3 pretrainedmodels efficientnet_pytorch resnest scikit-learn scikit-image imgaug albumentations cython pycocotools seaborn tensorboard thop graphviz lmdb
        ```
    
    * Ubuntu 18.04 with ppc64le Architecture
    
        ```
        $ pip install numpy
        $ conda install pillow
        # Install `PyTorch 1.10.0` and `torchvision 0.11.1` from source
        # Install `OpenCV` and `OpenCV Contrib` from source
        $ pip install matplotlib requests tqdm flask flask-wtf nvidia-ml-py3 pretrainedmodels efficientnet_pytorch resnest
        $ conda install scikit-learn scikit-image
        $ pip install imgaug --no-dependencies
        $ pip install albumentations
        $ pip install cython pycocotools
        $ pip install seaborn tensorboard thop graphviz lmdb
        ```
      
    * CentOS 7 with x86_64 Architecture
    
        ```
        $ CUDA_VERSION=11.1
        $ conda install pytorch=1.10.0 torchvision=0.11.1 cudatoolkit=$CUDA_VERSION numpy pillow -c pytorch -c nvidia
        $ pip install matplotlib opencv-python opencv-contrib-python requests tqdm flask flask-wtf nvidia-ml-py3 pretrainedmodels efficientnet_pytorch resnest scikit-learn scikit-image imgaug albumentations cython pycocotools seaborn tensorboard thop graphviz lmdb
        ```
      
    > If error like `The function is not implemented. ..., install libgtk..., then re-run cmake...` has occurred, try to install OpenCV by conda:
    > ```
    > $ conda install opencv -c conda-forge
    > ```
    > or re-install after installing `libgtk2.0-dev`, e.g.:
    > ```
    > $ sudo apt install libgtk2.0-dev  # for Ubuntu
    > ```
    

## Setup

1. Install package

    1. Install

        ```
        $ python ./setup.py develop
        ```

    1. Uninstall and clean

        ```
        $ python ./setup.py develop --uninstall
        $ rm -rf ./src/aibox_vision.egg-info
        ```

1. Prepare data

    * Make your folder structure as below:
    
        ```
        - /path/to/dataset
            - images
                00000001.jpg
                00000002.jpg
                ...
            - annotations
                00000001.xml
                00000002.xml
                ...
            - splits
                train.txt
                val.txt
                test.txt
            - segmentations
                00000001.png
                00000002.png
                ...
            meta.json
        ```
        
    * Annotation Format

        ```xml
        <annotation>
            <filename>00000001.jpg</filename>
            <size>
                <width>353</width>
                <height>500</height>
                <depth>3</depth>
            </size>
            <category>dog</category>
            <object>
                <name>dog</name>
                <difficult>0</difficult>
                <bbox>
                    <left>48</left>
                    <top>240</top>
                    <right>195</right>
                    <bottom>371</bottom>
                </bbox>
                <mask>
                    <color>1</color>
                </mask>
            </object>
        </annotation>
        ```
        
        or

        ```xml
        <annotation>
            <filename>00000001.jpg</filename>
            <size>
                <width>353</width>
                <height>500</height>
                <depth>3</depth>
            </size>
            <object>
                <name>dog</name>
                <difficult>0</difficult>
                <bbox>
                    <left>48</left>
                    <top>240</top>
                    <right>195</right>
                    <bottom>371</bottom>
                </bbox>
                <mask>
                    <color>1</color>
                </mask>
            </object>
            <object>
                <name>cat</name>
                <difficult>1</difficult>
                <bbox>
                    <left>8</left>
                    <top>12</top>
                    <right>352</right>
                    <bottom>498</bottom>
                </bbox>
                <mask>
                    <color>2</color>
                </mask>
            </object>
        </annotation>
        ```
        
        > `filename` and `size` tags are required
        
        > `name` must not be conflicted with reserved keywords: `background` and `mean`
        
        > note that `right` and `bottom` are excluded from object pixels, i.e., an object with bounding box `(l, t, r, b) = (1, 2, 3, 4)` should occupy pixels `{(x, y)} = {(1, 2), (2, 2), (1, 3), (2, 3)}`
        
        > `difficult` indicates whether the object is too difficult to detect, for example: a person in a crowd or a mostly truncated sofa, this is typically subjective.

    * Segmentation Format
    
        * Segmentation images should be in 8-bit color PNG format and have a consistent size with the corresponding image
        
        * Each color belongs to a single object (instance-level) or a particular class (semantic-level), note that `0` is reserved for the `background`
        
        * Sample as below
        
            ![Segmentation Sample](docs/images/segmentation-sample.png)

    * Split Format
    
        ```
        00000001.jpg
        00000002.jpg
        ```
        
        > each line contains an image filename
        
    * Metadata Format
    
        ```json
        {
          "background": 0,
          "cat": 1,
          "dog": 2
        }
        ```
        
        > `background` must be associated with class 0
        
1. Set working directory

    * For some IDEs (such as PyCharm), set working directory to `/path/to/mirle-vision.pytorch` for all configurations
    
    * For terminal, it's recommended to switch current directory to `/path/to/mirle-vision.pytorch`
    
1. Pre-download pre-trained model if you want

    | Name                       | URL                                                                                                    |
    |:--------------------------:|:-------------------------------------------------------------------------------------------------------|
    | EfficientNet_B0            | https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth   |
    | EfficientNet_B1            | https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth   |
    | EfficientNet_B2            | https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth   |
    | EfficientNet_B3            | https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth   |
    | EfficientNet_B4            | https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth   |
    | EfficientNet_B5            | https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth   |
    | EfficientNet_B6            | https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth   |
    | EfficientNet_B7            | https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth   |
    | GoogLeNet                  | https://download.pytorch.org/models/googlenet-1378be20.pth                                             |
    | Inception_v3               | https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth                                   |
    | ResNet18                   | https://download.pytorch.org/models/resnet18-5c106cde.pth                                              |
    | ResNet34                   | https://download.pytorch.org/models/resnet34-333f7ec4.pth                                              |
    | ResNet50                   | https://download.pytorch.org/models/resnet50-19c8e357.pth                                              |
    | ResNet101                  | https://download.pytorch.org/models/resnet101-5d3b4d8f.pth                                             |
    | ResNet152                  | https://download.pytorch.org/models/resnet152-b121ed2d.pth                                             |
    | ResNeSt50                  | https://hangzh.s3.amazonaws.com/encoding/models/resnest50-528c19ca.pth                                 |
    | ResNeSt101                 | https://hangzh.s3.amazonaws.com/encoding/models/resnest101-22405ba7.pth                                |
    | ResNeSt200                 | https://hangzh.s3.amazonaws.com/encoding/models/resnest200-75117900.pth                                |
    | ResNeSt269                 | https://hangzh.s3.amazonaws.com/encoding/models/resnest269-0cc87c48.pth                                |
    | ResNeXt50_32x4d            | https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth                                       |
    | ResNeXt101_32x8d           | https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth                                      |
    | SENet154                   | http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth                                      |
    | SEResNeXt50_32x4d          | http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth                            |
    | SEResNeXt101_32x4d         | http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth                           |
    | MaskRCNN_ResNet50_FPN_COCO | https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth                            |
    | MobileNet_v2               | https://download.pytorch.org/models/mobilenet_v2-b0353104.pth                                          |
    | NASNet_A_Large             | http://data.lip6.fr/cadene/pretrainedmodels/nasnetalarge-a1897284.pth                                  |
    | PNASNet_5_Large            | http://data.lip6.fr/cadene/pretrainedmodels/pnasnet5large-bf079911.pth                                 |
    | WideResNet50_2             | https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth                                       |
    | WideResNet101_2            | https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth                                      |
    
    > Put files under `~/.cache/torch/hub/checkpoints/`


## Usage

* Launch Web Service

    ```
    $ env FLASK_ENV=development FLASK_APP=./webservice/launcher.py flask run --host 0.0.0.0 --port 5000
    ```
    
    or 
    
    ```
    $ bash ./bin/launch-dev.sh
    ```
    
* Test Web API

    ```
    $ curl -X GET http://127.0.0.1:5000/api
    ```
    
* Shutdown Web Service

    ```
    $ lsof -i:5000
    $ ps -aux | grep python
    
    kill all relative processes
    
    $ kill xxx
    ```

    
## Web API

see [WEB_API.md](docs/WEB_API.md)


## TEST

see [TEST.md](docs/TEST.md)


## TensorRT
see [TENSORRT.md](docs/TENSORRT.md)
