## Benchmarks

* For classification task

    * CIFAR-10 Dataset
    
        | Version | Date       | Algorithm        | Accuracy | Elapsed | GPU                   | Batch Size | Learning Rate |
        |:-------:|:----------:|:----------------:|:--------:|:-------:|:---------------------:|:----------:|:-------------:|
        | 1.1.1   | 2020-09-07 | ResNet-50        | 0.9438   | 0.41 hr | NVIDIA RTX2080-Ti x 2 | 64         | 0.064         |
        | 1.3.0   | 2020-12-12 | ResNet-50        | 0.9452   | 0.43 hr | NVIDIA RTX2080-Ti x 2 | 64         | 0.064         |

    * CIFAR-100 Dataset
    
        | Version | Date       | Algorithm        | Accuracy | Elapsed | GPU                   | Batch Size | Learning Rate |
        |:-------:|:----------:|:----------------:|:--------:|:-------:|:---------------------:|:----------:|:-------------:|
        | 1.1.1   | 2020-09-07 | ResNet-50        | 0.7870   | 0.41 hr | NVIDIA RTX2080-Ti x 2 | 64         | 0.064         |
        | 1.3.0   | 2020-12-12 | ResNet-50        | 0.7886   | 0.43 hr | NVIDIA RTX2080-Ti x 2 | 64         | 0.064         |

* For detection task

    * VOC2007 Dataset
    
        | Version | Date       | Algorithm        | Backbone       | mean AP@[.5:.95:.05]  | mean AP@0.5 | Elapsed   | GPU                    | Batch Size | Learning Rate |
        |:-------:|:----------:|:----------------:|:--------------:|:---------------------:|:-----------:|:---------:|:----------------------:|:----------:|:-------------:|
        | 1.1.1   | 2020-09-07 | Faster RCNN      | ResNet-50      | 0.4036                | 0.7503      | 1.25 hr   | NVIDIA P100 x 4        | 8          | 0.008         |
        | 1.1.1   | 2020-09-07 | FPN              | ResNet-50      | 0.4164                | 0.7539      | 1.39 hr   | NVIDIA P100 x 4        | 8          | 0.008         |
        | 1.3.0   | 2020-11-24 | Faster RCNN      | ResNet-50      | 0.4047                | 0.7450      | 1.34 hr   | NVIDIA RTX2080-Ti x 2  | 8          | 0.008         |
        | 1.3.0   | 2020-11-24 | FPN              | ResNet-50      | 0.4141                | 0.7499      | 1.43 hr   | NVIDIA RTX2080-Ti x 2  | 8          | 0.008         |
        | 1.4.0   | 2021-06-07 | TorchFPN         | ResNet-50      | 0.4335                | 0.7602      | 1.39 hr   | NVIDIA RTX2080-Ti x 2  | 8          | 0.008         |

    * COCO2017 Dataset
    
        | Version | Date       | Algorithm        | Backbone       | mean AP@[.5:.95:.05]  | mean AP@0.5 | Elapsed   | GPU                    | Batch Size | Learning Rate |
        |:-------:|:----------:|:----------------:|:--------------:|:---------------------:|:-----------:|:---------:|:----------------------:|:----------:|:-------------:|
        | 1.3.0   | 2020-12-12 | Faster RCNN      | ResNet-50      | 0.3234                | 0.5297      | 30.59 hr  | NVIDIA RTX2080-Ti x 2  | 8          | 0.01          |
        | 1.3.0   | 2020-12-10 | Faster RCNN      | ResNet-50      | 0.3232                | 0.5305      | 30.33 hr  | NVIDIA RTX2080-Ti x 2  | 8          | 0.008         |
        | 1.3.0   | 2020-12-09 | FPN              | ResNet-50      | 0.3411                | 0.5635      | 38.74 hr  | NVIDIA RTX2080-Ti x 2  | 4          | 0.004         |

* For instance segmentation task

    * VOC2007 Dataset
    
        | Version | Date       | Algorithm        | mean AP@0.5  | Elapsed   | GPU                | Batch Size | Learning Rate |
        |:-------:|:----------:|:----------------:|:------------:|:---------:|:------------------:|:----------:|:-------------:|
        | 1.0.6   | 2020-08-13 | Mask RCNN        | 0.xxxx       | 0.xx hr   | NVIDIA P100 x 4    | 16         | 0.016         |

    * COCO2017 Dataset
    
        | Version | Date       | Algorithm        | mean AP@0.5  | Elapsed   | GPU                | Batch Size | Learning Rate |
        |:-------:|:----------:|:----------------:|:------------:|:---------:|:------------------:|:----------:|:-------------:|
        | 1.0.4   | 2020-04-04 | Mask RCNN        | 0.xxxx       | x.xx hr   | NVIDIA P100 x 4    | 16         | 0.016         |
