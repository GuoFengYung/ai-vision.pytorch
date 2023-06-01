## dataset-converter


### Requirements

* lxml

    ```
    $ pip install lxml
    ```
    

### Conversion

##### From image folder

* Task: Classification

* Structure

    ```
    - /path/to/src/dataset
        - bird
            foo.jpg
            bar.bmp
            123.png
        - cat
        - dog
        - rabbit
    ```
  
    > Specify `--split` to a list of floating values
  
    or

    ```
    - /path/to/src/dataset
        - train
            - bird
                foo.jpg
                bar.bmp
                123.png
            - cat
            - dog
            - rabbit
        - val
            - ...
        - test
            - ...
    ```
  
    > Specify `--split` to `dir`

* Script

    ```
    $ python ./tools/dataset-converter/from_image_folder.py -s=/path/to/src/dataset -d=/path/to/dst/dataset --split="[0.8, 0.2, 0]"
    ```
  
    > split 0.8, 0.2 and 0 to `train`, `val` and `test` respectively
  
    or
    
    ```
    $ python ./tools/dataset-converter/from_image_folder.py -s=/path/to/src/dataset -d=/path/to/dst/dataset --split=dir
    ```

##### From CIFAR-10 dataset

* Task: Classification

* Download [CIFAR-10 python version](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) and extract it

* Structure

    ```
    - /path/to/src/dataset
        data_batch_1
        data_batch_2
        data_batch_3
        data_batch_4
        data_batch_5
        test_batch
    ```

* Script

    ```
    $ python ./tools/dataset-converter/from_cifar10.py -s=/path/to/src/dataset -d=/path/to/dst/dataset
    ```

##### From CIFAR-100 dataset

* Task: Classification

* Download [CIFAR-100 python version](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz) and extract it

* Structure

    ```
    - /path/to/src/dataset
        meta
        test
        train
    ```

* Script

    ```
    $ python ./tools/dataset-converter/from_cifar100.py -s=/path/to/src/dataset -d=/path/to/dst/dataset
    ```

##### From labelme JSON format

* Task: Detection and Instance Segmentation

* Structure

    ```
    - /path/to/src/dataset
        001.jpg
        001.json
        002.jpg
        002.json
        ...
    ```

* Script

    ```
    $ python ./tools/dataset-converter/from_labelme_json.py -s=/path/to/src/dataset -d=/path/to/dst/dataset --val_ratio=0.1 --test_ratio=0.1
    ```

> You can convert back to `labelme` format by running:
> ```
> $ python ./tools/dataset-converter/to_labelme_json.py -s=/path/to/src/dataset -d=/path/to/dst/dataset
> ```
  
> Open `labelme` with `labelflags`
> ```
> $ labelme . --nodata --labelflags "{.*: [difficult]}"
> ``` 

##### From VOC XML format

* Task: Detection

* Structure

    ```
    - /path/to/src/dataset
        001.jpg
        001.xml
        002.jpg
        002.xml
        ...
    ```

* Script

    ```
    $ python ./tools/dataset-converter/from_voc_xml.py -s=/path/to/src/dataset -d=/path/to/dst/dataset -r=0.8 -e=jpg
    ```

##### From VOC2007 dataset

* Task: Detection and Instance Segmentation

* Download [VOC2007 training/validation data](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) and extract it

* Structure

    ```
    - /path/to/src/dataset
        - Annotations
        - ImageSets
        - JPEGImages
        - SegmentationClass
        - SegmentationObject
    ```

* Script

    ```
    $ python ./tools/dataset-converter/from_voc2007.py -s=/path/to/src/dataset -d=/path/to/dst/dataset
    ```

##### From COCO2017 dataset

* Task: Detection and Instance Segmentation

* Download following files and extract it

    * [COCO2017 Train images (18GB)](http://images.cocodataset.org/zips/train2017.zip)
    * [COCO2017 Val images (1GB)](http://images.cocodataset.org/zips/val2017.zip)
    * [COCO2017 Train/Val annotations (241MB)](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

* Structure

    ```
    - /path/to/src/dataset
        - annotations
        - train2017
        - val2017
    ```

* Script

    ```
    $ python ./tools/dataset-converter/from_coco2017.py -s=/path/to/src/dataset -d=/path/to/dst/dataset
    ```

##### From CUB-200-2011 dataset

* Task: Classification

* Download [CUB-200-2011](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz) and extract it

* Structure

    ```
    - /path/to/src/dataset
        - images
            - 001.Black_footed_Albatross
                Black_Footed_Albatross_0001_796111.jpg
                Black_Footed_Albatross_0002_55.jpg
                ...
            - 002.Laysan_Albatross
            - ...
            - 200.Common_Yellowthroat
        classes.txt
        images.txt
        train_test_split.txt
    ```

* Script

    ```
    $ python ./tools/dataset-converter/from_cub_200_2011.py -s=/path/to/src/dataset -d=/path/to/dst/dataset
    ```
      
### Sampling

```
$ python ./tools/dataset-converter/sample_dataset.py -s=/path/to/src/dataset -d=/path/to/dst/dataset --strategy=random -train_ratio=0.5 --val_ratio=0.2 --test_ratio=0.1
```

> strategy can be one of `random`, `stratified` or `category`
>   * for `stratified`, dataset will be split by category individually, this is useful for which has imbalanced classes
>   * for `category`, user will be requested to provide certain categories to be sampled

> ratios are not necessary to sum to 1

### Generate LMDB

```
$ python ./tools/dataset-converter/generate_lmdb.py -d=/path/to/dataset
```

> `lmdb` directory will be generated under dataset directory
