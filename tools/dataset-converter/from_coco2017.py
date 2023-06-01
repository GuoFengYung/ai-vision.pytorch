import argparse
import json
import os
from dataclasses import dataclass
from shutil import copyfile
from typing import List

import lxml.builder
import lxml.etree
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

from aibox_vision.lib.task.instance_segmentation.palette import Palette

coco2017_category_to_class_dict = {
    'background': 0, 'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4,
    'airplane': 5, 'bus': 6, 'train': 7, 'truck': 8, 'boat': 9,
    'traffic light': 10, 'fire hydrant': 11, 'street sign': 12, 'stop sign': 13, 'parking meter': 14,
    'bench': 15, 'bird': 16, 'cat': 17, 'dog': 18, 'horse': 19,
    'sheep': 20, 'cow': 21, 'elephant': 22, 'bear': 23, 'zebra': 24,
    'giraffe': 25, 'hat': 26, 'backpack': 27, 'umbrella': 28, 'shoe': 29,
    'eye glasses': 30, 'handbag': 31, 'tie': 32, 'suitcase': 33, 'frisbee': 34,
    'skis': 35, 'snowboard': 36, 'sports ball': 37, 'kite': 38, 'baseball bat': 39,
    'baseball glove': 40, 'skateboard': 41, 'surfboard': 42, 'tennis racket': 43, 'bottle': 44,
    'plate': 45, 'wine glass': 46, 'cup': 47, 'fork': 48, 'knife': 49,
    'spoon': 50, 'bowl': 51, 'banana': 52, 'apple': 53, 'sandwich': 54,
    'orange': 55, 'broccoli': 56, 'carrot': 57, 'hot dog': 58, 'pizza': 59,
    'donut': 60, 'cake': 61, 'chair': 62, 'couch': 63, 'potted plant': 64,
    'bed': 65, 'mirror': 66, 'dining table': 67, 'window': 68, 'desk': 69,
    'toilet': 70, 'door': 71, 'tv': 72, 'laptop': 73, 'mouse': 74,
    'remote': 75, 'keyboard': 76, 'cell phone': 77, 'microwave': 78, 'oven': 79,
    'toaster': 80, 'sink': 81, 'refrigerator': 82, 'blender': 83, 'book': 84,
    'clock': 85, 'vase': 86, 'scissors': 87, 'teddy bear': 88, 'hair drier': 89,
    'toothbrush': 90, 'hair brush': 91
}
coco2017_class_to_category_dict = {v: k for k, v in coco2017_category_to_class_dict.items()}


@dataclass
class Annotation:
    @dataclass
    class Size:
        width: int
        height: int
        depth: int

    @dataclass
    class Object:
        @dataclass
        class BBox:
            left: int
            top: int
            right: int
            bottom: int

        @dataclass
        class Mask:
            color: int

        name: str
        difficult: bool
        bbox: BBox
        mask: Mask

    filename: str
    size: Size
    objects: List[Object]


def write_dst_annotation(annotation: Annotation, path_to_annotation_xml: str):
    E = lxml.builder.ElementMaker()

    annotation_node = E.annotation
    filename_node = E.filename
    size_node = E.size
    width_node = E.width
    height_node = E.height
    depth_node = E.depth
    object_node = E.object
    name_node = E.name
    difficult_node = E.difficult
    bbox_node = E.bbox
    left_node = E.left
    top_node = E.top
    right_node = E.right
    bottom_node = E.bottom
    mask_node = E.mask
    color_node = E.color

    root = annotation_node(
        filename_node(annotation.filename),
        size_node(
            width_node(str(annotation.size.width)),
            height_node(str(annotation.size.height)),
            depth_node(str(annotation.size.depth))
        )
    )
    for obj in annotation.objects:
        object_node_ = object_node(
            name_node(obj.name),
            difficult_node('1' if obj.difficult else '0'),
            bbox_node(
                left_node(str(obj.bbox.left)),
                top_node(str(obj.bbox.top)),
                right_node(str(obj.bbox.right)),
                bottom_node(str(obj.bbox.bottom))
            )
        )
        if obj.mask is not None:
            object_node_.append(
                mask_node(
                    color_node(str(obj.mask.color))
                )
            )
        root.append(object_node_)

    tree = lxml.etree.ElementTree(root)
    tree.write(path_to_annotation_xml, pretty_print=True)


def generate_splits(train_image_filenames: List[str], val_image_filenames: List[str], test_image_filenames: List[str],
                    path_to_dst_dir: str):
    path_to_splits_dir = os.path.join(path_to_dst_dir, 'splits')

    with open(os.path.join(path_to_splits_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_image_filenames))
    with open(os.path.join(path_to_splits_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_image_filenames))
    with open(os.path.join(path_to_splits_dir, 'test.txt'), 'w') as f:
        f.write('\n'.join(test_image_filenames))


def convert(path_to_src_dir: str, path_to_dst_dir: str):
    path_to_annotations_dir = os.path.join(path_to_src_dir, 'annotations')

    train_image_filenames = []
    val_image_filenames = []
    test_image_filenames = []

    for phase in ['train', 'val']:
        if phase == 'train':
            path_to_jpeg_images_dir = os.path.join(path_to_src_dir, 'train2017')
            path_to_annotation = os.path.join(path_to_annotations_dir, 'instances_train2017.json')
        elif phase == 'val':
            path_to_jpeg_images_dir = os.path.join(path_to_src_dir, 'val2017')
            path_to_annotation = os.path.join(path_to_annotations_dir, 'instances_val2017.json')
        else:
            raise ValueError

        coco_dataset = COCO(annotation_file=path_to_annotation)
        image_ids = []
        image_id_to_annotation_dict = {}
        palette = Palette.get_flatten_palette()

        for idx, (coco_image_id, coco_annotations) in enumerate(tqdm(coco_dataset.imgToAnns.items())):
            if len(coco_annotations) > 0:
                image_id = f'{coco_image_id:012d}'
                image_ids.append(image_id)
                filename = f'{image_id}.jpg'

                if phase == 'train':
                    train_image_filenames.append(filename)
                elif phase == 'val':
                    val_image_filenames.append(filename)
                    test_image_filenames.append(filename)
                else:
                    raise ValueError

                info_dict = coco_dataset.imgs[coco_image_id]
                image_width = info_dict['width']
                image_height = info_dict['height']

                objects = []
                mask_image = np.zeros((len(coco_annotations), image_height, image_width), dtype=np.uint8)
                for i, coco_annotation in enumerate(coco_annotations):
                    color = i + 1
                    coco_mask = coco_dataset.annToMask(coco_annotation)
                    mask_image[i] = coco_mask * color

                    objects.append(Annotation.Object(
                        name=coco2017_class_to_category_dict[coco_annotation['category_id']],
                        difficult=coco_annotation['iscrowd'] == 1,
                        bbox=Annotation.Object.BBox(
                            left=round(coco_annotation['bbox'][0]),
                            top=round(coco_annotation['bbox'][1]),
                            right=round(coco_annotation['bbox'][0] + coco_annotation['bbox'][2]),
                            bottom=round(coco_annotation['bbox'][1] + coco_annotation['bbox'][3])
                        ),
                        mask=Annotation.Object.Mask(color=color)
                    ))
                mask_image = np.max(mask_image, axis=0)  # select maximum for overlapping masks

                annotation = Annotation(
                    filename=filename,
                    size=Annotation.Size(
                        width=image_width,
                        height=image_height,
                        depth=3
                    ),
                    objects=objects
                )
                image_id_to_annotation_dict[image_id] = annotation

                # handle images
                path_to_src_image = os.path.join(path_to_jpeg_images_dir, filename)
                path_to_dst_image = os.path.join(path_to_dst_dir, 'images', filename)
                copyfile(path_to_src_image, path_to_dst_image)

                # handle annotations
                path_to_dst_annotation_xml = os.path.join(path_to_dst_dir, 'annotations', f'{image_id}.xml')
                write_dst_annotation(annotation, path_to_dst_annotation_xml)

                # handle segmentations
                path_to_dst_mask_image = os.path.join(path_to_dst_dir, 'segmentations', f'{image_id}.png')
                mask_image = Image.fromarray(mask_image).convert('P')
                mask_image.putpalette(palette)
                mask_image.save(path_to_dst_mask_image)

    # handle splits
    generate_splits(train_image_filenames, val_image_filenames, test_image_filenames, path_to_dst_dir)

    # handle meta
    with open(os.path.join(path_to_dst_dir, 'meta.json'), 'w') as f:
        json_dict = coco2017_category_to_class_dict
        json.dump(json_dict, f, indent=2)


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('-s', '--src_dir', type=str, required=True, help='path to source directory')
        parser.add_argument('-d', '--dst_dir', type=str, required=True, help='path to destination directory')
        args = parser.parse_args()

        path_to_src_dir = args.src_dir
        path_to_dst_dir = args.dst_dir

        assert os.path.isdir(path_to_src_dir)
        assert not os.path.exists(path_to_dst_dir)

        os.makedirs(path_to_dst_dir)
        os.makedirs(os.path.join(path_to_dst_dir, 'images'))
        os.makedirs(os.path.join(path_to_dst_dir, 'annotations'))
        os.makedirs(os.path.join(path_to_dst_dir, 'segmentations'))
        os.makedirs(os.path.join(path_to_dst_dir, 'splits'))

        convert(path_to_src_dir, path_to_dst_dir)

    main()
