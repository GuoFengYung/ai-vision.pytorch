import argparse
import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
from typing import List, Optional

import lxml.builder
import lxml.etree
import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_iou
from tqdm import tqdm


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
        mask: Optional[Mask]

    filename: str
    size: Size
    objects: List[Object]


def read_src_annotation(path_to_annotation_xml: str) -> Annotation:
    tree = ET.ElementTree(file=path_to_annotation_xml)
    root = tree.getroot()

    annotation = Annotation(
        filename=root.findtext('filename', ''),  # for compatibility with lack of filename tag in some annotations
        size=Annotation.Size(
            width=int(root.findtext('size/width')),
            height=int(root.findtext('size/height')),
            depth=int(root.findtext('size/depth'))
        ),
        objects=[Annotation.Object(
            name=object_tag.findtext('name'),
            difficult=object_tag.findtext('difficult', '0') == '1',  # for compatibility with lack of difficult tag in some annotations
            bbox=Annotation.Object.BBox(
                left=int(object_tag.findtext('bndbox/xmin')),
                top=int(object_tag.findtext('bndbox/ymin')),
                right=int(object_tag.findtext('bndbox/xmax')),
                bottom=int(object_tag.findtext('bndbox/ymax'))
            ),
            mask=None
        ) for object_tag in root.iterfind('object')]
    )

    return annotation


def generate_segmentation(annotation: Annotation, path_to_src_mask_image: str, path_to_dst_mask_image: str):
    if os.path.exists(path_to_src_mask_image):
        mask_image = Image.open(path_to_src_mask_image)
        palette = mask_image.getpalette()

        mask_image = np.array(mask_image)
        mask_image[(mask_image == 255).nonzero()] = 0  # remove contour color

        mask_colors = np.unique(mask_image)[1:]  # ignore background color

        mask_bboxes = []
        for color in mask_colors:
            nonzero = (mask_image == color).nonzero()
            left, top = nonzero[1].min(), nonzero[0].min()
            right, bottom = nonzero[1].max(), nonzero[0].max()
            mask_bboxes.append([left, top, right, bottom])
        mask_bboxes = torch.tensor(mask_bboxes, dtype=torch.float)

        annotation_bboxes = torch.tensor([[a.bbox.left, a.bbox.top, a.bbox.right, a.bbox.bottom]
                                          for a in annotation.objects],
                                         dtype=torch.float)

        mask_to_annotation_ious = box_iou(mask_bboxes, annotation_bboxes)
        _, mask_assign_annotation_indices = mask_to_annotation_ious.max(dim=1)

        for mask_index, annotation_index in enumerate(mask_assign_annotation_indices):
            mask_color = mask_colors[mask_index]
            annotation.objects[annotation_index].mask = Annotation.Object.Mask(color=mask_color)

        mask_image = Image.fromarray(mask_image).convert('P')
        mask_image.putpalette(palette)
        mask_image.save(path_to_dst_mask_image)


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


def generate_splits(path_to_src_trainval_txt: str, path_to_src_test_txt: str,
                    path_to_dst_dir: str):
    path_to_splits_dir = os.path.join(path_to_dst_dir, 'splits')

    train_image_filenames = []
    val_image_filenames = []
    test_image_filenames = []

    with open(path_to_src_trainval_txt, 'r') as src_trainval_txt_file:
        for line in src_trainval_txt_file.readlines():
            image_id = line.strip()
            filename = f'{image_id}.jpg'
            train_image_filenames.append(filename)

    with open(path_to_src_test_txt, 'r') as src_test_txt_file:
        for line in src_test_txt_file.readlines():
            image_id = line.strip()
            filename = f'{image_id}.jpg'
            val_image_filenames.append(filename)
            test_image_filenames.append(filename)

    with open(os.path.join(path_to_splits_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_image_filenames))
    with open(os.path.join(path_to_splits_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_image_filenames))
    with open(os.path.join(path_to_splits_dir, 'test.txt'), 'w') as f:
        f.write('\n'.join(test_image_filenames))


def convert(path_to_src_dir: str, path_to_dst_dir: str):
    path_to_src_images = sorted([path.as_posix() for path in Path(path_to_src_dir).rglob(f'*.jpg')])
    print('Found {:d} images'.format(len(path_to_src_images)))

    categories = []

    for path_to_src_image in tqdm(path_to_src_images):
        filename = os.path.basename(path_to_src_image)
        image_id = os.path.splitext(filename)[0]
        path_to_src_annotation_xml = os.path.join(path_to_src_dir, 'Annotations', f'{image_id}.xml')
        path_to_src_mask_image = os.path.join(path_to_src_dir, 'SegmentationObject', f'{image_id}.png')
        path_to_dst_annotation_xml = os.path.join(path_to_dst_dir, 'annotations', f'{image_id}.xml')
        path_to_dst_mask_image = os.path.join(path_to_dst_dir, 'segmentations', f'{image_id}.png')

        # handle images
        path_to_dst_image = os.path.join(path_to_dst_dir, 'images', filename)
        copyfile(path_to_src_image, path_to_dst_image)

        # handle annotations and segmentations
        annotation = read_src_annotation(path_to_src_annotation_xml)
        generate_segmentation(annotation, path_to_src_mask_image, path_to_dst_mask_image)
        write_dst_annotation(annotation, path_to_dst_annotation_xml)

        categories += [obj.name for obj in annotation.objects]

    # handle splits
    path_to_src_trainval_txt = os.path.join(path_to_src_dir, 'ImageSets', 'Main', 'trainval.txt')
    path_to_src_test_txt = os.path.join(path_to_src_dir, 'ImageSets', 'Main', 'test.txt')
    generate_splits(path_to_src_trainval_txt, path_to_src_test_txt, path_to_dst_dir)

    # handle meta
    categories = sorted(list(set(categories)))
    categories.insert(0, 'background')
    with open(os.path.join(path_to_dst_dir, 'meta.json'), 'w') as f:
        json_dict = {category: i for i, category in enumerate(categories)}
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
