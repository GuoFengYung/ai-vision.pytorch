import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
from typing import List, Optional, Tuple

import lxml.builder
import lxml.etree
import numpy as np
from PIL import Image
from aibox_vision.lib.task.instance_segmentation.palette import Palette
from skimage.measure import grid_points_in_poly
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
            left: float
            top: float
            right: float
            bottom: float

        @dataclass
        class Mask:
            color: int

        name: str
        difficult: bool
        bbox: BBox
        mask: Optional[Mask]
        polygon: List[Tuple[float, float]]

    filename: str
    size: Size
    objects: List[Object]


def read_src_annotation(path_to_annotation_json: str) -> Annotation:
    with open(path_to_annotation_json, 'r', errors='ignore') as f:
        annotation_dict = json.load(f)

    annotation = Annotation(
        filename=annotation_dict['imagePath'],
        size=Annotation.Size(
            width=annotation_dict['imageWidth'],
            height=annotation_dict['imageHeight'],
            depth=3
        ),
        objects=[Annotation.Object(
            name=shape['label'],
            difficult=shape['flags']['difficult'] if 'difficult' in shape['flags'] else False,
            bbox=Annotation.Object.BBox(
                left=float(min(p[0] for p in shape['points'])),
                top=float(min(p[1] for p in shape['points'])),
                right=float(max(p[0] for p in shape['points'])),
                bottom=float(max(p[1] for p in shape['points']))
            ),
            mask=None if shape['shape_type'] != 'polygon' else Annotation.Object.Mask(color=i + 1),  # ignore background color
            polygon=None if shape['shape_type'] != 'polygon' else [tuple(p) for p in shape['points']]
        ) for i, shape in enumerate(annotation_dict['shapes'])]
    )
    return annotation


def generate_segmentation(annotation: Annotation, path_to_dst_mask_image: str):
    if all(obj.mask is None for obj in annotation.objects):
        return

    os.makedirs(os.path.dirname(path_to_dst_mask_image), exist_ok=True)

    mask_image = np.zeros((annotation.size.height, annotation.size.width), dtype=np.uint8)
    for obj in annotation.objects:
        if obj.mask is not None and obj.polygon is not None:
            color = obj.mask.color
            mask = grid_points_in_poly(shape=mask_image.shape,
                                       verts=[(y, x) for x, y in obj.polygon]).astype(np.uint8) * color
            mask_image = np.maximum(mask_image, mask)  # make the latter object overwrite the former object

    mask_image = Image.fromarray(mask_image).convert('P')
    mask_image.putpalette(Palette.get_flatten_palette())
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


def generate_splits(filenames: List[str], path_to_dst_dir: str, val_split_ratio: float, test_split_ratio: float):
    path_to_splits_dir = os.path.join(path_to_dst_dir, 'splits')

    random.shuffle(filenames)
    num_examples = len(filenames)
    num_val_examples = int(num_examples * val_split_ratio)
    num_test_examples = int(num_examples * test_split_ratio)
    num_train_examples = num_examples - num_val_examples - num_test_examples

    train_split_filenames = filenames[:num_train_examples]
    filenames = filenames[num_train_examples:]

    val_split_filenames = filenames[:num_val_examples]
    filenames = filenames[num_val_examples:]

    test_split_filenames = filenames[:num_test_examples]
    filenames = filenames[num_test_examples:]

    assert len(filenames) == 0

    with open(os.path.join(path_to_splits_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_split_filenames))
    with open(os.path.join(path_to_splits_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_split_filenames))
    with open(os.path.join(path_to_splits_dir, 'test.txt'), 'w') as f:
        f.write('\n'.join(test_split_filenames))


def convert(path_to_src_dir: str, path_to_dst_dir: str, val_split_ratio: float, test_split_ratio: float):
    path_to_src_annotation_jsons = sorted([path.as_posix() for path in Path(path_to_src_dir).rglob(f'*.json')])
    print('Found {:d} JSON files'.format(len(path_to_src_annotation_jsons)))

    filenames = []
    categories = []

    for path_to_src_annotation_json in tqdm(path_to_src_annotation_jsons):
        image_name = os.path.splitext(os.path.basename(path_to_src_annotation_json))[0]
        path_to_dst_annotation_xml = os.path.join(path_to_dst_dir, 'annotations', f'{image_name}.xml')
        path_to_dst_mask_image = os.path.join(path_to_dst_dir, 'segmentations', f'{image_name}.png')

        # handle annotations and segmentations
        annotation = read_src_annotation(path_to_src_annotation_json)
        annotation.filename = image_name + '.jpg'
        generate_segmentation(annotation, path_to_dst_mask_image)
        write_dst_annotation(annotation, path_to_dst_annotation_xml)

        # handle images
        print(annotation.filename)
        path_to_src_image = os.path.join(path_to_src_dir, annotation.filename)
        path_to_dst_image = os.path.join(path_to_dst_dir, 'images', annotation.filename)
        try:
            copyfile(path_to_src_image, path_to_dst_image)
        except:
            pass

        filenames.append(annotation.filename)
        categories += [obj.name for obj in annotation.objects]

    # handle splits
    generate_splits(filenames, path_to_dst_dir, val_split_ratio, test_split_ratio)

    # handle meta
    categories = sorted(list(set(categories)))
    categories = list(filter(lambda x: x != 'background', categories))
    categories.insert(0, 'background')
    with open(os.path.join(path_to_dst_dir, 'meta.json'), 'w') as f:
        json_dict = {category: i for i, category in enumerate(categories)}
        json.dump(json_dict, f, indent=2)

    print(f'Done! Dataset has converted to {path_to_dst_dir}')


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('-s', '--src_dir', type=str, required=True, help='path to source directory')
        parser.add_argument('-d', '--dst_dir', type=str, required=True, help='path to destination directory')
        parser.add_argument('--val_ratio', type=float, default=0.1, help='validation set split ratio')
        parser.add_argument('--test_ratio', type=float, default=0.1, help='test set split ratio')
        args = parser.parse_args()

        path_to_src_dir = args.src_dir
        path_to_dst_dir = args.dst_dir
        val_split_ratio = args.val_ratio
        test_split_ratio = args.test_ratio

        assert os.path.isdir(path_to_src_dir)
        assert not os.path.exists(path_to_dst_dir)

        os.makedirs(path_to_dst_dir)
        os.makedirs(os.path.join(path_to_dst_dir, 'images'))
        os.makedirs(os.path.join(path_to_dst_dir, 'annotations'))
        os.makedirs(os.path.join(path_to_dst_dir, 'splits'))

        convert(path_to_src_dir, path_to_dst_dir, val_split_ratio, test_split_ratio)

    main()
