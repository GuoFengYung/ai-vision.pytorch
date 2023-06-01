import argparse
import glob
import json
import os
import random
from ast import literal_eval
from dataclasses import dataclass
from typing import List, Union, Tuple

import lxml.builder
import lxml.etree
from PIL import Image
from tqdm import tqdm


@dataclass
class Annotation:
    @dataclass
    class Size:
        width: int
        height: int
        depth: int

    filename: str
    size: Size
    category: str


def write_dst_annotation(annotation: Annotation, path_to_annotation_xml: str):
    E = lxml.builder.ElementMaker()

    annotation_node = E.annotation

    filename_node = E.filename
    size_node = E.size
    width_node = E.width
    height_node = E.height
    depth_node = E.depth
    category_node = E.category

    root = annotation_node(
        filename_node(annotation.filename),
        size_node(
            width_node(str(annotation.size.width)),
            height_node(str(annotation.size.height)),
            depth_node(str(annotation.size.depth))
        ),
        category_node(annotation.category)
    )

    tree = lxml.etree.ElementTree(root)
    tree.write(path_to_annotation_xml, pretty_print=True)


def generate_splits_by_ratios(filenames: List[str], path_to_dst_dir: str,
                              train_split_ratio: float, val_split_ratio: float, test_split_ratio: float):
    assert train_split_ratio > 0
    assert 0 < train_split_ratio + val_split_ratio + test_split_ratio <= 1.0

    path_to_splits_dir = os.path.join(path_to_dst_dir, 'splits')

    random.shuffle(filenames)
    num_examples = int(len(filenames) * (train_split_ratio + val_split_ratio + test_split_ratio))

    if num_examples == 2:
        if val_split_ratio > 0:
            num_train_examples = 1
            num_val_examples = 1
            num_test_examples = 0
        elif test_split_ratio > 0:
            num_train_examples = 1
            num_val_examples = 0
            num_test_examples = 1
        else:
            num_train_examples = 2
            num_val_examples = 0
            num_test_examples = 0
    elif num_examples == 3:
        if val_split_ratio > 0 and test_split_ratio > 0:
            num_train_examples = 1
            num_val_examples = 1
            num_test_examples = 1
        elif val_split_ratio > 0 and test_split_ratio == 0:
            num_train_examples = 2
            num_val_examples = 1
            num_test_examples = 0
        elif val_split_ratio == 0 and test_split_ratio > 0:
            num_train_examples = 2
            num_val_examples = 0
            num_test_examples = 1
        else:
            num_train_examples = 3
            num_val_examples = 0
            num_test_examples = 0
    else:
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


def generate_splits_by_dir(filenames: List[str], path_to_dst_dir: str, split_dirs: List[str]):
    path_to_splits_dir = os.path.join(path_to_dst_dir, 'splits')

    train_split_filenames = []
    val_split_filenames = []
    test_split_filenames = []
    for filename, split_dir in zip(filenames, split_dirs):
        if split_dir == 'train':
            train_split_filenames.append(filename)
        elif split_dir == 'val':
            val_split_filenames.append(filename)
        elif split_dir == 'test':
            test_split_filenames.append(filename)

    with open(os.path.join(path_to_splits_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_split_filenames))
    with open(os.path.join(path_to_splits_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_split_filenames))
    with open(os.path.join(path_to_splits_dir, 'test.txt'), 'w') as f:
        f.write('\n'.join(test_split_filenames))


def convert(path_to_src_dir: str, path_to_dst_dir: str, dir_or_ratios: Union[str,
                                                                             Tuple[float, float, float]]):
    is_split_by_ratios_or_dir = type(dir_or_ratios) is not str

    if is_split_by_ratios_or_dir:
        path_to_image_pattern = os.path.join(path_to_src_dir, '*', '*')
    else:
        path_to_image_pattern = os.path.join(path_to_src_dir, '*', '*', '*')

    path_to_src_image_list = []
    for path_to_file in glob.glob(path_to_image_pattern):
        ext = os.path.splitext(path_to_file)[-1]
        if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            path_to_src_image_list.append(path_to_file)

    filenames = []
    categories = []
    split_dirs = []

    for index, path_to_src_image in enumerate(tqdm(path_to_src_image_list)):
        filename = os.path.basename(path_to_src_image)
        image_name = os.path.splitext(filename)[0]
        category = path_to_src_image.split(os.path.sep)[-2]

        filenames.append(filename)
        categories.append(category)

        # handle images
        path_to_dst_image = os.path.join(path_to_dst_dir, 'images', filename)
        image = Image.open(path_to_src_image)
        image.save(path_to_dst_image)

        # handle annotations
        path_to_dst_annotation_xml = os.path.join(path_to_dst_dir, 'annotations', f'{image_name}.xml')
        annotation = Annotation(filename=filename,
                                size=Annotation.Size(width=image.width, height=image.height, depth=len(image.getbands())),
                                category=category)
        write_dst_annotation(annotation, path_to_dst_annotation_xml)

        if not is_split_by_ratios_or_dir:
            split_dir = path_to_src_image.split(os.path.sep)[-3]
            assert split_dir in ['train', 'val', 'test']
            split_dirs.append(split_dir)

    # handle splits
    if is_split_by_ratios_or_dir:
        train_split_ratio, val_split_ratio, test_split_ratio = dir_or_ratios
        generate_splits_by_ratios(filenames, path_to_dst_dir, train_split_ratio, val_split_ratio, test_split_ratio)
    else:
        assert len(split_dirs) == len(filenames)
        generate_splits_by_dir(filenames, path_to_dst_dir, split_dirs)

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
        parser.add_argument('--split', type=str, required=True)
        args = parser.parse_args()

        path_to_src_dir = args.src_dir
        path_to_dst_dir = args.dst_dir
        split = args.split

        assert os.path.isdir(path_to_src_dir)
        assert not os.path.exists(path_to_dst_dir)

        if split == 'dir':
            dirs = os.listdir(path_to_src_dir)
            for expected_dir_name in ['train', 'val', 'test']:
                assert expected_dir_name in dirs
            dir_or_ratios = split
        else:
            ratios = literal_eval(split)
            assert len(ratios) == 3
            dir_or_ratios = ratios

        os.makedirs(path_to_dst_dir)
        os.makedirs(os.path.join(path_to_dst_dir, 'images'))
        os.makedirs(os.path.join(path_to_dst_dir, 'annotations'))
        os.makedirs(os.path.join(path_to_dst_dir, 'splits'))

        convert(path_to_src_dir, path_to_dst_dir, dir_or_ratios)

    main()
