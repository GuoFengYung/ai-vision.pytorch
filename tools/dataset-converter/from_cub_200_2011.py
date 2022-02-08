import argparse
import json
import os
from dataclasses import dataclass
from shutil import copyfile
from typing import List

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
    class_to_category_dict = {}
    path_to_classes_txt = os.path.join(path_to_src_dir, 'classes.txt')
    with open(path_to_classes_txt, 'r') as f:
        for line in f:
            cls, category = line.strip().split(' ')
            class_to_category_dict[cls] = category

    image_sn_to_path_to_image_dict = {}
    path_to_images_txt = os.path.join(path_to_src_dir, 'images.txt')
    with open(path_to_images_txt, 'r') as f:
        for line in f:
            image_sn, path_to_image = line.strip().split(' ')
            path_to_image = os.path.join(path_to_src_dir, 'images', path_to_image)
            image_sn_to_path_to_image_dict[image_sn] = path_to_image

    image_sn_to_is_train_dict = {}
    path_to_train_test_split_txt = os.path.join(path_to_src_dir, 'train_test_split.txt')
    with open(path_to_train_test_split_txt, 'r') as f:
        for line in f:
            image_sn, is_train = line.strip().split(' ')
            is_train = is_train == '1'
            image_sn_to_is_train_dict[image_sn] = is_train

    train_image_filenames = []
    val_image_filenames = []
    test_image_filenames = []
    for image_sn in tqdm(image_sn_to_path_to_image_dict.keys()):
        path_to_image = image_sn_to_path_to_image_dict[image_sn]
        is_train = image_sn_to_is_train_dict[image_sn]

        image_filename = os.path.basename(path_to_image)
        image_id, _ = os.path.splitext(image_filename)
        path_to_dst_annotation_xml = os.path.join(path_to_dst_dir, 'annotations', f'{image_id}.xml')

        # handle images
        path_to_dst_image = os.path.join(path_to_dst_dir, 'images', image_filename)
        copyfile(path_to_image, path_to_dst_image)

        # handle annotations
        category = path_to_image.split(os.path.sep)[-2]
        image = Image.open(path_to_image)
        annotation = Annotation(filename=image_filename,
                                size=Annotation.Size(width=image.width, height=image.height, depth=3),
                                category=category)
        write_dst_annotation(annotation, path_to_dst_annotation_xml)

        if is_train:
            train_image_filenames.append(image_filename)
        else:
            val_image_filenames.append(image_filename)
            test_image_filenames.append(image_filename)

    # handle splits
    generate_splits(train_image_filenames, val_image_filenames, test_image_filenames, path_to_dst_dir)

    # handle meta
    categories = list(class_to_category_dict.values())
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
        os.makedirs(os.path.join(path_to_dst_dir, 'splits'))

        convert(path_to_src_dir, path_to_dst_dir)

    main()
