import argparse
import json
import os
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
from typing import List

import lxml.builder
import lxml.etree
from tqdm import tqdm


@dataclass
class Annotation:
    @dataclass
    class Size:
        width: float
        height: float
        depth: int

    @dataclass
    class Object:
        @dataclass
        class BBox:
            left: float
            top: float
            right: float
            bottom: float

        name: str
        difficult: bool
        bbox: BBox

    filename: str
    size: Size
    objects: List[Object]


def read_src_annotation(path_to_annotation_xml: str) -> Annotation:
    tree = ET.ElementTree(file=path_to_annotation_xml)
    root = tree.getroot()

    annotation = Annotation(
        filename=root.findtext('filename', ''),  # for compatibility with lack of filename tag in some annotations
        size=Annotation.Size(
            width=float(root.findtext('size/width')),
            height=float(root.findtext('size/height')),
            depth=int(root.findtext('size/depth'))
        ),
        objects=[Annotation.Object(
            name=object_tag.findtext('name'),
            difficult=object_tag.findtext('difficult', '0') == 1,  # for compatibility with lack of difficult tag in some annotations
            bbox=Annotation.Object.BBox(
                left=float(object_tag.findtext('bndbox/xmin')),
                top=float(object_tag.findtext('bndbox/ymin')),
                right=float(object_tag.findtext('bndbox/xmax')),
                bottom=float(object_tag.findtext('bndbox/ymax'))
            )
        ) for object_tag in root.iterfind('object')]
    )
    return annotation


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

    root = annotation_node(
        filename_node(annotation.filename),
        size_node(
            width_node(str(annotation.size.width)),
            height_node(str(annotation.size.height)),
            depth_node(str(annotation.size.depth))
        )
    )
    for obj in annotation.objects:
        root.append(
            object_node(
                name_node(obj.name),
                difficult_node('1' if obj.difficult else '0'),
                bbox_node(
                    left_node(str(obj.bbox.left)),
                    top_node(str(obj.bbox.top)),
                    right_node(str(obj.bbox.right)),
                    bottom_node(str(obj.bbox.bottom))
                )
            )
        )

    tree = lxml.etree.ElementTree(root)
    tree.write(path_to_annotation_xml, pretty_print=True)


def generate_splits(filenames: List[str], path_to_dst_dir: str, split_ratio: float):
    path_to_splits_dir = os.path.join(path_to_dst_dir, 'splits')

    random.shuffle(filenames)
    num_filenames = len(filenames)
    num_train_examples = int(num_filenames * split_ratio)

    train_split_filenames = filenames[:num_train_examples]
    val_split_filenames = filenames[num_train_examples:]
    test_split_filenames = filenames[num_train_examples:]

    with open(os.path.join(path_to_splits_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_split_filenames))
    with open(os.path.join(path_to_splits_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_split_filenames))
    with open(os.path.join(path_to_splits_dir, 'test.txt'), 'w') as f:
        f.write('\n'.join(test_split_filenames))


def convert(path_to_src_dir: str, path_to_dst_dir: str, split_ratio: float, extension: str):
    path_to_src_images = sorted([path.as_posix() for path in Path(path_to_src_dir).rglob(f'*.{extension}')])
    print('Found {:d} images'.format(len(path_to_src_images)))

    filenames = []
    categories = []

    for path_to_src_image in tqdm(path_to_src_images):
        filename = os.path.basename(path_to_src_image)
        image_name = os.path.splitext(filename)[0]
        path_to_src_annotation_xml = path_to_src_image.replace(extension, 'xml')
        path_to_dst_annotation_xml = os.path.join(path_to_dst_dir, 'annotations', f'{image_name}.xml')

        # handle images
        path_to_dst_image = os.path.join(path_to_dst_dir, 'images', filename)
        copyfile(path_to_src_image, path_to_dst_image)

        # handle annotations
        annotation = read_src_annotation(path_to_src_annotation_xml)
        annotation.filename = filename
        write_dst_annotation(annotation, path_to_dst_annotation_xml)

        filenames.append(filename)
        categories += [obj.name for obj in annotation.objects]

    # handle splits
    generate_splits(filenames, path_to_dst_dir, split_ratio)

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
        parser.add_argument('-r', '--ratio', type=float, required=True, help='split ratio')
        parser.add_argument('-e', '--extension', type=str, required=True, help='extension name')
        args = parser.parse_args()

        path_to_src_dir = args.src_dir
        path_to_dst_dir = args.dst_dir
        split_ratio = args.ratio
        extension = args.extension

        assert os.path.isdir(path_to_src_dir)
        assert not os.path.exists(path_to_dst_dir)

        os.makedirs(path_to_dst_dir)
        os.makedirs(os.path.join(path_to_dst_dir, 'images'))
        os.makedirs(os.path.join(path_to_dst_dir, 'annotations'))
        os.makedirs(os.path.join(path_to_dst_dir, 'splits'))

        convert(path_to_src_dir, path_to_dst_dir, split_ratio, extension)

    main()
