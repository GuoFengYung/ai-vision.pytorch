import argparse
import json
import os
import pickle
from dataclasses import dataclass
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


cifar_class_to_category_dict = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}


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
    train_image_filenames = []
    val_image_filenames = []
    test_image_filenames = []
    image_id_counter = 1
    batch_filenames = [f'data_batch_{i}' for i in range(1, 6)] + ['test_batch']

    for batch_index, batch_filename in enumerate(batch_filenames):
        path_to_batch_file = os.path.join(path_to_src_dir, batch_filename)
        with open(path_to_batch_file, 'rb') as f:
            cifar_dict = pickle.load(f, encoding='bytes')

        for image_data, label in tqdm(zip(cifar_dict[b'data'], cifar_dict[b'labels']),
                                      desc=f'Converting batch {batch_index + 1}/{len(batch_filenames)}'):
            image = image_data.reshape((3, 32, 32)).transpose(1, 2, 0)
            image = Image.fromarray(image)

            image_id = f'{image_id_counter:06d}'
            image_filename = f'{image_id}.png'
            path_to_dst_annotation_xml = os.path.join(path_to_dst_dir, 'annotations', f'{image_id}.xml')

            # handle images
            path_to_dst_image = os.path.join(path_to_dst_dir, 'images', image_filename)
            image.save(path_to_dst_image)

            # handle annotations
            category = cifar_class_to_category_dict[label]
            annotation = Annotation(filename=image_filename,
                                    size=Annotation.Size(width=32, height=32, depth=3),
                                    category=category)
            write_dst_annotation(annotation, path_to_dst_annotation_xml)

            if image_id_counter <= 50000:
                train_image_filenames.append(image_filename)
            else:
                val_image_filenames.append(image_filename)
                test_image_filenames.append(image_filename)

            image_id_counter += 1

        # handle splits
        generate_splits(train_image_filenames, val_image_filenames, test_image_filenames, path_to_dst_dir)

        # handle meta
        categories = list(cifar_class_to_category_dict.values())
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
