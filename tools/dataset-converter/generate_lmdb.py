import argparse
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List

import lmdb
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

        name: str
        difficult: bool
        bbox: BBox

    filename: str
    size: Size
    objects: List[Object]


def read_annotation(path_to_annotation_xml: str) -> Annotation:
    tree = ET.ElementTree(file=path_to_annotation_xml)
    root = tree.getroot()

    annotation = Annotation(
        filename=root.findtext('filename'),
        size=Annotation.Size(
            width=int(root.findtext('size/width')),
            height=int(root.findtext('size/height')),
            depth=int(root.findtext('size/depth'))
        ),
        objects=[Annotation.Object(
            name=object_tag.findtext('name'),
            difficult=object_tag.findtext('difficult') == 1,
            bbox=Annotation.Object.BBox(
                left=int(float(object_tag.findtext('bbox/left'))),
                top=int(float(object_tag.findtext('bbox/top'))),
                right=int(float(object_tag.findtext('bbox/right'))),
                bottom=int(float(object_tag.findtext('bbox/bottom')))
            )
        ) for object_tag in root.iterfind('object')]
    )
    return annotation


def convert(path_to_dataset_dir: str, path_to_lmdb_dir: str):
    path_to_annotation_xmls = sorted([path.as_posix()
                                      for path in Path(os.path.join(path_to_dataset_dir, 'annotations')).rglob(f'*.xml')])
    print('Found {:d} XML files'.format(len(path_to_annotation_xmls)))

    lmdb_env = lmdb.open(path_to_lmdb_dir, map_size=10 * 1024 * 1024 * 1024)
    lmdb_txn = lmdb_env.begin(write=True)

    for path_to_annotation_xml in tqdm(path_to_annotation_xmls):
        annotation = read_annotation(path_to_annotation_xml)
        path_to_image = os.path.join(path_to_dataset_dir, 'images', annotation.filename)
        with open(path_to_image, 'rb') as f:
            lmdb_txn.put(key=annotation.filename.encode(), value=f.read())

    lmdb_txn.commit()
    lmdb_env.close()

    print(f'Done! Dataset has converted to {path_to_lmdb_dir}')


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--dataset_dir', type=str, required=True, help='path to dataset directory')
        args = parser.parse_args()

        path_to_dataset_dir = args.dataset_dir

        assert os.path.isdir(path_to_dataset_dir)
        assert os.path.isdir(os.path.join(path_to_dataset_dir, 'images'))
        assert os.path.isdir(os.path.join(path_to_dataset_dir, 'annotations'))

        path_to_lmdb_dir = os.path.join(path_to_dataset_dir, 'lmdb')
        assert not os.path.exists(path_to_lmdb_dir)

        convert(path_to_dataset_dir, path_to_lmdb_dir)

    main()
