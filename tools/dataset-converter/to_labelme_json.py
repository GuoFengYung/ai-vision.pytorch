import argparse
import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
from typing import List

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


def read_src_annotation(path_to_annotation_xml: str) -> Annotation:
    tree = ET.ElementTree(file=path_to_annotation_xml)
    root = tree.getroot()

    annotation = Annotation(
        filename=root.findtext('filename'),
        size=Annotation.Size(
            width=int(float(root.findtext('size/width'))),
            height=int(float(root.findtext('size/height'))),
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


def write_dst_annotation(annotation: Annotation, path_to_annotation_json: str):
    json_dict = {
        'shapes': [{
            'label': obj.name,
            'points': [[obj.bbox.left, obj.bbox.top], [obj.bbox.right, obj.bbox.bottom]],
            'group_id': None,
            'shape_type': 'rectangle',
            'flags': {
                'difficult': obj.difficult
            }
        } for obj in annotation.objects],
        'imagePath': annotation.filename,
        'imageData': None,
        'imageHeight': annotation.size.height,
        'imageWidth': annotation.size.width
    }

    with open(path_to_annotation_json, 'w') as f:
        json.dump(json_dict, f, indent=2)


def convert(path_to_src_dir: str, path_to_dst_dir: str):
    path_to_src_annotation_xmls = sorted([path.as_posix()
                                          for path in Path(os.path.join(path_to_src_dir, 'annotations')).rglob(f'*.xml')])
    print('Found {:d} XML files'.format(len(path_to_src_annotation_xmls)))

    filenames = []

    for path_to_src_annotation_xml in tqdm(path_to_src_annotation_xmls):
        image_name = os.path.splitext(os.path.basename(path_to_src_annotation_xml))[0]
        path_to_dst_annotation_json = os.path.join(path_to_dst_dir, f'{image_name}.json')

        # handle annotations
        annotation = read_src_annotation(path_to_src_annotation_xml)
        write_dst_annotation(annotation, path_to_dst_annotation_json)

        # handle images
        path_to_src_image = os.path.join(path_to_src_dir, 'images', annotation.filename)
        path_to_dst_image = os.path.join(path_to_dst_dir, annotation.filename)
        copyfile(path_to_src_image, path_to_dst_image)

        filenames.append(annotation.filename)

    print(f'Done! Dataset has converted to {path_to_dst_dir}')


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('-s', '--src_dir', type=str, required=True, help='path to source directory')
        parser.add_argument('-d', '--dst_dir', type=str, required=True, help='path to destination directory')
        args = parser.parse_args()

        path_to_src_dir = args.src_dir
        path_to_dst_dir = args.dst_dir

        assert os.path.isdir(path_to_src_dir)
        assert os.path.isdir(os.path.join(path_to_src_dir, 'images'))
        assert os.path.isdir(os.path.join(path_to_src_dir, 'annotations'))
        assert not os.path.exists(path_to_dst_dir)

        os.makedirs(path_to_dst_dir)

        convert(path_to_src_dir, path_to_dst_dir)

    main()
