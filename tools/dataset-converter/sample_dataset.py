import argparse
import json
import os
import random
from collections import defaultdict
from enum import Enum
from typing import List, Tuple

from aibox_vision.lib.preprocessor import Preprocessor
from aibox_vision.lib.task import Task


class Strategy(Enum):
    RANDOM = 'random'
    STRATIFIED = 'stratified'
    CATEGORY = 'category'


def random_split(filenames: List[str],
                 train_split_ratio: float, val_split_ratio: float, test_split_ratio: float) -> Tuple[List[str],
                                                                                                     List[str],
                                                                                                     List[str]]:
    assert train_split_ratio > 0
    assert 0 < train_split_ratio + val_split_ratio + test_split_ratio <= 1.0

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
        num_train_examples = int(len(filenames) * train_split_ratio)
        num_val_examples = int(len(filenames) * val_split_ratio)
        num_test_examples = int(len(filenames) * test_split_ratio)

        if num_train_examples + num_val_examples + num_test_examples < num_examples:
            num_train_examples += num_examples - (num_train_examples + num_val_examples + num_test_examples)

    train_split_filenames = filenames[:num_train_examples]
    filenames = filenames[num_train_examples:]

    val_split_filenames = filenames[:num_val_examples]
    filenames = filenames[num_val_examples:]

    test_split_filenames = filenames[:num_test_examples]

    assert len(train_split_filenames) + len(val_split_filenames) + len(test_split_filenames) == num_examples

    return train_split_filenames, val_split_filenames, test_split_filenames


def sample(task_name: Task.Name, path_to_src_dir: str, path_to_dst_dir: str, strategy: Strategy,
           train_split_ratio: float, val_split_ratio: float, test_split_ratio: float):
    path_to_src_annotations_dir = os.path.join(path_to_src_dir, 'annotations')
    path_to_src_images_dir = os.path.join(path_to_src_dir, 'images')
    path_to_src_meta_json = os.path.join(path_to_src_dir, 'meta.json')

    if task_name == Task.Name.CLASSIFICATION:
        from aibox_vision.lib.task.classification.dataset import Dataset
        dataset = Dataset(path_to_data_dir=path_to_src_dir, mode=Dataset.Mode.UNION,
                          preprocessor=Preprocessor.build_noop(), augmenter=None)
    elif task_name == Task.Name.DETECTION:
        from aibox_vision.lib.task.detection.dataset import Dataset
        dataset = Dataset(path_to_data_dir=path_to_src_dir, mode=Dataset.Mode.UNION,
                          preprocessor=Preprocessor.build_noop(), augmenter=None, exclude_difficulty=False)
    elif task_name == Task.Name.INSTANCE_SEGMENTATION:
        from aibox_vision.lib.task.instance_segmentation.dataset import Dataset
        dataset = Dataset(path_to_data_dir=path_to_src_dir, mode=Dataset.Mode.UNION,
                          preprocessor=Preprocessor.build_noop(), augmenter=None, exclude_difficulty=False)
    else:
        raise ValueError

    print('Found {:d} samples'.format(len(dataset)))

    if strategy == Strategy.RANDOM:
        filenames = [annotation.filename for annotation in dataset.annotations]
        train_split_filenames, val_split_filenames, test_split_filenames = \
            random_split(filenames, train_split_ratio, val_split_ratio, test_split_ratio)

        with open(path_to_src_meta_json, 'r') as f:
            meta_json = json.load(f)
    elif strategy == Strategy.STRATIFIED:
        if task_name == Task.Name.CLASSIFICATION:
            categories = [annotation.category for annotation in dataset.annotations]
            category_set = set(categories)

            category_to_filenames_dict = {}
            for category in category_set:
                filenames = [annotation.filename for annotation in dataset.annotations
                             if annotation.category == category]
                category_to_filenames_dict[category] = filenames
        elif task_name in [Task.Name.DETECTION, Task.Name.INSTANCE_SEGMENTATION]:
            category_in_single_to_count_dict = defaultdict(int)
            for annotation in dataset.annotations:
                names = [obj.name for obj in annotation.objects]
                if len(set(names)) == 1:
                    category = names[0]
                    category_in_single_to_count_dict[category] += 1

            category_in_multiple_to_weight_dict = {category: 1 / count
                                                   for category, count in category_in_single_to_count_dict.items()}

            category_to_filenames_dict = defaultdict(list)
            for annotation in dataset.annotations:
                names = [obj.name for obj in annotation.objects]

                if len(set(names)) == 1:
                    selected_category = names[0]
                else:
                    categories = names
                    weights = [category_in_multiple_to_weight_dict[category] for category in categories]
                    selected_category = random.choices(categories, weights, k=1)[0]
                category_to_filenames_dict[selected_category].append(annotation.filename)
        else:
            raise ValueError

        train_split_filenames = []
        val_split_filenames = []
        test_split_filenames = []
        for category, filenames in category_to_filenames_dict.items():
            category_train_split_filenames, category_val_split_filenames, category_test_split_filenames = \
                random_split(filenames, train_split_ratio, val_split_ratio, test_split_ratio)
            train_split_filenames += category_train_split_filenames
            val_split_filenames += category_val_split_filenames
            test_split_filenames += category_test_split_filenames

        with open(path_to_src_meta_json, 'r') as f:
            meta_json = json.load(f)
    elif strategy == Strategy.CATEGORY:
        if task_name == Task.Name.CLASSIFICATION:
            kept_categories = input('Please enter kept categories (e.g.: cat,dog): ').split(',')

            categories = [annotation.category for annotation in dataset.annotations
                          if annotation.category in kept_categories]
            category_set = set(categories)

            category_to_filenames_dict = {}
            for category in category_set:
                filenames = [annotation.filename for annotation in dataset.annotations
                             if annotation.category == category]
                category_to_filenames_dict[category] = filenames

            train_split_filenames = []
            val_split_filenames = []
            test_split_filenames = []
            for category, filenames in category_to_filenames_dict.items():
                category_train_split_filenames, category_val_split_filenames, category_test_split_filenames = \
                    random_split(filenames, train_split_ratio, val_split_ratio, test_split_ratio)
                train_split_filenames += category_train_split_filenames
                val_split_filenames += category_val_split_filenames
                test_split_filenames += category_test_split_filenames
        elif task_name in [Task.Name.DETECTION, Task.Name.INSTANCE_SEGMENTATION]:
            raise NotImplementedError
        else:
            raise ValueError

        categories = kept_categories
        categories.insert(0, 'background')
        meta_json = {category: i for i, category in enumerate(categories)}
    else:
        raise ValueError

    path_to_dst_annotations_dir = os.path.join(path_to_dst_dir, 'annotations')
    path_to_dst_images_dir = os.path.join(path_to_dst_dir, 'images')
    path_to_dst_meta_json = os.path.join(path_to_dst_dir, 'meta.json')
    path_to_dst_splits_dir = os.path.join(path_to_dst_dir, 'splits')

    os.makedirs(path_to_dst_dir)
    os.symlink(os.path.realpath(path_to_src_annotations_dir), os.path.realpath(path_to_dst_annotations_dir))
    os.symlink(os.path.realpath(path_to_src_images_dir), os.path.realpath(path_to_dst_images_dir))

    with open(path_to_dst_meta_json, 'w') as f:
        json.dump(meta_json, f, indent=2)

    os.makedirs(path_to_dst_splits_dir)
    with open(os.path.join(path_to_dst_splits_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_split_filenames))
    with open(os.path.join(path_to_dst_splits_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_split_filenames))
    with open(os.path.join(path_to_dst_splits_dir, 'test.txt'), 'w') as f:
        f.write('\n'.join(test_split_filenames))

    print(f'The new dataset has created at {path_to_dst_dir}')


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()

        # region ===== Common arguments =====
        parser.add_argument('-s', '--src_dir', type=str, required=True, help='path to source directory')
        parser.add_argument('-d', '--dst_dir', type=str, required=True, help='path to destination directory')
        parser.add_argument('--strategy', type=str, help='sampling strategy')
        parser.add_argument('--train_ratio', type=float, default=0.8, help='training set split ratio')
        parser.add_argument('--val_ratio', type=float, default=0.2, help='validation set split ratio')
        parser.add_argument('--test_ratio', type=float, default=0, help='test set split ratio')
        # endregion =========================

        subparsers = parser.add_subparsers(dest='task', help='task name')
        classification_subparser = subparsers.add_parser(Task.Name.CLASSIFICATION.value)
        detection_subparser = subparsers.add_parser(Task.Name.DETECTION.value)
        instance_segmentation_subparser = subparsers.add_parser(Task.Name.INSTANCE_SEGMENTATION.value)

        # region ===== Classification arguments =====
        # endregion =================================

        # region ===== Detection arguments =====
        # endregion ============================

        # region ===== Instance Segmentation arguments =====
        # endregion ========================================

        args = parser.parse_args()

        path_to_src_dir = args.src_dir
        path_to_dst_dir = args.dst_dir
        strategy = Strategy(args.strategy)
        train_split_ratio = args.train_ratio
        val_split_ratio = args.val_ratio
        test_split_ratio = args.test_ratio
        task_name = Task.Name(args.task)

        assert os.path.isdir(path_to_src_dir)
        assert not os.path.exists(path_to_dst_dir)

        sample(task_name, path_to_src_dir, path_to_dst_dir, strategy,
               train_split_ratio, val_split_ratio, test_split_ratio)

    main()
