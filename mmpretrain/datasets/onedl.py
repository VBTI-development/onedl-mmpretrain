# Copyright (c) VBTI. All rights reserved.
from collections import defaultdict
from typing import Any, Callable, List, Mapping, Optional, Union

import numpy as np
from onedl.datasets.specializations.classification import (
    ClassificationDataset, MultiLabelClassificationDataset)
from onedl.datasets.specializations.regression import RegressionDataset

from mmpretrain.datasets import BaseDataset
from mmpretrain.datasets.builder import DATASETS
from ._palette import DEFAULT_PALETTE


@DATASETS.register_module()
class OneDLDataset:
    """Wrapper for OneDL Classification and Regression datasets to mmpretrain.

    For more information about OneDL datasets, please refer to
    https://onedl.ai.
    """
    def __new__(  # noqa: PLR0913
        cls,
        dataset_name: str,
        pipeline: Optional[List[Union[dict, Callable]]] = None,  # noqa: FA100
        test_mode: bool = False,
    ) -> Union['OneDLDatasetAdapterSingleLabel',
               'OneDLDatasetAdapterMultiLabel',
               'OneDLDatasetAdapterRegressionValues', ]:
        """Wrapper for OneDL Classification and Regression datasets to
        mmpretrain.

        Automatically selects the right format for mmpretrain based on the
        input dataset.

        Args:
            dataset_name: OneDL dataset name to load
            pipeline: Processing pipeline. Defaults to an empty tuple
            test_mode: `test_mode=True`` means in test phase,
                an error will be raised when getting an item fails,
                ``test_mode=False`` means in training phase, another item will
                be returned randomly. Defaults to False.

        Returns:
            Any: specific dataset adapted
        """
        if pipeline is None:
            pipeline = []

        from onedl.client import Client

        client = Client()
        dataset = client.datasets.load(dataset_name,
                                       pull_blobs=True)  # type: ignore

        if dataset.targets.get_dtype_str() == 'RegressionValues':
            return OneDLDatasetAdapterRegressionValues(
                regression_dataset=dataset,
                dataset_name=dataset_name,
                pipeline=pipeline,
                test_mode=test_mode)
        elif dataset.targets.get_dtype_str() == 'MultiLabel':
            return OneDLDatasetAdapterMultiLabel(
                multilabel_classification_dataset=dataset,
                dataset_name=dataset_name,
                pipeline=pipeline,
                test_mode=test_mode,
            )
        return OneDLDatasetAdapterSingleLabel(classification_dataset=dataset,
                                              dataset_name=dataset_name,
                                              pipeline=pipeline,
                                              test_mode=test_mode)


class OneDLDatasetAdapterSingleLabel(BaseDataset):
    def __init__(  # noqa: PLR0913
        self,
        classification_dataset: ClassificationDataset,
        dataset_name: str,
        pipeline: Mapping[Any, Any],
        test_mode: bool = False,
        # Whether the annotation file includes ground truth labels,
        # or use sub-folders to specify categories.
        lazy_init: bool = False,
    ) -> None:
        """Adapter to load single label OneDL data in mmclassification.

        Args:
            classification_dataset: classification data set.
            dataset_name: name of the dataset.
            pipeline: Processing pipeline. Defaults to an empty tuple
            test_mode: `test_mode=True`` means in test phase,
                an error will be raised when getting an item fails,
                ``test_mode=False`` means in training phase, another item will
                be returned randomly. Defaults to False.
            lazy_init: Whether to load annotation during instantiation.
                In some cases, such as visualization, only the meta
                information of the dataset is needed, which is not necessary
                to load annotation file. ``Basedataset`` can skip load
                annotations to save time by set ``lazy_init=False``. Defaults
                to False.
        """
        self.classification_dataset = classification_dataset
        self.label_map = self.classification_dataset.label_map
        meta_info = {
            'classes': self.label_map.get_labels(),
            'palette': DEFAULT_PALETTE[:self.label_map.n_classes]
        }
        super(OneDLDatasetAdapterSingleLabel, self).__init__(
            ann_file=dataset_name,
            data_prefix='',
            data_root='',
            pipeline=pipeline,
            metainfo=meta_info,
            test_mode=test_mode,
            # Force to lazy_init for some modification before loading data.
            lazy_init=True,
        )

        # Full initialize the dataset.
        if not lazy_init:
            self.full_init()

    def load_data_list(self) -> List[dict]:
        """Load annotations.

        Returns:
            list[dict]: Annotation info from ClassificationDataset
        """
        dataset: ClassificationDataset = (
            self.classification_dataset.classification)
        self.cat_ids = self.label_map.get_class_ids()

        data_infos = []
        cat_to_imgs = defaultdict(list)
        for img_path, instances in zip(dataset.inputs.path_iterator(),
                                       dataset.targets):
            data_infos.append(
                dict(img_path=img_path,
                     gt_label=int(self.label_map[instances])))

        self.cat_to_imgs = cat_to_imgs
        return data_infos


class OneDLDatasetAdapterMultiLabel(BaseDataset):
    def __init__(
        self,
        multilabel_classification_dataset: MultiLabelClassificationDataset,
        dataset_name: str,
        pipeline: Mapping[Any, Any],
        test_mode: bool = False,
    ) -> None:
        """Adapter to load multi-label OneDL data in mmclassification.

        Args:
            multilabel_classification_dataset: multilabel classification data
                set.
            dataset_name: name of the dataset.
            pipeline: Processing pipeline. Defaults to an empty tuple
            test_mode: `test_mode=True`` means in test phase,
                an error will be raised when getting an item fails,
                ``test_mode=False`` means in training phase, another item will
                be returned randomly. Defaults to False.
        """
        self.multilabel_classification_dataset = (
            multilabel_classification_dataset)
        self.label_map = self.multilabel_classification_dataset.label_map
        meta_info = {
            'classes': self.label_map.get_labels(),
            'palette': DEFAULT_PALETTE[:self.label_map.n_classes]
        }
        super(OneDLDatasetAdapterMultiLabel,
              self).__init__(ann_file=dataset_name,
                             data_root='',
                             pipeline=pipeline,
                             metainfo=meta_info,
                             test_mode=test_mode)

    def load_data_list(self) -> List[dict]:
        """Load annotations.

        Returns:
            list[dict]: Annotation info from ClassificationDataset
        """
        dataset: MultiLabelClassificationDataset = (
            self.multilabel_classification_dataset.multilabelclassification)
        self.cat_ids = self.label_map.get_class_ids()

        cat_to_imgs = defaultdict(list)
        data_infos = []
        for img_path, instances in zip(dataset.inputs.path_iterator(),
                                       dataset.targets):
            gt_labels = [
                int(self.label_map[label]) for label in instances.labels
            ]
            data_infos.append(dict(img_path=img_path, gt_label=gt_labels))

        self.cat_to_imgs = cat_to_imgs
        return data_infos


class OneDLDatasetAdapterRegressionValues(BaseDataset):
    def __init__(
        self,
        regression_dataset: RegressionDataset,
        dataset_name: str,
        pipeline: Mapping[Any, Any],
        test_mode: bool = False,
    ) -> None:
        """Adapter to load regression OneDL data in mmclassification.

        Args:
            regression_dataset: regression data set.
            dataset_name: name of the dataset.
            pipeline: Processing pipeline. Defaults to an empty tuple
            test_mode: `test_mode=True`` means in test phase,
                an error will be raised when getting an item fails,
                ``test_mode=False`` means in training phase, another item will
                be returned randomly. Defaults to False.
        """
        self.regression_dataset = regression_dataset
        self.label_map = self.regression_dataset.label_map
        meta_info = {
            'classes': self.label_map.get_labels(),
            'palette': DEFAULT_PALETTE[:self.label_map.n_classes]
        }
        super(OneDLDatasetAdapterRegressionValues,
              self).__init__(ann_file=dataset_name,
                             pipeline=pipeline,
                             test_mode=test_mode,
                             metainfo=meta_info)

    def load_data_list(self) -> List[dict]:
        """Load annotations.

        Returns:
            list[dict]: Annotation info from RegressionDataset
        """
        dataset: RegressionDataset = self.regression_dataset.regression
        self.cat_ids = self.label_map.get_class_ids()

        cat_to_imgs = defaultdict(list)
        data_infos = []
        for img_path, instances in zip(dataset.inputs.path_iterator(),
                                       dataset.targets):
            gt_score = np.array(instances._values).astype(float)
            data_infos.append(dict(img_path=img_path, gt_score=gt_score))

        self.cat_to_imgs = cat_to_imgs
        return data_infos
