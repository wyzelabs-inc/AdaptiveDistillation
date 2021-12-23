import os
import os.path as osp

import numpy as np
import torch.distributed as dist
from mmcv.runner import get_dist_info, master_only

from mmcls.datasets.utils import download_and_extract_archive, rm_suffix
from mmcls.datasets.mnist import read_label_file, read_image_file

from .base_dataset import BaseDatasetKD
from .builder import DATASETS

# Load datasets with the modified BaseDataset
@DATASETS.register_module()
class MNISTKD(BaseDatasetKD):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    This implementation is modified from
    https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py  # noqa: E501
    """

    resource_prefix = 'http://yann.lecun.com/exdb/mnist/'
    resources = {
        'train_image_file':
        ('train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
        'train_label_file':
        ('train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
        'test_image_file':
        ('t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
        'test_label_file':
        ('t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')
    }

    CLASSES = [
        '0 - zero', '1 - one', '2 - two', '3 - three', '4 - four', '5 - five',
        '6 - six', '7 - seven', '8 - eight', '9 - nine'
    ]

    def load_annotations(self):
        train_image_file = osp.join(
            self.data_prefix, rm_suffix(self.resources['train_image_file'][0]))
        train_label_file = osp.join(
            self.data_prefix, rm_suffix(self.resources['train_label_file'][0]))
        test_image_file = osp.join(
            self.data_prefix, rm_suffix(self.resources['test_image_file'][0]))
        test_label_file = osp.join(
            self.data_prefix, rm_suffix(self.resources['test_label_file'][0]))

        if not osp.exists(train_image_file) or not osp.exists(
                train_label_file) or not osp.exists(
                    test_image_file) or not osp.exists(test_label_file):
            self.download()

        _, world_size = get_dist_info()
        if world_size > 1:
            dist.barrier()
            assert osp.exists(train_image_file) and osp.exists(
                train_label_file) and osp.exists(
                    test_image_file) and osp.exists(test_label_file), \
                'Shared storage seems unavailable. Please download dataset ' \
                f'manually through {self.resource_prefix}.'

        train_set = (read_image_file(train_image_file),
                     read_label_file(train_label_file))
        test_set = (read_image_file(test_image_file),
                    read_label_file(test_label_file))

        if not self.test_mode:
            imgs, gt_labels = train_set
        else:
            imgs, gt_labels = test_set

        data_infos = []
        for img, gt_label in zip(imgs, gt_labels):
            gt_label = np.array(gt_label, dtype=np.int64)
            info = {'img': img.numpy(), 'gt_label': gt_label}
            data_infos.append(info)
        return data_infos

    @master_only
    def download(self):
        os.makedirs(self.data_prefix, exist_ok=True)

        # download files
        for url, md5 in self.resources.values():
            url = osp.join(self.resource_prefix, url)
            filename = url.rpartition('/')[2]
            download_and_extract_archive(
                url,
                download_root=self.data_prefix,
                filename=filename,
                md5=md5)


@DATASETS.register_module()
class FashionMNISTKD(MNISTKD):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_
    Dataset."""

    resource_prefix = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'  # noqa: E501
    resources = {
        'train_image_file':
        ('train-images-idx3-ubyte.gz', '8d4fb7e6c68d591d4c3dfef9ec88bf0d'),
        'train_label_file':
        ('train-labels-idx1-ubyte.gz', '25c81989df183df01b3e8a0aad5dffbe'),
        'test_image_file':
        ('t10k-images-idx3-ubyte.gz', 'bef4ecab320f06d8554ea6380940ec79'),
        'test_label_file':
        ('t10k-labels-idx1-ubyte.gz', 'bb300cfdad3c16e7a12a480ee83cd310')
    }
    CLASSES = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
        'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

