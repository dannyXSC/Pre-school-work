# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

import torch
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from torch.utils.data.sampler import BatchSampler, Sampler

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, \
    IMAGENET_INCEPTION_STD
from timm.data import create_transform
from torchvision.datasets.vision import VisionDataset

import torch.utils.data as data

import addPepperNoise
import utils
import itertools

from randaugment import fixmatch_augment_pool, RandAugmentMC


class MultiImageFolder_AddOrigin(data.Dataset):
    def __init__(self, dataset_list, transform, loader=default_loader,
                 known_data_source=True, is_test=False) -> None:
        super().__init__()
        self.loader = loader
        self.transform = transform
        self.num_transform = len(transform) + 1

        samples_list = [x.samples for x in dataset_list]
        classes_list = [x.classes for x in dataset_list]
        self.classes_list = classes_list
        self.dataset_list = dataset_list
        self.classes = [y for x in self.classes_list for y in x]

        start_id = 0
        self.samples = []
        for dataset_id, (samples, classes) in enumerate(zip(samples_list, classes_list)):
            for i, data in enumerate(samples):
                if not is_test:
                    # concat the taxonomy of all datasets
                    img, target = data[:2]
                    self.samples.append((img, target + start_id, dataset_id))
                    samples[i] = (img, target + start_id)
                else:
                    img = data
                    self.samples.append((img, None, dataset_id))
            start_id += len(classes)

    def __len__(self, ):
        return len(self.samples) * self.num_transform

    def __getitem__(self, index):
        """
        Returns:
            sample: the tensor of the input image
            target: a int tensor of class id
            dataset_id: a int number indicating the dataset id
        """
        length = len(self.samples)
        type_idx = index // length
        index = index % length

        path, target, dataset_id = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None and type_idx > 0:
            sample = self.transform[type_idx - 1](sample)
        else:
            sample = build_standard_transform()(sample)

        return sample, target, dataset_id


class MultiImageFolder(data.Dataset):
    def __init__(self, dataset_list, transform, loader=default_loader,
                 known_data_source=True, is_test=False) -> None:
        super().__init__()
        self.loader = loader
        self.transform = transform

        samples_list = [x.samples for x in dataset_list]
        classes_list = [x.classes for x in dataset_list]
        self.classes_list = classes_list
        self.dataset_list = dataset_list
        self.classes = [y for x in self.classes_list for y in x]

        start_id = 0
        self.samples = []
        for dataset_id, (samples, classes) in enumerate(zip(samples_list, classes_list)):
            for i, data in enumerate(samples):
                if not is_test:
                    # concat the taxonomy of all datasets
                    img, target = data[:2]
                    self.samples.append((img, target + start_id, dataset_id))
                    samples[i] = (img, target + start_id)
                else:
                    img = data
                    self.samples.append((img, None, dataset_id))
            start_id += len(classes)

    def __len__(self, ):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Returns:
            sample: the tensor of the input image
            target: a int tensor of class id
            dataset_id: a int number indicating the dataset id
        """
        path, target, dataset_id = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target, dataset_id


class MultiImageFolder_Unlabel(data.Dataset):
    def __init__(self, dataset_list, loader=default_loader,
                 known_data_source=True) -> None:
        super().__init__()
        self.loader = loader

        samples_list = [x.samples for x in dataset_list]
        self.dataset_list = dataset_list

        self.samples = []
        for dataset_id, samples in enumerate(samples_list):
            for i, data in enumerate(samples):
                img = data
                self.samples.append((img, None, dataset_id))

        self.transform = TransformFixMatch(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD);

    def __len__(self, ):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Returns:
            sample: the tensor of the input image
            target: a int tensor of class id
            dataset_id: a int number indicating the dataset id
        """
        path, target, dataset_id = self.samples[index]
        sample = self.loader(path)

        # if self.transform is not None:
        #     sample = self.transform(sample)
        sample_w, sample_s = self.transform(sample)

        return sample_w, sample_s, path


class UnlabelFolder(data.Dataset):
    def __init__(self, image_root, loader=default_loader):
        self.loader = loader
        self.samples = []

        for file_name in os.listdir(image_root):
            self.samples.append(os.path.join(image_root, file_name))

    def __len__(self, ):
        return len(self.samples)

    def get_image_id(self, path):
        file_name = path.split('/')[-1]
        id_name = file_name.split('.')[0]
        return int(id_name)

    def __getitem__(self, index):
        """
        Returns:
            sample: the tensor of the input image
            image_id: a int number indicating the image id
        """
        path = self.samples[index]
        target = None
        image_id = self.get_image_id(path)
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, image_id


class TestFolder(data.Dataset):
    def __init__(self, image_root, transform, loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.samples = []
        self.classes = os.listdir(os.path.join(image_root, 'train'))
        image_root = os.path.join(image_root, 'test')

        for file_name in os.listdir(image_root):
            self.samples.append(os.path.join(image_root, file_name))

    def __len__(self, ):
        return len(self.samples)

    def get_image_id(self, path):
        file_name = path.split('/')[-1]
        id_name = file_name.split('.')[0]
        return int(id_name)

    def __getitem__(self, index):
        """
        Returns:
            sample: the tensor of the input image
            image_id: a int number indicating the image id
        """
        path = self.samples[index]
        target = None
        image_id = self.get_image_id(path)
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, image_id


def build_dataset(is_train, args, is_unlabel=False):
    is_test = not is_train and args.test_only
    transform = build_transform(is_train, args, is_unlabel)

    dataset_list = []
    nb_classes = 0

    if is_unlabel:
        for dataset in args.dataset_list:
            root = os.path.join(args.data_path, dataset, 'unlabel')
            dataset = UnlabelFolder(root)
            dataset_list.append(dataset)

        multi_dataset = MultiImageFolder_Unlabel(dataset_list)
        return multi_dataset
    elif is_test:
        for dataset in args.dataset_list:
            root = os.path.join(args.data_path, dataset)
            dataset = TestFolder(root, transform=transform)
            dataset_list.append(dataset)
            nb_classes += len(dataset.classes)
        multi_dataset = MultiImageFolder(dataset_list, transform, is_test=True)
        return multi_dataset, nb_classes, None
    else:
        for dataset in args.dataset_list:
            root = os.path.join(args.data_path, dataset, 'train' if is_train else 'val')
            dataset = datasets.ImageFolder(root, transform=transform)
            dataset_list.append(dataset)
            nb_classes += len(dataset.classes)

        if is_train:
            multi_dataset = MultiImageFolder_AddOrigin(dataset_list, transform,
                                                       known_data_source=args.known_data_source)
        else:
            # eval
            multi_dataset = MultiImageFolder(dataset_list, transform,
                                             known_data_source=args.known_data_source)

        return multi_dataset, nb_classes


def build_transform(is_train, args, img_size=224,
                    mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    # TODO: does any other data augmentation work better?
    if is_train:
        t_list = []
        t_list.append(transforms.Compose(
            [transforms.Resize(img_size), transforms.RandomCrop(img_size), transforms.ToTensor(),
             transforms.Normalize(mean, std)]))

        def cur_customized_transform(T):
            t_list.append(transforms.Compose(
                [transforms.Resize(256), T,
                 transforms.ToTensor(),
                 transforms.Normalize(mean, std)]))
            t_list.append(transforms.Compose(
                [transforms.Resize(256), T, transforms.RandomHorizontalFlip(p=1),
                 transforms.ToTensor(),
                 transforms.Normalize(mean, std)]))

        # t_list.append(transforms.Compose(
        #     [transforms.Resize(img_size), transforms.RandomCrop(img_size), transforms.ToTensor(),
        #      transforms.Normalize(mean, std)]))
        # cur_customized_transform(transforms.Lambda(my_crop_top_left))
        # cur_customized_transform(transforms.Lambda(my_crop_top_right))
        # cur_customized_transform(transforms.Lambda(my_crop_down_right))
        # cur_customized_transform(transforms.Lambda(my_crop_down_left))
        t_list.append(transforms.Compose(
            [transforms.Resize(img_size), transforms.RandomCrop(img_size),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)]))

        if args.flip and args.rotation:
            # t_list.append(build_customerised_transform(transforms.RandomChoice(
            #     [transforms.RandomVerticalFlip(p=args.flip), transforms.RandomHorizontalFlip(p=args.flip),
            #      transforms.RandomRotation(args.rotation)]), img_size=img_size, mean=mean, std=std))
            t_list.append(build_customerised_transform(transforms.RandomHorizontalFlip(p=1)))
            # t_list.append(
            #     build_customerised_transform(transforms.RandomAffine(0, translate=(0.5, 0.5)), img_size=img_size, mean=mean,
            #                                  std=std))
            # t_list.append(build_customerised_transform(
            #     transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=2, fill=0), img_size=img_size,
            #     mean=mean, std=std))
        # 增加白噪音
        t_list.append(
            build_customerised_transform(addPepperNoise.AddPepperNoise(0.9, p=0.5), img_size=img_size, mean=mean,
                                         std=std))
        t_list.append(
            build_customerised_transform(transforms.ColorJitter(hue=0.5), img_size=img_size, mean=mean,
                                         std=std))
        t_list.append(
            build_customerised_transform(transforms.RandomGrayscale(p=1), img_size=img_size, mean=mean,
                                         std=std))
        return t_list

    return build_standard_transform(img_size, mean, std)


def build_standard_transform(img_size=224,
                             mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    t = []
    t.append(transforms.Resize(img_size))
    t.append(transforms.CenterCrop(img_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def build_customerised_transform(T, img_size=224,
                                 mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    t = []
    t.append(transforms.Resize(img_size))
    t.append(transforms.CenterCrop(img_size))
    if isinstance(T, list):
        for item in T:
            t.append(item)
    else:
        t.append(T)
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=224,
                                  padding=int(224 * 0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=224,
                                  padding=int(224 * 0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class GroupedDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, batch_sizes, num_datasets):
        """
        Group images from the same dataset into a batch
        """
        self.dataset = dataset
        self.batch_sizes = batch_sizes
        self._buckets = [[] for _ in range(num_datasets)]

    def __len__(self):
        return len(self.dataset) // self.batch_sizes

    def __iter__(self):
        iter_id = 0
        while True:
            for d in self.dataset:
                image, target, dataset_id = d
                bucket = self._buckets[dataset_id]
                bucket.append(d)
                if len(bucket) == self.batch_sizes:
                    images, targets, dataset_ids = list(zip(*bucket))
                    images = torch.stack(images)
                    targets = torch.tensor(targets)
                    dataset_ids = torch.tensor(dataset_ids)
                    del bucket[:]
                    yield images, targets, dataset_ids
                    iter_id += 1
                    if iter_id == len(self):
                        return
