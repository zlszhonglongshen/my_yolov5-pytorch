import torch
from torch.utils.data import Dataset
import random
import numpy as np
import cv2
import os
import xml.etree.ElementTree as ET
import json

class Data_Name_Color():
    def __init__(self, labels_txt):
        self.class_names = list(map(lambda x: x.strip(), open(labels_txt, 'r').readlines()))
        self.cats_to_ids = dict(map(reversed, enumerate(self.class_names)))
        self.ids_to_cats = dict(enumerate(self.class_names))
        self.num_classes = len(self.class_names)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.class_names))]

class MyDataset_xml(Dataset):
    def __init__(self, imgroot, labroot, imgs_txt, labels_txt, image_size=416, input_norm=False, augment=False, keep_difficult=False):
        self.imgroot = imgroot
        self.labroot = labroot
        self.image_size = image_size  ###square
        # self.img_list = list(map(lambda x: x.strip(), open(imgs_txt, 'r').readlines()))
        self.img_list = list(map(lambda x: x.strip()+'.jpg', open(imgs_txt, 'r').readlines()))
        self.nSamples = len(self.img_list)
        self.augment = augment
        self.input_norm = input_norm
        self.class_names = list(map(lambda x: x.strip(), open(labels_txt, 'r').readlines()))
        self.num_classes = len(self.class_names)
        self.keep_difficult = keep_difficult

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        name = self.img_list[index]
        imgpath = os.path.join(self.imgroot, name)
        img = cv2.imread(imgpath)
        dh, dw = 1. / (img.shape[0]), 1. / (img.shape[1])
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        xmlpath = os.path.join(self.labroot, os.path.splitext(os.path.basename(name))[0] + '.xml')
        labels = np.empty((0,5), dtype=np.float32)
        if os.path.exists(xmlpath):
            tree = ET.parse(xmlpath)
            root = tree.getroot()
            for obj in root.findall('object'):
                # 判断difficult
                difficult = 0
                if obj.find('difficult') != None:
                    difficult = obj.find('difficult').text
                if not self.keep_difficult and int(difficult)==1:
                    continue
                label = obj.find('name').text
                cat = float(self.class_names.index(label))
                bbox_tag = obj.find('bndbox')
                xmin = float(bbox_tag.find('xmin').text)
                ymin = float(bbox_tag.find('ymin').text)
                xmax = float(bbox_tag.find('xmax').text)
                ymax = float(bbox_tag.find('ymax').text)
                obj_arr = np.array(
                    [[cat, (xmin + xmax) * 0.5 * dw, (ymin + ymax) * 0.5 * dh, (xmax - xmin) * dw, (ymax - ymin) * dh]],
                    dtype=np.float32)
                labels = np.append(labels, obj_arr, axis=0)

        # labels = labels.reshape(-1, 5)
        nL = labels.shape[0]  # number of labels
        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip and random.random() < 0.5:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)
        img = np.ascontiguousarray(img, dtype=np.float32)
        if self.input_norm:
            img /= 255.0
        return torch.from_numpy(img), labels_out

    def collate_fn(self, batch):
        img, label = list(zip(*batch))  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0)

class MyDataset_xml_ratio(Dataset):
    def __init__(self, imgroot, labroot, imgs_txt, labels_txt, image_size=416, input_norm=False, augment=False, keep_ratio=False, keep_difficult=False):
        self.imgroot = imgroot
        self.labroot = labroot
        self.image_size = image_size  ###square
        self.img_list = list(map(lambda x: x.strip(), open(imgs_txt, 'r').readlines()))
        self.nSamples = len(self.img_list)
        self.augment = augment
        self.input_norm = input_norm
        self.class_names = list(map(lambda x: x.strip(), open(labels_txt, 'r').readlines()))
        self.num_classes = len(self.class_names)
        self.keep_ratio = keep_ratio
        self.keep_difficult = keep_difficult
    def __len__(self):
        return self.nSamples
    def resize_image(self, srcimg):
        top, left, newh, neww = 0, 0, self.image_size, self.image_size
        if self.keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.image_size, int(self.image_size / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.image_size - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.image_size - neww - left, cv2.BORDER_CONSTANT, value=0)  # add border
            else:
                newh, neww = int(self.image_size * hw_scale), self.image_size
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.image_size - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.image_size - newh - top, 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.resize(srcimg, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left

    def load_xml(self, xmlpath, srcshape, newshape, pad_hw):
        if self.keep_ratio and srcshape[0] != srcshape[1]:
            dh, dw = 1. / self.image_size, 1. / self.image_size  ### max(newshape)
            ratioh, ratiow = newshape[0] / (srcshape[0]), newshape[1] / (srcshape[1])
            labels = np.empty((0, 5), dtype=np.float32)
            tree = ET.parse(xmlpath)
            root = tree.getroot()
            for obj in root.findall('object'):
                # 判断difficult
                difficult = 0
                if obj.find('difficult') != None:
                    difficult = obj.find('difficult').text
                if not self.keep_difficult and int(difficult) == 1:
                    continue
                label = obj.find('name').text
                cat = float(self.class_names.index(label))
                bbox_tag = obj.find('bndbox')
                xmin = float(bbox_tag.find('xmin').text) * ratiow + pad_hw[1]
                ymin = float(bbox_tag.find('ymin').text) * ratioh + pad_hw[0]
                xmax = float(bbox_tag.find('xmax').text) * ratiow + pad_hw[1]
                ymax = float(bbox_tag.find('ymax').text) * ratioh + pad_hw[0]
                obj_arr = np.array(
                    [[cat, (xmin + xmax) * 0.5 * dw, (ymin + ymax) * 0.5 * dh, (xmax - xmin) * dw, (ymax - ymin) * dh]],
                    dtype=np.float32)
                labels = np.append(labels, obj_arr, axis=0)
        else:
            dh, dw = 1. / (srcshape[0]), 1. / (srcshape[1])
            labels = np.empty((0, 5), dtype=np.float32)
            tree = ET.parse(xmlpath)
            root = tree.getroot()
            for obj in root.findall('object'):
                label = obj.find('name').text
                cat = float(self.class_names.index(label))
                bbox_tag = obj.find('bndbox')
                xmin = float(bbox_tag.find('xmin').text)
                ymin = float(bbox_tag.find('ymin').text)
                xmax = float(bbox_tag.find('xmax').text)
                ymax = float(bbox_tag.find('ymax').text)
                obj_arr = np.array(
                    [[cat, (xmin + xmax) * 0.5 * dw, (ymin + ymax) * 0.5 * dh, (xmax - xmin) * dw, (ymax - ymin) * dh]],
                    dtype=np.float32)
                labels = np.append(labels, obj_arr, axis=0)
        return labels

    def __getitem__(self, index):
        name = self.img_list[index]
        imgpath = os.path.join(self.imgroot, name)
        srcimg = cv2.imread(imgpath)
        img, newh, neww, top, left = self.resize_image(srcimg)
        xmlpath = os.path.join(self.labroot, os.path.splitext(os.path.basename(name))[0] + '.xml')
        if os.path.exists(xmlpath):
            labels = self.load_xml(xmlpath, (srcimg.shape[0], srcimg.shape[1]), (newh, neww), (top, left))
        else:
            labels = np.empty((0, 5), dtype=np.float32)

        nL = labels.shape[0]  # number of labels
        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip and random.random() < 0.5:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)
        img = np.ascontiguousarray(img, dtype=np.float32)
        if self.input_norm:
            img /= 255.0
        return torch.from_numpy(img), labels_out

    def collate_fn(self, batch):
        img, label = list(zip(*batch))  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0)

class MyDataset_json(Dataset):
    def __init__(self, imgroot, labroot, imgs_txt, labels_txt, image_size=416, input_norm=False, augment=False, train=True):
        self.imgroot = os.path.join(imgroot, 'train') if train else os.path.join(imgroot, 'test')
        self.labroot = os.path.join(labroot, 'train') if train else os.path.join(labroot, 'test')
        self.image_size = image_size  ###square
        self.img_list = list(map(lambda x: x.strip(), open(imgs_txt, 'r').readlines()))
        self.nSamples = len(self.img_list)
        self.augment = augment
        self.input_norm = input_norm
        self.class_names = list(map(lambda x: x.strip(), open(labels_txt, 'r').readlines()))
        self.num_classes = len(self.class_names)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        name = os.path.basename(self.img_list[index])
        imgpath = os.path.join(self.imgroot, name)
        img = cv2.imread(imgpath)
        dh, dw = 1. / (img.shape[0]), 1. / (img.shape[1])
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        jsonpath = os.path.join(self.labroot, os.path.splitext(name)[0] + '.xml')
        labels = np.empty((0,5), dtype=np.float32)
        if os.path.exists(jsonpath):
            f = open(jsonpath)
            info = json.load(f)
            objects = info['frames'][0]['objects']
            for i in objects:
                label = i['category']
                cat = float(self.class_names.index(label))
                xmin = float(i['box2d']['x1'])
                ymin = float(i['box2d']['y1'])
                xmax = float(i['box2d']['x2'])
                ymax = float(i['box2d']['y2'])
                obj_arr = np.array(
                    [[cat, (xmin + xmax) * 0.5 * dw, (ymin + ymax) * 0.5 * dh, (xmax - xmin) * dw, (ymax - ymin) * dh]],
                    dtype=np.float32)
                labels = np.append(labels, obj_arr, axis=0)

        # labels = labels.reshape(-1, 5)
        nL = labels.shape[0]  # number of labels
        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip and random.random() < 0.5:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)
        img = np.ascontiguousarray(img, dtype=np.float32)
        if self.input_norm:
            img /= 255.0
        return torch.from_numpy(img), labels_out

    def collate_fn(self, batch):
        img, label = list(zip(*batch))  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0)