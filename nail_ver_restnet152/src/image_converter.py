#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: A.Akdogan
"""

### 데이터 전처리 과정 ###
import os
import cv2
import json
import numpy as np
from pathlib import Path

class ImageConverter:

    def __init__(self, row, col):
        self.row = row
        self.col = col

    @staticmethod
    def get_final_path(sub_count, join_list):
        path = os.path.dirname(os.path.realpath(__file__))
        for i in range(sub_count):
            path = os.path.dirname(os.path.normpath(path))
        for item in join_list:
            path = os.path.join(path, item)
        return path

    def create_masks_from_json(self, json_path, img_shape):
        with open(json_path, 'r') as file:
            data = json.load(file)

        image_masks = {}
        for image in data['images']:
            image_id = image['id']
            image_file_name = image['file_name']
            mask = np.zeros(img_shape, dtype=np.uint8)

            for annotation in data['annotations']:
                if annotation['image_id'] == image_id:
                    segmentation = annotation['segmentation'][0]
                    poly = np.array(segmentation).reshape((-1, 1, 2)).astype(np.int32)
                    cv2.fillPoly(mask, [poly], 255)

            image_masks[image_file_name] = mask

        return image_masks

    def converter(self, files, base_folder, dest_folder, tp):
        Path(os.path.join(dest_folder, tp)).mkdir(parents=True, exist_ok=True)
        for file_name in files:
            file_path = os.path.join(base_folder, tp, file_name)
            if file_name.endswith('.json'):
                print(f"Processing {file_name}")
                img_shape = (self.row, self.col)
                image_masks = self.create_masks_from_json(file_path, img_shape)
                for image_file_name, mask in image_masks.items():
                    resized_mask = cv2.resize(mask, (self.col, self.row))
                    save_filename = os.path.join(dest_folder, tp, image_file_name)  # 원본 파일명과 동일하게 저장
                    cv2.imwrite(save_filename, resized_mask)
                    print(f"Saved {save_filename}")
            elif file_name.lower().endswith(('.jpg', '.jpeg')):
                img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                resized_img = cv2.resize(img, (self.col, self.row))
                save_filename = os.path.join(dest_folder, tp, file_name)
                cv2.imwrite(save_filename, resized_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                print(f"Processed and saved {save_filename}")

    def main(self):
        raw_base_path = self.get_final_path(1, ['dataset', 'raw'])
        processed_base_path = self.get_final_path(1, ['dataset', 'processed'])

        folders = ['train', 'test', 'val', 'train_labels', 'test_labels', 'val_labels']
        for folder in folders:
            folder_path = os.path.join(raw_base_path, folder)
            files = os.listdir(folder_path)
            self.converter(files, raw_base_path, processed_base_path, folder)

if __name__ == '__main__':
    ImageConverter(800, 800).main()
