# Copyright (c) OpenMMLab. All rights reserved.
import json
import logging
import os

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from PIL import Image
from torch.utils.data import Dataset

from xtuner.registry import BUILDER
from xtuner.dataset.huggingface import process_hf_dataset
import copy
from .utils.internvl_data import build_transform, dynamic_preprocess
import numpy as np
import torch.nn.functional as F
from omg_llava.dataset.utils import expand2square, expand2square_mask, expand2square_points
from pycocotools import mask

# import copy
# from utils import expand2square
# from utils.internvl_data import build_transform, dynamic_preprocess
class VRPSamDataset(Dataset):

    def __init__(self,
                 image_folder,
                 image_processor,
                 data_path=None,
                 tokenizer=None,
                 offline_processed_text_folder=None,
                 max_dataset_length=None,
                 dataset_map_fn=None,
                 template_map_fn=None,
                 max_length=2048,
                 pad_image_to_square=False,
                 debug=False,
                 max_patch=12,
                 multi_image=True,
                 internvl_pad=None):
        super().__init__()
        assert offline_processed_text_folder or (data_path and tokenizer)

        self.tokenizer = tokenizer
        if isinstance(tokenizer, dict) or isinstance(
                tokenizer, Config) or isinstance(tokenizer, ConfigDict):
            tokenizer_type = self.tokenizer['type']
            del self.tokenizer['type']
            self.tokenizer = tokenizer_type(**self.tokenizer)
            self._add_special_tokens()

        if offline_processed_text_folder and data_path:
            print_log(
                'Both `offline_processed_text_folder` and '
                '`data_path` are set, and we load dataset from'
                '`offline_processed_text_folder` '
                f'({offline_processed_text_folder})',
                logger='current',
                level=logging.WARNING)
        if offline_processed_text_folder is not None:
            self.text_data = load_from_disk(offline_processed_text_folder)
        else:
            json_data = json.load(open(data_path))
            if debug:
                json_data = json_data[:1000]
            for idx in range(len(json_data)):
                if isinstance(json_data[idx]['id'], int):
                    json_data[idx]['id'] = str(json_data[idx]['id'])
            json_data = DatasetDict({'train': HFDataset.from_list(json_data)})
            self.text_data = process_hf_dataset(
                dataset=json_data,
                tokenizer=self.tokenizer,
                max_length=max_length,
                dataset_map_fn=dataset_map_fn,
                template_map_fn=template_map_fn,
                split='train',
                max_dataset_length=max_dataset_length,
                remove_unused_columns=False,
                pack_to_max_length=False,
                with_image_token=True,
                map_num_proc=32,  # because limited mem
                multi_image=multi_image,
            )
        self.image_folder = image_folder
        if isinstance(image_processor, dict) or isinstance(
                image_processor, Config) or isinstance(image_processor,
                                                       ConfigDict):
            self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = image_processor
        self.pad_image_to_square = pad_image_to_square
        self.transform = build_transform(is_train=False, input_size=448,
                                    pad2square=False, normalize_type="imagenet")
        self.max_patch = max_patch
        self.down_ratio = 1
        size = image_processor.crop_size
        if isinstance(size, int):
            self.image_h, self.image_w = size, size
        else:
            self.image_w, self.image_h = size

    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.text_data:
            cur_len = len(data_dict['input_ids'])
            if data_dict.get('image', None) is None:
                cur_len = -cur_len
            length_list.append(cur_len)
        return length_list

    def __len__(self):
        return len(self.text_data)
    
    def decode_mask(self, object_masks):
        output_image = []
        binary_masks = []
        for object_mask in object_masks:
            binary_mask = mask.decode(object_mask)
            binary_mask =  np.expand_dims(binary_mask, axis=0)
            if self.pad_image_to_square:
                binary_mask = expand2square_mask(binary_mask)
            binary_mask = torch.from_numpy(binary_mask)
            binary_mask = F.interpolate(binary_mask.unsqueeze(0), size=(self.image_h // self.down_ratio, self.image_w // self.down_ratio), mode='nearest').squeeze(0).squeeze(0)
            binary_masks.append(binary_mask)
            output_image.append(object_mask["seg_img_path"])
        if len(binary_masks) == 0:
            return None
        masks = torch.stack(binary_masks, axis=0)
        return masks, output_image

    def __getitem__(self, index):
        data_dict = copy.deepcopy(self.text_data[index])
        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            # print(image_file)
            if type(image_file)==list:
                data_dict['is_multi_image'] = True
                multi_pixel_values = []
                internvl_multi_pixel_values = []
                for per_img_file in image_file:
                    if per_img_file[0] == '/':
                        per_img_file = per_img_file[1:]
                    image = Image.open(os.path.join(self.image_folder,per_img_file)).convert('RGB')
                    #add
                    internvl_images = dynamic_preprocess(image, min_num=1, max_num=self.max_patch,
                                                image_size=448, use_thumbnail=True)
                    internvl_pixel_values = [self.transform(internvl_image) for internvl_image in internvl_images]
                    internvl_pixel_values = torch.stack(internvl_pixel_values)
                    num_patches = internvl_pixel_values.size(0)
                    #add
                    ori_width, ori_height = image.size
                    if self.pad_image_to_square:
                        image = expand2square(
                            image,
                            tuple(
                                int(x * 255) for x in self.image_processor.image_mean))
                    image = self.image_processor.preprocess(
                        image, return_tensors='pt')['pixel_values'][0]
                    multi_pixel_values.append(image)
                    internvl_multi_pixel_values.append(internvl_pixel_values)
                data_dict['pixel_values'] = multi_pixel_values
                data_dict['internvl_pixel_values'] = internvl_multi_pixel_values
                # process and get masks
                points = data_dict["points"]
                points = np.array(points)
                if self.pad_image_to_square:
                    points = expand2square_points(points, height=ori_height, width=ori_width)
                    points[:, 0] = points[:, 0] / max(ori_height, ori_width) * self.image_w
                    points[:, 1] = points[:, 1] / max(ori_height, ori_width) * self.image_h
                else:
                    points[:, 0] = points[:, 0] / ori_width * self.image_w
                    points[:, 1] = points[:, 1] / ori_height * self.image_h
                data_dict['points'] = torch.from_numpy(points)
                if data_dict['points'] is None:
                    return self.__getitem__(0)
                
                data_dict['masks'] = self.decode_mask(data_dict['rle_masks'])

                # import torchvision.transforms as transforms
                # import matplotlib.pyplot as plt
                # transform = transforms.ToPILImage()
                # image = transform(data_dict['pixel_values'][1])
                # query_image = transform(data_dict['pixel_values'][0])
                # mask = transform(data_dict['masks'][0].squeeze(0).squeeze(0))
                # fig, ax = plt.subplots(1, 3, figsize=(18, 6))
                # ax[0].imshow(image)
                # ax[0].set_title('Image')
                # ax[0].axis('off')  # 不显示坐标轴

                # # 显示掩码
                # ax[1].imshow(mask, cmap='gray')
                # ax[1].set_title('Mask')
                # ax[1].axis('off')  # 不显示坐标轴

                # ax[2].imshow(query_image)
                # ax[2].set_title('Image 2 with Point')
                # ax[2].scatter([float(data_dict['points'][0][0])], [float(data_dict['points'][0][1])], c='red', s=100)  # 在中央绘制红点
                # ax[2].axis('off')

                # # 显示子图
                # plt.savefig('/home/kas/lz_new/image_and_mask.png', bbox_inches='tight')
                # import pdb;pdb.set_trace()
                data_dict['regions'] = None
            else:
                image = Image.open(os.path.join(self.image_folder,image_file)).convert('RGB')
                ori_width, ori_height = image.size
                #add
                internvl_images = dynamic_preprocess(image, min_num=1, max_num=self.max_patch,
                                            image_size=448, use_thumbnail=True)
                internvl_pixel_values = [self.transform(internvl_image) for internvl_image in internvl_images]
                internvl_pixel_values = torch.stack(internvl_pixel_values)
                num_patches = internvl_pixel_values.size(0)
                data_dict['internvl_pixel_values'] = internvl_pixel_values
                #add
                if self.pad_image_to_square:
                    image = expand2square(
                        image,
                        tuple(
                            int(x * 255) for x in self.image_processor.image_mean))
                image = self.image_processor.preprocess(
                    image, return_tensors='pt')['pixel_values'][0]
                data_dict['pixel_values'] = image
                
                # process and get masks
                if data_dict.get("points", 0)==0:
                    data_dict['points']=None
                else:
                    points = data_dict["points"]
                    points = np.array(points)
                    if self.pad_image_to_square:
                        points = expand2square_points(points, height=ori_height, width=ori_width)
                        points[:, 0] = points[:, 0] / max(ori_height, ori_width) * self.image_w
                        points[:, 1] = points[:, 1] / max(ori_height, ori_width) * self.image_h
                    else:
                        points[:, 0] = points[:, 0] / ori_width * self.image_w
                        points[:, 1] = points[:, 1] / ori_height * self.image_h
                    data_dict['points'] = torch.from_numpy(points)
                    if data_dict['points'] is None:
                        return self.__getitem__(0)
                
                data_dict['masks'], output_image = self.decode_mask(data_dict['rle_masks'])
                #add
                output_images_pixel_values = []
                for out_image_path in output_image:
                    output_image = Image.open(out_image_path).convert('RGB')
                    # output_image.save("/home/kas/lz_new/omginternvl/test.jpg")
                    # import pdb;pdb.set_trace()
                    pr_output_images = dynamic_preprocess(output_image, min_num=1, max_num=1,image_size=448, use_thumbnail=True)
                    output_image_pixel_values = [self.transform(pr_output_image) for pr_output_image in pr_output_images]
                    output_image_pixel_values = torch.stack(output_image_pixel_values)
                    output_images_pixel_values.append(output_image_pixel_values)
                data_dict['output_images_pixel_values'] = output_images_pixel_values
                #add



                data_dict['regions'] = None
        else:
            if hasattr(self.image_processor, 'crop_size'):
                crop_size = self.image_processor.crop_size
            else:
                crop_size = self.image_processor.size
            data_dict['pixel_values'] = torch.zeros(3, crop_size['height'],
                                                    crop_size['width'])
            #add
            image = Image.new('RGB', (224, 224), (255, 255, 255))
            internvl_images = dynamic_preprocess(image, min_num=1, max_num=self.max_patch,
                                        image_size=448, use_thumbnail=True)
            internvl_pixel_values = [self.transform(internvl_image) for internvl_image in internvl_images]
            internvl_pixel_values = torch.stack(internvl_pixel_values)
            num_patches = internvl_pixel_values.size(0)
            data_dict['internvl_pixel_values'] = internvl_pixel_values
            #add
            data_dict['points']=None
            data_dict['masks'] = None
            data_dict['regions'] = None
        return data_dict

if __name__=="__main__":
    vrpsam_dataset_shot1 = dict(
    type=VRPSamDataset,
    data_path=vrpsam_data_path_shot1,
    image_folder=vrpsam_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=mmdu_map_fn,
    debug=debug,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    max_patch=multi_img_patch)