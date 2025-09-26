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
from pycocotools import mask
import numpy as np
import torch.nn.functional as F
import copy

from xtuner.registry import BUILDER
from omg_llava.dataset.utils import expand2square, expand2square_mask
from xtuner.dataset.huggingface import process_hf_dataset
from .utils.internvl_data import build_transform, dynamic_preprocess
class GCGDataset(Dataset):

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
                 num_proc=8,
                 debug=False,
                 repeats=1,
                 max_patch=12,
                 internvl_pad=None,
                 use_thumbnail=True):
        super().__init__()

        assert offline_processed_text_folder or (data_path and tokenizer)
        self.use_thumbnail=use_thumbnail
        self.debug = debug
        if offline_processed_text_folder and data_path:
            print_log(
                'Both `offline_processed_text_folder` and '
                '`data_path` are set, and we load dataset from'
                '`offline_processed_text_folder` '
                f'({offline_processed_text_folder})',
                logger='current',
                level=logging.WARNING)

        if offline_processed_text_folder is not None:
            raise NotImplementedError
        else:
            json_data = self.json_file_preprocess(data_path)
            json_data = DatasetDict({'train': HFDataset.from_list(json_data)})
            self.text_data = process_hf_dataset(
                dataset=json_data,
                tokenizer=tokenizer,
                max_length=max_length,
                dataset_map_fn=dataset_map_fn,
                template_map_fn=template_map_fn,
                split='train',
                max_dataset_length=max_dataset_length,
                remove_unused_columns=False,
                pack_to_max_length=False,
                with_image_token=True,
                map_num_proc=num_proc,  # because limited mem
            )

        self.image_folder = image_folder
        size = image_processor.crop_size
        if isinstance(size, int):
            self.image_h, self.image_w = size, size
        else:
            self.image_w, self.image_h = size

        if isinstance(image_processor, dict) or isinstance(
                image_processor, Config) or isinstance(image_processor,
                                                       ConfigDict):
            self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = image_processor
        self.pad_image_to_square = pad_image_to_square
        self.down_ratio = 1
        self.repeats = repeats
        self.transform = build_transform(is_train=False, input_size=448,
                                    pad2square=False, normalize_type="imagenet")
        self.max_patch = max_patch
        self.internvl_pad=internvl_pad

    def json_file_preprocess(self, data_path):
        with open(data_path, 'r') as f:
            json_data = json.load(f)

        # for quickly debug with mini split
        if self.debug:
            json_data = json_data[:100]
        return json_data

    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.text_data:
            cur_len = len(data_dict['input_ids'])
            if data_dict.get('image', None) is None:
                cur_len = -cur_len
            length_list.append(cur_len)
        length_list = length_list * self.repeats
        return length_list

    def __len__(self):
        return len(self.text_data) * self.repeats

    def real_len(self):
        return len(self.text_data)

    def decode_mask(self, object_masks, ori_height, ori_width):
        binary_masks = []
        for object_mask in object_masks:
            binary_mask = np.zeros((ori_height, ori_width), dtype=np.uint8)
            for seg in object_mask:
                rles = mask.frPyObjects([seg], ori_height, ori_width)
                m = mask.decode(rles)
                m = m.astype(np.uint8)
                binary_mask += m.squeeze()

            binary_masks.append(binary_mask)
        if len(binary_masks) == 0:
            return None
        masks = np.stack(binary_masks, axis=0)
        if self.pad_image_to_square:
            masks = expand2square_mask(masks)
        masks = torch.from_numpy(masks)
        masks = F.interpolate(masks.unsqueeze(0), size=(self.image_h // self.down_ratio, self.image_w // self.down_ratio), mode='nearest').squeeze(0)
        return masks

    def __getitem__(self, index):
        index = index % self.real_len()
        data_dict = copy.deepcopy(self.text_data[index])

        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            image = Image.open(os.path.join(self.image_folder,
                                            image_file)).convert('RGB')
            #add
            internvl_images = dynamic_preprocess(image, min_num=1, max_num=self.max_patch,
                                        image_size=448, use_thumbnail=self.use_thumbnail)
            internvl_pixel_values = [self.transform(internvl_image) for internvl_image in internvl_images]
            internvl_pixel_values = torch.stack(internvl_pixel_values)
            num_patches = internvl_pixel_values.size(0)
            data_dict['internvl_pixel_values'] = internvl_pixel_values
            #add

            #add
            output_images_pixel_values = []
            
            for out_image_path in data_dict['output_image']:
                output_image = Image.open(out_image_path).convert('RGB')
                if self.internvl_pad is not None:
                    try:
                        if self.internvl_pad=="white":
                            output_image = expand2square(output_image, tuple([255,255,255]))
                                
                        elif self.internvl_pad=="black":
                            output_image = expand2square(output_image,tuple([0,0,0]))
                        else:
                            print(self.internvl_pad)
                    except:
                        print(output_image.size())

                    # output_image.save("/home/kas/lz_new/omginternvl/test.jpg")
                    # import pdb;pdb.set_trace()
                pr_output_images = dynamic_preprocess(output_image, min_num=1, max_num=1,image_size=448, use_thumbnail=True)
                output_image_pixel_values = [self.transform(pr_output_image) for pr_output_image in pr_output_images]
                output_image_pixel_values = torch.stack(output_image_pixel_values)
                output_images_pixel_values.append(output_image_pixel_values)
            data_dict['output_images_pixel_values'] = output_images_pixel_values
            #add



            ori_width, ori_height = image.size
            if self.pad_image_to_square:
                image = expand2square(
                    image,
                    tuple(
                        int(x * 255) for x in self.image_processor.image_mean))
            image = self.image_processor.preprocess(
                image, return_tensors='pt')['pixel_values'][0]
            data_dict['pixel_values'] = image

            # process and get masks
            data_dict['masks'] = self.decode_mask(data_dict['masks'], ori_height=ori_height, ori_width=ori_width)
            if data_dict['masks'] is None:
                return self.__getitem__(0)
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
                                        image_size=448, use_thumbnail=self.use_thumbnail)
            internvl_pixel_values = [self.transform(internvl_image) for internvl_image in internvl_images]
            internvl_pixel_values = torch.stack(internvl_pixel_values)
            num_patches = internvl_pixel_values.size(0)
            data_dict['internvl_pixel_values'] = internvl_pixel_values
            #add
            data_dict['masks'] = None
        return data_dict

class RefCOCOgGCGDataset(GCGDataset):
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
                 repeats=1,
                 max_patch=12,
                 internvl_pad=None,
                 use_thumbnail=True):
        super().__init__(
            image_folder=image_folder,
            image_processor=image_processor,
            data_path=data_path,
            tokenizer=tokenizer,
            offline_processed_text_folder=offline_processed_text_folder,
            max_dataset_length=max_dataset_length,
            dataset_map_fn=dataset_map_fn,
            template_map_fn=template_map_fn,
            max_length=max_length,
            pad_image_to_square=pad_image_to_square,
            debug=debug,
            repeats=repeats,
            max_patch=max_patch,
            internvl_pad=internvl_pad,
            use_thumbnail=use_thumbnail
        )

    def json_file_preprocess(self, data_path):
        json_data = json.load(open(data_path))
        if self.debug:
            json_data = json_data[:1000]

        # convert {id: dict} to dict(..., id=xx)
        for idx in range(len(json_data)):
            id = list(json_data[idx].keys())[0]
            json_data[idx] = json_data[idx][id]
            json_data[idx].update({'id': id})
        return json_data

class GranDfGCGDataset(GCGDataset):
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
                 num_proc=8,
                 debug=False,
                 repeats=1,
                 max_patch=12,
                 internvl_pad=None,
                 use_thumbnail=True):
        super().__init__(
            image_folder=image_folder,
            image_processor=image_processor,
            data_path=data_path,
            tokenizer=tokenizer,
            offline_processed_text_folder=offline_processed_text_folder,
            max_dataset_length=max_dataset_length,
            dataset_map_fn=dataset_map_fn,
            template_map_fn=template_map_fn,
            max_length=max_length,
            pad_image_to_square=pad_image_to_square,
            num_proc=num_proc,
            debug=debug,
            repeats=repeats,
            max_patch=max_patch,
            internvl_pad=internvl_pad,
            use_thumbnail=use_thumbnail
        )

    def decode_mask(self, object_masks, ori_height, ori_width):
        binary_masks = []
        for object_mask in object_masks:
            binary_mask = np.zeros((ori_height, ori_width), dtype=np.uint8)

            for rle in object_mask:
                m = mask.decode(rle).astype(np.uint8)
                binary_mask += m.squeeze()

            binary_masks.append(binary_mask)
        if len(binary_masks) == 0:
            return None
        masks = np.stack(binary_masks, axis=0)
        if self.pad_image_to_square:
            masks = expand2square_mask(masks)
        masks = torch.from_numpy(masks)
        masks = F.interpolate(masks.unsqueeze(0), size=(self.image_h // self.down_ratio, self.image_w // self.down_ratio), mode='nearest').squeeze(0)
        return masks

class OpenPsgGCGDataset(GranDfGCGDataset):
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
                 num_proc=8,
                 debug=False,
                 repeats=1,
                 max_patch=12,
                 internvl_pad=None,
                 use_thumbnail=True):
        super().__init__(
            image_folder=image_folder,
            image_processor=image_processor,
            data_path=data_path,
            tokenizer=tokenizer,
            offline_processed_text_folder=offline_processed_text_folder,
            max_dataset_length=max_dataset_length,
            dataset_map_fn=dataset_map_fn,
            template_map_fn=template_map_fn,
            max_length=max_length,
            pad_image_to_square=pad_image_to_square,
            num_proc=num_proc,
            debug=debug,
            repeats=repeats,
            max_patch=max_patch,
            internvl_pad=internvl_pad,
            use_thumbnail=use_thumbnail
        )

class FlickrGCGDataset(GCGDataset):
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
                 num_proc=8,
                 debug=False,
                 repeats=1,
                 max_patch=12,
                 internvl_pad=None,
                 use_thumbnail=True):
        super().__init__(
            image_folder=image_folder,
            image_processor=image_processor,
            data_path=data_path,
            tokenizer=tokenizer,
            offline_processed_text_folder=offline_processed_text_folder,
            max_dataset_length=max_dataset_length,
            dataset_map_fn=dataset_map_fn,
            template_map_fn=template_map_fn,
            max_length=max_length,
            pad_image_to_square=pad_image_to_square,
            num_proc=num_proc,
            debug=debug,
            repeats=repeats,
            max_patch=max_patch,
            internvl_pad=internvl_pad,
            use_thumbnail=use_thumbnail
        )

    def json_file_preprocess(self, data_path):
        def filter_images(data_infos, min_size):
            return [i for i, info in enumerate(data_infos) if min(info['width'], info['height']) >= min_size]

        # convert {id: dict} to dict(..., id=xx)
        from pycocotools.coco import COCO
        self.coco = COCO(data_path)
        self.image_ids = self.coco.getImgIds()
        data_infos = []
        total_ann_ids = []
        removed_img_count = 0
        for img_id in self.image_ids:
            info = self.coco.loadImgs([img_id])[0]
            if len(info['caption'].split(' ')) < 3:
                removed_img_count += 1
                continue
            info['filename'] = info['file_name'].split('_')[-1]
            info['height'] = int(info['height'])
            info['width'] = int(info['width'])
            data_infos.append(info)
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(total_ann_ids), f"Non-unique annotation IDs in '{data_path}'!"
        print(f'Removed {removed_img_count} images.')
        data_infos = [data_infos[i] for i in filter_images(data_infos, min_size=32)]

        # obtain_annotations
        for data_info in data_infos:
            ann_ids = self.coco.getAnnIds(imgIds=data_info['id'])
            ann_info = self.coco.loadAnns(ann_ids)
            data_info.update({'ann_info': ann_info})
        if self.debug:
            data_infos = data_infos[:1000]
        return data_infos

    def decode_mask(self, object_masks, ori_height, ori_width):
        binary_masks = []
        for object_mask in object_masks:
            binary_mask = mask.decode(object_mask).astype(np.uint8)
            binary_masks.append(binary_mask)
        if len(binary_masks) == 0:
            return None
        masks = np.stack(binary_masks, axis=0)
        if self.pad_image_to_square:
            masks = expand2square_mask(masks)
        masks = torch.from_numpy(masks)
        masks = F.interpolate(masks.unsqueeze(0), size=(self.image_h // self.down_ratio, self.image_w // self.down_ratio), mode='nearest').squeeze(0)
        return masks