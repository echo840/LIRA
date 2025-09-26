import argparse
import base64
import itertools
import json
import os
import random
import time
from functools import partial
from io import BytesIO

import pandas as pd
import torch
# from internvl.model.internvl_chat import InternVLChatModel
# from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

import sys
from omg_llava.tools.app_utils import process_markdown, show_mask_pred, parse_visual_prompts, description

import torch
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)
from transformers.generation.streamers import TextStreamer

from xtuner.dataset.utils import expand2square, load_image
from omg_llava.dataset.utils import expand2square_bbox, expand2square_mask, expand2square_points
from omg_llava.model.utils import prepare_inputs_labels_for_multimodal_with_visual_prompts
from xtuner.tools.utils import get_stop_criteria
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          PROMPT_TEMPLATE, SYSTEM_TEMPLATE)

import argparse
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.fileio import PetrelBackend, get_file_backend

from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint
from xtuner.registry import BUILDER

from gradio_image_prompter import ImagePrompter
from omg_llava.dataset.utils.internvl_data import build_transform, dynamic_preprocess
from PIL import Image,ImageDraw
from pathlib import Path
import os
import tempfile
import uuid
import re
import json
from tqdm import tqdm
TORCH_DTYPE_MAP = dict(fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')

ds_collections = {
    'mmbench_dev_20230712': {
        'root': 'internvl/data/mmbench/mmbench_dev_20230712.tsv',
        'max_new_tokens': 100,
        'min_new_tokens': 1,
        'type': 'dev',
        'language': 'en'
    },
    'mmbench_dev_cn_20231003': {
        'root': 'internvl/data/mmbench/mmbench_dev_cn_20231003.tsv',
        'max_new_tokens': 100,
        'min_new_tokens': 1,
        'type': 'dev',
        'language': 'cn'
    },
    'mmbench_dev_en_20231003': {
        'root': 'internvl/data/mmbench/mmbench_dev_en_20231003.tsv',
        'max_new_tokens': 100,
        'min_new_tokens': 1,
        'type': 'dev',
        'language': 'en'
    },
    'mmbench_test_cn_20231003': {
        'root': 'internvl/data/mmbench/mmbench_test_cn_20231003.tsv',
        'max_new_tokens': 100,
        'min_new_tokens': 1,
        'type': 'test',
        'language': 'cn'
    },
    'mmbench_test_en_20231003': {
        'root': 'internvl/data/mmbench/mmbench_test_en_20231003.tsv',
        'max_new_tokens': 100,
        'min_new_tokens': 1,
        'type': 'test',
        'language': 'en'
    },
    'ccbench_dev_cn': {
        'root': 'internvl/data/mmbench/CCBench_legacy.tsv',
        'max_new_tokens': 100,
        'min_new_tokens': 1,
        'type': 'dev',
        'language': 'cn'
    }
}


def collate_fn(batches):
    pixel_values = torch.stack([_['pixel_values'] for _ in batches])
    questions = [_['question'] for _ in batches]
    answers = [_['answer'] for _ in batches]
    indexes = [_['index'] for _ in batches]
    options = [_['option'] for _ in batches]
    internvl_pixel_values = [_['internvl_pixel_values'] for _ in batches]
    return pixel_values, questions, answers, indexes, options, internvl_pixel_values


class MMBenchDataset(torch.utils.data.Dataset):

    def __init__(self, root, prompt, language, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6,min_num=1,image_processor=None):
        self.df = pd.read_csv(root, sep='\t')
        self.prompt = prompt
        self.language = language
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.min_num = min_num
        self.root = root

        self.image_processor = image_processor
        self.transform = build_transform(is_train=False,input_size=448, pad2square=False, normalize_type="imagenet")
        self.pad_image_to_square = True

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        index = self.df.iloc[idx]['index']
        image = self.df.iloc[idx]['image']
        question = self.df.iloc[idx]['question']
        answer = self.df.iloc[idx]['answer'] if 'answer' in self.df.iloc[0].keys() else None
        # catetory = self.df.iloc[idx]['category']
        # l2_catetory = self.df.iloc[idx]['l2-category']

        # image = Image.open(BytesIO(base64.b64decode(image))).convert('RGB')
        # if self.dynamic_image_size:
        #     images = dynamic_preprocess(image, image_size=self.input_size,
        #                                 use_thumbnail=self.use_thumbnail,
        #                                 max_num=self.max_num)
        # else:
        #     images = [image]
        # pixel_values = [self.transform(image) for image in images]
        # pixel_values = torch.stack(pixel_values)


        catetory = self.df.iloc[idx]['category']
        # l2_catetory = self.df.iloc[idx]['l2-category']

        ori_image = Image.open(BytesIO(base64.b64decode(image))).convert('RGB')
        # if self.root == 'internvl/data/mmbench/mmbench_test_cn_20231003.tsv':
        #     internvl_images = dynamic_preprocess(ori_image, min_num=self.min_num, max_num=self.max_num,image_size=448, use_thumbnail=True)
        # else:
        #     internvl_images = dynamic_preprocess(ori_image, min_num=self.min_num, max_num=self.max_num,image_size=448, use_thumbnail=True)
        internvl_images = dynamic_preprocess(ori_image, min_num=self.min_num, max_num=self.max_num,image_size=448, use_thumbnail=True)
        internvl_pixel_values = [self.transform(internvl_image) for internvl_image in internvl_images]
        internvl_pixel_values = torch.stack(internvl_pixel_values)
        ori_width, ori_height = ori_image.size
        if self.pad_image_to_square:
            image = expand2square(
                ori_image,
                tuple(
                    int(x * 255) for x in self.image_processor.image_mean))
        image = self.image_processor.preprocess(
            image, return_tensors='pt')['pixel_values'][0]
        pixel_values = image

        option_candidate = ['A', 'B', 'C', 'D', 'E']
        options = {
            cand: self.load_from_df(idx, cand)
            for cand in option_candidate
            if self.load_from_df(idx, cand) is not None
        }

        hint = self.load_from_df(idx, 'hint')
        if hint is not None:
            question = hint + '\n' + question
        for key, item in options.items():
            question += f'\n{key}. {item}'
        if self.language == 'cn':
            question = question + '\n' + self.prompt['cn']
        else:
            question = question + '\n' + self.prompt['en']

        return {
            'question': question,
            'pixel_values': pixel_values,
            "internvl_pixel_values":internvl_pixel_values,
            'answer': answer,
            'index': index,
            'option': options
        }

    def load_from_df(self, idx, key):
        if key in self.df.iloc[idx] and not pd.isna(self.df.iloc[idx][key]):
            return self.df.iloc[idx][key]
        else:
            return None


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def post_process(pred, option):
    pred = pred.strip()
    option_candidate = list(option.keys())
    if len(pred) == 1:
        return pred
    elif len(pred) != 1 and pred[0] in option_candidate:
        return pred[0]
    elif len(pred) != 1 and pred[0] not in option_candidate:
        for k, v in option.items():
            if v in pred:
                return k

    return pred


def evaluate_chat_model():
    random.seed(args.seed)

    for ds_name in args.datasets:
        dataset = MMBenchDataset(
            root=ds_collections[ds_name]['root'],
            prompt=prompt,
            language=ds_collections[ds_name]['language'],
            input_size=448,
            image_processor=image_processor,
            max_num=args.max_num,
            min_num=args.min_num,

        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn),
        )

        outputs = []
        for pixel_values, questions, answers, indexes, options,internvl_pixel_values in tqdm(dataloader):
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            generation_config = GenerationConfig(
                num_beams=args.num_beams,
                max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
                min_new_tokens=ds_collections[ds_name]['min_new_tokens'],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                eos_token_id=model.tokenizer.eos_token_id,
                pad_token_id=model.tokenizer.pad_token_id
                if model.tokenizer.pad_token_id is not None else model.tokenizer.eos_token_id,
            )
            pred = model.chat_clean(pixel_values=pixel_values.cuda(),internvl_pixel_values=[internvl_pixel_values[0].cuda()],question=questions[0],generation_config=generation_config)
            preds = [post_process(pred, options[0])]

            for question, pred, answer, index in zip(questions, preds, answers, indexes):
                outputs.append({
                    'question': question,
                    'answer': pred,
                    'gt_answers': answer,
                    'index': int(index)
                })

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            right = 0
            for item in merged_outputs:
                if item["gt_answers"] == item["answer"]:
                    right += 1
            print("MMbench accuracy:{}".format(right/len(merged_outputs)))
            print(f'Evaluating {ds_name} ...')
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'{ds_name}_{time_prefix}.xlsx'
            output_path = os.path.join(args.out_dir, results_file)
            df = pd.read_table(ds_collections[ds_name]['root'])
            cur_df = df.copy()
            if 'mmbench' in ds_name:
                cur_df = cur_df.drop(columns=['hint', 'category', 'source', 'image', 'comment', 'l2-category'])
                cur_df.insert(6, 'prediction', None)
            else:
                cur_df = cur_df.drop(columns=['category', 'image'])
                cur_df.insert(8, 'prediction', None)
            for item in merged_outputs:
                cur_df.loc[df['index'] == item['index'], 'prediction'] = item['answer']

            cur_df.to_excel(output_path, index=False, engine='openpyxl')
            print('Results saved to {}'.format(output_path))

def init_models(args):
    torch.manual_seed(args.seed)
    # parse config
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f'Cannot find {args.config}')

    # load config
    cfg = Config.fromfile(args.config)

    model_name = cfg.model.type if isinstance(cfg.model.type,str) else cfg.model.type.__name__

    cfg.model.pretrained_pth = None

    model = BUILDER.build(cfg.model)

    backend = get_file_backend(args.pth_model)
    if isinstance(backend, PetrelBackend):
        from xtuner.utils.fileio import patch_fileio
        with patch_fileio():
            state_dict = guess_load_checkpoint(args.pth_model)
    else:
        state_dict = guess_load_checkpoint(args.pth_model)
    
    model.load_state_dict(state_dict, strict=False)
    print(f'Load PTH model from {args.pth_model}')

    image_processor = cfg.image_processor
    image_processor_type = image_processor['type']
    del image_processor['type']
    image_processor = image_processor_type(**image_processor)

    return model, image_processor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./omg_llava/configs/finetune/LIRA-2B.py',help='config file name or path.')
    parser.add_argument('--pth_model', default='./model_weight/LIRA-2B.pth',help='pth model file')
    parser.add_argument(
        '--torch-dtype',
        default='bf16',
        choices=TORCH_DTYPE_MAP.keys(),
        help='Override the default `torch.dtype` and load the model under '
             'a specific `dtype`.')
    parser.add_argument('--datasets', type=str, default='mmbench_dev_20230712')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--min-num', type=int, default=1)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--exit_stage_infer', type=int, default=-1)
    args = parser.parse_args()
    print(f"max num: {args.max_num}, min num: {args.min_num}")

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    if args.auto:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    kwargs = {'device_map': 'auto'} if args.auto else {}

    model, image_processor = init_models(args)
    model=model.eval()
    model.to(torch.bfloat16)
    model=model.cuda()



    model = model.cuda()

    args.num_beams = 1

    prompt = {
        'en': "Answer with the option's letter from the given choices directly.",
        'cn': "Answer with the option's letter from the given choices directly."
    }
    evaluate_chat_model()
