import argparse
import itertools
import json
import os
import random
import time
from functools import partial

import torch
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
    'sqa_test': {
        'root': 'internvl/data/scienceqa/scienceqa_test_img.jsonl',
        'max_new_tokens': 100,
        'min_new_tokens': 1,
    },
}


def collate_fn(batches):
    pixel_values = torch.stack([_['pixel_values'] for _ in batches])
    questions = [_['question'] for _ in batches]
    answers = [_['answer'] for _ in batches]
    image_paths = [_['image_path'] for _ in batches]
    options = [_['option'] for _ in batches]
    internvl_pixel_values = [_['internvl_pixel_values'] for _ in batches]
    return pixel_values, questions, answers, image_paths, options, internvl_pixel_values


class ScienceQADataset(torch.utils.data.Dataset):

    def __init__(self, root, prompt, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6,min_num=1,image_processor=None,):
        f = open(root, 'r', encoding='utf-8')
        self.data = [json.loads(line) for line in f.readlines()]
        self.prompt = prompt
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.min_num = min_num

        self.image_processor = image_processor
        self.transform = build_transform(is_train=False,input_size=448, pad2square=False, normalize_type="imagenet")
        self.pad_image_to_square = True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        image_path = "internvl/"+data['image']
        hint = data['hint'] if data['hint'] else None
        question = data['question']

        choices = data['choices']
        answer = data['answer']
        choice_list = []

        options = {}
        multiple_choices = ['A', 'B', 'C', 'D', 'E']
        for i, c in enumerate(choices):
            choice_list.append('{}. {}'.format(multiple_choices[i], c))
            options[multiple_choices[i]] = c
        choice_txt = '\n'.join(choice_list)

        # image = Image.open(image_path).convert('RGB')
        # if self.dynamic_image_size:
        #     images = dynamic_preprocess(image, image_size=self.input_size,
        #                                 use_thumbnail=self.use_thumbnail,
        #                                 max_num=self.max_num)
        # else:
        #     images = [image]
        # pixel_values = [self.transform(image) for image in images]
        # pixel_values = torch.stack(pixel_values)
        ori_image = Image.open(image_path).convert('RGB')
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



        if hint is not None:
            question = hint + '\n' + question
        question += '\n' + choice_txt
        question += '\n' + self.prompt

        return {
            'question': question,
            'pixel_values': pixel_values,
            "internvl_pixel_values":internvl_pixel_values,
            'answer': multiple_choices[answer],
            'image_path': image_path,
            'option': options
        }


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
    prompt = "Answer with the option's letter from the given choices directly."
    random.seed(args.seed)

    for ds_name in args.datasets:
        dataset = ScienceQADataset(
            root=ds_collections[ds_name]['root'],
            prompt=prompt,
            input_size=448,
            dynamic_image_size=args.dynamic,
            max_num=args.max_num,
            min_num=args.min_num,
            image_processor=image_processor,
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
        for pixel_values,  questions, answers, image_paths, options,internvl_pixel_values in tqdm(dataloader):
            # pixel_values = pixel_values.to(torch.bfloat16).cuda()
            generation_config = GenerationConfig(
                num_beams=args.num_beams,
                max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
                min_new_tokens=1,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                eos_token_id=model.tokenizer.eos_token_id,
                pad_token_id=model.tokenizer.pad_token_id
                if model.tokenizer.pad_token_id is not None else model.tokenizer.eos_token_id,
            )
            # model.only_internvl=True
            pred = model.chat_clean(pixel_values=pixel_values.cuda(),internvl_pixel_values=[internvl_pixel_values[0].cuda()],question=questions[0],generation_config=generation_config,only_internvl=model.only_internvl)
            preds = [post_process(pred, options[0])]

            for question, pred, answer, image_path in zip(questions, preds, answers, image_paths):
                outputs.append({
                    'question': question,
                    'answer': pred,
                    'gt_answers': answer,
                    'image_path': image_path
                })

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:

            print(f'Evaluating {ds_name} ...')
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'{ds_name}_{time_prefix}.jsonl'
            output_path = os.path.join(args.out_dir, results_file)
            with open(output_path, 'w') as f:
                for output in merged_outputs:
                    f.write(json.dumps(output) + '\n')
            print('Results saved to {}'.format(output_path))
            cnt = 0
            for item in merged_outputs:
                if item['answer'] == item['gt_answers']:
                    cnt += 1
            print(f'Acc@1: {cnt / len(merged_outputs)}')

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
    if 'LLaVAModel' or 'OMG' in model_name:
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
    parser.add_argument('--config', default='/home/kas/lz_new/ref_monkey/OMG-Seg/omg_llava/omg_llava/configs/finetune/internvl2/omg_internvl2_psaml_common.py',help='config file name or path.')
    parser.add_argument('--pth_model', default='/home/kas/lz_new/ref_monkey/OMG-Seg/omg_llava/saved_path/internvl2/internvl_all/fine_230000.pth',help='pth model file')
    parser.add_argument(
        '--torch-dtype',
        default='bf16',
        choices=TORCH_DTYPE_MAP.keys(),
        help='Override the default `torch.dtype` and load the model under '
             'a specific `dtype`.')
    parser.add_argument(
        '--prompt-template',
        choices=PROMPT_TEMPLATE.keys(),
        default="internlm2_chat",
        help='Specify a prompt template')
    parser.add_argument('--datasets', type=str, default='sqa_test')
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

    if not args.load_in_8bit and not args.auto:
        model = model.cuda()
    # image_size = model.config.force_image_size or model.config.vision_config.image_size
    # use_thumbnail = model.config.use_thumbnail

    # if getattr(model.config.llm_config,"use_multi_early_exit",False):
    #     model.language_model.model.exit_stage_infer = args.exit_stage_infer
    #     print("setting early stop: {}".format(args.exit_stage_infer))
    # total_params = sum(p.numel() for p in model.parameters()) / 1e9
    args.num_beams = 1
    # if total_params > 20 or args.dynamic:
    #     args.num_beams = 1
    #     print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    # else:
    #     print(f'[test] total_params: {total_params}B')
    # print(f'[test] image_size: {image_size}')
    # print(f'[test] template: {model.config.template}')
    # print(f'[test] dynamic_image_size: {args.dynamic}')
    # print(f'[test] use_thumbnail: {use_thumbnail}')

    evaluate_chat_model()
