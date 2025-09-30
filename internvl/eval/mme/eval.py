import argparse
import os
import re

import torch
from PIL import Image
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

def load_image(image_file, input_size=448, image_processor=None,max_num=6,min_num=1):

    transform = build_transform(is_train=False,input_size=448, pad2square=False, normalize_type="imagenet")

    ori_image = Image.open(image_file).convert('RGB')
    internvl_images = dynamic_preprocess(ori_image, min_num=min_num, max_num=max_num,image_size=448, use_thumbnail=True)
    internvl_pixel_values = [transform(internvl_image) for internvl_image in internvl_images]
    internvl_pixel_values = torch.stack(internvl_pixel_values)
    ori_width, ori_height = ori_image.size
    image = expand2square(
        ori_image,
        tuple(
            int(x * 255) for x in image_processor.image_mean))
    image = image_processor.preprocess(
        image, return_tensors='pt')['pixel_values'][0]
    pixel_values = image
    return pixel_values.unsqueeze(0),internvl_pixel_values


def post_processing(response):
    response = response.replace('\n', '').replace('不是', 'No').replace('是', 'Yes').replace('否', 'No')
    response = response.lower().replace('true', 'yes').replace('false', 'no')
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    response = re.sub(pattern, '', response)
    return response

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
    parser.add_argument('--root', type=str, default='./internvl/eval/mme/Your_Results')
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--sample', type=bool, default=False)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--min-num', type=int, default=1)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--exit_stage_infer', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=0.0)
    args = parser.parse_args()
    print(f"max num: {args.max_num}, min num: {args.min_num}")


    if args.auto:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    kwargs = {'device_map': 'auto'} if args.auto else {}
    prompt = 'Answer the question using a single word or phrase.'

    model, image_processor = init_models(args)
    model=model.eval()
    model.to(torch.bfloat16)
    model=model.cuda()

    model = model.cuda()

    args.num_beams = 1
    
    image_size = 448
    output = "./internvl/eval/mme/output_files"
    os.makedirs(output, exist_ok=True)
    num=1
    for filename in os.listdir(args.root):
        fin = open(os.path.join(args.root, filename), 'r', encoding='utf-8')
        fout = open(os.path.join(output, filename), 'w', encoding='utf-8')
        lines = fin.readlines()
        filename = filename.replace('.txt', '')
        for line in tqdm(lines):
            img, question, gt = line.strip().split('\t')
            question = question + ' ' + prompt

            num+=1
            if num%200==0:
                print(question)

            
            img_path = os.path.join('data/MME/MME_Benchmark_release_version', filename, img)
            if os.path.exists(img_path):
                pass
            else:
                img_path = os.path.join('data/MME/MME_Benchmark_release_version', filename,'images', img)
            pixel_values,internvl_pixel_values = load_image(img_path, image_size, image_processor,max_num=args.max_num,min_num=args.min_num)
            generation_config = GenerationConfig(
                num_beams=args.num_beams,
                max_new_tokens=20,
                min_new_tokens=1,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                eos_token_id=model.tokenizer.eos_token_id,
                pad_token_id=model.tokenizer.pad_token_id
                if model.tokenizer.pad_token_id is not None else model.tokenizer.eos_token_id,
            )
            response = model.chat_clean(pixel_values=pixel_values.cuda(),internvl_pixel_values=[internvl_pixel_values.cuda()],question=question,generation_config=generation_config)
            
            response = post_processing(response)
            question = question.replace("\n"," ")
            print(img, question, gt, response, sep='\t', file=fout)
        fin.close()
        fout.close()
