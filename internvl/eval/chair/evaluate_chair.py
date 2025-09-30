import argparse
import itertools
import json
import os
import random
import subprocess
import time
from functools import partial
from typing import Optional

from textvqa_eval import TextVQAAccuracyEvaluator
from tqdm import tqdm
import numpy as np

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
from PIL import Image, ImageChops, ImageEnhance
from pathlib import Path
import os
import tempfile
import uuid
import re
import json
from tqdm import tqdm
from xtuner.utils import  PROMPT_TEMPLATE
import torch.nn.functional as F
from pycocotools import mask as mask_utils
import json
from transformers import top_k_top_p_filtering
TORCH_DTYPE_MAP = dict(fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')


ds_collections = {
    'chair_test': {
        'train': None,
        'test': "./data/chair/chair_test.jsonl",
        'metric': None,
        'max_new_tokens': 512,
    },
}
def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def normANLS(s1,s2):
    dist = levenshtein_distance(s1.lower().strip(),s2.lower().strip())
    length = max(len(s1),len(s2))
    value =  0.0 if length == 0 else float(dist) / float(length) 
    return value 

def evaluateANLS(ans_list):
    anls_threshold = 0.5
    anls_list = []
    for predict_pair in ans_list:
        answer = predict_pair["answer"].strip()
        gt_list = predict_pair["annotation"]
        
        value_list = []
        for gt_single in gt_list:
            # if gt_single.strip().lower() in answer.strip().lower():
            #     value_list.append(0)
            value_list.append(normANLS(gt_single,answer))
        question_result = 1 - min(value_list)

        if (question_result < anls_threshold) :
            question_result = 0
        anls_list.append(question_result)
    return np.mean(anls_list)

# https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py#L81
def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float -
                              target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


def evaluate_relaxed_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            relaxed_correctness(elem['answer'].strip(), ann)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)


def evaluate_exact_match_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            (1.0 if
             (elem['answer'].strip().lower() == ann.strip().lower()) else 0.0)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)


def collate_fn(batches):
    pixel_values = torch.stack([_['pixel_values'] for _ in batches])
    questions = [_['question'] for _ in batches]
    question_ids = [_['question_id'] for _ in batches]
    annotations = [_['annotation'] for _ in batches]
    internvl_pixel_values = [_['internvl_pixel_values'] for _ in batches]
    ori_images = [_['ori_image'] for _ in batches]

    return pixel_values, questions, question_ids, annotations, internvl_pixel_values, ori_images


class VQADataset(torch.utils.data.Dataset):

    def __init__(self, train, test, prompt, few_shot, input_size=448, dynamic_image_size=False, max_num=6, min_num=1,image_processor=None,use_thumbnail_=None):
        self.test = open(test).readlines()
        self.prompt = prompt
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.few_shot = few_shot
        self.max_num = max_num
        self.min_num = min_num
        self.use_thumbnail_=use_thumbnail_
        if few_shot > 0:
            self.train = open(train).readlines()
        self.transform = build_transform(is_train=False,input_size=448, pad2square=False, normalize_type="imagenet")
        self.image_processor = image_processor
        self.pad_image_to_square = True

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        data = json.loads(self.test[idx].strip())
        image, question, question_id, annotation = data['image'], data[
            'question'], data['question_id'], data.get('answer', None)
        few_shot_prompt = ''
        if self.few_shot > 0:
            few_shot_samples = random.sample(self.train, self.few_shot)
            for sample in few_shot_samples:
                sample = json.loads(sample.strip())
                few_shot_prompt += self.prompt.format(
                    sample['image'],
                    sample['question']) + f" {sample['answer']}"

        ori_image = Image.open(image).convert('RGB')
        if self.use_thumbnail_ is None:
            internvl_images = dynamic_preprocess(ori_image, min_num=self.min_num, max_num=self.max_num,image_size=448, use_thumbnail=True)
        elif self.use_thumbnail_==True:
            internvl_images = dynamic_preprocess(ori_image, min_num=1, max_num=1,image_size=448, use_thumbnail=True)
        elif self.use_thumbnail_==False:
            internvl_images = dynamic_preprocess(ori_image, min_num=self.min_num, max_num=self.max_num,image_size=448, use_thumbnail=False)
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
        if len(self.prompt) != 0:
            question = question + ' ' + self.prompt
        return {
            'question_id': question_id,
            'question': question,
            'pixel_values': pixel_values,
            "internvl_pixel_values":internvl_pixel_values,
            'annotation': annotation,
            "ori_image":ori_image
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


def post_process(response):
    response = response.strip().split('.')[0].split(
        ',')[0].split('!')[0].lower()
    if 'is ' in response:
        response = response.split('is ')[1]
    if 'are ' in response:
        response = response.split('are ')[1]
    if 'a ' in response:
        response = response.split('a ')[1]
    if 'an ' in response:
        response = response.split('an ')[1]
    if 'the ' in response:
        response = response.split('the ')[1]
    if ' of' in response:
        response = response.split(' of')[0]
    response = response.strip()
    return response

def sample(logits, return_probs: bool=False, do_sample: bool=False, top_k: int=50, top_p: float=0.7, temperature: float=0.7):

    if return_probs:

        all_probs = logits.softmax(-1)
        if do_sample and top_k != 1 and top_p != 0.0 and temperature != 0.0:
            _logits = top_k_top_p_filtering(logits.view(-1, logits.size(-1)) / temperature, top_k=top_k, top_p=top_p)
            output_ids = torch.multinomial(_logits.softmax(-1), num_samples=1).view(logits.shape[:-1])
            probs = torch.gather(all_probs, -1, output_ids.unsqueeze(-1)).squeeze(-1)
        else:
            probs, output_ids = torch.max(all_probs, dim=-1)
            
        return output_ids, probs

    else:

        if do_sample and top_k != 1 and top_p != 0.0 and temperature != 0.0:
            _logits = top_k_top_p_filtering(logits.view(-1, logits.size(-1)) / temperature, top_k=top_k, top_p=top_p)
            output_ids = torch.multinomial(_logits.softmax(-1), num_samples=1).view(logits.shape[:-1])
        else:
            output_ids = torch.argmax(logits, dim=-1)
                
            
        return output_ids

def extract_object_with_black_background(image_ori, binary_mask, background_color=(0, 0, 0)):
    # Convert binary mask to a PIL image with the same mode as the original image
    mask_img = Image.fromarray(binary_mask * 255, mode='L')
    if background_color=="highlightall":
        result_img = image_ori.copy()
        enhancer = ImageEnhance.Brightness(result_img)
        result_img = enhancer.enhance(0.3)
        result_img.paste(image_ori, (0, 0), mask_img)
        return result_img

    bbox = mask_img.getbbox()
    image_cropped = image_ori.crop(bbox)
    if background_color=="ori":
        return image_cropped
    if background_color=="highlight":
        mask_cropped = mask_img.crop(bbox)
        result_img = image_cropped.copy()
        enhancer = ImageEnhance.Brightness(result_img)
        result_img = enhancer.enhance(0.3)
    else:
        mask_cropped = mask_img.crop(bbox)
        # Create a new image with a black background
        result_img = Image.new('RGB', image_cropped.size, background_color)
    
    # Composite the cropped image onto the black background using the cropped mask
    result_img.paste(image_cropped, (0, 0), mask_cropped)
    return result_img
def base_generate(model,  question, ori_image, pixel_values, internvl_pixel_values, tokenizer, prompt_template="internvl2_chat", max_new_tokens=100, do_sample=False, top_k=50, top_p=1.0, temperature=0, early_stop=True):
    
    if '<image>' not in question:
        question = '<image>\n' + question
    template = PROMPT_TEMPLATE[prompt_template]
    if prompt_template == "internvl2_chat":
        prompt_text = template['SYSTEM']
        # question = question.replace("<image>","<img><image></img>")
    else:
        prompt_text = ""
    prompt_text+=template['INSTRUCTION'].format(input=question)
    chunk_encode = []
    for idx, chunk in enumerate(prompt_text.split(DEFAULT_IMAGE_TOKEN)):
        cur_encode = tokenizer.encode(chunk, add_special_tokens=False)
        chunk_encode.append(cur_encode)
    ids = []
    for idx, cur_chunk_encode in enumerate(chunk_encode):
        ids.extend(cur_chunk_encode)
        if idx != len(chunk_encode) - 1:
            ids.append(IMAGE_TOKEN_INDEX)
    ids = torch.tensor(ids).to(model.device).unsqueeze(0)
    visual_outputs = model.visual_encoder(pixel_values.to(model.visual_encoder.dtype),output_hidden_states=True)
    
    internvl_vit_embeds=[]
    for internvl_pixel_value in internvl_pixel_values:
        internvl_vit_embed = model.extract_feature(internvl_pixel_value.to(model.vlm_enc.dtype))
        internvl_vit_embed = internvl_vit_embed.reshape(-1, model.llm.config.hidden_size)
        internvl_vit_embeds.append(internvl_vit_embed)
    
    image_features = model.projector(visual_outputs, internvl_vit_embeds)
    mm_inputs =prepare_inputs_labels_for_multimodal_with_visual_prompts(
                llm=model.llm, region_id=model.region_token_idx,
                regions_feats=None,
                mark_id=model.mark_token_idx,
                mark_feats=None,
                pixel_values=image_features,
                input_ids=ids)
    inputs_embeds = mm_inputs['inputs_embeds']
    current_input_ids = [[-100]]
    current_inputs_embeds = inputs_embeds
    generate_ids = torch.empty([inputs_embeds.size(0), max_new_tokens], dtype=torch.long, device=model.device)
    predict = ""
    past_key_values = None
    output_masks = []
    internvl_transform = build_transform(is_train=False, input_size=448,
                                    pad2square=False, normalize_type="imagenet")


    ori_masks = []
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            output = model.llm(inputs_embeds=current_inputs_embeds,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=True)

            logits = output['logits'][:,-1:]
            output_ids = sample(logits, do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature)
            predict+=tokenizer.decode(output_ids[0])
            if current_input_ids[0][0]==92553:
                seg_hidden = output['hidden_states'][-1][-1]
                seg_hidden = model.projector_text2vision(seg_hidden)
                batch_idxs = torch.zeros((seg_hidden.shape[0],),dtype=torch.int64).to(seg_hidden.device)
                pred_masks_list = model.visual_encoder.forward_llm_seg(seg_hidden, batch_idxs)
                pred_masks = pred_masks_list[-1]
                w, h = ori_image.size
                masks = F.interpolate(pred_masks, size=(max(w, h), max(w, h)),mode='bilinear', align_corners=False)
                masks = masks[:, 0]
                # remove padding
                if w == h:
                    pass
                elif w > h:
                    n_pad = w - h
                    n_pad_1 = n_pad // 2
                    n_pad_2 = n_pad - n_pad_1
                    masks = masks[:, n_pad_1: w - n_pad_2]
                else:
                    n_pad = h - w
                    n_pad_1 = n_pad // 2
                    n_pad_2 = n_pad - n_pad_1
                    masks = masks[:, :, n_pad_1: h - n_pad_2]
                # binary
                ori_masks.append(masks)
                masks = masks.sigmoid() > 0.5
                masks = masks.int()
                output_masks.append(masks)
            current_input_ids = output_ids
            if output_ids[0][0]==92558:
                object_image_black = extract_object_with_black_background(ori_image, output_masks[-1][0].cpu().numpy().astype(np.uint8), "highlight")
                internvl_images = dynamic_preprocess(object_image_black, min_num=1, max_num=1, image_size=448, use_thumbnail=True)
                internvl_pixel_values = [internvl_transform(internvl_image) for internvl_image in internvl_images]
                internvl_pixel_values = torch.stack(internvl_pixel_values)
                internvl_vit_embed = model.extract_feature(internvl_pixel_values.to(model.vlm_enc.dtype).cuda())
                internvl_vit_embed = internvl_vit_embed.reshape(-1, model.llm.config.hidden_size)
                end_embed = model.llm.get_input_embeddings()(torch.tensor(tokenizer.encode(" <p>")[1:]).to(model.llm.device))
                predict+=" <p>"
                current_inputs_embeds = torch.cat([model.llm.get_input_embeddings()(current_input_ids), internvl_vit_embed.unsqueeze(0),end_embed.unsqueeze(0) ],dim=1)
            else:
                current_inputs_embeds = model.llm.get_input_embeddings()(current_input_ids)
            past_key_values = output['past_key_values']
            if early_stop and current_input_ids.item() == 92542:
                break
    return {
        'predict': predict.replace("[SEG]","").replace("<image_id>","").replace("<p>","").replace("</p>","").replace("<|im_end|>","").replace("   "," ").replace("  "," ").strip(),
        "output_masks": output_masks
    }

def evaluate_chat_model():
    base_prompt = 'Answer the question using a single word or phrase.'
    vizwiz_prompt = "When the provided information is insufficient, respond with 'unanswerable'. "
    # infovqa_prompt = 'Answer the question directly.'
    infovqa_prompt = 'Answer the question using a single word or phrase.'
    
    ai2d_prompt = ''
    random.seed(args.seed)
    summaries = []
    for ds_name in args.datasets:
        input_prompt = ""

        dataset = VQADataset(
            train=ds_collections[ds_name]['train'],
            test=ds_collections[ds_name]['test'],
            prompt=input_prompt,
            few_shot=args.few_shot,
            input_size=448,
            dynamic_image_size=args.dynamic,
            max_num=args.max_num,
            min_num=args.min_num,
            image_processor=image_processor,
            use_thumbnail_=use_thumbnail_,
        )
        use_len_dataset = len(dataset)
        if len(dataset)>20000:
            print(ds_name)
            use_len_dataset = 4000
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(use_len_dataset),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn),
        )
        print(len(dataset))
        outputs = []
        for pixel_values, questions, question_ids, annotations,internvl_pixel_values, ori_images in tqdm(dataloader):

            predict = base_generate(model, 'Provide an in-depth description of this image.', ori_images[0], pixel_values.cuda(), [internvl_pixel_values[0].cuda()],model.tokenizer, max_new_tokens=ds_collections[ds_name]['max_new_tokens'])
            
            answers = [predict['predict']]
            for question, question_id, answer, annotation,internvl_pixel_values in zip(questions, question_ids, answers, annotations,internvl_pixel_values):
                outputs.append({
                    'question': question,
                    'image_id': question_id,
                    'caption': answer,
                })

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')
            results_file = "./internvl/results/chair_results.jsonl"
            with open(results_file, 'w') as f:
                for entry in merged_outputs:
                    f.write(json.dumps(entry) + '\n')
        torch.distributed.barrier()

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
    parser.add_argument('--datasets', type=str,
                        default='chair_test')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--few-shot', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--min-num', type=int, default=1)
    parser.add_argument('--local-rank', type=int, default=1)
    parser.add_argument('--isgcg', type=bool, default=False)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--exit_stage_infer', type=int, default=-1)
    args = parser.parse_args()
    print(f"max num: {args.max_num}, min num: {args.min_num}, isgcg: {args.isgcg}")

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

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    args.num_beams = 1
    use_thumbnail_ = Config.fromfile(args.config).get("use_thumbnail",None)
    evaluate_chat_model()
