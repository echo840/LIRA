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
    'vqav2_val': {
        'train': 'internvl/data/vqav2/vqav2_train.jsonl',
        'test': 'internvl/data/vqav2/vqav2_val.jsonl',
        'question': 'internvl/data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json',
        'annotation': 'internvl/data/vqav2/v2_mscoco_val2014_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vqav2_testdev': {
        'train': 'internvl/data/vqav2/vqav2_train.jsonl',
        'test': 'internvl/data/vqav2/vqav2_testdev.jsonl',
        'metric': None,
        'max_new_tokens': 10,
    },
    'okvqa_val': {
        'train': 'internvl/data/okvqa/okvqa_train.jsonl',
        'test': 'internvl/data/okvqa/okvqa_val.jsonl',
        'question': 'internvl/data/okvqa/OpenEnded_mscoco_val2014_questions.json',
        'annotation': 'internvl/data/okvqa/mscoco_val2014_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'textvqa_val_ocr': {
        'train': 'internvl/data/textvqa/textvqa_train.jsonl',
        'test': 'internvl/data/textvqa/textvqa_val_llava.jsonl',
        'question': 'internvl/data/textvqa/textvqa_val_questions.json',
        'annotation': 'internvl/data/textvqa/textvqa_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vizwiz_val': {
        'train': 'internvl/data/vizwiz/vizwiz_train.jsonl',
        'test': 'internvl/data/vizwiz/vizwiz_val.jsonl',
        'question': 'internvl/data/vizwiz/vizwiz_val_questions.json',
        'annotation': 'internvl/data/vizwiz/vizwiz_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vizwiz_test': {
        'train': 'internvl/data/vizwiz/vizwiz_train.jsonl',
        'test': 'internvl/data/vizwiz/vizwiz_test.jsonl',
        'metric': None,
        'max_new_tokens': 10,
    },
    'refcoco_attri': {
        'train': '',
        'test': 'internvl/data/refcoco_attri/refcoco_attri.jsonl',
        'metric': "accuracy2",
        'max_new_tokens': 10,
    },
    'gqa_testdev': {
        'train': 'internvl/data/gqa/train.jsonl',
        'test': 'internvl/data/gqa/test_balanced.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    },
    'gqa_testdev_llava': {
        'train': 'internvl/data/gqa/train.jsonl',
        'test': 'internvl/data/gqa/testdev_balanced.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    },
    'ai2diagram_test': {
        'train': 'internvl/data/ai2diagram/train.jsonl',
        # 'test': 'internvl/data/ai2d/test_vlmevalkit.jsonl',
        'test': "internvl/data/ai2d/ai2d_test_1014.jsonl",
        'metric': 'accuracy',
        'max_new_tokens': 10,
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

def evaluate_exact_match_accuracy2(entries):
    scores = []
    for i in range(len(entries)//2):
        if entries[2*i]['answer'].strip().lower() == entries[2*i]['annotation'].strip().lower() and entries[2*i+1]['answer'].strip().lower() == entries[2*i+1]['annotation'].strip().lower():
            score = 1
        else:
            score = 0
        scores.append(score)
    return sum(scores) / (len(scores))

def equal(entries):
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

    return pixel_values, questions, question_ids, annotations, internvl_pixel_values


class VQADataset(torch.utils.data.Dataset):

    def __init__(self, train, test, prompt, few_shot, input_size=448, dynamic_image_size=False, max_num=6, min_num=1, image_processor=None,use_thumbnail_=None):
        self.test = open(test).readlines()


        self.prompt = prompt
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.few_shot = few_shot
        self.max_num = max_num
        self.min_num = min_num
        self.use_thumbnail_ = use_thumbnail_
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
        image = "internvl/"+image
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
        # internvl_images = dynamic_preprocess(ori_image, min_num=4, max_num=6,image_size=448, use_thumbnail=True)
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
            'annotation': annotation
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


def evaluate_chat_model():
    base_prompt = 'Answer the question using a single word or phrase.'
    vizwiz_prompt = "When the provided information is insufficient, respond with 'unanswerable'. "
    infovqa_prompt = 'Answer the question using a single word or phrase.'
    
    ai2d_prompt = ''
    random.seed(args.seed)
    summaries = []

    for ds_name in args.datasets:
        if 'vizwiz' in ds_name:
            input_prompt = vizwiz_prompt + base_prompt
        elif 'ai2d' in ds_name:
            input_prompt = ai2d_prompt
        elif 'infographicsvqa' in ds_name:
            input_prompt = infovqa_prompt
        elif 'refcoco_attri' in ds_name:
            input_prompt = ""
        elif 'refcocog_attri' in ds_name:
            input_prompt = ""
        else:
            input_prompt = base_prompt
        
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
        for pixel_values, questions, question_ids, annotations,internvl_pixel_values in tqdm(dataloader):
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
            pred = model.chat_clean(pixel_values=pixel_values.cuda(),internvl_pixel_values=[internvl_pixel_values[0].cuda()],question=questions[0],generation_config=generation_config)
            answers = [pred]

            for question, question_id, answer, annotation,internvl_pixel_values in zip(questions, question_ids, answers, annotations,internvl_pixel_values):
                if ds_name in ['vqav2_val', 'vqav2_testdev', 'okvqa_val', 'textvqa_val',
                               'vizwiz_val', 'textvqa_val_ocr']:
                    outputs.append({
                        'question': question,
                        'question_id': question_id,
                        'answer': answer,
                    })
                elif ds_name in ['docvqa_val', 'infographicsvqa_val', 'gqa_testdev', 'ocrvqa_val',
                                 'ocrvqa_test', 'gqa_testdev_llava', 'infographicsvqa_test','docvqa_test']:
                    outputs.append({
                        'question': question,
                        'questionId': question_id,
                        'answer': answer,
                        'annotation': annotation,
                    })
                elif ds_name in ['ai2diagram_test']:
                    outputs.append({
                        'question': question,
                        'image': question_id,
                        'answer': answer,
                        'annotation': annotation,
                    })
                elif ds_name in ['chartqa_test_human', 'chartqa_test_augmented']:
                    outputs.append({
                        'question': question,
                        'answer': answer,
                        'annotation': annotation,
                    })
                elif ds_name in ['docvqa_test']:
                    outputs.append({
                        'questionId': question_id,
                        'answer': answer,
                    })
                elif ds_name in ['vizwiz_test']:
                    outputs.append({
                        'image': question_id.replace('internvl/data/vizwiz/test/', ''),
                        'answer': answer,
                    })
                elif ds_name in ['refcoco_attri',"refcocog_attri"]:
                    outputs.append({
                        'image': question_id,
                        'question': question,
                        'answer': answer,
                        'annotation': annotation,
                    })
                else:
                    raise NotImplementedError

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'{ds_name}_{time_prefix}.json'
            results_file = os.path.join(args.out_dir, results_file)
            json.dump(merged_outputs, open(results_file, 'w'))
            # print('Results saved to {}'.format(results_file))

            if ds_collections[ds_name]['metric'] == 'vqa_score':
                evaluator = TextVQAAccuracyEvaluator()
                annotation = json.load(open(ds_collections[ds_name]['annotation'], 'r'))['annotations']
                question_id2answers = {}
                for item in annotation:
                    question_id = item['question_id']
                    answers = [answer['answer'] for answer in item['answers']]
                    question_id2answers[question_id] = answers
                for item in merged_outputs:
                    item['pred_answer'] = item['answer']
                    item['gt_answers'] = question_id2answers[item['question_id']]
                accuracy = evaluator.eval_pred_list(merged_outputs)
                print(ds_name, accuracy)
                summaries.append([args.pth_model, ds_name, accuracy])

            elif ds_collections[ds_name]['metric'] == 'anls':
                json.dump(merged_outputs,
                          open(results_file, 'w'),
                          ensure_ascii=False)
                anls_res = evaluateANLS(merged_outputs)
                print("anls: ",anls_res)
            elif ds_collections[ds_name]['metric'] == 'relaxed_accuracy':
                relaxed_accuracy = evaluate_relaxed_accuracy(merged_outputs)
                print(ds_name, {'relaxed_accuracy': relaxed_accuracy})
                summaries.append([ds_name, {'relaxed_accuracy': relaxed_accuracy}])
            elif ds_collections[ds_name]['metric'] == 'accuracy':
                if 'gqa' in ds_name:

                    for entry in merged_outputs:
                        response = entry['answer']
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
                        entry['answer'] = response
                    accuracy = {'accuracy': evaluate_exact_match_accuracy(merged_outputs)}
                else:
                    accuracy = {'accuracy': evaluate_exact_match_accuracy(merged_outputs)}
                print(ds_name, accuracy)
                summaries.append([args.pth_model, ds_name, accuracy])
            elif ds_collections[ds_name]['metric'] == 'accuracy2':
                accuracy = {'accuracy': evaluate_exact_match_accuracy2(merged_outputs)}
                print(ds_name, accuracy)
                summaries.append([args.pth_model, ds_name, accuracy])

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
                        default='okvqa_val,textvqa_val,vizwiz_val,ai2diagram_test,gqa_testdev_llava,refcoco_attri')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--few-shot', type=int, default=0)
    parser.add_argument('--local-rank', type=int, default=0)
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

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    args.num_beams = 1
    use_thumbnail_ = Config.fromfile(args.config).get("use_thumbnail",None)
    evaluate_chat_model()
