# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import math
import os
import os.path as osp
import re
import torch
import tqdm
import sys

from mmengine.dist import (collect_results, get_dist_info, get_rank, init_dist,
                           master_only)
from mmengine.utils.dl_utils import set_multi_processing
from torch.utils.data import Dataset
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)

from xtuner.model.utils import LoadWoInit, prepare_inputs_labels_for_multimodal
from xtuner.tools.utils import get_stop_criteria, is_cn_string
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          PROMPT_TEMPLATE)
from omg_llava.model.utils import prepare_inputs_labels_for_multimodal_with_visual_prompts

from xtuner.registry import BUILDER
from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint
from mmengine.config import Config
from mmengine.fileio import PetrelBackend, get_file_backend
from mmengine.config import ConfigDict

from PIL import Image, ImageChops, ImageEnhance
import torch.nn.functional as F
from xtuner.dataset.utils import expand2square
from pycocotools import mask as mask_utils
from omg_llava.dataset.utils.internvl_data import build_transform, dynamic_preprocess
import numpy as np


def convert_dict2config_dict(input):
    input = ConfigDict(**input)
    for key in input.keys():
        if isinstance(input[key], dict):
            input[key] = convert_dict2config_dict(input[key])
    return input

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')


# GCG_QUESTIONS = [
#     'Could you please give me a detailed description of the image? Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
#     'Can you provide a thorough description of the this image? Please output with interleaved segmentation masks for the corresponding phrases.',
#     'Please describe in detail the contents of the image. Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
#     'Could you give a comprehensive explanation of what can be found within this picture? Please output with interleaved segmentation masks for the corresponding phrases.',
#     'Could you give me an elaborate explanation of this picture? Please respond with interleaved segmentation masks for the corresponding phrases.',
#     'Could you provide me with a detailed analysis of this photo? Please output with interleaved segmentation masks for the corresponding parts of the answer.',
# ]

GCG_QUESTIONS = [
    'Can you provide a thorough description of the this image? Please output with interleaved segmentation masks for the corresponding phrases.',
    'Could you please give me a detailed description of the image? Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
    'Please describe in detail the contents of the image. Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
    'Could you give a comprehensive explanation of what can be found within this picture? Please output with interleaved segmentation masks for the corresponding phrases.',
    'Could you give me an elaborate explanation of this picture? Please respond with interleaved segmentation masks for the corresponding phrases.',
    'Could you provide me with a detailed analysis of this photo? Please output with interleaved segmentation masks for the corresponding parts of the answer.',
]

def parse_args():
    parser = argparse.ArgumentParser(description='RefCocoSeg')
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('pth_model', help='pth model file')
    parser.add_argument(
        '--output-name', type=str, default='gcg', help='save folder name')
    parser.add_argument(
        '--prompt-template',
        choices=PROMPT_TEMPLATE.keys(),
        default='internvl2_chat',
        help='Specify a prompt template')
    parser.add_argument(
        '--stop-words', nargs='+', type=str, default=[], help='Stop words')
    parser.add_argument(
        '--torch-dtype',
        default='fp16',
        choices=TORCH_DTYPE_MAP.keys(),
        help='Override the default `torch.dtype` and load the model under '
        'a specific `dtype`.')
    parser.add_argument(
        '--bot-name', type=str, default='BOT', help='Name for Bot')
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=100,
        help='Maximum number of new tokens allowed in generated text')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducible text generation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument(
        '--max_patch',
        type=int,
        default=6)
    parser.add_argument(
        '--output_name',
        type=str,
        default="test_gcg")
    args = parser.parse_args()
    
    args = parser.parse_args()
    return args


@master_only
def master_print(msg):
    print(msg)

class GCD_Inference_Dataset(Dataset):
    def __init__(self,
                 image_folder,
                 image_processor,
                 debug=False,
                 pad_image_to_square=True,
                max_patch=12,
                 ):
        self.max_patch = max_patch
        self.debug = debug
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
        self.transform = build_transform(is_train=False, input_size=448,
                                    pad2square=False, normalize_type="imagenet")

        self.images = os.listdir(image_folder)
        if debug:
            self.images = self.images[:20]

    def __len__(self):
        return len(self.images)

    def get_questions(self):
        question = 'Can you provide a thorough description of the this image? Please output with interleaved segmentation masks for the corresponding phrases.'
        return question

    def __getitem__(self, index):

        data_dict = {}

        questions = self.get_questions()
        image_file = self.images[index]
        data_dict['image_file'] = image_file
        image_file = os.path.join(self.image_folder, image_file)
        # print(image_file)
        ori_image = Image.open(image_file).convert('RGB')


        #add
        internvl_images = dynamic_preprocess(ori_image, min_num=1, max_num=self.max_patch,
                                    image_size=448, use_thumbnail=True)
        internvl_pixel_values = [self.transform(internvl_image) for internvl_image in internvl_images]
        internvl_pixel_values = torch.stack(internvl_pixel_values)
        num_patches = internvl_pixel_values.size(0)
        data_dict['internvl_pixel_values'] = internvl_pixel_values
        #add

        ori_width, ori_height = ori_image.size
        if self.pad_image_to_square:
            image = expand2square(
                ori_image,
                tuple(
                    int(x * 255) for x in self.image_processor.image_mean))
        image = self.image_processor.preprocess(
            image, return_tensors='pt')['pixel_values'][0]
        data_dict['pixel_values'] = image
        data_dict['ori_image'] = ori_image
        data_dict['ori_size'] = (ori_width, ori_height)
        data_dict['questions'] = questions
        return data_dict
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
    final_output_masks = []
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
            if output_ids[0][0]==92558 and len(output_masks)!=0:
                final_output_masks.append(output_masks[-1])
                object_image_black = extract_object_with_black_background(ori_image, output_masks[-1][0].cpu().numpy().astype(np.uint8), "highlight")
                internvl_images = dynamic_preprocess(object_image_black, min_num=1, max_num=1, image_size=448, use_thumbnail=True)
                internvl_pixel_values = [internvl_transform(internvl_image) for internvl_image in internvl_images]
                internvl_pixel_values = torch.stack(internvl_pixel_values)
                internvl_vit_embed = model.extract_feature(internvl_pixel_values.to(model.vlm_enc.dtype).cuda())
                internvl_vit_embed = internvl_vit_embed.reshape(-1, model.llm.config.hidden_size)
                end_embed = model.llm.get_input_embeddings()(torch.tensor(tokenizer.encode(" <p>")[1:]).to(model.llm.device))
                predict+=" <p>"
                current_inputs_embeds = torch.cat([model.llm.get_input_embeddings()(current_input_ids), internvl_vit_embed.unsqueeze(0),end_embed.unsqueeze(0) ],dim=1)
                # current_inputs_embeds = torch.cat([llm.get_input_embeddings()(current_input_ids), internvl_vit_embed.unsqueeze(0)],dim=1)
                has_image_id_flag = 1
            else:
                current_inputs_embeds = model.llm.get_input_embeddings()(current_input_ids)
            past_key_values = output['past_key_values']
            if early_stop and current_input_ids.item() == 92542:
                break
    return {
        'predict': predict.replace("[SEG]<image_id>",""),
        "output_masks": final_output_masks
    }

def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    if args.launcher != 'none':
        set_multi_processing(distributed=True)
        init_dist(args.launcher)

        rank, world_size = get_dist_info()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1

    # build model
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f'Cannot find {args.config}')

    # load config
    cfg = Config.fromfile(args.config)

    model_name = cfg.model.type if isinstance(cfg.model.type,
                                              str) else cfg.model.type.__name__
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


    model.to(torch.bfloat16)



    llm = model.llm
    tokenizer = model.tokenizer

    model.cuda()
    model.eval()
    llm.eval()
    visual_encoder = model.visual_encoder
    projector = model.projector
    projector_text2vision = model.projector_text2vision

    projector.cuda()
    projector.eval()

    visual_encoder.cuda()
    visual_encoder.eval()

    stop_words = args.stop_words
    if args.prompt_template:
        template = PROMPT_TEMPLATE[args.prompt_template]
        stop_words += template.get('STOP_WORDS', [])
    stop_criteria = get_stop_criteria(
        tokenizer=tokenizer, stop_words=stop_words)

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )

    dataset = GCD_Inference_Dataset(
        image_folder='./data/glamm_data/images/grandf/val_test/',
        image_processor=image_processor,
        pad_image_to_square=True,
        debug=False,
        max_patch=args.max_patch,
        # debug=True,
    )
    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size)

    per_rank_ids = range(per_rank_samples * rank,
                         min(n_samples, per_rank_samples * (rank + 1)))

    for i in tqdm.tqdm(per_rank_ids, desc=f'Rank {rank}'):
        # pixel feature
        
        data_sample = dataset[i]
        image = data_sample['pixel_values']  # ()

        questions = GCG_QUESTIONS
        for question in questions:
            output_res = base_generate(model, question, data_sample['ori_image'], data_sample['pixel_values'].unsqueeze(0).cuda(), [data_sample['internvl_pixel_values'].cuda()],model.tokenizer, max_new_tokens=args.max_new_tokens)
            predict = output_res['predict']
            masks = output_res['output_masks']
            if len(masks) != 0:
                break
        ori_size = data_sample['ori_size']
        if len(masks) == 0:
            print("Warnning !!! No mask Pred !!!")
            w, h = ori_size
            masks = torch.zeros((0, h, w), dtype=torch.bool)
        else:
            masks = torch.cat(masks)
            masks = (masks==1)
        process_and_save_output(
            "./results/{}/".format(args.output_name),
            data_sample['image_file'],
            predict,
            masks
        )

def forward_model(question, pixel_values,
                  tokenizer, model, llm,
                  projector_text2vision,
                  gen_config, stop_criteria):

    inputs = question
    # print("Question: ", inputs)
    chunk_encode = []
    for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
        if idx == 0:
            cur_encode = tokenizer.encode(chunk)
        else:
            cur_encode = tokenizer.encode(chunk, add_special_tokens=False)
        chunk_encode.append(cur_encode)
    assert len(chunk_encode) == 2
    ids = []
    for idx, cur_chunk_encode in enumerate(chunk_encode):
        ids.extend(cur_chunk_encode)
        if idx != len(chunk_encode) - 1:
            ids.append(IMAGE_TOKEN_INDEX)
    ids = torch.tensor(ids).cuda().unsqueeze(0)
    mm_inputs = prepare_inputs_labels_for_multimodal(
        llm=llm, input_ids=ids, pixel_values=pixel_values)
    # mm_inputs['inputs_embeds'] = mm_inputs['inputs_embeds'].to(torch.float16)
    generate_output = llm.generate(
        **mm_inputs,
        generation_config=gen_config,
        streamer=None,
        bos_token_id=tokenizer.bos_token_id,
        stopping_criteria=stop_criteria,
        output_hidden_states=True,
        return_dict_in_generate=True
    )
    predict = tokenizer.decode(
        # generate_output.sequences[0], skip_special_tokens=True).strip()
        generate_output.sequences[0]).strip()
    # print("Answer:", predict)

    hidden_states = generate_output.hidden_states
    last_hidden_states = [item[-1][0] for item in hidden_states]
    last_hidden_states = torch.cat(last_hidden_states, dim=0)
    seg_hidden_states = get_seg_hidden_states(
        last_hidden_states, generate_output.sequences[0][:-1],
        seg_id=model.seg_token_idx
    )
    return predict, seg_hidden_states

def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    return hidden_states[-n_out:][seg_mask]

def process_and_save_output(output_dir, image_name, text_output, pred_masks):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    text_output = text_output.replace("<s>", "").replace("\n", "").replace("  ", " ")
    text_output = text_output.split("ASSISTANT: ")[-1]

    cleaned_str = re.sub(r'<.*?>', '', text_output)

    pattern = re.compile(r'<p>(.*?)<\/p>')
    phrases = pattern.findall(text_output)
    phrases = [p.strip() for p in phrases]

    # Remove the [SEG] token
    cleaned_str = cleaned_str.replace('[SEG]', '')

    # Strip unnecessary spaces
    cleaned_str = ' '.join(cleaned_str.split()).strip("'")
    cleaned_str = cleaned_str.strip()

    # Convert the predicted masks into RLE format
    pred_masks_tensor = pred_masks.cpu()
    uncompressed_mask_rles = mask_to_rle_pytorch(pred_masks_tensor)
    rle_masks = []
    for m in uncompressed_mask_rles:
        rle_masks.append(coco_encode_rle(m))
    # Create results dictionary
    result_dict = {
        "image_id": image_name[:-4],
        "caption": cleaned_str,
        "phrases": phrases,
        "pred_masks": rle_masks
    }

    output_path = f"{output_dir}/{image_name[:-4]}.json"

    with open(output_path, 'w') as f:
        json.dump(result_dict, f)

    return

def mask_to_rle_pytorch(tensor: torch.Tensor):
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    # Put in fortran order and flatten h,w
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)

    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat(
            [torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device), cur_idxs + 1,
             torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device), ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [h, w], "counts": counts})

    return out

def coco_encode_rle(uncompressed_rle):
    h, w = uncompressed_rle["size"]
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    rle["counts"] = rle["counts"].decode("utf-8")  # Necessary to serialize with json

    return rle

if __name__ == '__main__':

    main()
