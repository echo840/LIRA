import gradio as gr
import numpy as np

import sys
from omg_llava.tools.app_utils import process_markdown, show_mask_pred,show_mask_pred_new, parse_visual_prompts, description
import torch.nn.functional as F
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
import numpy as np
TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')
internvl_transform = build_transform(is_train=False, input_size=448, pad2square=False, normalize_type="imagenet")
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
objects = [
    'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
    'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

relations = ['Above', 'Below', 'Beside', 'Between', 'Behind', 'In front of', 'Next to', 'Near', 'Far', 'Over', 'Under', 'Inside', 'Outside', 'Adjacent', 'Opposite', 'Surrounding', 'Within', 'Alongside']

positions = ['North', 'South', 'East', 'West', 'Northeast', 'Northwest', 'Southeast', 'Southwest', 'Up', 'Down', 'Left', 'Right', 'Top', 'Bottom', 'Front', 'Back', 'Center', 'Edge', 'Corner']

colors = ['Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Purple', 'Pink', 'Brown', 'Cyan', 'Silver', 'Gold', 'Gray', 'White', 'Black']


def process_logits(tokenizer,logits,input_list):
    return_dict = {}

    for item in input_list:
        item = item.lower()
        input_ids = tokenizer.encode(item)[1:]
        value_list = []
        if(len(input_ids)>1):
            continue
        for input_id in input_ids:
            value_list.append(logits[0,0,input_id].item())
        return_dict[item] = np.mean(value_list)
    return return_dict

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

    has_image_id_flag = 0
    ori_masks = []
    show_analyse = {}
    with torch.no_grad():
        seg_flag = True
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
                if seg_flag:
                    show_analyse["objects"] = process_logits(tokenizer,logits,objects)
                    show_analyse["relations"] = process_logits(tokenizer,logits,relations)
                    show_analyse["positions"] = process_logits(tokenizer,logits,positions)
                    show_analyse["colors"] = process_logits(tokenizer,logits,colors)
                    seg_flag = False


                batch_idxs = torch.zeros((seg_hidden.shape[0],),dtype=torch.int64).to(seg_hidden.device)
                pred_masks_list = model.visual_encoder.forward_llm_seg(seg_hidden, batch_idxs)
                pred_masks = pred_masks_list[-1]
                w, h = ori_image.size
                masks = F.interpolate(pred_masks, size=(max(w, h), max(w, h)),mode='bilinear', align_corners=False)
                masks = masks[:, 0]
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
            else:
                current_inputs_embeds = model.llm.get_input_embeddings()(current_input_ids)
            past_key_values = output['past_key_values']
            if early_stop and current_input_ids.item() == 92542:
                break
    print(show_analyse)
    return {
        'predict': predict.replace("[SEG]<image_id>",""),
        "output_masks": final_output_masks
    }


def parse_args(args):
    parser = argparse.ArgumentParser(description="OMG-LLaVA Demo")
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('pth_model', help='pth model file')

    parser.add_argument(
        '--torch-dtype',
        default='fp16',
        choices=TORCH_DTYPE_MAP.keys(),
        help='Override the default `torch.dtype` and load the model under '
             'a specific `dtype`.')
    parser.add_argument(
        '--prompt-template',
        choices=PROMPT_TEMPLATE.keys(),
        default="internvl2_chat",
        help='Specify a prompt template')
    system_group = parser.add_mutually_exclusive_group()
    system_group.add_argument(
        '--system', default=None, help='Specify the system text')
    system_group.add_argument(
        '--system-template',
        choices=SYSTEM_TEMPLATE.keys(),
        default=None,
        help='Specify a system template')
    parser.add_argument(
        '--stop-words', nargs='+', type=str, default=[], help='Stop words')
    parser.add_argument(
        '--bot-name', type=str, default='BOT', help='Name for Bot')
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=256,
        help='Maximum number of new tokens allowed in generated text')
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='The value used to modulate the next token probabilities.')
    parser.add_argument(
        '--top-k',
        type=int,
        default=40,
        help='The number of highest probability vocabulary tokens to '
             'keep for top-k-filtering.')
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.75,
        help='If set to float < 1, only the smallest set of most probable '
             'tokens with probabilities that add up to top_p or higher are '
             'kept for generation.')
    parser.add_argument(
        '--repetition-penalty',
        type=float,
        default=1.0,
        help='The parameter for repetition penalty. 1.0 means no penalty.')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducible text generation')
    return parser.parse_args(args)



def inference(input_str, all_inputs):

    input_image = all_inputs['image']

    
    # reset
    print('Log: History responses have been removed!')
    global_infos.n_turn = 0
    global_infos.inputs = ''
    # reset prompts
    global_infos.point_prompts = []
    global_infos.box_prompts = []
    global_infos.mask_prompts = []

    # first conversation, add image tokens
    text = input_str

    # prepare image
    ori_image = load_image(input_image)

    internvl_vit_embeds = []
    #add
    internvl_images = dynamic_preprocess(ori_image, min_num=1, max_num=6,
                                image_size=448, use_thumbnail=True)
    internvl_pixel_values = [internvl_transform(internvl_image) for internvl_image in internvl_images]
    internvl_pixel_values = torch.stack(internvl_pixel_values)
    #add


    width, height = ori_image.size
    global_infos.image_width = width
    global_infos.image_height = height
    image = expand2square(
        ori_image, tuple(int(x * 255) for x in image_processor.image_mean))
    global_infos.image_for_show = image
    image = image_processor.preprocess(
        image, return_tensors='pt')['pixel_values'][0]
    pixel_values = image.unsqueeze(0)
    global_infos.pixel_values = pixel_values

    # for remove padding
    if width == height:
        sx, ex, sy, ey = 0, width, 0, height
    elif width > height:
        sy = int((width - height) / 2.0)
        ey = width - sy
        sx, ex = 0, width
    else:
        sx = int((height - width) / 2.0)
        ex = height - sx
        sy, ey = 0, height

    global_infos.sx = sx
    global_infos.sy = sy
    global_infos.ex = ex
    global_infos.ey = ey

    print(text)
    predict = base_generate(model, text, ori_image, pixel_values.cuda(), [internvl_pixel_values.cuda()],model.tokenizer, max_new_tokens=512)
    answer = predict['predict']
    print(answer)
    output_masks = predict['output_masks']
    if len(output_masks)==0:
        return ori_image, answer
    else:
        panoptic_show, selected_colors = show_mask_pred_new(ori_image, output_masks)

        predict = process_markdown(answer, selected_colors)
        return panoptic_show, predict

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
    model_name = cfg.model.type if isinstance(cfg.model.type, str) else cfg.model.type.__name__

    cfg.model.pretrained_pth = None

    model = BUILDER.build(cfg.model)

    backend = get_file_backend(args.pth_model)
    if isinstance(backend, PetrelBackend):
        from xtuner.utils.fileio import patch_fileio
        with patch_fileio():
            state_dict = guess_load_checkpoint(args.pth_model)
    else:
        state_dict = guess_load_checkpoint(args.pth_model)
    old_embedding = state_dict['llm.base_model.model.model.tok_embeddings.weight']
    new_embedding = model.llm.get_output_embeddings().weight
    if old_embedding.shape[0]!=new_embedding.shape[0]:
        new_add_num = new_embedding.shape[0]-old_embedding.shape[0]
        mean_value = old_embedding.mean(dim=0, keepdim=True).repeat(new_add_num, 1)
        old_embedding = torch.cat((old_embedding, mean_value), dim=0)
        state_dict['llm.base_model.model.model.tok_embeddings.weight']=old_embedding

        mean_value1 = state_dict['llm.base_model.model.output.base_layer.weight'].mean(dim=0, keepdim=True).repeat(new_add_num, 1)
        state_dict['llm.base_model.model.output.base_layer.weight']=torch.cat((state_dict['llm.base_model.model.output.base_layer.weight'], mean_value1), dim=0)


        mean_value2 = torch.zeros(new_add_num,state_dict['llm.base_model.model.output.lora_B.default.weight'].shape[1]).to(state_dict['llm.base_model.model.output.lora_B.default.weight'].device)
        state_dict['llm.base_model.model.output.lora_B.default.weight']=torch.cat((state_dict['llm.base_model.model.output.lora_B.default.weight'], mean_value2), dim=0)
        print("add new tokens leads to size mismatch, manually fix it!!!")


    model.load_state_dict(state_dict, strict=False)
    print(f'Load PTH model from {args.pth_model}')

    image_processor = cfg.image_processor
    image_processor_type = image_processor['type']
    del image_processor['type']
    image_processor = image_processor_type(**image_processor)

    # build llm
    quantization_config = None
    load_in_8bit = False
    model_kwargs = {
        'quantization_config': quantization_config,
        'load_in_8bit': load_in_8bit,
        'device_map': 'auto',
        'offload_folder': None,
        'trust_remote_code': True,
        'torch_dtype': TORCH_DTYPE_MAP[args.torch_dtype]
    }

    inner_thoughts_open = False
    calculate_open = False
    solve_open = False
    search_open = False
    model.to(torch.bfloat16)
    # build llm
    llm = model.llm
    tokenizer = model.tokenizer

    model.cuda()
    model.eval()
    llm.eval()
    visual_encoder = model.visual_encoder
    projector = model.projector
    projector_text2vision = model.projector_text2vision

    visual_encoder.eval()
    projector.eval()
    projector_text2vision.eval()
    return model, llm, tokenizer, image_processor, visual_encoder, projector, projector_text2vision

def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    return hidden_states[-n_out:][seg_mask]

class global_infos:
    inputs = ''
    n_turn = 0
    image_width = 0
    image_height = 0

    image_for_show = None
    pixel_values = None
    panoptic_masks = None

    sx, sy, ex, ey = 0, 0 ,1024, 1024

    point_prompts = []
    box_prompts = []
    mask_prompts = []

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    model, llm, tokenizer, image_processor, visual_encoder, projector, projector_text2vision = \
        init_models(args)

    stop_words = args.stop_words
    sep = ''
    if args.prompt_template:
        template = PROMPT_TEMPLATE[args.prompt_template]
        stop_words += template.get('STOP_WORDS', [])
        sep = template.get('SEP', '')
    stop_criteria = get_stop_criteria(
        tokenizer=tokenizer, stop_words=stop_words)


    streamer = TextStreamer(tokenizer, skip_prompt=True)

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )

    demo = gr.Interface(
        inference, inputs=[gr.Textbox(lines=1, placeholder=None, label="Text Instruction"), ImagePrompter(
            type='filepath', label='Input Image', interactive=True,
            elem_id='image_upload', height=360, visible=True, render=True
            )],
        outputs=[
            gr.Image(type="pil", label="Output Image"),
            gr.Markdown()],
        theme=gr.themes.Soft(), allow_flagging="auto", description=description,
        title='LIRA'
    )

    demo.queue()
    demo.launch(share=True)