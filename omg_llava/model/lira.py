from collections import OrderedDict
import torch
import torch.nn as nn
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from peft import get_peft_model, prepare_model_for_kbit_training

from xtuner.registry import BUILDER
from .modules import ProjectorConfig_OMG_LLaVA, ProjectorModel_OMG_LLaVA
from xtuner.model.modules import ProjectorModel, ProjectorConfig
from xtuner.model.modules import dispatch_modules
from .utils import (LoadWoInit, find_all_linear_names,
                    get_peft_model_state_dict, guess_load_checkpoint,
                    make_inputs_require_grad,
                    traverse_dict,
                    prepare_inputs_labels_for_multimodal_with_visual_prompts,
                    prepare_internvl_inputs_labels_for_multimodal_with_visual_prompts)
from .convnext_clip import OpenCLIPBackbone
from .omg_seg import OMGSegVisualEncoder
import copy
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          PROMPT_TEMPLATE, SYSTEM_TEMPLATE)
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)
from xtuner.tools.utils import get_stop_criteria
import torch.nn.functional as F
from omg_llava.dataset.utils import expand2square_bbox, expand2square_mask, expand2square_points
import numpy as np
from omg_llava.model.modules.projector.modeling_projector import CrossAttentionLayer, FFNLayer

def find_enclosed_values(tensor, start_value, end_value):
    values = []
    i = 0
    while i < len(tensor):
        if tensor[i] == start_value:
            j = i + 1
            while j < len(tensor) and tensor[j] != end_value:
                if tensor[j] == start_value:
                    break
                j += 1
            if j < len(tensor) and tensor[j] == end_value:
                values.append(tensor[i+1:j].tolist())
                i = j
        i += 1
    return values
def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    return hidden_states[-n_out:][seg_mask]
class LIRA(BaseModel):
    def __init__(self,
                 llm,
                 visual_encoder,
                 visual_select_layer=-2,
                 freeze_llm=False,
                 freeze_visual_encoder=False,
                 require_omg_decoder=False,
                 pretrained_pth=None,
                 llm_lora=None,
                 visual_encoder_lora=None,
                 use_activation_checkpointing=True,
                 projector_depth=2,
                 text2vision_projector=False,
                 tokenizer=None,
                 keep_omg_decoder_frozen=False,
                 add_seg_pretrain=False,
                 additional_cross_attn_layers=False,
                 pixel_shuffle_ratio=None,
                 train_vocabulary=False,
                 freeze_llm_with_lora=False,
                 freeze_visual_projector=False,
                 rm_prior_embedding=False,
                 rm_query=False,
                 clip_feat_channel=1536,
                 is_init_new_decoder=True,
                 unfreeze_mlp=False,
                 add_cross=False,
                 prefix = False,
                 cross_head_num=32,

                 onlyomg_pretrain=False,
                 ):
        super().__init__()

        self.freeze_llm_with_lora = freeze_llm_with_lora
        self.freeze_visual_projector = freeze_visual_projector

        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        self.add_cross = add_cross
        self.onlyomg_pretrain = onlyomg_pretrain
        self.prefix = prefix
        with LoadWoInit():
            vlm = self._build_from_cfg_or_module(llm)
            
            self.llm = vlm.language_model
            self.vlm_enc = vlm.vision_model
            self.vlm_mlp = vlm.mlp1

            if visual_encoder.type == OpenCLIPBackbone or visual_encoder.type == OMGSegVisualEncoder:
                self.visual_encoder = visual_encoder.type(**visual_encoder)
            else:
                self.visual_encoder = self._build_from_cfg_or_module(
                    visual_encoder)
        self.llm.config.use_cache = False
        dispatch_modules(self.llm)

        projector_config = ProjectorConfig_OMG_LLaVA(
            query_channels=256,
            feat_channels=clip_feat_channel,
            llm_hidden_size=self.llm.config.hidden_size,
            depth=projector_depth,
            pixel_shuffle_ratio=pixel_shuffle_ratio,
        )
        self.projector = ProjectorModel_OMG_LLaVA(projector_config,self.add_cross, self.prefix, cross_head_num).to(
            self.visual_encoder.dtype)

        self.text2vision_projector = text2vision_projector
        if text2vision_projector:
            projector_config = ProjectorConfig(
                visual_hidden_size=self.llm.config.hidden_size,
                llm_hidden_size=256 * 2,
                depth=projector_depth)
            self.projector_text2vision = ProjectorModel(projector_config).to(
                self.visual_encoder.dtype)


        if rm_query:
            self.projector.model.rm_query = rm_query
        if rm_prior_embedding:
            self.projector.model.rm_prior_embedding = rm_prior_embedding

        if self.freeze_llm:
            self.llm.requires_grad_(False)
        if self.freeze_visual_encoder:
            self.visual_encoder.requires_grad_(False)
        self.vlm_enc.requires_grad_(False)
        self.unfreeze_mlp = unfreeze_mlp
        if self.unfreeze_mlp:
            self.vlm_mlp.requires_grad_(True)
        else:
            self.vlm_mlp.requires_grad_(False)

        self.use_activation_checkpointing = use_activation_checkpointing
        if use_activation_checkpointing:
            # For backward compatibility
            if hasattr(self.llm, 'enable_input_require_grads'):
                self.llm.enable_input_require_grads()
            else:
                self.llm.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad)
            if hasattr(self.visual_encoder, 'enable_input_require_grads'):
                self.visual_encoder.enable_input_require_grads()
            else:
                self.visual_encoder.get_input_embeddings(
                ).register_forward_hook(make_inputs_require_grad)
            self.projector.enable_input_require_grads()
            if text2vision_projector:
                self.projector_text2vision.enable_input_require_grads()

            # enable gradient (activation) checkpointing for memory efficiency
            self.gradient_checkpointing_enable()

        # resize input embed before add llm lora
        self.added_special_token = False
        if tokenizer is not None:
            self.tokenizer = tokenizer
            tokenizer_type = self.tokenizer['type']
            del self.tokenizer['type']
            self.tokenizer = tokenizer_type(**self.tokenizer)
            self._add_special_tokens()

        self.use_llm_lora = llm_lora is not None
        self.use_visual_encoder_lora = visual_encoder_lora is not None

        if self.use_llm_lora:
            self._prepare_llm_for_lora(llm_lora, use_activation_checkpointing)
            if self.freeze_llm_with_lora:
                for name, param in self.llm.named_parameters():
                    param.requires_grad_(False)
        else:
            if train_vocabulary:
                # train vocabulary embedding and logit head when pretrain
                for name, param in self.named_parameters():
                    if 'tok_' in name or 'lm_head' in name:
                        print("Unfrozen {} !!!".format(name))
                        param.requires_grad_(True)
                    if 'output.' in name and 'llm' in name and 'lora' not in name:
                        print("Unfrozen {} !!!".format(name))
                        param.requires_grad_(True)

        if self.use_visual_encoder_lora:
            self._prepare_visual_encoder_for_lora(
                visual_encoder_lora, use_activation_checkpointing)


        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            if 'llm.base_model.model.model.tok_embeddings.weight' in pretrained_state_dict.keys():
                old_embedding = pretrained_state_dict['llm.base_model.model.model.tok_embeddings.weight']
                new_embedding = self.llm.get_output_embeddings().weight
                if old_embedding.shape[0]!=new_embedding.shape[0]:
                    new_add_num = new_embedding.shape[0]-old_embedding.shape[0]
                    mean_value = old_embedding.mean(dim=0, keepdim=True).repeat(new_add_num, 1)
                    old_embedding = torch.cat((old_embedding, mean_value), dim=0)
                    pretrained_state_dict['llm.base_model.model.model.tok_embeddings.weight']=old_embedding

                    mean_value1 = pretrained_state_dict['llm.base_model.model.output.base_layer.weight'].mean(dim=0, keepdim=True).repeat(new_add_num, 1)
                    pretrained_state_dict['llm.base_model.model.output.base_layer.weight']=torch.cat((pretrained_state_dict['llm.base_model.model.output.base_layer.weight'], mean_value1), dim=0)


                    mean_value2 = torch.zeros(new_add_num,pretrained_state_dict['llm.base_model.model.output.lora_B.default.weight'].shape[1]).to(pretrained_state_dict['llm.base_model.model.output.lora_B.default.weight'].device)
                    pretrained_state_dict['llm.base_model.model.output.lora_B.default.weight']=torch.cat((pretrained_state_dict['llm.base_model.model.output.lora_B.default.weight'], mean_value2), dim=0)
                    print("add new tokens leads to size mismatch, manually fix it!!!")
            elif 'llm.model.tok_embeddings.weight' in pretrained_state_dict.keys():
                old_embedding = pretrained_state_dict['llm.model.tok_embeddings.weight']
                new_embedding = self.llm.get_output_embeddings().weight
                if old_embedding.shape[0]!=new_embedding.shape[0]:
                    new_add_num = new_embedding.shape[0]-old_embedding.shape[0]
                    mean_value = old_embedding.mean(dim=0, keepdim=True).repeat(new_add_num, 1)
                    old_embedding = torch.cat((old_embedding, mean_value), dim=0)
                    pretrained_state_dict['llm.model.tok_embeddings.weight']=old_embedding
                    print("add new tokens leads to size mismatch, manually fix it!!!")

            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')

        self.visual_select_layer = visual_select_layer

        self._is_init = True

        self.require_omg_decoder = require_omg_decoder
        self.is_init_new_decoder = is_init_new_decoder
        if require_omg_decoder and is_init_new_decoder:
            self.visual_encoder.init_new_decoder()
            if keep_omg_decoder_frozen:
                for name, param in self.visual_encoder.panoptic_head.transformer_decoder_llm.named_parameters():
                    param.requires_grad_(False)
                print("Frozen all the omg seg decoder !!!(nochange)")
        elif require_omg_decoder and not is_init_new_decoder:
            for name, param in self.visual_encoder.named_parameters():
                if 'panoptic_head.transformer_decoder' in name:
                    param.requires_grad_(True)
                elif 'panoptic_head.mask_embed' in name:
                    param.requires_grad_(True)

        self.additional_cross_attn_layers = additional_cross_attn_layers
        if self.additional_cross_attn_layers:
            self.visual_encoder.init_cross_attn_layer()

        if self.freeze_visual_projector:
            for name, param in self.projector.named_parameters():
                param.requires_grad_(False)

        self.add_seg_pretrain = add_seg_pretrain
    def _add_special_tokens(self):
        assert hasattr(self, "tokenizer")

        segmentation_tokens = ['[SEG]']
        # Adding tokens for GCG
        phrase_tokens = ['<p>', '</p>']
        # add for visual prompt
        region_tokens = ['<region>']
        point_tokens = ['<mark>']
        img_id_tokens = ['<image_id>']
        special_tokens = segmentation_tokens + phrase_tokens + region_tokens + point_tokens+ img_id_tokens
        num_new_tokens = self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        self.seg_token_idx = self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        self.bop_token_idx = self.tokenizer("<p>", add_special_tokens=False).input_ids[0]
        self.eop_token_idx = self.tokenizer("</p>", add_special_tokens=False).input_ids[0]
        self.region_token_idx = self.tokenizer("<region>", add_special_tokens=False).input_ids[0]

        self.mark_token_idx = self.tokenizer("<mark>", add_special_tokens=False).input_ids[0]

        self.img_id_idx = self.tokenizer("<image_id>", add_special_tokens=False).input_ids[0]
        if num_new_tokens>0:
            self.llm.resize_token_embeddings(len(self.tokenizer))
            output_embeddings = self.llm.get_output_embeddings().weight.data
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings[-num_new_tokens:] = output_embeddings_avg
            self.llm.config.vocab_size = len(self.tokenizer)

        if self.use_activation_checkpointing or self.use_llm_lora or not self.freeze_llm:
            self.llm.enable_input_require_grads()
        self.added_special_token = True
        print("[SEG]: {}, <p>: {}, </p>: {}, <region>: {}, <mark>: {}, <image_id>: {}" \
              .format(self.seg_token_idx, self.bop_token_idx,
                      self.eop_token_idx, self.region_token_idx, self.mark_token_idx, self.img_id_idx))
        print('****************************Add special tokens ********************************************')
        return

    def _parse_lora_config(self, lora_config):
        if isinstance(lora_config, dict) or isinstance(
                lora_config, Config) or isinstance(lora_config, ConfigDict):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self,
                              lora_config,
                              use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.llm = prepare_model_for_kbit_training(
            self.llm, use_activation_checkpointing)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.llm)
            lora_config.target_modules = modules
        self.llm = get_peft_model(self.llm, lora_config)
        for name, param in self.named_parameters():
            if 'tok_' in name or 'lm_head' in name:
                print("Unfrozen {} !!!".format(name))
                param.requires_grad_(True)
            if 'output.' in name and 'llm' in name and 'lora' not in name:
                print("Unfrozen {} !!!".format(name))
                param.requires_grad_(True)

    def _prepare_visual_encoder_for_lora(self,
                                         lora_config,
                                         use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.visual_encoder)
            lora_config.target_modules = modules
        self.visual_encoder = get_peft_model(self.visual_encoder, lora_config)

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()
        if hasattr(self.visual_encoder, 'gradient_checkpointing_enable'):
            self.visual_encoder.gradient_checkpointing_enable()
        elif hasattr(self.visual_encoder, 'clip_model'):
            if self.visual_encoder.clip_model is not None:
                self.visual_encoder.clip_model.gradient_checkpointing_enable()
        if hasattr(self.projector, 'gradient_checkpointing_enable'):
            self.projector.gradient_checkpointing_enable()
        if self.text2vision_projector and hasattr(self.projector_text2vision, 'gradient_checkpointing_enable'):
            self.projector_text2vision.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        if hasattr(self.visual_encoder, 'gradient_checkpointing_disable'):
            self.visual_encoder.gradient_checkpointing_disable()
        if hasattr(self.projector, 'gradient_checkpointing_disable'):
            self.projector.gradient_checkpointing_disable()
        if self.text2vision_projector and hasattr(self.projector_text2vision, 'gradient_checkpointing_disable'):
            self.projector_text2vision.gradient_checkpointing_disable()

    def init_weights(self):
        pass

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)

        to_return = OrderedDict()

        # vocabulary embedding
        to_return.update({k: v for k, v in state_dict.items() if 'tok_' in k})
        # logit head
        to_return.update({k: v for k, v in state_dict.items() if 'output.' in k and 'llm.' in k and 'lora' not in k})

        if self.unfreeze_mlp:
            to_return.update({k: v for k, v in state_dict.items() if 'vlm_mlp.' in k})
        
        # Step 1. visual_encoder
        if self.use_visual_encoder_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.visual_encoder, state_dict=state_dict))
        elif not self.freeze_visual_encoder:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'visual_encoder.' in k
            })
        # Step 2. LLM
        if self.use_llm_lora:
            to_return.update(get_peft_model_state_dict(self.llm, state_dict=state_dict))
        elif not self.freeze_llm:
            to_return.update(
                {k: v
                 for k, v in state_dict.items() if 'llm.' in k})
        # Step 3. Projector
        to_return.update({k: v for k, v in state_dict.items() if 'projector.' in k})

        # projector text2vision
        to_return.update({k: v for k, v in state_dict.items() if 'projector_text2vision' in k})

        # visual_encoder.adapter_proj
        if self.freeze_visual_encoder:
            to_return.update({k: v for k, v in state_dict.items() if 'visual_encoder.adapter_proj' in k})

        # git_clip lora
        if hasattr(self.visual_encoder, 'clip_model'):
            if self.visual_encoder.clip_lora is not None:
                to_return.update(
                    get_peft_model_state_dict(self.visual_encoder.clip_model,
                                              state_dict=state_dict))
        # omg decoder for llm
        if self.require_omg_decoder and self.is_init_new_decoder:
            to_return.update(
                {k: v
                for k, v in state_dict.items()
                if 'visual_encoder.panoptic_head.transformer_decoder_llm' in k or
                   'visual_encoder.panoptic_head.mask_embed_llm' in k or
                   'visual_encoder.panoptic_head.pixel_decoder_llm' in k or
                   'visual_encoder.panoptic_head.additional_cross_attn_layers' in k or
                   'visual_encoder.panoptic_head.additional_ffn' in k or
                   'visual_encoder.downsample_layer' in k
                 })
            print("is_init_new_decoder")
        elif self.require_omg_decoder and not self.is_init_new_decoder:
            to_return.update(
                {k: v
                for k, v in state_dict.items()
                if 'visual_encoder.panoptic_head.transformer_decoder' in k or
                   'visual_encoder.panoptic_head.mask_embed' in k
                })
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(f"unforze {name}")
        # import pdb;pdb.set_trace()
        return to_return

    def _build_from_cfg_or_module(self, cfg_or_mod):
        if isinstance(cfg_or_mod, nn.Module):
            return cfg_or_mod
        elif isinstance(cfg_or_mod, dict):
            traverse_dict(cfg_or_mod)
            return BUILDER.build(cfg_or_mod)
        else:
            raise NotImplementedError
    
    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        # if self.ps_version == 'v1':
        if False:
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x
    
    def extract_feature(self, pixel_values):
        # if self.select_layer == -1:
        if True:
            vit_embeds = self.vlm_enc(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vlm_enc(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        # if self.training and self.neftune_alpha is not None:
        #     vit_embeds = self.noised_embed(vit_embeds, self.neftune_alpha)

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.vlm_mlp(vit_embeds)#.to(pixel_values.device)
        return vit_embeds

    def forward(self, data, data_samples=None, mode='loss'):

        if 'pixel_values' in data:
            if 'masks' in data:
                masks = data['masks']
                del data['masks']
            else:
                masks = None
            if 'regions' in data:
                regions = data['regions']
                del data['regions']
            else:
                regions = None
            if 'points' in data:
                points = data['points']
                del data['points']
            else:
                points = None
            if 'has_multi_image' in data:
                has_multi_image = data['has_multi_image']
                image_nums = data['image_nums']
                del data['has_multi_image']
                del data['image_nums']
            else:
                has_multi_image=False
                image_nums=None
            if not has_multi_image:
                visual_outputs = self.visual_encoder(
                    data['pixel_values'].to(self.visual_encoder.dtype),
                    output_hidden_states=True)
                
                internvl_vit_embeds=[]
                for internvl_pixel_value in data['internvl_pixel_values']:
                    internvl_vit_embed = self.extract_feature(internvl_pixel_value.to(self.vlm_enc.dtype))
                    internvl_vit_embed = internvl_vit_embed.reshape(-1, self.llm.config.hidden_size)
                    internvl_vit_embeds.append(internvl_vit_embed)
                output_images_vit_embeds = []
                for output_images_pixel_values in data['output_images_pixel_values']:
                    output_images_vit_embeds_per = []
                    for output_images_pixel_value_per in output_images_pixel_values:
                        output_images_pixel_value_per_embed = self.extract_feature(output_images_pixel_value_per.to(self.vlm_enc.dtype))
                        output_images_pixel_value_per_embed = output_images_pixel_value_per_embed.reshape(-1, self.llm.config.hidden_size)
                        output_images_vit_embeds_per.append(output_images_pixel_value_per_embed)
                    output_images_vit_embeds.append(output_images_vit_embeds_per)
            else:

                tmp_visual_outputs = self.visual_encoder(
                        data['pixel_values'].to(self.visual_encoder.dtype),
                        output_hidden_states=True)
                visual_outputs = []
                start_idx = 0
                for num in image_nums:
                    end_idx = start_idx + num
                    visual_outputs.append([tmp_visual_output[start_idx:end_idx]  for tmp_visual_output in tmp_visual_outputs])
                    start_idx = end_idx

                internvl_vit_embeds=[]
                for internvl_pixel_value_perconv in data['internvl_pixel_values']:
                    internvl_vit_embeds_perconv = []
                    if type(internvl_pixel_value_perconv)==list:
                        for internvl_pixel_value in internvl_pixel_value_perconv:
                            internvl_vit_embed = self.extract_feature(internvl_pixel_value.to(self.vlm_enc.dtype))
                            internvl_vit_embed = internvl_vit_embed.reshape(-1, self.llm.config.hidden_size)
                            internvl_vit_embeds_perconv.append(internvl_vit_embed)
                    else:
                        internvl_vit_embed = self.extract_feature(internvl_pixel_value_perconv.to(self.vlm_enc.dtype))
                        internvl_vit_embed = internvl_vit_embed.reshape(-1, self.llm.config.hidden_size)
                        internvl_vit_embeds_perconv.append(internvl_vit_embed)
                    internvl_vit_embeds.append(internvl_vit_embeds_perconv)

            if self.add_seg_pretrain:
                #pred_obj_query  经过query_in_proj和query_out_proj  query_in_proj先经过linear与clip feature对齐再经过mlp与llm对齐
                #gt_obj_query    只经过query_in_proj
                if self.add_cross:
                    pred_obj_query, gt_obj_query = prepare_seg_pretrain_data(
                        visual_outputs,
                        [self.projector.model.query_proj, self.projector.model.model],
                        self.projector_text2vision.model, self.projector.model.cross_beforellm,self.projector.model.ffn_beforellm,internvl_vit_embeds
                    )
                else:
                    pred_obj_query, gt_obj_query = prepare_seg_pretrain_data(
                        visual_outputs,
                        [self.projector.model.query_proj, self.projector.model.model],
                        self.projector_text2vision.model
                    )
            if isinstance(visual_outputs, list) or isinstance(visual_outputs, tuple)\
                    or isinstance(visual_outputs, torch.Tensor):
                #clip_feat  + seperate_embed   +   有意义的obj query对齐到llm
                if has_multi_image:
                    pixel_values = []
                    for i in range(len(internvl_vit_embeds)):
                        pixel_value_perconv = self.projector(visual_outputs[i], internvl_vit_embeds[i],  onlyomg_pretrain=self.onlyomg_pretrain)
                        pixel_values.extend(pixel_value_perconv)
                else:
                    pixel_values = self.projector(visual_outputs, internvl_vit_embeds, onlyomg_pretrain=self.onlyomg_pretrain)
                    new_pixel_values = []
                    for i in range(len(pixel_values)):
                        new_pixel_values.append(pixel_values[i])
                        new_pixel_values.extend(output_images_vit_embeds[i])
                    pixel_values = new_pixel_values
            else:
                assert has_multi_image==False
                pixel_values = self.projector(
                    visual_outputs.hidden_states[self.visual_select_layer][:, 1:])
            if regions is not None:
                region_embeddings, region_success = self.get_region_embeddings(
                    regions, data['input_ids'],image_nums=image_nums
                )
                del regions
            else:
                region_success = True
                region_embeddings = []

            if points is not None:
                points_mark_embedding, mark_success = self.get_points_embeddings(
                    points, data['input_ids'],
                    width=data['pixel_values'].shape[-1],
                    height=data['pixel_values'].shape[-2],
                    image_nums=image_nums
                )
            else:
                points_mark_embedding = []
                mark_success = True
            data['pixel_values'] = pixel_values

            data = prepare_inputs_labels_for_multimodal_with_visual_prompts(
                    llm=self.llm, region_id=self.region_token_idx,
                    regions_feats=region_embeddings,
                    mark_id=self.mark_token_idx,
                    mark_feats=points_mark_embedding,
                    **data)
        else:
            masks = None

        if mode == 'loss':
            if self.add_seg_pretrain:
                return self.compute_loss(data, data_samples, masks=masks, region_success=region_success,
                                         pred_gt_obj_query=(pred_obj_query, gt_obj_query),
                                         mark_success=mark_success,image_nums=image_nums)
            else:
                return self.compute_loss(data, data_samples, masks=masks,
                                         pred_gt_obj_query=None,
                                         region_success=region_success,
                                         mark_success=mark_success,image_nums=image_nums)
        elif mode == 'predict':
            return self.predict(data, data_samples)
        elif mode == 'tensor':
            return self._forward(data, data_samples)
        else:
            raise NotImplementedError

    def _forward(self, data, data_samples=None):

        outputs = self.llm(**data)

        return outputs

    def predict(self, data, data_samples=None):
        outputs = self.llm(**data)
        logits_dict = [{'logits': logits} for logits in outputs.logits]
        return logits_dict

    def compute_loss(self, data, data_samples=None, masks=None, pred_gt_obj_query=None,
                     region_success=True, mark_success=True,image_nums=None):
        if 'original_labels' in data.keys():
            input_ids = data['original_labels']
            del data['original_labels']
        else:
            input_ids = data['labels']
        outputs = self.llm(**data, output_hidden_states=True)
        loss_dice, loss_mask = self.compute_seg_loss(
            input_ids, outputs.hidden_states[-1], masks, image_nums)

        if pred_gt_obj_query is not None:
            pred_obj_query, gt_obj_query = pred_gt_obj_query
            proj_loss = torch.mean((pred_obj_query - gt_obj_query) ** 2) * 10
        else:
            proj_loss = 0

        if not region_success:
            loss = outputs.loss * 0
        else:
            loss = outputs.loss

        if not mark_success:
            loss = outputs.loss * 0

        loss = loss + self.get_visual_prompts_projector_zero()
        loss_dict = {'loss': loss, 'loss_dice': outputs.loss* 0 + loss_dice * 0.1,
                     'loss_mask': outputs.loss * 0 + loss_mask * 0.4,
                     'loss_proj': outputs.loss * 0 + proj_loss}
        return loss_dict

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.llm, name)

    def get_region_embeddings(self, regions, input_ids, image_nums=None, pre_batch_ids=None):
        success = True
        if regions is None or len(regions) == 0:
            return [], success
        else:
            region_token_mask = input_ids == self.region_token_idx
            if pre_batch_ids is not None:
                batch_idxs=torch.tensor(pre_batch_ids).to(region_token_mask.device)
            else:
                if image_nums is not None:
                    new_image_nums = copy.deepcopy(image_nums)
                    new_image_nums.insert(0, 0)
                    sum_image_nums = [sum(new_image_nums[:i+1]) for i in range(len(new_image_nums))]
                    batch_idxs = torch.tensor(sum_image_nums[:-1]).unsqueeze(1).repeat(1, input_ids.shape[1]).to(region_token_mask.device)
                    batch_idxs = batch_idxs[region_token_mask]
                else:
                    batch_idxs = torch.arange(input_ids.shape[0]).unsqueeze(1).repeat(1, input_ids.shape[1]).to(
                        input_ids.device)
                    batch_idxs = batch_idxs[region_token_mask]  # (N, ) batch_size number

            if len(regions) != len(batch_idxs):
                # There is a bug !!! skip it.
                success = False
                if len(regions) > len(batch_idxs):
                    regions = regions[:len(batch_idxs)]
                else:
                    n_pad = len(batch_idxs) - len(regions)
                    pad_region = regions[:1].repeat(n_pad, 1, 1)
                    regions = torch.cat([pad_region, regions])

            regions_embeddings = self.visual_encoder.forward_region_sam(
                regions, batch_idxs
            )[:, 0]  # (N, C)

            regions_embeddings = self.projector.model.forward_visual_prompts_embeddings(
                regions_embeddings, batch_idxs)
            return regions_embeddings, success  # (N, C)

    def get_points_embeddings(self, points, input_ids, width, height,image_nums=None, pre_batch_ids=None):
        success = True
        if points is None or len(points) == 0:
            return []

        mark_token_mask = input_ids == self.mark_token_idx
        img_id_token_mask = input_ids == self.img_id_idx
        if pre_batch_ids is not None:
            batch_idxs=torch.tensor(pre_batch_ids).to(mark_token_mask.device)
        else:
            if image_nums is not None:
                new_image_nums = copy.deepcopy(image_nums)
                new_image_nums.insert(0, 0)
                sum_image_nums = [sum(new_image_nums[:i+1]) for i in range(len(new_image_nums))]
                batch_idxs = torch.tensor(sum_image_nums[:-1]).unsqueeze(1).repeat(1, input_ids.shape[1]).to(mark_token_mask.device)
                if int(img_id_token_mask.sum())==0:
                    batch_idxs = batch_idxs[mark_token_mask]
                else:
                    new_batch_idxs = []
                    for i in range(input_ids.shape[0]):
                        if int(img_id_token_mask[i].sum())!=0:
                            decode_num = []
                            results = find_enclosed_values(input_ids[i], self.img_id_idx, self.mark_token_idx)
                            for result in results:
                                decode_num.append(int(self.tokenizer.decode(result)))
                            decode_num = torch.tensor(decode_num).to(mark_token_mask.device)
                            new_batch_idxs.extend(batch_idxs[i][mark_token_mask[i]]+decode_num)
                        else:
                            new_batch_idxs.extend(batch_idxs[i][mark_token_mask[i]])
                    batch_idxs = torch.tensor(new_batch_idxs).to(mark_token_mask.device)
            else:
                batch_idxs = torch.arange(input_ids.shape[0]).unsqueeze(1).repeat(1, input_ids.shape[1]).to(
                    input_ids.device)
                batch_idxs = batch_idxs[mark_token_mask]  # (N, ) batch_size number
        if len(points) != len(batch_idxs):
            # There is a bug !!! skip it.
            success = False
            if len(points) > len(batch_idxs):
                points = points[:len(batch_idxs)]
            else:
                n_pad = len(batch_idxs) - len(points)
                pad_region = points[:1].repeat(n_pad, 1, 1)
                points = torch.cat([pad_region, points])
        marks_embeddings = self.visual_encoder.forward_point_sam(
            points.float(), batch_idxs, width=width, height=height
        )[:, 0]  # (N, C) torch.Size([46, 512])
        marks_embeddings = self.projector.model.forward_visual_prompts_embeddings(
            marks_embeddings, batch_idxs)
        return marks_embeddings, success  # (N, C)

    def get_visual_prompts_projector_zero(self):
        return self.projector.model.visual_prompt_zero

    def compute_seg_loss(self, input_ids, hidden_states, gt_masks, image_nums=None):
        if not self.text2vision_projector or self.add_seg_pretrain:
            return 0.0, 0.0
        success = True
        if gt_masks is None or len(gt_masks) == 0:
            batch_idxs = torch.arange(input_ids.shape[0]).unsqueeze(1).repeat(1, input_ids.shape[1]).to(
                input_ids.device)
            batch_idxs = batch_idxs[0, :1]  # (N, ) batch_size number
            gt_masks = [None]
            hidden_states = hidden_states[0, :1]
            hidden_states = self.projector_text2vision(hidden_states)  # (N, C)

            pred_masks_list = self.visual_encoder.forward_llm_seg(hidden_states, batch_idxs)
            dice_loss, mask_loss = self.visual_encoder.loss_llm_seg(pred_masks_list, gt_masks)

            return dice_loss * 0.0, mask_loss * 0.0

        seg_tokens_mask = input_ids == self.seg_token_idx
        img_id_token_mask = input_ids == self.img_id_idx
        if image_nums is None:
            batch_idxs = torch.arange(input_ids.shape[0]).unsqueeze(1).repeat(1, input_ids.shape[1]).to(seg_tokens_mask.device)
            batch_idxs = batch_idxs[seg_tokens_mask]  # (N, ) batch_size number
        else:
            new_image_nums = copy.deepcopy(image_nums)
            new_image_nums.insert(0, 0)
            sum_image_nums = [sum(new_image_nums[:i+1]) for i in range(len(new_image_nums))]
            batch_idxs = torch.tensor(sum_image_nums[:-1]).unsqueeze(1).repeat(1, input_ids.shape[1]).to(seg_tokens_mask.device)
            if int(img_id_token_mask.sum())==0:
                try:
                    batch_idxs = batch_idxs[seg_tokens_mask]
                except:
                    prin("batch_idxs error")
                    return dice_loss * 0.0, mask_loss * 0.0
            else:
                new_batch_idxs = []
                for i in range(input_ids.shape[0]):
                    if int(img_id_token_mask[i].sum())!=0:
                        decode_num = []
                        results = find_enclosed_values(input_ids[i], self.img_id_idx, self.seg_token_idx)
                        for result in results:
                            decode_num.append(int(self.tokenizer.decode(result)))
                        decode_num = torch.tensor(decode_num).to(seg_tokens_mask.device)
                        new_batch_idxs.extend(batch_idxs[i][seg_tokens_mask[i]]+decode_num)
                    else:
                        new_batch_idxs.extend(batch_idxs[i][seg_tokens_mask[i]])
                batch_idxs = torch.tensor(new_batch_idxs).to(seg_tokens_mask.device)
        ori_hidden_states = hidden_states
        hidden_states = hidden_states[seg_tokens_mask]

        gt_masks = gt_masks.repeat_interleave(2, dim=0)
        if len(hidden_states) != len(gt_masks) or len(hidden_states) == 0:
            # drop this batch
            print("Drop the batch because the number of [SEG] and masks not equal !!!")
            hidden_states = ori_hidden_states
            batch_idxs = torch.arange(input_ids.shape[0]).unsqueeze(1).repeat(1, input_ids.shape[1]).to(
                input_ids.device)
            batch_idxs = batch_idxs[0, :1]  # (N, ) batch_size number
            gt_masks = [None]
            hidden_states = hidden_states[0, :1]
            hidden_states = self.projector_text2vision(hidden_states)  # (N, C)

            pred_masks_list = self.visual_encoder.forward_llm_seg(hidden_states, batch_idxs)
            dice_loss, mask_loss = self.visual_encoder.loss_llm_seg(pred_masks_list, gt_masks)

            return dice_loss * 0.0, mask_loss * 0.0

        assert len(hidden_states) == len(gt_masks), "expect [seg] number equal to mask number, but get {} [seg] and {} masks".format(len(hidden_states), len(gt_masks))
        hidden_states = self.projector_text2vision(hidden_states)  # (N, C)
        
        pred_masks_list = self.visual_encoder.forward_llm_seg(hidden_states, batch_idxs)
        
        dice_loss, mask_loss = self.visual_encoder.loss_llm_seg(pred_masks_list, gt_masks)

        if not success:
            return dice_loss * 0.0, mask_loss * 0.0

        return dice_loss, mask_loss
    #history = {"images":[],"points":[],"boxes":[],conversation:[{user:ques,assistant:answer}]}
    def chat(self, question, ori_images=None, pixel_values=None, internvl_pixel_values=None, points=None, regions=None, generation_config=None, history=None,  prompt_template="internlm2_chat",system_text="You are an AI assistant whose name is InternLM (书生·浦语)."):
        if generation_config is None:
            generation_config = GenerationConfig(
                max_new_tokens=200,
                do_sample=False,
                temperature=0,
                top_p=0.75,
                top_k=1,
                repetition_penalty=1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
            )
        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        #get input ids start
        prompt_text = ''
        template = PROMPT_TEMPLATE[prompt_template]
        # prompt_text = template['SYSTEM'].format(system=system_text)
        if history is None:
            prompt_text+=template['INSTRUCTION'].format(input=question)
        else:
            conversation = history['conversation']
            for i in range(conversation):
                prompt_text+=template['INSTRUCTION'].format(input=conversation[i]['user'])
                prompt_text+=template['INSTRUCTION'].format(input=conversation[i]['assistant'])
            prompt_text+=template['INSTRUCTION'].format(input=question)
        if prompt_text.count("<image>")!=len(ori_images):
            print("error  ")
            return "ABCD",None
        assert prompt_text.count("<image>")==len(ori_images)
        chunk_encode = []
        for idx, chunk in enumerate(prompt_text.split(DEFAULT_IMAGE_TOKEN)):
            cur_encode = self.tokenizer.encode(chunk, add_special_tokens=False)
            chunk_encode.append(cur_encode)
        ids = []
        for idx, cur_chunk_encode in enumerate(chunk_encode):
            ids.extend(cur_chunk_encode)
            if idx != len(chunk_encode) - 1:
                ids.append(IMAGE_TOKEN_INDEX)
        ids = torch.tensor(ids).to(self.device).unsqueeze(0)
        visual_outputs = self.visual_encoder(pixel_values.to(self.visual_encoder.dtype),output_hidden_states=True)\
        
        internvl_vit_embeds=[]
        for internvl_pixel_value in internvl_pixel_values:
            internvl_vit_embed = self.extract_feature(internvl_pixel_value.to(self.vlm_enc.dtype))
            internvl_vit_embed = internvl_vit_embed.reshape(-1, self.llm.config.hidden_size)
            internvl_vit_embeds.append(internvl_vit_embed)
        image_features = self.projector(visual_outputs, internvl_vit_embeds)
        if regions is not None:
            batch_ids = []
            input_regions = []
            for i in range(len(regions)):
                for key, value in regions[i].items():
                    batch_ids.append(key)
                    w,h = ori_images[key].size
                    value = np.array([value])
                    value = expand2square_bbox(value, height=h, width=w)
                    value[:, [0, 2]] = value[:, [0, 2]] / max(h, w) * 1024
                    value[:, [1, 3]] = value[:, [1, 3]] / max(h, w) * 1024
                    input_regions.append(list(value[0]))
            input_regions = np.array(input_regions) 
            input_regions = torch.from_numpy(input_regions)
            mask_tensors = []
            for input_region in input_regions:
                mask_tensor = torch.zeros((1024, 1024), dtype=torch.uint8)
                x1, y1, x2, y2 = map(int, input_region)
                mask_tensor[y1:y2, x1:x2] = 1
                mask_tensors.append(mask_tensor)
            mask_tensors = torch.stack(mask_tensors, dim=0).to(self.device)
            region_embeddings, region_success = self.get_region_embeddings(
                mask_tensors, ids, pre_batch_ids=batch_ids
            )
        else:
            region_success = True
            region_embeddings = []

        if points is not None:
            batch_ids = []
            input_points = []
            for i in range(len(points)):
                 for key, value in points[i].items():
                    batch_ids.append(key)
                    w,h = ori_images[key].size
                    value = np.array([value])
                    value = expand2square_points(value, height=h, width=w)
                    value[:, 0] = value[:, 0] / max(h, w) * 1024
                    value[:, 1] = value[:, 1] / max(h, w) * 1024
                    input_points.append(list(value[0]))
            input_points = np.array(input_points) 
            input_points = torch.from_numpy(input_points)
            input_points = input_points.to(self.device)
            points_mark_embedding, mark_success = self.get_points_embeddings(
                input_points, ids,
                width=pixel_values.shape[-1],
                height=pixel_values.shape[-2],
                pre_batch_ids=batch_ids
            )
        else:
            points_mark_embedding = []
            mark_success = True
        mm_inputs =prepare_inputs_labels_for_multimodal_with_visual_prompts(
                llm=self.llm, region_id=self.region_token_idx,
                regions_feats=region_embeddings,
                mark_id=self.mark_token_idx,
                mark_feats=points_mark_embedding,
                pixel_values=image_features,
                input_ids=ids)
        stop_criteria = get_stop_criteria(tokenizer=self.tokenizer, stop_words=template.STOP_WORDS)
        generate_output = self.llm.generate(
            **mm_inputs,
            generation_config=generation_config,
            bos_token_id=self.tokenizer.bos_token_id,
            stopping_criteria=stop_criteria,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
        predict = self.tokenizer.decode(generate_output.sequences[0])
        hidden_states = generate_output.hidden_states
        last_hidden_states = [item[-1][0] for item in hidden_states]
        last_hidden_states = torch.cat(last_hidden_states, dim=0)
        seg_hidden_states = get_seg_hidden_states(
            last_hidden_states, generate_output.sequences[0][:-1],
            seg_id=self.seg_token_idx
        )
        if len(seg_hidden_states) != 0:
            seg_hidden_states = self.projector_text2vision(seg_hidden_states)
            if self.img_id_idx in generate_output.sequences[0]:
                results = find_enclosed_values(generate_output.sequences[0], self.img_id_idx, self.seg_token_idx)
                batch_idxs = []
                for result in results:
                    batch_idxs.append(int(self.tokenizer.decode(result))+(len(ori_images)-question.count("<image>")))#need to add
                batch_idxs = torch.tensor(batch_idxs).to(seg_hidden_states.device)
            else:
                batch_idxs = torch.full((seg_hidden_states.shape[0],),fill_value=int(pixel_values.shape[0]-1),dtype=torch.int64,device=seg_hidden_states.device)
            pred_masks_list = self.visual_encoder.forward_llm_seg(seg_hidden_states, batch_idxs)[-1]
            pred_masks = []
            for i in range(pred_masks_list.shape[0]):
                pred_mask = pred_masks_list[i]
                w,h = ori_images[int(batch_idxs[i])].size
                pred_mask = F.interpolate(pred_mask.unsqueeze(0), size=(max(w, h), max(w, h)), mode='bilinear', align_corners=False)
                pred_mask = pred_mask[:, 0]
                if w == h:
                    pass
                elif w > h:
                    n_pad = w - h
                    n_pad_1 = n_pad // 2
                    n_pad_2 = n_pad - n_pad_1
                    pred_mask = pred_mask[:, n_pad_1: w - n_pad_2]
                else:
                    n_pad = h - w
                    n_pad_1 = n_pad // 2
                    n_pad_2 = n_pad - n_pad_1
                    pred_mask = pred_mask[:, :, n_pad_1: h - n_pad_2]
                pred_mask=pred_mask.squeeze(0)
                pred_mask = pred_mask.sigmoid() > 0.5
                pred_mask = pred_mask.to(torch.uint8).cpu().numpy()
                pred_masks.append(pred_mask)
            return predict, pred_masks
        else:
            return predict, None
    def chat_clean(self, question, pixel_values=None, internvl_pixel_values=None, generation_config=None, history=None, prompt_template="internvl2_chat"):
        if generation_config is None:
            generation_config = GenerationConfig(
                max_new_tokens=200,
                do_sample=False,
                temperature=0,
                top_p=0.75,
                top_k=1,
                repetition_penalty=1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
            )
        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question
        template = PROMPT_TEMPLATE[prompt_template]
        if prompt_template == "internvl2_chat":
            prompt_text = template['SYSTEM']
            # question = question.replace("<image>","<img><image></img>")
        else:
            prompt_text = ""
        if history is None:
            prompt_text+=template['INSTRUCTION'].format(input=question)
        else:
            conversation = history['conversation']
            for i in range(conversation):
                prompt_text+=template['INSTRUCTION'].format(input=conversation[i]['user'])
                prompt_text+=template['INSTRUCTION'].format(input=conversation[i]['assistant'])
            prompt_text+=template['INSTRUCTION'].format(input=question)
        chunk_encode = []
        for idx, chunk in enumerate(prompt_text.split(DEFAULT_IMAGE_TOKEN)):
            cur_encode = self.tokenizer.encode(chunk, add_special_tokens=False)
            chunk_encode.append(cur_encode)
        ids = []
        for idx, cur_chunk_encode in enumerate(chunk_encode):
            ids.extend(cur_chunk_encode)
            if idx != len(chunk_encode) - 1:
                ids.append(IMAGE_TOKEN_INDEX)
        ids = torch.tensor(ids).to(self.device).unsqueeze(0)
        visual_outputs = self.visual_encoder(pixel_values.to(self.visual_encoder.dtype),output_hidden_states=True)
        
        internvl_vit_embeds=[]
        for internvl_pixel_value in internvl_pixel_values:
            internvl_vit_embed = self.extract_feature(internvl_pixel_value.to(self.vlm_enc.dtype))
            internvl_vit_embed = internvl_vit_embed.reshape(-1, self.llm.config.hidden_size)
            internvl_vit_embeds.append(internvl_vit_embed)
        

        image_features = self.projector(visual_outputs, internvl_vit_embeds)
        

        mm_inputs =prepare_inputs_labels_for_multimodal_with_visual_prompts(
                llm=self.llm, region_id=self.region_token_idx,
                regions_feats=None,
                mark_id=self.mark_token_idx,
                mark_feats=None,
                pixel_values=image_features,
                input_ids=ids)
        stop_criteria = get_stop_criteria(tokenizer=self.tokenizer, stop_words=template.STOP_WORDS)
        generate_output = self.llm.generate(
            **mm_inputs,
            generation_config=generation_config,
            bos_token_id=self.tokenizer.bos_token_id,
            stopping_criteria=stop_criteria,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
        predict = self.tokenizer.decode(generate_output.sequences[0],skip_special_tokens=True)
        return predict


def prepare_seg_pretrain_data(visual_outputs,
                              query_in_proj, query_out_proj,cross_beforellm=None,ffn_beforellm=None,internvl_vit_embeds=None):
    clip_feature, query_feat, attention_mask = visual_outputs#[2, 256, 6656], [2, 300, 512], [2, 300, 256]

    bs, q, _ = query_feat.shape
    pred_query_embed = []
    gt_query_embed = []
    for i in range(bs):
        valid = attention_mask[i].sum(-1) > 0
        valid_query_feat = query_feat[i][valid]  # (n, 2c)
        gt_query_embed.append(valid_query_feat)

        if isinstance(query_in_proj, list):#三层linear
            llm_query = valid_query_feat
            for proj in query_in_proj:
                llm_query = proj(llm_query)
            if cross_beforellm is not None:
                llm_query = cross_beforellm(llm_query.unsqueeze(1), internvl_vit_embeds[i].unsqueeze(1),)[:, 0]
                llm_query = ffn_beforellm(llm_query)
        else:
            llm_query = query_in_proj(valid_query_feat)
            if cross_beforellm is not None:
                llm_query = cross_beforellm(llm_query.unsqueeze(1), internvl_vit_embeds[i].unsqueeze(1),)[:, 0]
                llm_query = ffn_beforellm(llm_query)
        
        pred_query_embed.append(query_out_proj(llm_query))#两层linear，接llm的输出

    pred_query_embed = torch.cat(pred_query_embed, dim=0)#经过query_in_proj和query_out_proj
    gt_query_embed = torch.cat(gt_query_embed, dim=0) #只经过query_in_proj
    return pred_query_embed, gt_query_embed

