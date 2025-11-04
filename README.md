<div align="center" xmlns="http://www.w3.org/1999/html">
<h1 align="center">
LIRA: Inferring Segmentation in Large Multi-modal Models with Local Interleaved Region Assistance
</h1>

[![arXiv](https://img.shields.io/badge/Arxiv-LIRA-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2507.06272)
[![Model Weight](https://img.shields.io/badge/HuggingFace-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/LIRA)
[![Data](https://img.shields.io/badge/Data-yellow)](https://huggingface.co/datasets/echo840/LIRA_Data)
[![GitHub issues](https://img.shields.io/github/issues/echo840/LIRA?color=critical&label=Issues)](https://github.com/echo840/LIRA/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/echo840/LIRA?color=success&label=Issues)](https://github.com/echo840/LIRA/issues?q=is%3Aissue+is%3Aclosed)
[![GitHub views](https://komarev.com/ghpvc/?username=echo840&repo=LIRA&color=brightgreen&label=Views)](https://github.com/echo840/LIRA)
</div>


> **LIRA: Inferring Segmentation in Large Multi-modal Models with Local Interleaved Region Assistance**<br>
> Zhang Li, Biao Yang, Qiang Liu, Shuo Zhang, Zhiyin Ma, Liang Yin, Linger Deng, Yabo Sun, Yuliang Liu, Xiang Bai <br>
[![arXiv](https://img.shields.io/badge/Arxiv-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2507.06272) 
[![Source_code](https://img.shields.io/badge/Code-Available-white)](README.md)
[![Model Weight](https://img.shields.io/badge/HuggingFace-gray)](https://huggingface.co/echo840/LIRA)


## Abstract
While large multi-modal models (LMMs) demonstrate promising capabilities in segmentation and comprehension, they still struggle with two limitations: inaccurate segmentation and hallucinated comprehension. These challenges stem primarily from constraints in weak visual comprehension and a lack of fine-grained perception. To alleviate these limitations, we propose LIRA, a framework that capitalizes on the complementary relationship between visual comprehension and segmentation via two key components: (1) Semantic-Enhanced Feature Extractor (SEFE) improves object attribute inference by fusing semantic and pixel-level features, leading to more accurate segmentation; (2) Interleaved Local Visual Coupling (ILVC) autoregressively generates local descriptions after extracting local features based on segmentation masks, offering fine-grained supervision to mitigate hallucinations. Furthermore, we find that the precision of object segmentation is positively correlated with the latent related semantics of the <seg> token. To quantify this relationship and the model's potential semantic inferring ability, we introduce the Attributes Evaluation (AttrEval) dataset. Our experiments show that LIRA achieves state-of-the-art performance in both segmentation and comprehension tasks.


## Overview
<a href="https://zimgs.com/i/EjHWis"><img src="https://v1.ax1x.com/2025/09/26/EjHWis.png" alt="EjHWis.png" border="0" /></a>


## Results
<a href="https://zimgs.com/i/EjHv7a"><img src="https://v1.ax1x.com/2025/09/26/EjHv7a.jpg" alt="EjHv7a.jpg" border="0" /></a>



## Install

Please follow the instructions in [omg_llava](https://github.com/lxtGH/OMG-Seg/tree/main/omg_llava) and ensure that transformers ≥ 4.37.2 is installed. We will update the environment configuration within November.

## Dataset
Download data from [data](https://huggingface.co/datasets/echo840/LIRA_Data).
```python 
cat data.zip.part* > data.zip
unzip data.zip
```


## Weights
1. Download model
```python 
python download_model.py -n echo840/LIRA
```

2. Download InternVL
```python 
python download_model.py -n OpenGVLab/InternVL2-2B # OpenGVLab/InternVL2-8B
```


## Demo
```python 
python ./omg_llava/tools/app_lira.py ./omg_llava/configs/finetune/LIRA-2B.py ./model_weight/LIRA-2B.pth
```

## Train

1. Pretrain
```python 
bash ./scripts/pretrain.sh 
```

2. After train, please use the tools to convert deepspeed chekpoint to pth format
```python 
python omg_llava/tools/convert_deepspeed2pth.py
    ${PATH_TO_CONFIG} \
    ${PATH_TO_DeepSpeed_PTH} \
    --save-path ./pretrained/${PTH_NAME.pth}
```

3. Finetune
```python 
bash ./scripts/finetune.sh
```


## Evaluation
```python 
bash ./scripts/eval_gcg.sh #  Evaluation on Grounded Conversation Generation Tasks.

bash ./scripts/eval_refseg.sh # Evaluation on Referring Segmentation Tasks.

bash ./scripts/eval_vqa.sh # Evaluation on Comprehension Tasks.
```


## Acknowledgments
Our code is built upon [OMGLLaVA](https://github.com/lxtGH/OMG-Seg) and [InternVL2](https://github.com/OpenGVLab/InternVL), and we sincerely thank them for providing the code and base models. We also thank [OPERA](https://github.com/shikiw/OPERA) for providing the evaluation code for chair.


## Copyright
If you have any questions, please feel free to contact us at zhangli123@hust.edu.cn.

## Citation
If you wish to refer to the baseline results published here, please use the following BibTeX entries:
```BibTeX
@inproceedings{li2025lirainferringsegmentationlarge,
  title={LIRA: Inferring Segmentation in Large Multi-modal Models with Local Interleaved Region Assistance},
  author={Zhang Li and Biao Yang and Qiang Liu and Shuo Zhang and Zhiyin Ma and Liang Yin and Linger Deng and Yabo Sun and Yuliang Liu and Xiang Bai},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2025}
}
```

