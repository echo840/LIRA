<div align="center" xmlns="http://www.w3.org/1999/html">
<h1 align="center">
LIRA: Inferring Segmentation in Large Multi-modal Models with Local Interleaved Region Assistance
</h1>

[![arXiv](https://img.shields.io/badge/Arxiv-LIRA-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2507.06272)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/LIRA)
[![GitHub issues](https://img.shields.io/github/issues/echo840/LIRA?color=critical&label=Issues)](https://github.com/echo840/LIRA/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/echo840/LIRA?color=success&label=Issues)](https://github.com/echo840/LIRA/issues?q=is%3Aissue+is%3Aclosed)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=LIRA&color=brightgreen&label=Views)](https://github.com/echo840/LIRA)
</div>


> **LIRA: Inferring Segmentation in Large Multi-modal Models with Local Interleaved Region Assistance**<br>
> Zhang Li, Biao Yang, Qiang Liu, Shuo Zhang, Zhiyin Ma, Liang Yin, Linger Deng, Yabo Sun, Yuliang Liu, Xiang Bai <br>
[![arXiv](https://img.shields.io/badge/Arxiv-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2507.06272) 
[![Source_code](https://img.shields.io/badge/Code-Available-white)](README.md)
[![Model Weight](https://img.shields.io/badge/HuggingFace-gray)](https://huggingface.co/echo840/)


## Abstract
While large multi-modal models (LMMs) demonstrate promising capabilities in segmentation and comprehension, they still struggle with two limitations: inaccurate segmentation and hallucinated comprehension. These challenges stem primarily from constraints in weak visual comprehension and a lack of fine-grained perception. To alleviate these limitations, we propose LIRA, a framework that capitalizes on the complementary relationship between visual comprehension and segmentation via two key components: (1) Semantic-Enhanced Feature Extractor (SEFE) improves object attribute inference by fusing semantic and pixel-level features, leading to more accurate segmentation; (2) Interleaved Local Visual Coupling (ILVC) autoregressively generates local descriptions after extracting local features based on segmentation masks, offering fine-grained supervision to mitigate hallucinations. Furthermore, we find that the precision of object segmentation is positively correlated with the latent related semantics of the <seg> token. To quantify this relationship and the model's potential semantic inferring ability, we introduce the Attributes Evaluation (AttrEval) dataset. Our experiments show that LIRA achieves state-of-the-art performance in both segmentation and comprehension tasks.


## Overview



## Results


## Install



## Weights


## Demo


## Train


## Evaluation


## Acknowledgments

## Citation


