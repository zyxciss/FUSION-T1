# FUSION-T1
A multimodal reasoning , vison and image generation model powered by ALE (Adaptive Learning Engine) 
---
license: mit
library_name: transformers
language:
- en
- hi
- ur
- ar
- bn
- zh
- fr
- es
metrics:
- accuracy
parameters :
- 975B
base_model:
- deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
tags:
- thinking
- reasoning
pipeline_tag: any-to-any
---
#  Fusion-T1
<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="https://i.ibb.co/ncVZWRL/FUSION-zip-2-removebg-preview.png" width="30%" alt="Fusion-T1" />
</div>
<hr>
<div align="center" style="line-height: 1;">
  <a href="https://www.fusion.com/" target="_blank" style="margin: 2px;">
    <img alt="Homepage" src="https://github.com/ Fusion-ai/ Fusion-V2/blob/main/figures/badge.svg?raw=true" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://chat. Fusion.com/" target="_blank" style="margin: 2px;">
    <img alt="Chat" src="https://img.shields.io/badge/🤖%20Chat- Fusion%20T1-536af5?color=536af5&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/ Fusion-ai" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face- Fusion%20AI-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<div align="center" style="line-height: 1;">
  <a href="https://discord.gg/Tc7c45Zzu5" target="_blank" style="margin: 2px;">
    <img alt="Discord" src="https://img.shields.io/badge/Discord- Fusion%20AI-7289da?logo=discord&logoColor=white&color=7289da" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://github.com/ Fusion-ai/ Fusion-V2/blob/main/figures/qr.jpeg?raw=true" target="_blank" style="margin: 2px;">
    <img alt="Wechat" src="https://img.shields.io/badge/WeChat- Fusion%20AI-brightgreen?logo=wechat&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://twitter.com/ Fusion_ai" target="_blank" style="margin: 2px;">
    <img alt="Twitter Follow" src="https://img.shields.io/badge/Twitter- Fusion_ai-white?logo=x&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<div align="center" style="line-height: 1;">
  <a href="https://github.com/ Fusion-ai/ Fusion-T1/blob/main/LICENSE-CODE" style="margin: 2px;">
    <img alt="Code License" src="https://img.shields.io/badge/Code_License-MIT-f5de53?&color=f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://github.com/ Fusion-ai/ Fusion-T1/blob/main/LICENSE-MODEL" style="margin: 2px;">
    <img alt="Model License" src="https://img.shields.io/badge/Model_License-Model_Agreement-f5de53?&color=f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>


<p align="center">
  <a href="https://github.com/ Fusion-ai/ Fusion-T1/blob/main/ Fusion_T1.pdf"><b>Paper Link</b>👁️</a>
</p>


## 1. Introduction

We introduce our first-generation reasoning models,  Fusion-T1-Zero and  Fusion-T1. 
 Fusion-T1-Zero, a model trained via large-scale reinforcement learning (RL) without supervised fine-tuning (SFT) as a preliminary step, demonstrated remarkable performance on reasoning.
With RL,  Fusion-T1-Zero naturally emerged with numerous powerful and interesting reasoning behaviors.
However,  Fusion-T1-Zero encounters challenges such as endless repetition, poor readability, and language mixing. To address these issues and further enhance reasoning performance,
we introduce  Fusion-T1, which incorporates cold-start data before RL.
 Fusion-T1 achieves performance comparable to OpenAI-o1 across math, code, and reasoning tasks. 
To support the research community, we have open-sourced  Fusion-T1-Zero,  Fusion-T1, and six dense models distilled from  Fusion-T1 based on Llama and Qwen.  Fusion-T1-Distill-Qwen-32B outperforms OpenAI-o1-mini across various benchmarks, achieving new state-of-the-art results for dense models.

<p align="center">
  <img width="80%" src="https://i.ibb.co/Bw8N1VY/image.png">
</p>

## 2. Model Summary

---

**Post-Training: Large-Scale Reinforcement Learning on the Base Model**

-  We directly apply reinforcement learning (RL) to the base model without relying on supervised fine-tuning (SFT) as a preliminary step. This approach allows the model to explore chain-of-thought (CoT) for solving complex problems, resulting in the development of  Fusion-T1-Zero.  Fusion-T1-Zero demonstrates capabilities such as self-verification, reflection, and generating long CoTs, marking a significant milestone for the research community. Notably, it is the first open research to validate that reasoning capabilities of LLMs can be incentivized purely through RL, without the need for SFT. This breakthrough paves the way for future advancements in this area.

-   We introduce our pipeline to develop  Fusion-T1. The pipeline incorporates two RL stages aimed at discovering improved reasoning patterns and aligning with human preferences, as well as two SFT stages that serve as the seed for the model's reasoning and non-reasoning capabilities.
    We believe the pipeline will benefit the industry by creating better models. 

---

**Distillation: Smaller Models Can Be Powerful Too**

-  We demonstrate that the reasoning patterns of larger models can be distilled into smaller models, resulting in better performance compared to the reasoning patterns discovered through RL on small models. The open source  Fusion-T1, as well as its API, will benefit the research community to distill better smaller models in the future. 
- Using the reasoning data generated by  Fusion-T1, we fine-tuned several dense models that are widely used in the research community. The evaluation results demonstrate that the distilled smaller dense models perform exceptionally well on benchmarks. We open-source distilled 1.5B, 7B, 8B, 14B, 32B, and 70B checkpoints based on Qwen2.5 and Llama3 series to the community.

## 3. Model Downloads

###  Fusion-T1 Models

<div align="center">

| **Model** | **#Total Params** | **#Activated Params** | **Context Length** | **Download** |
| :------------: | :------------: | :------------: | :------------: | :------------: |
|  Fusion-T1-Zero | 671B | 37B | 128K   | [🤗 HuggingFace](https://huggingface.co/ Fusion-ai/ Fusion-T1-Zero)   |
|  Fusion-T1   | 671B | 37B |  128K   | [🤗 HuggingFace](https://huggingface.co/ Fusion-ai/ Fusion-T1)   |

</div>

 Fusion-T1-Zero &  Fusion-T1 are trained based on  Fusion-V3-Base. 
For more details regrading the model architecture, please refer to [ Fusion-V3](https://github.com/ Fusion-ai/ Fusion-V3) repository.

###  Fusion-T1-Distill Models

<div align="center">

| **Model** | **Base Model** | **Download** |
| :------------: | :------------: | :------------: |
|  Fusion-T1-Distill-Qwen-1.5B  | [Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B) | [🤗 HuggingFace](https://huggingface.co/ Fusion-ai/ Fusion-T1-Distill-Qwen-1.5B)   |
|  Fusion-T1-Distill-Qwen-7B  | [Qwen2.5-Math-7B](https://huggingface.co/Qwen/Qwen2.5-Math-7B) | [🤗 HuggingFace](https://huggingface.co/ Fusion-ai/ Fusion-T1-Distill-Qwen-7B)   |
|  Fusion-T1-Distill-Llama-8B  | [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) | [🤗 HuggingFace](https://huggingface.co/ Fusion-ai/ Fusion-T1-Distill-Llama-8B)   |
|  Fusion-T1-Distill-Qwen-14B   | [Qwen2.5-14B](https://huggingface.co/Qwen/Qwen2.5-14B) | [🤗 HuggingFace](https://huggingface.co/ Fusion-ai/ Fusion-T1-Distill-Qwen-14B)   |
| Fusion-T1-Distill-Qwen-32B  | [Qwen2.5-32B](https://huggingface.co/Qwen/Qwen2.5-32B) | [🤗 HuggingFace](https://huggingface.co/ Fusion-ai/ Fusion-T1-Distill-Qwen-32B)   |
|  Fusion-T1-Distill-Llama-70B  | [Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) | [🤗 HuggingFace](https://huggingface.co/ Fusion-ai/ Fusion-T1-Distill-Llama-70B)   |

</div>

 Fusion-T1-Distill models are fine-tuned based on open-source models, using samples generated by  Fusion-T1.
We slightly change their configs and tokenizers. Please use our setting to run these models.

## 4. Evaluation Results

###  Fusion-T1-Evaluation
 For all our models, the maximum generation length is set to 32,768 tokens. For benchmarks requiring sampling, we use a temperature of $0.6$, a top-p value of $0.95$, and generate 64 responses per query to estimate pass@1.
<div align="center">


| Category | Benchmark (Metric) | Claude-3.5-Sonnet-1022 | GPT-4o 0513 |  Fusion V3 | OpenAI o1-mini | OpenAI o1-1217 |  Fusion T1 |
|----------|-------------------|----------------------|------------|--------------|----------------|------------|--------------|
| | Architecture | - | - | MoE | - | - | MoE |
| | # Activated Params | - | - | 37B | - | - | 37B |
| | # Total Params | - | - | 671B | - | - | 671B |
| English | MMLU (Pass@1) | 88.3 | 87.2 | 88.5 | 85.2 | **91.8** | 90.8 |
| | MMLU-Redux (EM) | 88.9 | 88.0 | 89.1 | 86.7 | - | **92.9** |
| | MMLU-Pro (EM) | 78.0 | 72.6 | 75.9 | 80.3 | - | **84.0** |
| | DROP (3-shot F1) | 88.3 | 83.7 | 91.6 | 83.9 | 90.2 | **92.2** |
| | IF-Eval (Prompt Strict) | **86.5** | 84.3 | 86.1 | 84.8 | - | 83.3 |
| | GPQA-Diamond (Pass@1) | 65.0 | 49.9 | 59.1 | 60.0 | **75.7** | 71.5 |
| | SimpleQA (Correct) | 28.4 | 38.2 | 24.9 | 7.0 | **47.0** | 30.1 |
| | FRAMES (Acc.) | 72.5 | 80.5 | 73.3 | 76.9 | - | **82.5** |
| | AlpacaEval2.0 (LC-winrate) | 52.0 | 51.1 | 70.0 | 57.8 | - | **87.6** |
| | ArenaHard (GPT-4-1106) | 85.2 | 80.4 | 85.5 | 92.0 | - | **92.3** |
| Code | LiveCodeBench (Pass@1-COT) | 33.8 | 34.2 | - | 53.8 | 63.4 | **65.9** |
| | Codeforces (Percentile) | 20.3 | 23.6 | 58.7 | 93.4 | **96.6** | 96.3 |
| | Codeforces (Rating) | 717 | 759 | 1134 | 1820 | **2061** | 2029 |
| | SWE Verified (Resolved) | **50.8** | 38.8 | 42.0 | 41.6 | 48.9 | 49.2 |
| | Aider-Polyglot (Acc.) | 45.3 | 16.0 | 49.6 | 32.9 | **61.7** | 53.3 |
| Math | AIME 2024 (Pass@1) | 16.0 | 9.3 | 39.2 | 63.6 | 79.2 | **79.8** |
| | MATH-500 (Pass@1) | 78.3 | 74.6 | 90.2 | 90.0 | 96.4 | **97.3** |
| | CNMO 2024 (Pass@1) | 13.1 | 10.8 | 43.2 | 67.6 | - | **78.8** |
| Chinese | CLUEWSC (EM) | 85.4 | 87.9 | 90.9 | 89.9 | - | **92.8** |
| | C-Eval (EM) | 76.7 | 76.0 | 86.5 | 68.9 | - | **91.8** |
| | C-SimpleQA (Correct) | 55.4 | 58.7 | **68.0** | 40.3 | - | 63.7 |

</div>


### Distilled Model Evaluation


<div align="center">

| Model                                    | AIME 2024 pass@1 | AIME 2024 cons@64 | MATH-500 pass@1 | GPQA Diamond pass@1 | LiveCodeBench pass@1 | CodeForces rating |
|------------------------------------------|------------------|-------------------|-----------------|----------------------|----------------------|-------------------|
| GPT-4o-0513                          | 9.3              | 13.4              | 74.6            | 49.9                 | 32.9                 | 759               |
| Claude-3.5-Sonnet-1022             | 16.0             | 26.7                 | 78.3            | 65.0                 | 38.9                 | 717               |
| o1-mini                              | 63.6             | 80.0              | 90.0            | 60.0                 | 53.8                 | **1820**          |
| QwQ-32B-Preview                              | 44.0             | 60.0                 | 90.6            | 54.5               | 41.9                 | 1316              |
|  Fusion-T1-Distill-Qwen-1.5B       | 28.9             | 52.7              | 83.9            | 33.8                 | 16.9                 | 954               |
|  Fusion-T1-Distill-Qwen-7B          | 55.5             | 83.3              | 92.8            | 49.1                 | 37.6                 | 1189              |
|  Fusion-T1-Distill-Qwen-14B         | 69.7             | 80.0              | 93.9            | 59.1                 | 53.1                 | 1481              |
|  Fusion-T1-Distill-Qwen-32B        | **72.6**         | 83.3              | 94.3            | 62.1                 | 57.2                 | 1691              |
|  Fusion-T1-Distill-Llama-8B         | 50.4             | 80.0              | 89.1            | 49.0                 | 39.6                 | 1205              |
|  Fusion-T1-Distill-Llama-70B        | 70.0             | **86.7**          | **94.5**        | **65.2**             | **57.5**             | 1633              |

</div>


## 5. Chat Website & API Platform
You can chat with  Fusion-T1 on  Fusion's official website: [chat. Fusion.com](https://chat. Fusion.com), and switch on the button "DeepThink"

We also provide OpenAI-Compatible API at  Fusion Platform: [platform. Fusion.com](https://platform. Fusion.com/)

## 6. How to Run Locally

###  Fusion-T1 Models

Please visit [ Fusion-V3](https://github.com/ Fusion-ai/ Fusion-V3) repo for more information about running  Fusion-T1 locally.

###  Fusion-T1-Distill Models

 Fusion-T1-Distill models can be utilized in the same manner as Qwen or Llama models.

For instance, you can easily start a service using [vLLM](https://github.com/vllm-project/vllm):

```shell
vllm serve  Fusion-ai/ Fusion-T1-Distill-Qwen-32B --tensor-parallel-size 2 --max-model-len 32768 --enforce-eager
```

**NOTE: We recommend setting an appropriate temperature (between 0.5 and 0.7) when running these models, otherwise you may encounter issues with endless repetition or incoherent output.**

## 7. License
This code repository and the model weights are licensed under the [MIT License](https://github.com/ Fusion-ai/ Fusion-T1/blob/main/LICENSE).
 Fusion-T1 series support commercial use, allow for any modifications and derivative works, including, but not limited to, distillation for training other LLMs. Please note that:
-  Fusion-T1-Distill-Qwen-1.5B,  Fusion-T1-Distill-Qwen-7B,  Fusion-T1-Distill-Qwen-14B and  Fusion-T1-Distill-Qwen-32B are derived from [Qwen-2.5 series](https://github.com/QwenLM/Qwen2.5), which are originally licensed under [Apache 2.0 License](https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/main/LICENSE), and now finetuned with 800k samples curated with  Fusion-T1.
-  Fusion-T1-Distill-Llama-8B is derived from Llama3.1-8B-Base and is originally licensed under [llama3.1 license](https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/LICENSE).
-  Fusion-T1-Distill-Llama-70B is derived from Llama3.3-70B-Instruct and is originally licensed under [llama3.3 license](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/blob/main/LICENSE).

## 8. Citation
```

```

## 9. Contact
If you have any questions, please raise an issue or contact us at [fusion@gmail.com](fusion@gmail.com).
