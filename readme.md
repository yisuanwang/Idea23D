
<h1 style="font-weight: bold">
  <!-- <a href="https://idea23d.github.io/" target="_blank"> -->
     <span style="background: linear-gradient(90deg, rgba(131,58,180,1) 0%, rgba(253,29,29,1) 50%, rgba(252,176,69,1) 100%); -webkit-background-clip: text; color: transparent; background-clip: text;">Idea23D</span>:
    Collaborative LMM Agents Enable 3D Model Generation from Interleaved Multimodal Inputs
  <!-- </a> -->
</h1>

2024.11: 🎉 Idea-2-3D has been accepted by COLING 2025! 🎉 See you in Abu Dhabi, UAE, from January 19 to 24, 2025!

<div align="left">
  <!-- <a href='https://idea23d.github.io/'>
    <img src='https://img.shields.io/badge/Project-Page-green' alt="Project Page">
  </a>&ensp; -->
  <a href="https://idea23d.github.io/"><img src="https://img.shields.io/static/v1?label=Homepage&message=Idea23D&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://github.com/yisuanwang/Idea23D"><img src="https://img.shields.io/github/stars/yisuanwang/Idea23D?label=stars&logo=github&color=brightgreen" alt="GitHub Repo Stars"></a> &ensp;
  <a href="https://colab.research.google.com/drive/1u_lJRvxIlBUPjC_Lou57SWLEnc5vLgQ6?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> &ensp;
  <a href="https://arxiv.org/abs/2404.04363"><img src="https://img.shields.io/badge/arXiv-2404.04363-b31b1b.svg?style=flat-square" alt="arXiv"></a> &ensp;
</div>


<a href="https://scholar.google.com/citations?hl=en&user=uVMnzPMAAAAJ" target="_blank">Junhao Chen *</a>,
<a href="https://scholar.google.com/citations?hl=en&user=_wyYvQsAAAAJ" target="_blank">Xiang Li *</a>,
<a href="https://scholar.google.com/citations?user=BKMYsm4AAAAJ&hl=en" target="_blank">Xiaojun Ye</a>,
<a href="" target="_blank">Chao Li</a>,
<a href="https://scholar.google.com/citations?user=JHvyYDQAAAAJ" target="_blank">Zhaoxin Fan †</a>,
<a href="https://scholar.google.com/citations?hl=en&user=ygQznUQAAAAJ" target="_blank">Hao Zhao †</a>

----

## ✨Introduction
<!-- ![idea23d](./page/idea23d.gif) -->
![idea23d](./page/overview11.jpg)
Based on the LMM we developed Idea23D, a multimodal iterative self-refinement system that enhances any T2I model for automatic 3D model design and generation, enabling various new image creation functionalities togther with better visual qualities while understanding high level multimodal inputs.


## 📔Compatibility:
- LMM Agent:
[OpenAI GPT-4V](https://community.openai.com/t/how-can-i-get-a-gpt4-api-key/379141), 
[OpenAI GPT-4o](https://community.openai.com/t/how-can-i-get-a-gpt4-api-key/379141), 
[llava-v1.6-34b](https://github.com/haotian-liu/LLaVA),
[llava-v1.6-mistral-7b](https://huggingface.co/liuhaotian/llava-v1.6-mistral-7b), 
[llava-CoT-11B](https://github.com/PKU-YuanGroup/LLaVA-CoT),
[InternVL2.5-78B](https://huggingface.co/OpenGVLab/InternVL2_5-78B),
[Qwen-VL2-8B](https://github.com/QwenLM/Qwen2-VL), 
[llava-CoT-11B](https://huggingface.co/Xkev/Llama-3.2V-11B-cot),
[llama-3.2V-11B](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision), 
[intern-VL2-8B](https://huggingface.co/OpenGVLab/InternVL2-8B),

- Text-2-Image Agent: 
[SD-XL 1.0 base+refiner](https://huggingface.co/docs/diffusers/en/using-diffusers/sdxl), 
[DALL·E](https://platform.openai.com/docs/guides/images?context=node), 
[Deepfloyd IF](https://huggingface.co/docs/diffusers/en/api/pipelines/deepfloyd_if),
[FLUX.1.dev](https://huggingface.co/black-forest-labs/FLUX.1-dev),

- Image-2-3D Agent: 
[TripoSR](https://github.com/VAST-AI-Research/TripoSR), 
[Zero123](https://github.com/cvlab-columbia/zero123), 
[Wonder3D](https://github.com/xxlong0/Wonder3D),
[InstantMesh](https://github.com/TencentARC/InstantMesh), 
[LGM](https://github.com/3DTopia/LGM), 
[Hunyuan3D](https://github.com/Tencent/Hunyuan3D-1), 
[stable-fast-3d](https://huggingface.co/stabilityai/stable-fast-3d), 3DTopia, Hunyuan3D

## 🛠Run
<!-- ❗If different modules are used, install the corresponding dependency packages.

The code we have given to run locally uses llava-1.6, SD-XL and TripoSR. so [requirements-local.txt](./requirements-local.txt) is following that.

It's driven by GPT4V, [SD-XL(replicate)](https://replicate.com/stability-ai/sdxl/api), and TripoSR if you're using colab for testing, it uses this [requirements-colab.txt](./requirements-colab.txt). -->

### Colab
<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1u_lJRvxIlBUPjC_Lou57SWLEnc5vLgQ6?usp=sharing) -->

### Offline
<!-- ```
pip install -r requirements-local.txt
``` -->

<!-- Then change the path to your path in the "Initialize LMM, T2I, I23D" section of ipynb.
```
https://huggingface.co/llava-hf/llava-v1.6-34b-hf
https://huggingface.co/stabilityai/TripoSR
https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0
```

This section in [ipynb](./idea23d_pipeline.ipynb):
```
# init LMM,T2I,I23D
log('loading lmm...')

# lmm = lmm_gpt4v('sk-your open ai key')
lmm = lmm_llava_34b(model_path = "path_to_your/llava-v1.6-34b-hf", gpuid = 5)
# lmm = lmm_llava_7b(model_path = "path_to_your/llava-v1.6-mistral-7b-hf", gpuid = 2)

log('loading t2i...')
# t2i = text2img_sdxl_replicate(replicate_key='r8_ZCtxxxxxxxxxxxxxx')
t2i = text2img_sdxl(sdxl_base_path='path_to_your/stable-diffusion-xl-base-1.0', 
                    sdxl_refiner_path='path_to_your/stable-diffusion-xl-refiner-1.0', 
                    gpuid=2)

log('loading i23d...')
i23d = img23d_TripoSR(model_path = 'path_to_your/TripoSR' ,gpuid=2)
log('loading finish.')
```
open [Idea23D/idea23d_pipeline.ipynb](./idea23d_pipeline.ipynb), Explore freely in the notebook ~  -->

## 🧐Tips
<!-- Using [GPT4V](https://community.openai.com/t/how-can-i-get-a-gpt4-api-key/379141), [SD-XL](https://replicate.com/stability-ai/sdxl/api) or [DALL·E](https://platform.openai.com/docs/guides/images?context=node), [TripoSR](https://github.com/VAST-AI-Research/TripoSR) as LMM was able to get the best results so far.
The effects in the paper were obtained using [Zero123](https://github.com/cvlab-columbia/zero123), so they are inferior compared to [TripoSR](https://github.com/VAST-AI-Research/TripoSR).

If you don't have access to [GPT4V](https://community.openai.com/t/how-can-i-get-a-gpt4-api-key/379141) you can use [Qwen-VL](https://modelscope.cn/studios/qwen/Qwen-VL-Max/summary) or [LLaVA](https://github.com/haotian-liu/LLaVA), if you use LLaVA it is recommended to use the [llava-v1.6-34b](https://huggingface.co/llava-hf/llava-v1.6-34b-hf) model. Although we gave a pipeline built with [llava-v1.6-mistral-7b](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf), it works poorly, while [llava-v1.6-34b](https://huggingface.co/llava-hf/llava-v1.6-34b-hf) can correctly fulfill user commands. -->

## 🗓ToDO List
<!-- ✅1. Release offline version of Idea23D implementation (llava-1.6-34b, SD-XL, TripoSR)

✅2. Release online running version of Idea23D implementation (GPT4-V, SD-XL, TripoSR)

✅3. Release complete rendering script with 3d model input support.

✅4. Components supported by release: Qwen-VL, Zero123, DALL-E, Wonder3D, Stable Zero123, Deepfloyd IF. The release date for the complete set of all components will be delayed due to ongoing follow-up work. -->

## 📜Citations
```
@article{chen2024idea23d,
  title={Idea-2-3D: Collaborative LMM Agents Enable 3D Model Generation from Interleaved Multimodal Inputs}, 
  author={Junhao Chen and Xiang Li and Xiaojun Ye and Chao Li and Zhaoxin Fan and Hao Zhao},
  year={2024},
  eprint={2404.04363},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```


## 🧰Acknowledgement
We have intensively borrow codes from the following repositories. Many thanks to the authors for sharing their codes.

[Qwen-VL](https://modelscope.cn/studios/qwen/Qwen-VL-Max/summary),
[LLaVA](https://github.com/haotian-liu/LLaVA),
[TripoSR](https://github.com/VAST-AI-Research/TripoSR),
[Zero123](https://github.com/cvlab-columbia/zero123),
[Wonder3D](https://github.com/xxlong0/Wonder3D),
[SD-XL](https://huggingface.co/docs/diffusers/en/using-diffusers/sdxl),
[Deepfloyd IF](https://huggingface.co/docs/diffusers/en/api/pipelines/deepfloyd_if)
...

## ⭐️ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yisuanwang/Idea23D&type=Date)](https://star-history.com/#yisuanwang/Idea23D&Date)
