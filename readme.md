# Idea23D
The code is being organized and will be released in the next few weeks~

Showcase case available at: https://air-discover.github.io/Idea-2-3D/

----
## Introduction
Based on the Multimodal Big Model we developed Idea23D, a multimodal iterative self-refinement system that enhances any T2I model for automatic 3D model design and generation, enabling various new image creation functionalities togther with better visual qualities while understanding high level multimodal inputs.

## Prerequisites:
- LMM: prepare the [OpenAI GPT-4V API key](https://community.openai.com/t/how-can-i-get-a-gpt4-api-key/379141), or use another open source LMM (e.g., [LLaVA](https://github.com/haotian-liu/LLaVA), [Qwen-VL](https://modelscope.cn/studios/qwen/Qwen-VL-Max/summary)).

- Text-2-Image model: use [SD-XL](https://huggingface.co/docs/diffusers/en/using-diffusers/sdxl), [DALL·E](https://platform.openai.com/docs/guides/images?context=node), or [Deepfloyd IF](https://huggingface.co/docs/diffusers/en/api/pipelines/deepfloyd_if).

- Image-2-3D model: use [TripoSR](https://github.com/VAST-AI-Research/TripoSR), [Zero123](https://github.com/cvlab-columbia/zero123), [Stable Zero123](https://github.com/threestudio-project/threestudio?tab=readme-ov-file#stable-zero123), or [Wonder3D](https://github.com/xxlong0/Wonder3D)

## Run
❗If different modules are used, install the corresponding dependency packages.

The code we have given to run locally uses llava-1.6, SD-XL and TripoSR. so [requirements-local.txt](./requirements-local.txt) is following that.

It's driven by GPT4V, [SD-XL(replicate)](https://replicate.com/stability-ai/sdxl/api), and TripoSR if you're using colab for testing, it uses this [requirements-colab.txt](./requirements-colab.txt).

```
pip install -r requirements-local.txt
```

Then change the path to your path in the "Initialize LMM, T2I, I23D" section of ipynb.
```
https://huggingface.co/llava-hf/llava-v1.6-34b-hf
https://huggingface.co/stabilityai/TripoSR
https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0
```

This section in [ipynb](./idea23d_pipeline.ipynb):
```
# 初始化LMM,T2I,I23D
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

Explore freely in the notebook ~ 

Using [GPT4V](https://community.openai.com/t/how-can-i-get-a-gpt4-api-key/379141), [SD-XL](https://replicate.com/stability-ai/sdxl/api) or [DALL·E](https://platform.openai.com/docs/guides/images?context=node), [TripoSR](https://github.com/VAST-AI-Research/TripoSR) as LMM was able to get the best results so far.
The effects in the paper were obtained using [Zero123](https://github.com/cvlab-columbia/zero123), so they are inferior compared to [TripoSR](https://github.com/VAST-AI-Research/TripoSR).

If you don't have access to [GPT4V](https://community.openai.com/t/how-can-i-get-a-gpt4-api-key/379141) you can use [Qwen-VL](https://modelscope.cn/studios/qwen/Qwen-VL-Max/summary) or [LLaVA](https://github.com/haotian-liu/LLaVA), if you use LLaVA it is recommended to use the [llava-v1.6-34b](https://huggingface.co/llava-hf/llava-v1.6-34b-hf) model. Although we gave a pipeline built with [llava-v1.6-mistral-7b](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf), it works poorly, while [llava-v1.6-34b](https://huggingface.co/llava-hf/llava-v1.6-34b-hf) can correctly fulfill user commands.


## Acknowledgement
We have intensively borrow codes from the following repositories. Many thanks to the authors for sharing their codes.

[Qwen-VL](https://modelscope.cn/studios/qwen/Qwen-VL-Max/summary),
[LLaVA](https://github.com/haotian-liu/LLaVA),
[TripoSR](https://github.com/VAST-AI-Research/TripoSR),
[Zero123](https://github.com/cvlab-columbia/zero123),
[Stable Zero123](https://github.com/threestudio-project/threestudio?tab=readme-ov-file#stable-zero123),
[Wonder3D](https://github.com/xxlong0/Wonder3D),
[SD-XL](https://huggingface.co/docs/diffusers/en/using-diffusers/sdxl),
[Deepfloyd IF](https://huggingface.co/docs/diffusers/en/api/pipelines/deepfloyd_if)