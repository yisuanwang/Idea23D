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
We are in the process of constructing a colab and hugging face demo that can be demonstrated directly, which may take some time~.


## Acknowledgement
We have intensively borrow codes from the following repositories. Many thanks to the authors for sharing their codes.

[Qwen-VL](https://modelscope.cn/studios/qwen/Qwen-VL-Max/summary),
[LLaVA](https://github.com/haotian-liu/LLaVA),
[Idea2img](https://github.com/zyang-ur/idea2img?tab=readme-ov-file)
[TripoSR](https://github.com/VAST-AI-Research/TripoSR)
[Zero123](https://github.com/cvlab-columbia/zero123),
[Stable Zero123](https://github.com/threestudio-project/threestudio?tab=readme-ov-file#stable-zero123),
[Wonder3D](https://github.com/xxlong0/Wonder3D),
[SD-XL](https://huggingface.co/docs/diffusers/en/using-diffusers/sdxl),
[Deepfloyd IF](https://huggingface.co/docs/diffusers/en/api/pipelines/deepfloyd_if)