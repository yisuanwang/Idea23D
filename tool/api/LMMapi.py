import argparse
import json
import os
# import pygame
# from pygame.locals import *
# from OpenGL.GL import *
# from OpenGL.GLU import *
# import pywavefront
import re
import time
from http import HTTPStatus

import cv2
import numpy as np
import replicate
import requests
import torch
from PIL import Image
from transformers import (AutoProcessor, LlavaNextForConditionalGeneration,
                          LlavaNextProcessor, SegformerFeatureExtractor,
                          SegformerForSemanticSegmentation)


def log(text):
    print(f'\n[IDEA-2-3D]: {text}')

import base64
import json
from io import BytesIO

import openai
import requests
from PIL import Image

class lmm_gpt4v:
    def __init__(self, api_key=''):
        self.api_key = api_key

    def encode_image(self, image):
        """Encode PIL image to base64, converting RGBA images to RGB."""
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def inference(self, question: str, image):
        """Make an inference request to the GPT-4 Vision API with an image and a question."""
        base64_image = self.encode_image(image)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }
        
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        return response.json()['choices'][0]['message']['content']

# gpt4v
class lmm_gpt4v():
    def __str__(self):
        return 'gpt-4-vision-preview'

    def __init__(self, api_key=''):
        self.api_key = api_key

    def encode_image_to_base64(self, image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")


    def inference(self, prompt, image_paths, model="gpt-4-vision-preview", max_tokens=300):
        if not isinstance(image_paths, list):
            image_paths = [image_paths]
        openai.api_key = self.api_key

        base64_images = [self.encode_image_to_base64(image_path) for image_path in image_paths]
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ] + [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    } for base64_image in base64_images
                ]
            }
        ]

        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error during gpt-4-vision-preview request: {e}")
            return None
            

# gpt4o
class lmm_gpt4o_local():
    def __str__(self):
        return 'gpt4o'

    def __init__(self, api_key=''):
        self.api_key = api_key

    def encode_image_to_base64(self, image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")


    def inference(self, prompt, image_paths, model="gpt-4o", max_tokens=300, max_retries=6, retry_delay=3):
        if not isinstance(image_paths, list):
            image_paths = [image_paths]
        openai.api_key = self.api_key

        base64_images = [self.encode_image_to_base64(image_path) for image_path in image_paths]
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ] + [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    } for base64_image in base64_images
                ]
            }
        ]

        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.7,
                )
                return response['choices'][0]['message']['content']
            except Exception as e:
                print(f"Error during GPT-4o request (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:  
                    time.sleep(retry_delay)
                else:
                    print("Max retries reached. Returning None.")
                    return None
                


# need Idea23D/tool/api/gptweb.py
# gpt4o
class lmm_gpt4o():
    def __str__(self):
        return 'gpt4o'

    def __init__(self, api_key='', base_url="http://127.0.0.1:5000/test_gpt4o"):
        self.api_key = api_key
        self.base_url = base_url

    def encode_image_to_base64(self, image_path):
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode("utf-8")
        except Exception as e:
            raise ValueError(f"Failed to encode image {image_path} to Base64: {str(e)}")

    def inference(self, prompt, image_paths, model="gpt-4o", max_tokens=300):
        try:
            images_base64 = [self.encode_image_to_base64(image_path) for image_path in image_paths]

            data = {
                "prompt": prompt,
                "images": images_base64,
                "model": model,
                "max_tokens": max_tokens
            }

            response = requests.post(self.base_url, json=data)

            if response.status_code == 200:
                return response.json().get('response', 'No response in the response body')
            else:
                return f"Error: {response.json().get('error', 'Unknown error')}"
        except Exception as e:
            return f"Error during API call: {str(e)}"




class lmm_llava_v1_6_34b():
    
    def __init__(self, model_path = "llava-hf/llava-v1.6-34b-hf", gpuid = 0): 
        self.gpuid = gpuid
        from transformers import (LlavaNextForConditionalGeneration,
                                  LlavaNextProcessor)
        self.processor = LlavaNextProcessor.from_pretrained(model_path)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True) 
        self.model.to(f"cuda:{gpuid}")
        
    def inference(self, question: str, images_list):
        if type(images_list) == list:
            image = concatenate_images_with_number_label(images_list)
        else:
            image = images_list
        
        prompt = f"<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n{question}<|im_end|><|im_start|>assistant\n"
        inputs = self.processor(prompt, image, return_tensors="pt").to(f"cuda:{self.gpuid}")
        output = self.model.generate(**inputs, max_new_tokens=1000)
        res = self.processor.decode(output[0], skip_special_tokens=True)
        
        start_index = res.find("<|im_start|> assistant\n")
        if start_index != -1:
            content = res[start_index + len("<|im_start|> assistant\n"):]
        return content
    
    def image_caption(self, image):
        image_caption_prompt = 'Describe the details of this image in detail, including the color, pose, lighting, and environment of the target object.'
        return self.inference(image_caption_prompt, image)
    pass


class lmm_llava_v1_6_7b():
    
    def __init__(self, model_path = "llava-hf/llava-v1.6-mistral-7b-hf", gpuid = 0): 
        self.gpuid = gpuid
        from transformers import (LlavaNextForConditionalGeneration,
                                  LlavaNextProcessor)
        self.processor = LlavaNextProcessor.from_pretrained(model_path)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True) 
        self.model.to(f"cuda:{gpuid}")
        
    def inference(self, question: str, images_list):
        if type(images_list) == list:
            image = concatenate_images_with_number_label(images_list)
        else:
            image = images_list
        
        
        prompt = f"[INST] <image>\n{question} [/INST]"
        inputs = self.processor(prompt, image, return_tensors="pt").to(f"cuda:{self.gpuid}")
        output = self.model.generate(**inputs, max_new_tokens=1000)

        res = self.processor.decode(output[0], skip_special_tokens=True)
        
        result = re.search(r'\[/INST\](.*)', res)
        if result:
            res = result.group(1)
        return res
    
    def image_caption(self, image):
        image_caption_prompt = 'Describe the details of this image in detail, including the color, pose, lighting, and environment of the target object.'
        return self.inference(image_caption_prompt, image)
    pass

import re



import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration


class lmm_llava_CoT_11B:
    def __str__(self):
        return 'llava-CoT-11B'

    def __init__(self, model_path="Xkev/Llama-3.2V-11B-cot", gpuid=0):
        self.gpuid = gpuid
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16
        ).to(f"cuda:{gpuid}")
    
    def extract_reasoning_stage(self, text):
        start_tag = "(Here begins the REASONING stage)"
        end_tag = "(Here ends the REASONING stage)"
        
        # Find the start and end of the reasoning stage
        start_index = text.find(start_tag) + len(start_tag)
        end_index = text.find(end_tag)
        
        # Extract and return the content between the tags
        if start_index != -1 and end_index != -1:
            return text[start_index:end_index].strip()
        else:
            return "Reasoning stage not found in the text."
            
    def inference(self, question: str, images_list, max_new_tokens=1000):
        if isinstance(images_list, str):
            images_list = [images_list]
        
        # ./tool/api/view_0.png
        if not images_list:
            images_list = ['./tool/api/view_0.png']
            
        
        images = [Image.open(img_path).convert("RGB") for img_path in images_list]
        
        image_types = [{"type": "image"} for _ in range(len(images_list))]
        
        messages = [{"role": "user", "content": [{"type": "text", "text": question}] + image_types}]
        print(f'{messages=}')
        texts = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        
        inputs = self.processor(text=texts, images=images, return_tensors="pt").to(f"cuda:{self.gpuid}")
        
        generation_kwargs = dict(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=True, 
            temperature=0.6, 
            top_p=0.9
        )
        outputs = self.model.generate(**generation_kwargs)
        
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        generated_text = re.sub(r"<(\w+)>", r"(Here begins the \1 stage)", generated_text)
        generated_text = re.sub(r"</(\w+)>", r"(Here ends the \1 stage)", generated_text)
        
        return self.extract_reasoning_stage(generated_text)





import os


from lmdeploy import TurbomindEngineConfig, pipeline
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN
from PIL import Image


class lmm_InternVL2_8B:
    def __str__(self):
        return 'InternVL2-8B'

    def __init__(self, model_path='OpenGVLab/InternVL2-8B', gpuid=0, session_len=8192):
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path '{model_path}' does not exist.")
        self.device = f"cuda:{gpuid}"

        self.pipe = pipeline(
            model_path, 
            backend_config=TurbomindEngineConfig(session_len=session_len)
        )
        

    def load_images(self, image_paths):
        loaded_images = []
        for img_path in image_paths:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image path '{img_path}' does not exist.")
            loaded_images.append(Image.open(img_path).convert("RGB"))
        return loaded_images

    def inference(self, prompt, image_paths):
        if not isinstance(image_paths, list):
            image_paths = [image_paths]
        images = self.load_images(image_paths)
        response = self.pipe((prompt, images))
        return response.text


import math

import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


class lmm_InternVL2_5_78B:
    def __str__(self):
        return 'InternVL2-5-78B'

    def __init__(self, model_path='OpenGVLab/InternVL2_5-78B', gpuid=[0,1,2,3], load_in_8bit=True):
        self.model_path = model_path
        self.gpuid = gpuid

        self.device_map = self.split_model('InternVL2_5-78B', gpuid)

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=load_in_8bit,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=self.device_map
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    def split_model(self, model_name, gpuid):
        world_size = len(gpuid)
        num_layers = {'InternVL2_5-78B': 80}[model_name]
        num_layers_per_gpu = [math.ceil(num_layers / world_size)] * world_size
        device_map = {}
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = gpuid[i]
                layer_cnt += 1
        device_map['vision_model'] = gpuid[0]
        device_map['mlp1'] = gpuid[0]
        device_map['language_model.model.tok_embeddings'] = gpuid[0]
        device_map['language_model.model.embed_tokens'] = gpuid[0]
        device_map['language_model.output'] = gpuid[0]
        device_map['language_model.model.norm'] = gpuid[0]
        device_map['language_model.lm_head'] = gpuid[0]
        device_map[f'language_model.model.layers.{num_layers - 1}'] = gpuid[0]
        return device_map



    def build_transform(self, input_size):
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform
    

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio
        
    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images


    def load_image(self, image_file, input_size=448, max_num=12):
        image = Image.open(image_file).convert('RGB')
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
        
    def load_images(self, image_paths, max_num=12):
        pixel_values_list = []
        num_patches_list = []
        for img_path in image_paths:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image path '{img_path}' does not exist.")
            pixel_values = self.load_image(img_path, max_num=12).to(torch.bfloat16).cuda()
            pixel_values_list.append(pixel_values)
            num_patches_list.append(pixel_values.size(0))
        pixel_values = torch.cat(pixel_values_list, dim=0)
        return pixel_values, num_patches_list


    def inference(self, prompt, image_paths=None):
        if image_paths is None or len(image_paths) == 0:
            numbered_prompt = prompt
            pixel_values = None
            num_patches_list = None
        else:
            if not isinstance(image_paths, list):
                image_paths = [image_paths]
            
            pixel_values, num_patches_list = self.load_images(image_paths)
            
            numbered_prompt = ''.join([f'Image-{i + 1}: <image>\n' for i in range(len(image_paths))]) + prompt

        response, _ = self.model.chat(
            self.tokenizer, pixel_values, numbered_prompt, 
            generation_config=dict(max_new_tokens=1024, do_sample=True),
            num_patches_list=num_patches_list, 
            return_history=True
        )

        return response

import os
from typing import List

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


class lmm_qwen2vl_7b:
    def __str__(self):
        return 'qwen2vl-7b'

    def __init__(self, model_path="Qwen/Qwen2-VL-7B-Instruct", gpuid=0):
        self.gpuid = gpuid
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, 
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        self.model.to(f"cuda:{gpuid}")

    def prepare_input(self, question: str, image_paths: List[str]):
        images = [Image.open(path).convert("RGB") for path in image_paths]
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": img_path} for img_path in image_paths] + [{"type": "text", "text": question}]
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs.to(f"cuda:{self.gpuid}")

    def inference(self, question: str, image_paths, max_new_tokens=128):
        if not isinstance(image_paths, list):
            image_paths = [image_paths]
        inputs = self.prepare_input(question, image_paths)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]
    
