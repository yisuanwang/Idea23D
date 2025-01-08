# pipeline.py
import json
import logging
import os
import re
import shutil
import argparse

from tool.api.I23Dapi import *
from tool.api.LMMapi import *
from tool.api.module import Iter, Memory
from tool.api.T2Iapi import *
from tool.render.render_3d_model import render_images
from tool.utils.utils import (concatenate_images_with_number_label, readimage,
                              show_image, writeimage)
import os
from datetime import datetime

class Idea23DPipeline:
    def __init__(self, lmm, t2i, i23d, output_dir='./output', num_img=1, num_draft=3, max_iters=5):
        self.lmm = lmm
        self.t2i = t2i
        self.i23d = i23d
        self.output_dir = output_dir
        self.num_img = num_img
        self.num_draft = num_draft
        self.max_iters = max_iters

    def run(self, IDEA_input_path):
        case_name = os.path.basename(IDEA_input_path)
        model_name = f'{str(self.lmm)}_{str(self.t2i)}_{str(self.i23d)}'
        outpath = os.path.join(self.output_dir, model_name, case_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        outpath = os.path.join(self.output_dir, model_name, f"{case_name}_{timestamp}")

        os.makedirs(outpath, exist_ok=True)


        os.makedirs(outpath, exist_ok=True)
        print(f"Output path: {outpath}")

        # Configure logging
        log_file = os.path.join(outpath, "log.txt")
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[
                                logging.FileHandler(log_file),
                                logging.StreamHandler()
                            ])

        def log(message, level="INFO"):
            if level == "INFO":
                logging.info(message)
            elif level == "ERROR":
                logging.error(message)
            elif level == "DEBUG":
                logging.debug(message)
            elif level == "WARNING":
                logging.warning(message)

        with open(os.path.join(IDEA_input_path, 'idea.txt'), 'r', encoding='utf-8') as file:
            IdeaContent = file.read()
            log(f'IdeaContent={IdeaContent}')

        if len(IdeaContent.strip()) == 0:
            log('Error: empty Idea.txt')
            raise ValueError('Empty Idea.txt')

        memory = Memory(IdeaContent)

        prompt_imagecaption = 'Describe the image in detail.'
        prompt_3dcaption = 'These six images are rendered views of a 3D model from six perspectives: front, back, left, right, top, and bottom. Please provide a detailed description of this 3D model. Description from the 3D model as a whole.'

        tags = re.findall(r'(<IMG>(.*?)<\/IMG>)|(<OBJ>(.*?)<\/OBJ>)', IdeaContent)
        
        log(f'{tags=}')
        tag_img_info = {}

        for tag in tags:
            if tag[1]:
                img_tag = tag[1]
                img_path = os.path.join(IDEA_input_path, img_tag)
                if not os.path.exists(img_path):
                    log(f"Image {img_tag} not found.", level="ERROR")
                    continue
                caption = self.lmm.inference(prompt_imagecaption, [img_path]) 
                tag_img_info[img_tag] = {
                    "type": "IMG",
                    "file_name": img_tag,
                    "file_path": img_path,
                    "caption": caption
                }
                log(f"Processed IMG: {tag_img_info[img_tag]}")
                IdeaContent += f'. \n\n Image {img_tag}: {caption}'
                show_image(img_path)
            elif tag[3]: 
                obj_tag = tag[3]
                obj_path = os.path.join(IDEA_input_path, obj_tag)
                if not os.path.exists(obj_path):
                    log(f"3D Model {obj_tag} not found.", level="ERROR")
                    continue
                obj_name = obj_tag.replace('.', '_')
                obj_images_dir = os.path.join(outpath, 'images', obj_name)

                img_render_list, combined_image_path = render_images(
                    input_path=obj_path,
                    output_dir=obj_images_dir,
                    image_size=512,
                    distance=2.5,
                    light='AmbientLights'
                )

                caption = self.lmm.inference(prompt_3dcaption, img_render_list)

                tag_img_info[obj_tag] = {
                    "type": "OBJ",
                    "file_name": obj_tag,
                    "file_path": obj_path,
                    "rendered_images": img_render_list,
                    "combined_image": combined_image_path,
                    "caption": caption
                }
                log(f"Processed OBJ: {tag_img_info[obj_tag]}")
                IdeaContent += f' \n\n 3D model {obj_tag}: {caption}'
                show_image(combined_image_path)

        memory.idea_input_content = tag_img_info
        memory.idea_input_prompt = IdeaContent 

        log(f'{memory.idea_input_content=}')
        log(f'{memory.idea_input_prompt=}')
        log(f'{memory.idea_input_prompt_init=}')  # Ensure this attribute exists

        iters = Iter(0)

        for i in range(self.max_iters):
            log(f'iter = {i}')
            iters.clear()

            for k in range(self.num_draft):
                if i == 0:  # initial round
                    prompt_gen = (
                        'You are an image designer, optimize the text prompt design based on the descriptions of the image and 3D model in the input below to generate an image that matches the User Input.  \n '
                        f'<User Input>{memory.idea_input_prompt}</User Input>. Only the text prompt needs to be returned.'
                    )
                    
                    log(f'{prompt_gen=}')
                    input_info_path_list = memory.get_input_info_path_list()
                    log(f'init iter round, {input_info_path_list=}')
                    IdeaContent = self.lmm.inference(prompt_gen, input_info_path_list)
                    log(f'init iter round, {IdeaContent=}')
                else:
                    # The second round starts with memory+idea input, and the image and best prompt of the best model from the previous round.
                    idea_input_content_desc = memory.get_idea_input_content_desc()
                    prompt_rev = f"""
Optimize the text prompt based on image content and details to better match User Input and images.
<Prompt>{memory.best_prompt}</Prompt>

The first image in the input corresponds to the 6 view angles (top, bottom, left, right, front, and back) of the 3D model generated by the above prompt.

The feedback on modifications to the above 3D model is as follows:

<feedback>{memory.feedback}</feedback>

Describe the style of the image to be generated, the details of the objects, and the interactions between the different objects.
{idea_input_content_desc}

Make sure that the newly generated text prompt better aligns with the User Input.

<User Input>{memory.idea_input_prompt_init}</User Input>

Only the text prompt needs to be returned.
                    """
                    
                    tmplist = [memory.best_3d_img_path]
                    tmplist.extend(memory.get_input_info_path_list())
                    log(f'{tmplist=}')
                    IdeaContent = self.lmm.inference(prompt_rev, tmplist)
                    log(f'{i} iter round, {IdeaContent=}')

                # Generate end of prompt, convert to image
                log('Generate end of prompt, convert to image and 3D..')
                log(f'{IdeaContent=}')
                for j in range(self.num_img):  # Each prompt generates N * images
                    iters.prompt.append(IdeaContent)
                    imgtmp = self.t2i.inference(IdeaContent)
                    draft_dir = os.path.join(outpath, 'draft', f'iter_{i+1}-prompt_{k}-image_{j}')
                    os.makedirs(draft_dir, exist_ok=True)
                    imgpath = os.path.join(draft_dir, 'draft.png')
                    out3dpath = os.path.join(draft_dir, 'mesh.obj')
                    writeimage(imgtmp, imgpath)
                    
                    print(f'{imgpath=}')
                    print(f'{out3dpath=}')
                    
                    i23d_res = self.i23d.inference(imgpath, out3dpath)
                    iters.draft_3d_path.append(i23d_res)
                    log(f'{i23d_res=}')
                    # Save 6 rendered images, and then filter, filter out the best prompt into memory.
                    output_dir = os.path.join(draft_dir, 'images')
                    img_render_list, combined_output_path = render_images(
                        input_path=i23d_res, 
                        output_dir=output_dir, 
                        image_size=512, 
                        distance=2.5, 
                        light='AmbientLights'
                    )
                    show_image(imgpath)
                    show_image(combined_output_path)
                    iters.draft_img.append(imgpath)
                    iters.best_3d_img_path.append(combined_output_path)

            log("End 3D gen, select best 3d model...")
            # Stitch all the images into one big picture, each row is a draft model, and the best model is filtered together.
            
            if i != 0:  # not initial round
                iters.prompt.append(memory.best_prompt)
                iters.draft_img.append(memory.best_img)
                iters.draft_3d_path.append(memory.best_3d_path)
                iters.best_3d_img_path.append(memory.best_3d_img_path)

            log(f'{iters.prompt=}')
            log(f'{iters.draft_img=}')
            log(f'{iters.draft_3d_path=}')
            log(f'{iters.best_3d_img_path=}')

            # Selection of the best draft model for the current round
            select_index_list = list(range(self.num_draft * self.num_img + (1 if i != 0 else 0)))

            log(f'{select_index_list=}')
            prompt_select = (
                f'Each of these images shows 6 views of a 3D model. Which image best meets the user input? '
                f'<User Input>{memory.idea_input_prompt}</User Input>. '
                f'Only return a number in the list {select_index_list}, the number of rows. '
                'Such as, "1", "2" or "0".'
            )
            log(f'{prompt_select=}')
            
            best_row = self.lmm.inference(prompt_select, iters.best_3d_img_path)
            log(f'best_row answer = {best_row}')
            
            pattern = r"\d+"
            numbers = re.findall(pattern, best_row)
            numbers = [int(num) for num in numbers]
            log(f'{numbers=}')

            if len(numbers) ==  1 and numbers[0] in select_index_list:
                best_row = numbers[0]
            else:
                log('Failed to parse best_row as an integer. Using default value.', level="ERROR")
                best_row = len(iters.prompt) - 1 

            log(f'best_row = {best_row}')
            memory.best_prompt = iters.prompt[best_row]
            memory.best_img = iters.draft_img[best_row]
            memory.best_3d_path = iters.draft_3d_path[best_row]
            memory.best_3d_img_path = iters.best_3d_img_path[best_row]

            log(f'memory.best_prompt = {memory.best_prompt}')
            log(f'memory.best_img = {memory.best_img}')
            log(f'memory.best_3d_path = {memory.best_3d_path}')

            # Determine if the output condition is met
            # Give feedback
            prompt_feedback = (
                f'Does the images satisfy the user input? <User Input>{memory.idea_input_prompt}</User Input>. \n'
                'If it matches the User Input, return <no revision>. '
                'If not, provide feedback on where the depicted content of the images does not match the 3D model described in the User Input.'
            )
            log(f'prompt_feedback = {prompt_feedback}')
            feedback = self.lmm.inference(prompt_feedback, [memory.best_img])
            log(f'feedback answer = {feedback}')
            
            feedback_lower = feedback.lower()

            if ('no revision' in feedback_lower or 
                'satisfies the user input' in feedback_lower or 
                'satisfies user input' in feedback_lower):
                log('output include [no revision], finish.')
                break
            else:
                memory.feedback = feedback

        parent_dir = os.path.dirname(memory.best_3d_path)
        target_dir = os.path.join(outpath, 'result')
        # os.makedirs(target_dir, exist_ok=True)
        shutil.copytree(parent_dir, target_dir, dirs_exist_ok=True)
        # shutil.copytree(parent_dir, os.path.join(target_dir, os.path.basename(parent_dir)), dirs_exist_ok=True)

        log(f'Copied {parent_dir} to {target_dir}')

        # Specify file paths
        idea_input_content_file = os.path.join(target_dir, "idea_input_content.json")
        idea_input_prompt_file = os.path.join(target_dir, "idea_input_prompt.txt")
        idea_input_prompt_init_file = os.path.join(target_dir, "idea_input_prompt_init.txt")

        # Save the JSON content
        with open(idea_input_content_file, "w", encoding='utf-8') as json_file:
            json.dump(memory.idea_input_content, json_file, indent=4)

        # Save the text strings
        with open(idea_input_prompt_file, "w", encoding='utf-8') as text_file:
            text_file.write(memory.idea_input_prompt)

        with open(idea_input_prompt_init_file, "w", encoding='utf-8') as text_file:
            text_file.write(memory.idea_input_prompt_init)

        print(f'Idea23d pipeline {target_dir=}')
        return target_dir

# The rest of the script remains mostly the same, but you can remove the CLI part if not needed.