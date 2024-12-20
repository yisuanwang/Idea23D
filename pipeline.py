import json
import logging
import os
import re
import shutil

from tool.api.I23Dapi import *
from tool.api.LMMapi import *
from tool.api.module import Iter, Memory
from tool.api.T2Iapi import *
from tool.render.render_3d_model import render_images
from tool.utils.utils import (concatenate_images_with_number_label, readimage,
                              show_image, writeimage)


def main(lmm, t2i, i23d, IDEA_input_path, num_img, num_draft, max_iters):
    case_name = os.path.basename(IDEA_input_path)
    model_name = f'{str(lmm)}_{str(t2i)}_{str(i23d)}'
    outpath = os.path.join('./output/', model_name, case_name)


    if os.path.exists(f'{outpath}/result'):
        print(f'{outpath} exists, skip')
        return

    os.makedirs(outpath, exist_ok=True)
    print(f"Output path: {outpath}")

    # Configure logging
    log_file = f"{outpath}/log.txt"
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

    with open(f'{IDEA_input_path}/idea.txt', 'r') as file:
        IdeaContent = file.read()
        log(f'IdeaContent={IdeaContent}')

    if len(IdeaContent.strip()) == 0:
        log('Error: empty Idea.txt')
        exit()


    memory = Memory(IdeaContent)

    prompt_imagecaption = 'Describe the image in detail.'
    prompt_3dcaption = 'These six images are rendered views of a 3D model from six perspectives: front, back, left, right, top, and bottom. Please provide a detailed description of this 3D model. Description from the 3D model as a whole.'


    tags = re.findall(r'(<IMG>(.*?)<\/IMG>)|(<OBJ>(.*?)<\/OBJ>)', IdeaContent)
    
    log(f'{tags=}')
    tag_img_info = {}


    for tag in tags:
        if tag[1]:
            img_tag = tag[1]
            img_path = f'{IDEA_input_path}/{img_tag}'
            caption = lmm.inference(prompt_imagecaption, [img_path]) 
            tag_img_info[img_tag] = {
                "type": "IMG",
                "file_name": img_tag,
                "file_path": img_path,
                "caption": caption
            }
            log(f"Processed IMG: {tag_img_info[img_tag]}")
            IdeaContent = f'{IdeaContent}. \n\n Image {img_tag}: {caption}'
            show_image(img_path)
        elif tag[3]: 
            obj_tag = tag[3]
            obj_path = f'{IDEA_input_path}/{obj_tag}'
            obj_name = obj_tag.replace('.', '_')
            obj_images_dir = f'{outpath}/images/{obj_name}'

            img_render_list ,combined_image_path = render_images(
                input_path=obj_path,
                output_dir=obj_images_dir,
                image_size=512,
                distance=2.5,
                light='AmbientLights'
            )

            caption = lmm.inference(prompt_3dcaption, img_render_list)

            tag_img_info[obj_tag] = {
                "type": "OBJ",
                "file_name": obj_tag,
                "file_path": obj_path,
                "rendered_images": img_render_list,
                "combined_image": combined_image_path,
                "caption": caption
            }
            log(f"Processed OBJ: {tag_img_info[obj_tag]}")
            IdeaContent = f'{IdeaContent} \n\n 3D model {obj_tag}: {caption}'
            show_image(combined_image_path)


    memory.idea_input_content = tag_img_info
    memory.idea_input_prompt = IdeaContent 

    log(f'{memory.idea_input_content=}')
    log(f'{memory.idea_input_prompt=}')
    log(f'{memory.idea_input_prompt_init=}')


    iters = Iter(0)

    for i in range(max_iters):
        log(f'iter = {i}')
        iters.clear()

        for k in range(num_draft):
            if i == 0: # initial round
                prompt_gen = f'You are an image designer, optimize the text prompt design based on the descriptions of the image and 3D model in the input below to generate an image that matches the User Input.  \n <User Input>{memory.idea_input_prompt}</User Input>. Only the text prompt needs to be returned.'
                
                log(f'{prompt_gen=}')
                input_info_path_list = memory.get_input_info_path_list()
                log(f'init iter round ,{input_info_path_list=}')
                IdeaContent = lmm.inference(prompt_gen, input_info_path_list)
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
                IdeaContent = lmm.inference(prompt_rev, tmplist)
                log(f'{i} iter round, {IdeaContent=}')

            
            # Generate end of prompt, convert to image
            log('Generate end of prompt, convert to image and 3D..')
            log(f'{IdeaContent=}')
            for j in range(num_img): # Each prompt generates N * images
                iters.prompt.append(IdeaContent)
                imgtmp = t2i.inference(IdeaContent)
                imgpath = f'{outpath}/draft/iter_{i+1}-prompt_{k}-image_{j}/draft.png'
                out3dpath = f'{outpath}/draft/iter_{i+1}-prompt_{k}-image_{j}/mesh.obj'
                writeimage(imgtmp, imgpath)
                
                print(f'{imgpath=}')
                print(f'{out3dpath=}')
                
                i23d_res = i23d.inference(imgpath, out3dpath)
                iters.draft_3d_path.append(i23d_res)
                log(f'{i23d_res=}')
                #  Save 6 rendered images, and then filter, filter out the best prompt into memory.
                output_dir = f'{outpath}/draft/iter_{i+1}-prompt_{k}-image_{j}/images'
                img_render_list, combined_output_path = render_images(input_path=i23d_res, output_dir=output_dir , image_size=512, distance=2.5, light='AmbientLights')
                show_image(imgpath)
                show_image(combined_output_path)
                iters.draft_img.append(imgpath)
                iters.best_3d_img_path.append(combined_output_path)

        log("End 3D gen, select best 3d model...")
        # Stitch all the images into one big picture, each row is a draft model, and the best model is filtered together.
        
        
        if i != 0: # not initial round
            iters.prompt.append(memory.best_prompt)
            iters.draft_img.append(memory.best_img)
            iters.draft_3d_path.append(memory.best_3d_path)
            iters.best_3d_img_path.append(memory.best_3d_img_path)

        log(f'{iters.prompt=}')
        log(f'{iters.draft_img=}')
        log(f'{iters.draft_3d_path=}')
        log(f'{iters.best_3d_img_path=}')

        # draft_img_comp = concatenate_images_with_number_label(iters.draft_img, 'v')

        # Selection of the best draft model for the current round
        select_index_list = [kj for kj in range(num_draft * num_img + (1 if i != 0 else 0))]

        log(f'{select_index_list=}')
        prompt_select = f'Each of these images shows 6 views of a 3D model. Which image best meets the user input? <User Input>{memory.idea_input_prompt}</User Input>. Only return a number in the list {select_index_list}, the number of rows. Such as, \"1\", \"2\" or \"0\".'
        log(f'{prompt_select=}')
        
        best_row = lmm.inference(prompt_select, iters.best_3d_img_path)
        log(f'best_row answer = {best_row}')
        
        
        pattern = r"\d+"
        numbers = re.findall(pattern, best_row)
        numbers = [int(num) for num in numbers]
        log(f'{numbers=}')

        if len(numbers) ==  1 and numbers[0] in [kj for kj in range( num_draft * num_img)]:
            best_row = numbers[0]
        else:
            log('Failed to parse best_row as an integer. Using default value.', level="Error")
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
        prompt_feedback = f'Does the images satisfy the user input? <User Input>{memory.idea_input_prompt}</User Input>. \n If it matches the User Input, return <no revision>. If not, provide feedback on where the depicted content of the images does not match the 3D model described in the User Input.'
        log(f'prompt_feedback = {prompt_feedback}')
        feedback = lmm.inference(prompt_feedback, [memory.best_img])
        log(f'feedback answer = {feedback}')
        
        
        feedback_lower = feedback.lower()


        if 'no revision' in feedback_lower or 'satisfies the user input' in feedback_lower or 'satisfies user input' in feedback_lower:
            log('output include [no revision], finish.')
            break
        else:
            memory.feedback = feedback
        
        pass


    parent_dir = os.path.dirname(memory.best_3d_path)
    target_dir = os.path.join(outpath, 'result')
    os.makedirs(target_dir, exist_ok=True)
    shutil.copytree(parent_dir, os.path.join(target_dir, os.path.basename(parent_dir)), dirs_exist_ok=True)

    log(f'Copied {parent_dir} to {target_dir}')


    # Specify file paths
    idea_input_content_file = f"{target_dir}/idea_input_content.json"
    idea_input_prompt_file = f"{target_dir}/idea_input_prompt.txt"
    idea_input_prompt_init_file = f"{target_dir}/idea_input_prompt_init.txt"

    # Save the JSON content

    with open(idea_input_content_file, "w") as json_file:
        json.dump(memory.idea_input_content, json_file, indent=4)

    # Save the text strings
    with open(idea_input_prompt_file, "w") as text_file:
        text_file.write(memory.idea_input_prompt)

    with open(idea_input_prompt_init_file, "w") as text_file:
        text_file.write(memory.idea_input_prompt_init)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Select LMM, T2I, and I23D models")
    parser.add_argument('--lmm', type=str, choices=['gpt4o', 'gpt4v', 'llava_34b', 'llava_7b', 'internvl2_5_78b','InternVL2_8B', 'llava_cot_11b', 'qwen2vl_7b'],
                        required=True, help="Select the LMM model")
    parser.add_argument('--t2i', type=str, choices=['flux', 'sdxl'], help="Select the T2I model")
    parser.add_argument('--i23d', type=str, choices=['instantmesh', 'triposr', 'sf3d', 'hunyuan3d'], help="Select the I23D model")


    args = parser.parse_args()

    # Initialize LMM,T2I,I23D
    if args.lmm == 'gpt4o':
        lmm = lmm_gpt4o_local(api_key = 'sk-xxx your openai api key')
        # lmm = lmm_gpt4o(api_key = 'sk-xxx your openai api key')
    elif args.lmm == 'internvl2_5_78b':
        lmm = lmm_InternVL2_5_78B(model_path='OpenGVLab/InternVL2_5-78B', gpuid=[5,6], load_in_8bit=True)
        # lmm = lmm_InternVL2_5_78B(model_path='OpenGVLab/InternVL2_5-78B', gpuid=[0, 1, 2, 3], load_in_8bit=False)
    elif args.lmm == 'internvl2_8b':
        lmm = lmm_InternVL2_8B(model_path='OpenGVLab/InternVL2-8B', gpuid=0)
    elif args.lmm == 'llava_cot_11b':
        lmm = lmm_llava_CoT_11B(model_path='Xkev/Llama-3.2V-11B-cot', gpuid=7)
    elif args.lmm == 'qwen2vl_7b':
        lmm = lmm_qwen2vl_7b(model_path='Qwen/Qwen2-VL-7B-Instruct', gpuid=1)
    else:
        raise ValueError(f"Unsupported LMM type: {args.lmm}")

    # t2i = text2img_sdxl_replicate(replicate_key='your api key')
    if args.t2i == 'flux':
        t2i = t2i_flux(model_path='black-forest-labs/FLUX.1-dev', gpuid=1)
    elif args.t2i == 'sdxl':
        t2i = t2i_sdxl(sdxl_base_path='stabilityai/stable-diffusion-xl-base-1.0', 
                       sdxl_refiner_path='stabilityai/stable-diffusion-xl-refiner-1.0', 
                       gpuid=7)
    

    if args.i23d == 'instantmesh':
        i23d = i23d_InstantMesh(gpuid=2)
    elif args.i23d == 'triposr':
        i23d = i23d_TripoSR(model_path = 'stabilityai/TripoSR' ,gpuid=7)
    elif args.i23d == 'sf3d':
        i23d = i23d_stable_fast_3d(model_path='stabilityai/stable-fast-3d', gpuid=0)
    elif args.i23d == 'hunyuan3d':
        i23d = i23d_Hunyuan3D(
            mv23d_ckt_path="Hunyuan3D-1/weights/svrm/svrm.safetensors",
            text2image_path="Hunyuan3D-1/weights/hunyuanDiT",
            gpuid=0,
            # use_lite=True,  # Example of setting additional parameters
            save_memory=True,
            max_faces_num=120000,
            # do_texture_mapping=True,
            do_bake=True,
            bake_align_times=3
        )
    input_root = './dataset'
    num_img = 1 # Number of images generated per prompt
    num_draft = 3 # Number of prompts generated per round
    max_iters = 5 # Maximum number of iteration rounds

    # Iterate through all the case_xxx folders under the ./input directory
    for case_folder in os.listdir(input_root):
        IDEA_input_path = os.path.join(input_root, case_folder)
        
        print(f'{IDEA_input_path=}')

        main(lmm, t2i, i23d, IDEA_input_path, num_img, num_draft, max_iters)


# python pipeline.py --lmm gpt4o --t2i flux --i23d instantmesh