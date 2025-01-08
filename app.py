import os
os.environ["GRADIO_TEMP_DIR"] = "./tmp"

import gradio as gr
import tempfile
import shutil
import os
import base64
import logging

from apppipeline import Idea23DPipeline 
from tool.api.I23Dapi import *
from tool.api.LMMapi import *
from tool.api.module import Iter, Memory
from tool.api.T2Iapi import *

from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

import logging

model_cache = {}  # {(lmm_choice, t2i_choice, i23d_choice): (lmm, t2i, i23d)}

def initialize_models(lmm_choice, t2i_choice, i23d_choice):
    logging.info(f"Initializing models: LMM={lmm_choice}, T2I={t2i_choice}, I23D={i23d_choice}")

    cache_key = (lmm_choice, t2i_choice, i23d_choice)
    if cache_key in model_cache:
        logging.info("Found models in cache, no need to re-initialize.")
        return model_cache[cache_key]

    if lmm_choice == 'gpt-4o':
        lmm = lmm_gpt4o_local(api_key='sk-xxx your openai api key')
    elif lmm_choice == 'internvl2_5_78b':
        lmm = lmm_InternVL2_5_78B(model_path='OpenGVLab/InternVL2_5-78B', gpuid=[5,6], load_in_8bit=True)
    elif lmm_choice == 'internvl2_8b':
        lmm = lmm_InternVL2_8B(model_path='OpenGVLab/InternVL2-8B', gpuid=0)
    elif lmm_choice == 'llava_cot_11b':
        lmm = lmm_llava_CoT_11B(model_path='Xkev/Llama-3.2V-11B-cot', gpuid=7)
    elif lmm_choice == 'qwen2vl_7b':
        lmm = lmm_qwen2vl_7b(model_path='Qwen/Qwen2-VL-7B-Instruct', gpuid=1)
    else:
        raise ValueError(f"Unsupported LMM type: {lmm_choice}")

    if t2i_choice == 'flux':
        t2i = t2i_flux(model_path='black-forest-labs/FLUX.1-dev', gpuid=4)
    elif t2i_choice == 'sdxl':
        t2i = t2i_sdxl(
            sdxl_base_path='stabilityai/stable-diffusion-xl-base-1.0',
            sdxl_refiner_path='stabilityai/stable-diffusion-xl-refiner-1.0',
            gpuid=7
        )
    else:
        raise ValueError(f"Unsupported T2I type: {t2i_choice}")

    if i23d_choice == 'instantmesh':
        i23d = i23d_InstantMesh(gpuid=4)
    elif i23d_choice == 'triposr':
        i23d = i23d_TripoSR(model_path='stabilityai/TripoSR', gpuid=7)
    elif i23d_choice == 'sf3d':
        i23d = i23d_stable_fast_3d(model_path='stabilityai/stable-fast-3d', gpuid=0)
    elif i23d_choice == 'hunyuan3d':
        i23d = i23d_Hunyuan3D(
            mv23d_ckt_path="Hunyuan3D-1/weights/svrm/svrm.safetensors",
            text2image_path="Hunyuan3D-1/weights/hunyuanDiT",
            gpuid=0,
            save_memory=True,
            max_faces_num=120000,
            do_bake=True,
            bake_align_times=3
        )
    else:
        raise ValueError(f"Unsupported I23D type: {i23d_choice}")

    model_cache[cache_key] = (lmm, t2i, i23d)
    logging.info("Models initialized successfully and stored in cache.")
    return lmm, t2i, i23d

def encode_file_to_data_uri(file_path, file_type):
    logging.info(f"Encoding file to data URI: {file_path}, type={file_type}")
    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{file_type};base64,{encoded}"

def convert_obj_to_glb(obj_path, output_dir):
    logging.info(f"Converting .obj to .glb: {obj_path}, out_dir={output_dir}")
    try:
        import trimesh
        from pygltflib import GLTF2

        file_ext = os.path.splitext(obj_path)[1].lower()
        if file_ext in ['.glb', '.gltf']:
            logging.info(f"File is already .glb/.gltf: {obj_path}")
            return obj_path
        elif file_ext == '.obj':
            mesh = trimesh.load(obj_path)
            if not isinstance(mesh, trimesh.Trimesh):
                logging.info("Loaded mesh is a Scene, merging into single Trimesh...")
                mesh = mesh.dump().sum()

            glb_filename = os.path.splitext(os.path.basename(obj_path))[0] + '.glb'
            glb_path = os.path.join(output_dir, glb_filename)
            logging.info(f"Exporting mesh to glb: {glb_path}")
            mesh.export(glb_path, file_type='glb')
            return glb_path
        else:
            logging.warning(f"Unsupported file extension for conversion: {file_ext}")
            return None
    except Exception as e:
        logging.error(f"Error converting OBJ to GLB: {e}")
        return None


def process_pipeline(
    text_input,
    file_inputs,
    lmm_choice,
    t2i_choice,
    i23d_choice,
    num_img,
    num_draft,
    max_iters
):
    logging.info("==> process_pipeline called.")
    logging.info(f"pipeline params: text={text_input}, file_inputs={file_inputs}")
    logging.info(f"model choices: LMM={lmm_choice}, T2I={t2i_choice}, I23D={i23d_choice}")
    logging.info(f"gen params: num_img={num_img}, num_draft={num_draft}, max_iters={max_iters}")

    try:
        lmm, t2i, i23d = initialize_models(lmm_choice, t2i_choice, i23d_choice)
    except Exception as e:
        logging.error(f"Model initialize error: {e}")
        return None

    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)

    idea_input_path = os.path.join(output_dir, 'input')
    os.makedirs(idea_input_path, exist_ok=True)
    logging.info(f"Created input dir in output: {idea_input_path}")

    auto_tags = []

    uploaded_files_info = []
    if file_inputs:
        for file in file_inputs:
            file_path = file.name if hasattr(file, 'name') else file
            filename = os.path.basename(file_path)
            dest_path = os.path.join(idea_input_path, filename)
            shutil.copy(file_path, dest_path)
            uploaded_files_info.append(dest_path)
            logging.info(f"Copied uploaded file {file_path} => {dest_path}")

            ext = os.path.splitext(filename)[1].lower()
            if ext in [".png", ".jpg", ".jpeg"]:
                auto_tags.append(f"<IMG>{filename}</IMG>")
            elif ext in [".obj", ".glb", ".gltf"]:
                auto_tags.append(f"<OBJ>{filename}</OBJ>")
            else:
                pass  

    extra_tags_str = ""
    if auto_tags:
        img_tags = [tag for tag in auto_tags if tag.startswith("<IMG>")]
        obj_tags = [tag for tag in auto_tags if tag.startswith("<OBJ>")]

        if img_tags:
            extra_tags_str += f"{img_tags}"

        if obj_tags:
            extra_tags_str += f"{obj_tags}"

    idea_txt_path = os.path.join(idea_input_path, 'idea.txt')
    with open(idea_txt_path, 'w', encoding='utf-8') as f:
        f.write(text_input.strip())
        if extra_tags_str:
            f.write(extra_tags_str)

    logging.info(f"Saved user text + appended tags to {idea_txt_path}")

    pipeline = Idea23DPipeline(
        lmm, t2i, i23d,
        num_img=num_img,
        num_draft=num_draft,
        max_iters=max_iters
    )
    logging.info("Running pipeline.run()...")

    try:
        logging.info(f"{idea_input_path=}")
        result_dir = pipeline.run(idea_input_path)
        logging.info(f"pipeline.run() done, result_dir={result_dir}")
    except Exception as e:
        logging.error(f"Error running pipeline: {e}")
        return None

    generated_model_path = None
    for root, dirs, files in os.walk(result_dir):
        for file in files:
            if file.endswith(('.obj', '.glb', '.gltf')):
                generated_model_path = os.path.join(root, file)
                break
        if generated_model_path:
            break

    if not generated_model_path:
        msg = "no found -> (.obj/.glb/.gltf)！"
        logging.error(msg)
        return None

    logging.info(f"Found pipeline 3D result: {generated_model_path}")
    return generated_model_path



def decode_glb_to_local(generated_model_data_uri, user_text_md, user_images, user_models_html):
    logging.info("Decoding pipeline's .glb base64 => local file in ./output")
    base64_str = generated_model_data_uri.split(",", 1)[1]
    model_bin = base64.b64decode(base64_str)

    os.makedirs("./output", exist_ok=True)

    import time
    timestamp = int(time.time())
    glb_filename = f"my_model_{timestamp}.glb"
    glb_path = os.path.join("./output", glb_filename)

    with open(glb_path, "wb") as f:
        f.write(model_bin)

    logging.info(f"Decoded pipeline .glb to: {glb_path}")
    return (user_text_md, user_images, user_models_html, glb_path)

def on_submit(
    text_input,
    file_inputs,
    lmm_choice,
    t2i_choice,
    i23d_choice,
    num_img,
    num_draft,
    max_iters,
    test_mode
):
    logging.info("======== on_submit clicked ========")
    logging.info(f"text_input={text_input}, file_inputs={file_inputs}")
    logging.info(f"modelChoice=({lmm_choice},{t2i_choice},{i23d_choice}), test_mode={test_mode}")
    logging.info(f"num_img={num_img}, num_draft={num_draft}, max_iters={max_iters}")

    if not text_input.strip():
        err = "error: empty text_input"
        logging.warning(err)
        return None

    if test_mode:
        logging.info("test mode...")
        obj_path = "./Idea23D/input/case_013/cat.obj"
        if not os.path.exists(obj_path):
            msg = "no found cat.obj"
            logging.error(msg)
            return None

        tmpdir = './tmp'
        os.makedirs("./tmp", exist_ok=True)
        glb_path = convert_obj_to_glb(obj_path, tmpdir)
        if not glb_path or not os.path.exists(glb_path):
            msg = "Unable to convert cat.obj to .glb in test mode"
            logging.error(msg)
            return None

        logging.info(f"Test mode: cat.obj -> {glb_path}")
        return glb_path

    # Normal mode: Actually invoke the multi-stage generation pipeline
    logging.info("Entering normal mode: invoking process_pipeline")
    generated_model_path = process_pipeline(
        text_input,
        file_inputs,
        lmm_choice,
        t2i_choice,
        i23d_choice,
        num_img,
        num_draft,
        max_iters
    )
    if not generated_model_path:
        return None, None  

    model_dir = os.path.dirname(generated_model_path)
    view_all_path = os.path.join(model_dir, "images", "view_all.png")

    if not os.path.exists(view_all_path):
        logging.warning(f"Warning: view_all.png not found in {view_all_path}")
        view_all_path = None

    return generated_model_path, view_all_path


def file_to_data_uri(filepath):
    """简单根据文件类型返回 data URI."""
    ext = os.path.splitext(filepath)[1].lower()
    mime_type = "application/octet-stream"  
    if ext in [".png", ".jpg", ".jpeg"]:
        mime_type = "image/png" if ext == ".png" else "image/jpeg"
    elif ext in [".glb", ".gltf"]:
        mime_type = "model/gltf-binary"
    with open(filepath, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"

def handle_upload(files, current_files):
    """
    当文件上传完成后，将其加入到 current_files(state) 中，
    并返回新的图片列表、模型列表，以便在前端分别显示。
    """
    if not files: 
        return [], [], []

    new_files = list(current_files) 

    for f in files:
        new_files.append(f)

    images_data_uris = []
    models_data_uris = []
    for f in new_files:
        ext = os.path.splitext(f.name)[1].lower()
        if ext in [".png", ".jpg", ".jpeg"]:
            images_data_uris.append(file_to_data_uri(f.name))
        elif ext in [".obj", ".glb", ".gltf"]:
            models_data_uris.append(file_to_data_uri(f.name))
        else:
            pass

    return new_files, images_data_uris, models_data_uris

def remove_file(file_index, current_files):
    if file_index is None:
        return current_files, [], []
    new_files = list(current_files)
    if 0 <= file_index < len(new_files):
        del new_files[file_index]

    images_data_uris = []
    models_data_uris = []
    for f in new_files:
        ext = os.path.splitext(f.name)[1].lower()
        if ext in [".png", ".jpg", ".jpeg"]:
            images_data_uris.append(file_to_data_uri(f.name))
        elif ext in [".obj", ".glb", ".gltf"]:
            models_data_uris.append(file_to_data_uri(f.name))

    return new_files, images_data_uris, models_data_uris

def build_ui():
    with gr.Blocks(css="""
    .my_model_viewer canvas {
        filter: brightness(1);
    }
    """) as demo:
        # gr.Markdown("# [COLING 2025] Idea23D: Collaborative LMM Agents Enable 3D Model Generation from Interleaved Multimodal Inputs")
        # gr.Markdown("Based on the LMM we developed Idea23D, a multimodal iterative self-refinement system that enhances any T2I model for automatic 3D model design and generation, enabling various new image creation functionalities togther with better visual qualities while understanding high level multimodal inputs.")
        gr.HTML("""
<h1 style="font-weight: bold">
  <!-- <a href="https://idea23d.github.io/" target="_blank"> -->
     <span style="background: linear-gradient(90deg, rgba(131,58,180,1) 0%, rgba(253,29,29,1) 50%, rgba(252,176,69,1) 100%); -webkit-background-clip: text; color: transparent; background-clip: text;">Idea23D</span>:
    Collaborative LMM Agents Enable 3D Model Generation from Interleaved Multimodal Inputs
  <!-- </a> -->
</h1>

2024.11: 🎉 Idea-2-3D has been accepted by COLING 2025! 🎉 See you in Abu Dhabi, UAE, from January 19 to 24, 2025!

<div align="left" style="white-space: nowrap;">
  <a href="https://idea23d.github.io/"><img src="https://img.shields.io/static/v1?label=Homepage&message=Idea23D&color=blue&logo=github-pages" style="display: inline-block; margin-right: 10px;"></a>
  <a href="https://github.com/yisuanwang/Idea23D"><img src="https://img.shields.io/github/stars/yisuanwang/Idea23D?label=stars&logo=github&color=brightgreen" alt="GitHub Repo Stars" style="display: inline-block; margin-right: 10px;"></a>
  <a href="https://arxiv.org/abs/2404.04363"><img src="https://img.shields.io/badge/arXiv-2404.04363-b31b1b.svg?style=flat-square" alt="arXiv" style="display: inline-block; margin-right: 10px;"></a>
  <a href="https://huggingface.co/yisuanwang/Idea23D"><img src="https://img.shields.io/static/v1?label=Dataset&message=HuggingFace&color=yellow" style="display: inline-block; margin-right: 10px;"></a>
  <a href="https://idea23d.github.io"><img src="https://img.shields.io/static/v1?label=Demo&message=Gradio&color=yellow" style="display: inline-block;"></a>
</div>
        """)

        with gr.Row():
            text_input = gr.Textbox(
                lines=3,
                label="Text Command",
                placeholder="Please enter a descriptive text..."
            )
            file_input = gr.File(
                file_count="multiple",
                type="filepath",
                file_types=[".png", ".jpg", ".jpeg", ".obj", ".glb", ".gltf"],
                label="Upload Images or 3D Models"
            )

        # Set open=False to open=True here
        with gr.Accordion("⚙️Advanced Settings", open=True):
            with gr.Row():
                lmm_choice = gr.Dropdown(
                    choices=["gpt-4o"],
                    value="gpt-4o", label="Language Model (LMM)"
                )
                t2i_choice = gr.Dropdown(
                    choices=["flux"],
                    value="flux", label="Text to Image (T2I)"
                )
                i23d_choice = gr.Dropdown(
                    choices=["instantmesh"],
                    value="instantmesh", label="Image to 3D (I23D)"
                )
            with gr.Row():
                num_img = gr.Number(value=1, label="Number of Images")
                num_draft = gr.Number(value=2, label="Number of Drafts")
                max_iters = gr.Number(value=3, label="Max Iterations")
            test_mode = gr.Checkbox(value=False, label="Test Mode (using cat.obj)", visible=False)

        submit_button = gr.Button("Generate 3D Model")

        
        with gr.Row():
            generated_model_output = gr.Model3D(
                label="Final 3D model output",
                elem_classes="my_model_viewer"
            )
            view_all_output = gr.Image(label="View All Image Preview")


        submit_button.click(
            fn=on_submit,
            inputs=[
                text_input,
                file_input,
                lmm_choice,
                t2i_choice,
                i23d_choice,
                num_img,
                num_draft,
                max_iters,
                test_mode
            ],
            outputs=[
                generated_model_output,  
                view_all_output          
            ]
        )

    return demo

if __name__ == "__main__":
    logging.info("Launching Gradio demo ...")
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7867, share=True)