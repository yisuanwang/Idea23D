# Add the tool directory to sys.path so Python can find your modules

import argparse
import os
# from i23d.TripoSR.run import TripoSRmain
import shutil
import sys
from contextlib import nullcontext

import rembg
import torch
from PIL import Image
from tqdm import tqdm

# import tempfile


# Ensure the current script directory and its parent are in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

# Add the tool directory to sys.path
tool_path = parent_dir
print(f'{tool_path=}')
if tool_path not in sys.path:
    sys.path.append(tool_path)

from render.rotate import rotate_InstantMesh, rotate_TripoSR

# Add the stablefast3d directory to sys.path
stablefast3d_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../i23d/stablefast3d"))
if stablefast3d_path not in sys.path:
    sys.path.append(stablefast3d_path)

from sf3d.system import SF3D
from sf3d.utils import get_device, remove_background, resize_foreground



class image23d:
    def __init__(self, model_path='', gpuid=1):
        self.model_path = model_path
        self.gpuid = gpuid

    def inference(self, png_path, output_path):
        raise NotImplementedError("This method should be overridden in subclasses.")



class i23d_TripoSR():
    def __str__(self):
        return 'TripoSR'
    
    def __init__(self, model_path='stabilityai/TripoSR', gpuid=1):
        self.gpuid = gpuid
        self.model_path = model_path

    def inference(self, png_path, output_path):
        """
        Perform inference using TripoSR and manage input/output paths.
        Args:
            png_path (str): Path to the input PNG file.
            output_path (str): Path to the directory where output should be stored.
        Returns:
            str: Path to the output mesh.obj file.
        """
        # Create a temporary directory for the input
        temp_input_dir = tempfile.mkdtemp()
        temp_input_path = os.path.join(temp_input_dir, os.path.basename(png_path))

        # Copy the input file to the temporary directory
        shutil.copy(png_path, temp_input_path)

        # Create the output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Perform the TripoSR inference
        print(f"Running inference on: {temp_input_path}")
        TripoSRmain(self.gpuid, self.model_path, temp_input_path, temp_input_dir)

        # Move the generated mesh.obj file to the output path
        generated_mesh = os.path.join(temp_input_dir, 'mesh.obj')
        output_mesh = os.path.join(output_path, 'mesh.obj')

        if os.path.exists(generated_mesh):
            shutil.move(generated_mesh, output_mesh)
        else:
            raise FileNotFoundError(f"Generated mesh.obj not found in {temp_input_dir}")

        # Clean up the temporary directory
        shutil.rmtree(temp_input_dir)

        # Rotate TripoSR result
        rotate_TripoSR(output_mesh, output_mesh)
        return output_mesh

# Add the InstantMesh directory to sys.path
InstantMesh_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../i23d/InstantMesh"))
if InstantMesh_path not in sys.path:
    sys.path.append(InstantMesh_path)
    
import numpy as np
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from einops import rearrange, repeat
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from src.utils.camera_util import (FOV_to_intrinsics,
                                   get_circular_camera_poses,
                                   get_zero123plus_input_cameras)
from src.utils.infer_util import (remove_background, resize_foreground,
                                  save_video)
from src.utils.mesh_util import save_obj, save_obj_with_mtl
from src.utils.train_util import instantiate_from_config
from torchvision.transforms import v2


class i23d_InstantMesh:
    def __str__(self):
        return 'InstantMesh'

    def __init__(
        self, 
        config_path='tool/i23d/InstantMesh/configs/instant-mesh-large.yaml', 
        model_path='', 
        gpuid=0,
        diffusion_steps=75,
        seed=42,
        scale=1.0,
        distance=4.5,
        view=6,
        no_rembg=False,
        export_texmap=False,
        save_video_flag=False,
    ):
        self.config_path = config_path
        self.model_path = model_path
        self.gpuid = gpuid
        self.diffusion_steps = diffusion_steps
        self.seed = seed
        self.scale = scale
        self.distance = distance
        self.view = view
        self.no_rembg = no_rembg
        self.export_texmap = export_texmap
        self.save_video_flag = save_video_flag

        self.device = torch.device(f'cuda:{gpuid}' if torch.cuda.is_available() else 'cpu')
        seed_everything(self.seed)

        # load config
        self.config = OmegaConf.load(self.config_path)
        self.config_name = os.path.basename(self.config_path).replace('.yaml', '')
        self.model_config = self.config.model_config
        self.infer_config = self.config.infer_config

        # check if model is flexicubes type
        self.IS_FLEXICUBES = True if self.config_name.startswith('instant-mesh') else False

        # load diffusion pipeline
        print('Loading diffusion model ...')
        self.pipeline = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.2",
            custom_pipeline="tool/i23d/InstantMesh/zero123plus/pipeline.py",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipeline.scheduler.config, timestep_spacing='trailing'
        )

        # load custom white-background UNet
        print('Loading custom white-background unet ...')
        if os.path.exists(self.infer_config.unet_path):
            unet_ckpt_path = self.infer_config.unet_path
        else:
            unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
        state_dict = torch.load(unet_ckpt_path, map_location='cpu')
        self.pipeline.unet.load_state_dict(state_dict, strict=True)
        self.pipeline = self.pipeline.to(self.device)

        # load reconstruction model
        print('Loading reconstruction model ...')
        self.model = instantiate_from_config(self.model_config)
        if os.path.exists(self.infer_config.model_path):
            model_ckpt_path = self.infer_config.model_path
        else:
            model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename=f"{self.config_name.replace('-', '_')}.ckpt", repo_type="model")
        state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
        state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
        self.model.load_state_dict(state_dict, strict=True)

        self.model = self.model.to(self.device)
        if self.IS_FLEXICUBES:
            self.model.init_flexicubes_geometry(self.device, fovy=30.0)
        self.model = self.model.eval()

    def get_render_cameras(self, batch_size=1, M=120, radius=4.0, elevation=20.0, is_flexicubes=False):
        c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
        if is_flexicubes:
            cameras = torch.linalg.inv(c2ws)
            cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        else:
            extrinsics = c2ws.flatten(-2)
            intrinsics = FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
            cameras = torch.cat([extrinsics, intrinsics], dim=-1)
            cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
        return cameras

    def render_frames(self, model, planes, render_cameras, render_size=512, chunk_size=1, is_flexicubes=False):
        frames = []
        for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
            if is_flexicubes:
                frame = model.forward_geometry(
                    planes,
                    render_cameras[:, i:i+chunk_size],
                    render_size=render_size,
                )['img']
            else:
                frame = model.forward_synthesizer(
                    planes,
                    render_cameras[:, i:i+chunk_size],
                    render_size=render_size,
                )['images_rgb']
            frames.append(frame)

        frames = torch.cat(frames, dim=1)[0]  # batch size = 1
        return frames

    def inference(self, png_path, output_path):
        """
        Inference method to take a single input image, generate a 3D mesh, and save results.

        Args:
            png_path (str): Path to the input PNG image.
            output_path (str): Path to the output OBJ file.
        """
        # Make output directories
        base_dir = os.path.dirname(output_path)
        if base_dir == '':
            base_dir = '.'
        image_path = os.path.join(base_dir, 'images')
        # mesh_path = os.path.join(base_dir, 'meshes')
        # video_path = os.path.join(base_dir, 'videos')
        os.makedirs(image_path, exist_ok=True)
        # os.makedirs(mesh_path, exist_ok=True)
        # os.makedirs(video_path, exist_ok=True)

        # Load single image
        name = os.path.basename(png_path).split('.')[0]
        print(f"Inferencing {name} ...")

        rembg_session = None if self.no_rembg else rembg.new_session()
        input_image = Image.open(png_path)
        if not self.no_rembg:
            input_image = remove_background(input_image, rembg_session)
            input_image = resize_foreground(input_image, 0.85)

        # Diffusion model sampling
        output_image = self.pipeline(
            input_image, 
            num_inference_steps=self.diffusion_steps, 
        ).images[0]

        # output_image.save(os.path.join(image_path, f'{name}.png'))
        # print(f"Image saved to {os.path.join(image_path, f'{name}.png')}")

        images = np.asarray(output_image, dtype=np.float32) / 255.0
        images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()  # (3, 960, 640)
        images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)     # (6, 3, 320, 320)

        images = images.unsqueeze(0).to(self.device)
        images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)

        # For fewer views if specified
        if self.view == 4:
            indices = torch.tensor([0, 2, 4, 5]).long().to(self.device)
            images = images[:, indices]

        input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0*self.scale).to(self.device)
        if self.view == 4:
            indices = torch.tensor([0, 2, 4, 5]).long().to(self.device)
            input_cameras = input_cameras[:, indices]

        with torch.no_grad():
            # get triplane
            planes = self.model.forward_planes(images, input_cameras)

            # get mesh
            if self.export_texmap:
                mesh_out = self.model.extract_mesh(
                    planes,
                    use_texture_map=True,
                    **self.infer_config,
                )
                vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
                save_obj_with_mtl(
                    vertices.data.cpu().numpy(),
                    uvs.data.cpu().numpy(),
                    faces.data.cpu().numpy(),
                    mesh_tex_idx.data.cpu().numpy(),
                    tex_map.permute(1, 2, 0).data.cpu().numpy(),
                    output_path,
                )
            else:
                mesh_out = self.model.extract_mesh(
                    planes,
                    use_texture_map=False,
                    **self.infer_config,
                )
                vertices, faces, vertex_colors = mesh_out
                save_obj(vertices, faces, vertex_colors, output_path)
            print(f"Mesh saved to {output_path}")

            # get video if required
            if self.save_video_flag:
                render_size = self.infer_config.render_resolution
                render_cameras = self.get_render_cameras(
                    batch_size=1, 
                    M=120, 
                    radius=self.distance, 
                    elevation=20.0,
                    is_flexicubes=self.IS_FLEXICUBES,
                ).to(self.device)

                chunk_size = 20 if self.IS_FLEXICUBES else 1
                frames = self.render_frames(
                    self.model, 
                    planes, 
                    render_cameras=render_cameras, 
                    render_size=render_size, 
                    chunk_size=chunk_size, 
                    is_flexicubes=self.IS_FLEXICUBES,
                )

                video_path_idx = os.path.join(video_path, f'{name}.mp4')
                save_video(
                    frames,
                    video_path_idx,
                    fps=30,
                )
                print(f"Video saved to {video_path_idx}")
        
        # 旋转 instantmesh 结果
        rotate_InstantMesh(output_path, output_path)
        return output_path


Hunyuan3D_path =  "tool/i23d/Hunyuan3D-1"

if Hunyuan3D_path not in sys.path:
    print(f'{Hunyuan3D_path=}')
    sys.path.append(Hunyuan3D_path)

import os
import time
import warnings
from glob import glob

from PIL import Image

# Suppress warnings
warnings.simplefilter('ignore', category=UserWarning)
warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=DeprecationWarning)

# Import necessary modules
from infer import GifRenderer, Image2Views, Removebg, Text2Image, Views2Mesh
from third_party.check import check_bake_available
from third_party.mesh_baker import MeshBaker


class i23d_Hunyuan3D:
    def __str__(self):
        return 'hunyuan3d'
    
    def __init__(
        self,
        use_lite=False,
        mv23d_cfg_path="tool/i23d/Hunyuan3D-1/svrm/configs/svrm.yaml",
        mv23d_ckt_path="weights/svrm/svrm.safetensors",
        text2image_path="weights/hunyuanDiT",
        gpuid=0,
        save_memory=False,
        max_faces_num=120000,
        do_texture_mapping=False,
        do_render=False,
        do_bake=False,
        bake_align_times=3,
        bake_available_check=True
    ):
        """
        Initialize the i23d_Hunyuan3D model with specified configurations.

        Parameters:
            use_lite (bool): Whether to use lite models.
            mv23d_cfg_path (str): Path to the MV23D configuration file.
            mv23d_ckt_path (str): Path to the MV23D checkpoint file.
            text2image_path (str): Path to the Text2Image model.
            gpuid (int): GPU ID to run the models on (e.g., 0 for 'cuda:0').
            save_memory (bool): Whether to save memory during processing.
            max_faces_num (int): Maximum number of faces in the mesh.
            do_texture_mapping (bool): Whether to perform texture mapping.
            do_render (bool): Whether to render the final mesh as a GIF.
            do_bake (bool): Whether to perform baking.
            bake_align_times (int): Number of alignment times for baking.
            bake_available_check (bool): Whether to check baking availability.
        """
        self.use_lite = use_lite
        self.mv23d_cfg_path = mv23d_cfg_path
        self.mv23d_ckt_path = mv23d_ckt_path
        self.text2image_path = text2image_path
        self.device = f'cuda:{gpuid}'
        self.save_memory = save_memory
        self.max_faces_num = max_faces_num
        self.do_texture_mapping = do_texture_mapping
        self.do_render = do_render
        self.do_bake = do_bake
        self.bake_align_times = bake_align_times

        # Initialize baking if available
        self.bake_available = False
        if self.do_bake and bake_available_check:
            try:
                self.mesh_baker = MeshBaker(
                    device=self.device,
                    align_times=self.bake_align_times
                )
                self.bake_available = True
            except Exception as err:
                print(err)
                print("Import baking related modules failed, running without baking.")
                self.bake_available = False

        # Initialize GIF renderer if baking is available
        if check_bake_available():
            self.gif_renderer = GifRenderer(device=self.device)
        else:
            self.gif_renderer = None

        # Initialize models
        self._initialize_models()

    def _initialize_models(self):
        """Initialize all necessary models."""
        st = time.time()
        self.rembg_model = Removebg()
        self.image_to_views_model = Image2Views(
            device=self.device,
            use_lite=self.use_lite,
            save_memory=self.save_memory
        )
        
        self.views_to_mesh_model = Views2Mesh(
            self.mv23d_cfg_path,
            self.mv23d_ckt_path,
            self.device,
            use_lite=self.use_lite,
            save_memory=self.save_memory
        )
        
        # Initialize Text2Image model if needed
        self.text_to_image_model = Text2Image(
            pretrain=self.text2image_path,
            device=self.device,
            save_memory=self.save_memory
        )
        
        print(f"Initialized models in {time.time() - st:.2f} seconds.")

    def inference(
        self,
        image_prompt,
        output_obj_path,
        text_prompt="",
        t2i_seed=0,
        t2i_steps=25,
        gen_seed=0,
        gen_steps=50
    ):
        """
        Perform inference to generate a 3D mesh from an image or text prompt.

        Parameters:
            image_prompt (str): Image path for image-to-views processing.
            output_obj_path (str): Path to save the generated .obj file.
            text_prompt (str, optional): Text prompt for image generation. Defaults to "".
            t2i_seed (int, optional): Seed for text-to-image generation. Defaults to 0.
            t2i_steps (int, optional): Steps for text-to-image generation. Defaults to 25.
            gen_seed (int, optional): Seed for view generation. Defaults to 0.
            gen_steps (int, optional): Steps for view generation. Defaults to 50.

        Returns:
            str: Path to the generated .obj file.
        """
        assert not (text_prompt and image_prompt), "Provide either text or image prompt, not both."
        assert text_prompt or image_prompt, "Either text or image prompt must be provided."

        # Determine output directory based on output_obj_path
        output_dir = os.path.dirname(output_obj_path)
        os.makedirs(output_dir, exist_ok=True)

        # Stage 1: Text to Image (if text_prompt is provided)
        if text_prompt:
            print("Stage 1: Generating image from text prompt...")
            res_rgb_pil = self.text_to_image_model(
                text_prompt,
                seed=t2i_seed,
                steps=t2i_steps
            )
            image_input_path = os.path.join(output_dir, "img.jpg")
            res_rgb_pil.save(image_input_path)
            print(f"Text-to-Image saved at {image_input_path}")
        else:
            print("Stage 1: Loading image from image prompt...")
            res_rgb_pil = Image.open(image_prompt)
            image_input_path = image_prompt

        # Stage 2: Remove Background
        print("Stage 2: Removing background...")
        res_rgba_pil = self.rembg_model(res_rgb_pil)
        img_nobg_path = os.path.join(output_dir, "img_nobg.png")
        res_rgba_pil.save(img_nobg_path)
        print(f"Background removed image saved at {img_nobg_path}")

        # Stage 3: Image to Views
        print("Stage 3: Generating views from image...")
        # Corrected unpacking here
        (views_grid_pil, cond_img), view_pil_list = self.image_to_views_model(
            res_rgba_pil,
            seed=gen_seed,
            steps=gen_steps
        )
        views_grid_path = os.path.join(output_dir, "views.jpg")
        views_grid_pil.save(views_grid_path)
        print(f"Views grid saved at {views_grid_path}")

        # Stage 4: Views to Mesh
        print("Stage 4: Generating mesh from views...")
        self.views_to_mesh_model(
            views_grid_pil,
            cond_img,
            seed=gen_seed,
            target_face_count=self.max_faces_num,
            save_folder=output_dir,
            do_texture_mapping=self.do_texture_mapping
        )
        print("Mesh generation completed.")

        # Locate the generated mesh file
        generated_mesh_path = os.path.join(output_dir, 'mesh_vertex_colors.obj')
        if not os.path.exists(generated_mesh_path):
            baked_fld_list = sorted(glob(os.path.join(output_dir, 'view_*/bake/mesh.obj')))
            if baked_fld_list:
                generated_mesh_path = baked_fld_list[-1]
            else:
                raise FileNotFoundError("Generated mesh_vertex_colors.obj not found.")

        # Stage 5: Baking (Optional)
        mesh_file_for_render = None
        if self.do_bake and self.bake_available:
            print("Stage 5: Baking mesh...")
            mesh_file_for_render = self.mesh_baker(output_dir)
            print(f"Baked mesh saved at {mesh_file_for_render}")

        # Move or copy the generated mesh to the desired output_obj_path
        if generated_mesh_path != output_obj_path:
            os.rename(generated_mesh_path, output_obj_path)
            print(f"Mesh moved to {output_obj_path}")
        else:
            print(f"Mesh saved at {output_obj_path}")

        # Stage 6: Render GIF (Optional) - Commented Out as per user request
        # if self.do_render:
        #     print("Stage 6: Rendering GIF...")
        #     if mesh_file_for_render and os.path.exists(mesh_file_for_render):
        #         mesh_to_render = mesh_file_for_render
        #     else:
        #         baked_fld_list = sorted(glob(os.path.join(output_dir, 'view_*/bake/mesh.obj')))
        #         if baked_fld_list:
        #             mesh_to_render = baked_fld_list[-1]
        #         else:
        #             mesh_to_render = os.path.join(output_dir, 'mesh.obj')
        #         assert os.path.exists(mesh_to_render), f"{mesh_to_render} file not found"

        #     print(f"Rendering 3D file: {mesh_to_render}")
        #     gif_output_path = os.path.join(output_dir, 'output.gif')
        #     self.gif_renderer(
        #         mesh_to_render,
        #         gif_dst_path=gif_output_path,
        #     )
        #     print(f"Rendered GIF saved at {gif_output_path}")

        # print("Inference completed successfully.")
        return output_obj_path  # Return the path to the generated .obj file



class i23d_stable_fast_3d(image23d):
    def __str__(self):
        return 'stable-fast-3d'

    def __init__(self, model_path='stabilityai/stable-fast-3d', gpuid=1):
        super().__init__(model_path, gpuid)
        self.device = f'cuda:{gpuid}'
        self.model = self._load_model()
        self.rembg_session = rembg.new_session()

    def _load_model(self):
        print("Loading model...")
        model = SF3D.from_pretrained(
            self.model_path,
            config_name="config.yaml",
            weight_name="model.safetensors",
        )
        model.to(self.device)
        model.eval()
        return model

    def inference(self, png_path, output_path, foreground_ratio=0.85, texture_resolution=1024, remesh_option="none", target_vertex_count=-1, batch_size=1):
        output_dir, output_file = os.path.split(output_path)
        os.makedirs(output_dir, exist_ok=True) 
        output_base, _ = os.path.splitext(output_file) 

        images = []

        def handle_image(image_path, idx):
            image = remove_background(
                Image.open(image_path).convert("RGBA"), self.rembg_session
            )
            image = resize_foreground(image, foreground_ratio)
            images.append(image)

        if os.path.isdir(png_path):
            image_paths = [
                os.path.join(png_path, f)
                for f in os.listdir(png_path)
                if f.endswith((".png", ".jpg", ".jpeg"))
            ]
            for idx, image_path in enumerate(image_paths):
                handle_image(image_path, idx)
        else:
            handle_image(png_path, 0)

        for i in tqdm(range(0, len(images), batch_size)):
            batch_images = images[i:i + batch_size]
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            with torch.no_grad(): 
                mesh, glob_dict = self.model.run_image(
                    batch_images,
                    bake_resolution=texture_resolution,
                    remesh=remesh_option,
                    vertex_count=target_vertex_count,
                )
            if torch.cuda.is_available():
                print("Peak Memory:", torch.cuda.max_memory_allocated() / 1024 / 1024, "MB")
            elif torch.backends.mps.is_available():
                print(
                    "Peak Memory:", torch.mps.driver_allocated_memory() / 1024 / 1024, "MB"
                )

            if len(batch_images) == 1:
                out_mesh_path = os.path.join(output_dir, f"{output_base}.glb")
                mesh.export(out_mesh_path, include_normals=True)
            else:
                for j in range(len(mesh)):
                    out_mesh_path = os.path.join(output_dir, f"{output_base}_{i + j}.glb")
                    mesh[j].export(out_mesh_path, include_normals=True)

        return out_mesh_path

