import argparse
import logging
import os
import time
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))


if current_dir not in sys.path:
    sys.path.append(current_dir)
    
import numpy as np
import rembg
import torch
from PIL import Image

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, save_video


class Timer:
    def __init__(self):
        self.items = {}
        self.time_scale = 1000.0  # ms
        self.time_unit = "ms"

    def start(self, name: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.items[name] = time.time()
        logging.info(f"{name} ...")

    def end(self, name: str) -> float:
        if name not in self.items:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = self.items.pop(name)
        delta = time.time() - start_time
        t = delta * self.time_scale
        logging.info(f"{name} finished in {t:.2f}{self.time_unit}.")

def TripoSRmain(gpuid, model_path, png_path, output_dir, render=True, no_remove_bg = False):
    timer = Timer()
    
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    device = f"cuda:{gpuid}" if torch.cuda.is_available() else "cpu"
    
    os.makedirs(output_dir, exist_ok=True)

    timer.start("Initializing model")
    model = TSR.from_pretrained(pretrained_model_name_or_path=model_path,
                                config_name="config.yaml",
                                weight_name="model.ckpt")
    model.renderer.set_chunk_size(8192)
    model.to(device)
    timer.end("Initializing model")

    timer.start("Processing image")
    if not os.path.isfile(png_path):
        logging.error(f"Provided path is not a file: {png_path}")
        return  

    if no_remove_bg:
        image = np.array(Image.open(png_path).convert("RGB"))
    else:
        rembg_session = rembg.new_session()
        image = remove_background(Image.open(png_path), rembg_session)
        image = resize_foreground(image, 0.85)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        image.save(os.path.join(output_dir, "input.png"))
    timer.end("Processing image")

    logging.info("Running model ...")
    timer.start("Running model")
    with torch.no_grad():
        scene_codes = model([image], device=device)
    timer.end("Running model")

    if render:
        timer.start("Rendering")
        render_images = model.render(scene_codes, n_views=6, return_type="pil")
        for ri, render_image in enumerate(render_images[0]):
            render_image.save(os.path.join(output_dir, f"render_{ri:03d}.png"))
        save_video(render_images[0], os.path.join(output_dir, "render.mp4"), fps=30)
        timer.end("Rendering")

    timer.start("Exporting mesh")
    meshes = model.extract_mesh(scene_codes, resolution=256)    
    meshes[0].export(os.path.join(output_dir, "mesh.obj"))
    timer.end("Exporting mesh")

