
from diffusers import DiffusionPipeline
import torch

class t2i_sdxl():
    def __str__(self):
        return 'SD-XL'

    def __init__(self, sdxl_base_path='stabilityai/stable-diffusion-xl-base-1.0', sdxl_refiner_path='stabilityai/stable-diffusion-xl-refiner-1.0', gpuid=1,variant="fp16"):
        self.sdxl_base_path=sdxl_base_path
        self.sdxl_refiner_path=sdxl_refiner_path
        self.gpuid=gpuid
        # load both base & refiner
        self.base = DiffusionPipeline.from_pretrained(
            sdxl_base_path, 
            torch_dtype=torch.float32,
            # variant="fp16", 
            use_safetensors=True
        )
        self.base.to(f"cuda:{gpuid}")
        self.refiner = DiffusionPipeline.from_pretrained(
            sdxl_refiner_path,
            text_encoder_2=self.base.text_encoder_2,
            vae=self.base.vae,
            torch_dtype=torch.float32,
            use_safetensors=True,
            # variant="fp16",
        )
        self.refiner.to(f"cuda:{gpuid}")

    def inference(self, prompt):
        # Define how many steps and what % of steps to be run on each experts (80/20) here
        n_steps = 40
        high_noise_frac = 0.8

        # run both experts
        image = self.base(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images
        image = self.refiner(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            image=image,
        ).images[0]
        
        return image
  
    pass


class t2i_sdxl_replicate():
    def __init__(self, replicate_key='your replicate_key, see https://replicate.com/stability-ai/sdxl/api'):
        self.replicate_key=replicate_key
        

    def inference(self, prompt):
        import replicate
        from PIL import Image
        import os

        replicate = replicate.Client(api_token=self.replicate_key)

        input = {
            "width": 1024,
            "height": 1024,
            "prompt": prompt,
            "refine": "expert_ensemble_refiner",
            "apply_watermark": False,
            "num_inference_steps": 25
        }

        output = replicate.run(
            "stability-ai/sdxl:xxxxx-xx-xx-xx-xx-xx",
            input=input
        )

        response = requests.get(output[0])
        image_data = BytesIO(response.content)
        image = Image.open(image_data)
        
        return image

    
    pass



from diffusers import FluxPipeline
import torch
from PIL import Image

class t2i_flux:
    def __str__(self):
        return 'FLUX-1-DEV'
    
    def __init__(self, model_path="black-forest-labs/FLUX.1-dev", gpuid=0, torch_dtype="bfloat16"):
        """
        Initialize the Flux pipeline.
        
        Args:
            model_path (str): The path to the pretrained model.
            gpuid (int): The GPU ID to use. Defaults to 0.
            torch_dtype (str): The torch data type. Defaults to 'bfloat16'.
        """
        self.model_path = model_path
        self.gpuid = gpuid
        self.torch_dtype = torch.bfloat16 if torch_dtype == "bfloat16" else torch.float32
        
        # Load the model
        self.pipe = FluxPipeline.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype
        )
        
        if gpuid is not None:
            self.pipe.to(f"cuda:{gpuid}")
        else:
            self.pipe.enable_model_cpu_offload()
    
    def inference(self, prompt, height=1024, width=1024, guidance_scale=3.5, num_inference_steps=50, max_sequence_length=512, seed=0):
        """
        Generate an image from a text prompt.
        
        Args:
            prompt (str): The text prompt for image generation.
            height (int): The height of the generated image. Defaults to 1024.
            width (int): The width of the generated image. Defaults to 1024.
            guidance_scale (float): The guidance scale for text conditioning. Defaults to 3.5.
            num_inference_steps (int): The number of inference steps. Defaults to 50.
            max_sequence_length (int): The maximum sequence length for the prompt. Defaults to 512.
            seed (int): The random seed for reproducibility. Defaults to 0.
        
        Returns:
            PIL.Image.Image: The generated image.
        """
        generator = torch.Generator("cpu").manual_seed(seed)
        
        image = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            generator=generator,
        ).images[0]
        
        return image

