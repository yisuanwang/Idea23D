import os
from PIL import Image
import time
from PIL import Image as PILImage
import matplotlib.pyplot as plt

def log(text, level='INFO'):
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f'\n[{current_time}][{level}][IDEA-2-3D]: {text}')

def show_image(path):
    # 替换为你的图片路径
    img = PILImage.open(path)
    plt.imshow(img)
    plt.axis('off')  # 不显示坐标轴
    plt.show()

def readimage(path):
    with open(path, 'rb') as file:
        image = Image.open(path).convert("RGB")
        resized_image = image.resize((256, 256))
    return resized_image

def writeimage(image, path):
    # Check if the directory exists, and create it if it doesn't
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save the image to the path
    with open(path, 'wb') as file:
        image.save(file, 'PNG')  # Use 'PNG' to ensure proper saving of PNG files


from PIL import Image
import numpy as np
import datetime
import os
import cv2

def concatenate_images_with_number_label(images_list, direction="h", output_folder=f'./outpath/tmp'):
    # Ensure all paths or images are loaded as PIL Image objects
    if all(isinstance(image, str) for image in images_list):
        try:
            images_list = [Image.open(image) for image in images_list]
        except Exception as e:
            raise ValueError(f"Error loading images: {e}")
    elif not all(isinstance(image, Image.Image) for image in images_list):
        raise ValueError("All items in images_list must be either file paths or PIL Image objects.")
    
    
    # Check direction parameter
    if direction not in ["h", "v"]:
        raise ValueError("Invalid direction parameter. It must be 'h' for horizontal or 'v' for vertical concatenation.")
    
    # Check output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Create the target folder if it doesn't exist
    
    # Convert PIL images to numpy arrays for processing
    np_images = [np.array(image) for image in images_list]
    
    # Check if we're concatenating horizontally or vertically and create a canvas
    if direction == "h":
        total_width = sum(image.size[0] for image in images_list)
        max_height = max(image.size[1] for image in images_list)
        concatenated_image = Image.new('RGB', (total_width, max_height))
    elif direction == "v":
        total_height = sum(image.size[1] for image in images_list)
        max_width = max(image.size[0] for image in images_list)
        concatenated_image = Image.new('RGB', (max_width, total_height))
    
    # Paste images onto the canvas
    x_offset, y_offset = 0, 0
    for image in images_list:
        concatenated_image.paste(image, (x_offset, y_offset))
        if direction == "h":
            x_offset += image.size[0]
        elif direction == "v":
            y_offset += image.size[1]
    
    # Save the image
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    output_path = os.path.join(output_folder, f"concatenated_image-{timestamp}.png")
    
    log(f'concatenated_image output_path={output_path}')
    concatenated_image.save(output_path)
    
    return concatenated_image
