import os
from http import HTTPStatus
from OpenGL.GL import *
from OpenGL.GLU import *
import re
import json
from io import BytesIO


class Memory():
    idea_input_content = {}  # Store info about img and obj, and render the image
    idea_input_prompt = ''  # Initial idea after replacement
    idea_input_prompt_init = ''  # Initial idea

    best_prompt = None
    best_img = None
    best_3d_path = None
    best_3d_img_path = None  # Rendered image merged into one
    
    feedback = ''
    
    def __init__(self, idea_input_prompt_init):
        self.idea_input_prompt_init = idea_input_prompt_init

    def get_input_info_path_list(self):
        input_info_path_list = []
        for key, value in self.idea_input_content.items():
            if value["type"] == "IMG":
                input_info_path_list.append(value["file_path"])
            elif value["type"] == "OBJ":
                input_info_path_list.append(value["combined_image"])
        return input_info_path_list
    

    def get_idea_input_content_desc(self):
        """
        Generate description. Describe the path of each image or 3D object.
        """
        desc_list = []
        for i, (key, value) in enumerate(self.idea_input_content.items()):
            if value["type"] == "IMG":
                desc = f"The {i+2}th image is from <User Input> with <IMG>{key}</IMG>"
                desc_list.append(desc)
            elif value["type"] == "OBJ":
                desc = f"The {i+2}th image is the rendered image of a 3D object from <User Input> with <OBJ>{key}</OBJ>"
                desc_list.append(desc)
        return desc_list

        

class Iter():
    def __init__(self, index):
        self.index = index
    idea_input_imglist = []
    prompt = []
    draft_img = []
    draft_3d_path = []
    best_img = None
    best_3d_path = ''
    best_prompt = ''
    best_3d_img_path = []
    
    def clear(self):
        self.idea_input_imglist = []
        self.prompt = []
        self.draft_img = []
        self.draft_3d_path = []
        self.best_img = None
        self.best_3d_path = ''
        self.best_prompt = ''
        self.best_3d_img_path = []
    
