import math
import os

import numpy as np
import torch
import trimesh
from PIL import Image
from pytorch3d.renderer import (AmbientLights, FoVPerspectiveCameras,
                                HardPhongShader, MeshRasterizer, MeshRenderer,
                                PointLights, RasterizationSettings, TexturesUV,
                                TexturesVertex, look_at_view_transform)
from pytorch3d.structures import Meshes


class MeshRenderer6View:
    def __init__(self, device=None, image_size=1024, distance=2.5, light=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.distance = distance

        self.raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        if light == 'PointLights':
            # 点光源
            self.lights = PointLights(device=self.device, location=[[0.0, 0.0, 3.0]])
        elif light == 'AmbientLights':
            # 使用环境光，提供均匀光照
            self.lights = AmbientLights(device=self.device)
        else:
            # 默认使用环境光或者点光源中的一种，不加判断也行
            # 这里选择默认环境光
            self.lights = AmbientLights(device=self.device)


    def compute_camera_distance(self, verts):
        """ 计算相机距离，使得模型填满画面 """
        bbox_min = verts.min(dim=0).values
        bbox_max = verts.max(dim=0).values
        bbox_center = (bbox_max + bbox_min) / 2
        bbox_size = (bbox_max - bbox_min).norm()

        # 设置相机距离，根据 FoV 计算
        fov = 58  # 默认相机视场角
        distance = bbox_size / (2 * torch.tan(torch.tensor(math.radians(fov / 2))))
        return bbox_center, distance
    
    def load_mesh_from_obj(self, mesh_path):
        verts = []
        verts_rgb = []
        faces = []

        with open(mesh_path, "r") as file:
            for line in file:
                if line.startswith("v "):  # 顶点行
                    parts = list(map(float, line.strip().split()[1:]))
                    verts.append(parts[:3])  # x, y, z
                    if len(parts) > 3:  # r, g, b
                        verts_rgb.append(parts[3:])
                elif line.startswith("f "):  # 面行
                    face = [int(idx.split("/")[0]) - 1 for idx in line.strip().split()[1:]]
                    faces.append(face)

        verts = torch.tensor(verts, dtype=torch.float32).to(self.device)
        faces = torch.tensor(faces, dtype=torch.int64).to(self.device)
        if verts_rgb:
            verts_rgb = torch.tensor(verts_rgb, dtype=torch.float32).to(self.device)
            textures = TexturesVertex(verts_features=verts_rgb.unsqueeze(0))
        else:
            print("未找到顶点颜色信息，使用默认灰色")
            textures = TexturesVertex(verts_features=torch.ones_like(verts[:, :3]).unsqueeze(0) * 0.5)

        mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
        return mesh, verts


    def load_mesh_from_glb(self, mesh_path):
        scene = trimesh.load(mesh_path)
        if isinstance(scene, trimesh.Scene):
            geometries = list(scene.geometry.values())
            if len(geometries) == 0:
                print("No geometry found in the GLB file. Using a default placeholder mesh.")
                mesh_trimesh = trimesh.primitives.Sphere(radius=0.5)
            else:
                mesh_trimesh = geometries[0]
        else:
            mesh_trimesh = scene

        verts = torch.tensor(mesh_trimesh.vertices, dtype=torch.float32).to(self.device)
        faces = torch.tensor(mesh_trimesh.faces, dtype=torch.int64).to(self.device)

        uv_coords = None
        uv_faces = None
        texture_image_tensor = None  # Initialize here

        # Check if UV data exists
        if hasattr(mesh_trimesh.visual, 'uv') and mesh_trimesh.visual.uv is not None:
            uv_coords = torch.tensor(mesh_trimesh.visual.uv, dtype=torch.float32).to(self.device)
            uv_faces = torch.tensor(mesh_trimesh.faces, dtype=torch.int64).to(self.device)
            print(f"UV coordinates and faces loaded: {uv_coords.shape}, {uv_faces.shape}")
        else:
            print("No UV data found.")

        # Check material information
        if hasattr(mesh_trimesh.visual, 'material'):
            material = mesh_trimesh.visual.material
            print(f"Material details: {material}")
            if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
                texture_image = material.baseColorTexture
                print(f"Base color texture found: {type(texture_image)}")

                if isinstance(texture_image, Image.Image):
                    texture_image_np = np.array(texture_image)
                elif isinstance(texture_image, np.ndarray):
                    texture_image_np = texture_image
                else:
                    raise TypeError(f"Unsupported texture image type: {type(texture_image)}")

                texture_image_tensor = torch.tensor(texture_image_np, dtype=torch.float32) / 255.0  
                texture_image_tensor = texture_image_tensor.unsqueeze(0)  # (1, H, W, 3)
        else:
            print("No material or base color texture found.")

        # Construct textures based on UV data and texture_image_tensor
        if texture_image_tensor is not None and uv_coords is not None and uv_faces is not None:
            # Ensure uv_coords and uv_faces have a batch dimension
            uv_coords = uv_coords.unsqueeze(0)
            uv_faces = uv_faces.unsqueeze(0)

            texture_image_tensor = texture_image_tensor.to(self.device)
            uv_coords = uv_coords.to(self.device)
            uv_faces = uv_faces.to(self.device)

            textures = TexturesUV(
                maps=texture_image_tensor,  
                faces_uvs=uv_faces,        
                verts_uvs=uv_coords       
            )
        else:
            # White color
            textures = TexturesVertex(verts_features=torch.ones((1, verts.shape[0], 3), device=self.device))

            # Default gray if UV and texture data cannot be used
            # textures = TexturesVertex(verts_features=torch.ones((1, verts.shape[0], 3), device=self.device) * 0.5)

        mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
        return mesh, verts


    def load_mesh(self, mesh_path):
        # Determine the file type based on the file extension
        ext = os.path.splitext(mesh_path)[1].lower()
        if ext == '.obj':
            return self.load_mesh_from_obj(mesh_path)
        elif ext == '.glb':
            return self.load_mesh_from_glb(mesh_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")


    def render_views(self, mesh, verts, save_dir, views=6):
        os.makedirs(save_dir, exist_ok=True)
        bbox_center, self.distance = self.compute_camera_distance(verts)
        # self.distance *= 0.5  # Shorten the camera distance
        print(f'{self.distance=}')

        # Fixed 6 views: front, back, left, right, top, bottom
        view_positions = [
            (0, 0),    # Front view
            (180, 0),  # Back view
            (90, 0),   # Left view
            (-90, 0),  # Right view
            (0, 90),   # Top view
            (0, -90),  # Bottom view
        ]

        image_paths = []  # List to store paths of rendered images

        for i, (azim, elev) in enumerate(view_positions):
            R, T = look_at_view_transform(dist=self.distance, elev=elev, azim=azim, at=bbox_center)
            T = T.squeeze(0)  # Ensure T has the shape (N, 3)
            cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)

            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=self.raster_settings,
                ),
                shader=HardPhongShader(
                    device=self.device,
                    cameras=cameras,
                    lights=self.lights,
                ),
            )

            try:
                # Render the image
                rendered_image = renderer(mesh.extend(len(cameras)))[0, ..., :3].cpu().numpy()
            except Exception as e:
                print(f"Error during rendering view {i}: {e}")
                # If rendering fails, generate a white image as a placeholder
                rendered_image = np.ones((self.image_size, self.image_size, 3), dtype=np.uint8)

            # Save the rendered image
            img = (rendered_image * 255).astype("uint8")
            output_path = os.path.join(save_dir, f"view_{i}.png")
            Image.fromarray(img).save(output_path)
            print(f"Saved view {i} to {output_path}")
            # Add the path to the list
            image_paths.append(output_path)
            

        # Horizontally concatenate the six images
        if len(image_paths) == 6:
            images = [Image.open(path) for path in image_paths]
            widths, heights = zip(*(img.size for img in images))
            
            # Calculate the total width and the maximum height of the concatenated image
            total_width = sum(widths)
            max_height = max(heights)
            
            # Create a blank image for concatenation
            combined_image = Image.new('RGB', (total_width, max_height))
            
            # Concatenate the images
            x_offset = 0
            for img in images:
                combined_image.paste(img, (x_offset, 0))
                x_offset += img.size[0]
            
            # Save the concatenated image
            combined_output_path = os.path.join(save_dir, "view_all.png")
            combined_image.save(combined_output_path)
            print(f"Saved combined image to {combined_output_path}")

        return image_paths, combined_output_path

    

    def render_views_evalclip(self, mesh, verts, views=4):
        """
        Render the mesh from a fixed number of views and return the rendered images.
        
        Args:
            mesh: The 3D mesh to render.
            verts: The vertices of the mesh.
            views: The number of views to render (default is 4).
        
        Returns:
            List of rendered images (PIL.Image objects).
        """
        bbox_center, self.distance = self.compute_camera_distance(verts)
        print(f'{self.distance=}')
        
        # Fixed views: Front, Back, Left, Right (you can modify or extend this list)
        view_positions = [
            (0, 0),    # Front view
            (180, 0),  # Back view
            (90, 0),   # Left view
            (-90, 0),  # Right view
            (0, 90),   # Top view (not included in the first 4)
            (0, -90),  # Bottom view (not included in the first 4)
        ]

        rendered_images = []  # List to store rendered images (as PIL.Image)

        for i, (azim, elev) in enumerate(view_positions[:views]):  # Limit to the first 'views' positions
            R, T = look_at_view_transform(dist=self.distance, elev=elev, azim=azim, at=bbox_center)
            T = T.squeeze(0)  # Ensure T is of shape (N, 3)
            cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)

            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=self.raster_settings,
                ),
                shader=HardPhongShader(
                    device=self.device,
                    cameras=cameras,
                    lights=self.lights,
                ),
            )

            try:
                # Render the image
                rendered_image = renderer(mesh.extend(len(cameras)))[0, ..., :3].cpu().numpy()
            except Exception as e:
                print(f"Error during rendering view {i}: {e}")
                # If rendering fails, create a white placeholder image
                rendered_image = np.ones((self.image_size, self.image_size, 3), dtype=np.uint8)

            # Convert to PIL Image and append to list
            img = (rendered_image * 255).astype("uint8")
            pil_image = Image.fromarray(img)
            rendered_images.append(pil_image)

        return rendered_images  # Return the list of PIL.Image objects


def render_images(input_path,output_dir,image_size=512,distance=2.5,light='AmbientLights'):
    renderer = MeshRenderer6View(image_size=512, distance=2.5, light='AmbientLights') # light = PointLights or AmbientLights
    mesh, verts = renderer.load_mesh(input_path)
    image_paths, combined_output_path = renderer.render_views(mesh, verts, output_dir)
    return image_paths, combined_output_path

