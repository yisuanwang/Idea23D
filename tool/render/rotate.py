import numpy as np

# Read .obj file
def read_obj_with_color(file_path):
    vertices = []
    faces = []
    materials = []
    mtllib = None

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):  # Vertex, possibly including color
                parts = line.strip().split()
                if len(parts) >= 4:
                    # Vertex may include color information
                    vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                    color = list(map(float, parts[4:])) if len(parts) > 4 else []
                    vertices.append(vertex + color)
            elif line.startswith('f '):  # Face
                parts = line.strip().split()
                faces.append([int(p.split('/')[0]) for p in parts[1:]])
            elif line.startswith('usemtl'):  # Material
                materials.append(line.strip())
            elif line.startswith('mtllib'):  # Material file path
                mtllib = line.strip()
    return np.array(vertices), faces, materials, mtllib

# Write a new .obj file
def write_obj_with_color(file_path, vertices, faces, materials, mtllib):
    with open(file_path, 'w') as file:
        # Write material file path
        if mtllib:
            file.write(f"{mtllib}\n")
        
        # Write vertices, preserving color information
        for v in vertices:
            if len(v) > 3:  # With color
                file.write(f"v {v[0]} {v[1]} {v[2]} {v[3]} {v[4]} {v[5]}\n")
            else:  # Without color
                file.write(f"v {v[0]} {v[1]} {v[2]}\n")
        
        # Write materials
        for material in materials:
            file.write(f"{material}\n")
        
        # Write faces
        for f in faces:
            file.write(f"f {' '.join(map(str, f))}\n")

# Rotation matrix
def rotate_vertices_TripoSR(vertices):
    # Define angles (in radians)
    theta_45 = np.pi / 4  # 45 degrees
    theta_10 = np.radians(20)  # Convert 10 degrees to radians
    cos_theta = np.cos(theta_10)
    sin_theta = np.sin(theta_10)

    # Counterclockwise 45° rotation matrix
    Rz_counterclockwise_45 = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta,  cos_theta, 0],
        [0,          0,         1]
    ])

    # Clockwise 45° rotation matrix
    Rz_clockwise_45 = np.array([
        [cos_theta, sin_theta, 0],
        [-sin_theta, cos_theta, 0],
        [0,          0,         1]
    ])
    Ry_counterclockwise_20 = np.array([
        [cos_theta, 0, sin_theta],
        [0,         1, 0],
        [-sin_theta, 0, cos_theta]
    ])

    # Clockwise 45° rotation matrix
    Ry_clockwise_45 = np.array([
        [cos_theta, 0, -sin_theta],
        [0,         1, 0],
        [sin_theta,  0, cos_theta]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    Rx2 = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])
    Ry = np.array([
        [0, 0, -1],
        [0, 1, 0],
        [1, 0, 0]
    ])
    Rz = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ])
    # Rz = np.array([
    #     [1, 0, 0],
    #     [0, 0, -1],
    #     [0, 1, 0]
    # ])
    theta_180 = np.radians(180)  # Convert 180 degrees to radians
    cos_theta = np.cos(theta_180)
    sin_theta = np.sin(theta_180)

    # 180° rotation matrix
    Ry_180 = np.array([
        [cos_theta, 0, sin_theta],
        [0,         1, 0],
        [-sin_theta, 0, cos_theta]
    ])
    # Total transformation matrix
    R = Ry_180 @ Ry_counterclockwise_20 @ Rx2 @ Rz

    
    # Extract vertex coordinates, apply rotation, and preserve original color information
    rotated_vertices = []
    for v in vertices:
        pos = np.array(v[:3])  # Take the first three as coordinates
        rotated_pos = np.dot(R, pos)  # Apply rotation
        rotated_vertices.append(list(rotated_pos) + list(v[3:]))  # Append color information (if any)
    
    return rotated_vertices

def rotate_vertices_InstantMesh(vertices):
    # Define angles (in radians)
    theta_180 = np.radians(180)  # Convert 180 degrees to radians
    cos_theta = np.cos(theta_180)
    sin_theta = np.sin(theta_180)

    # 180° rotation matrix along the Y-axis
    Ry_180 = np.array([
        [cos_theta, 0, sin_theta],
        [0,         1, 0],
        [-sin_theta, 0, cos_theta]
    ])

    # Other necessary rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])

    Ry = np.array([
        [0, 0, -1],
        [0, 1, 0],
        [1, 0, 0]
    ])

    # Total transformation matrix, applying Ry_180 at the end
    R = Ry_180 @ Ry @ Rx

    # Extract vertex coordinates, apply rotation, and preserve original color information
    rotated_vertices = []
    for v in vertices:
        pos = np.array(v[:3])  # Take the first three as coordinates
        rotated_pos = np.dot(R, pos)  # Apply rotation
        rotated_vertices.append(list(rotated_pos) + list(v[3:]))  # Append color information (if any)

    return rotated_vertices


def rotate_InstantMesh(input_file, output_file):
    vertices, faces, materials, mtllib = read_obj_with_color(input_file)
    rotated_vertices = rotate_vertices_InstantMesh(vertices)
    write_obj_with_color(output_file, rotated_vertices, faces, materials, mtllib)
    print(f"Rotation completed! Saved to {output_file}")

def rotate_TripoSR(input_file, output_file):
    vertices, faces, materials, mtllib = read_obj_with_color(input_file)
    rotated_vertices = rotate_vertices_TripoSR(vertices)
    write_obj_with_color(output_file, rotated_vertices, faces, materials, mtllib)
    print(f"Rotation completed! Saved to {output_file}")
