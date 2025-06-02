import open3d as o3d

class MeshSimplifier:
    def __init__(self, mesh=None, file_path=None, target_triangles=None, decimation_ratio=None, output_path=None):
        """
        Initialize the MeshSimplifier instance.

        Parameters:
            file_path (str): Path to the input mesh (e.g., an OBJ file).
            target_triangles (int, optional): Desired number of triangles after simplification.
            decimation_ratio (float, optional): Ratio (between 0 and 1) of original triangles to keep.
            output_path (str, optional): Path to save the simplified mesh.
                                         Defaults to a modified input filename.

        Note:
            - Either target_triangles or decimation_ratio must be provided.
            - If both are provided, target_triangles will be used.
        """
        if target_triangles is None and decimation_ratio is None:
            raise ValueError("Either target_triangles or decimation_ratio must be provided.")

        if decimation_ratio is not None:
            if not (0 < decimation_ratio < 1):
                raise ValueError("The decimation_ratio must be between 0 and 1 (non-inclusive).")
        
        self.file_path = file_path
        self.target_triangles = target_triangles
        self.decimation_ratio = decimation_ratio
        self.output_path = output_path #or file_path.replace(".obj", "_simplified.obj")
        
        # Load the mesh from file.
        if mesh is not None:
            self.mesh = mesh
        elif file_path is not None:
            self.mesh = o3d.io.read_triangle_mesh(self.file_path)
        else:
            raise ValueError("Either a mesh or a file_path must be provided.")
        if self.mesh.is_empty():
            raise ValueError(f"Failed to load mesh from {self.file_path}")
        
        self.simplified_mesh = None

    def simplify(self):
        if self.mesh.is_empty():
            raise RuntimeError("Mesh is empty. Cannot simplify an empty mesh.")

        # Determine the target number of triangles.
        if self.target_triangles is not None:
            target_count = self.target_triangles
        elif self.decimation_ratio is not None:
            original_triangle_count = len(self.mesh.triangles)
            target_count = int(original_triangle_count * (1 - self.decimation_ratio))
        else:
            # This branch should never be reached because of validation in __init__.
            raise RuntimeError("No valid simplification parameter provided.")

        # Perform the mesh simplification.
        self.simplified_mesh = self.mesh.simplify_quadric_decimation(
            target_number_of_triangles=target_count
        )
        return self.simplified_mesh
    
    def write_mesh(self):
        """
        Write the simplified mesh to disk.
        """
        if self.simplified_mesh is None:
            raise RuntimeError("You must run simplify() before writing the mesh to a file.")
        
        success = o3d.io.write_triangle_mesh(self.output_path, self.simplified_mesh)
        if not success:
            raise IOError(f"Failed to write simplified mesh to {self.output_path}")
        print(f"Simplified mesh written to {self.output_path}")
