import plotly.graph_objects as go
import trimesh
import numpy as np

def trimesh_to_plotly(mesh_or_scene):
    """
    Converteix un Trimesh o Scene a una figura plotly 3D, aplicant les transformacions.
    Manté la proporció real dels eixos.
    """
    meshes = []

    if isinstance(mesh_or_scene, trimesh.Scene):
        # Cal aplicar transformacions a cada geometria de la escena
        for name, geom in mesh_or_scene.geometry.items():
            # Copiem el mesh i apliquem la transformació de la escena
            transform = mesh_or_scene.graph.get(name)[0]  # transformació absoluta
            mesh_transformed = geom.copy()
            mesh_transformed.apply_transform(transform)
            meshes.append(mesh_transformed)
    elif isinstance(mesh_or_scene, trimesh.Trimesh):
        meshes.append(mesh_or_scene.copy())
    else:
        raise ValueError("El tipus ha de ser Trimesh o Scene")

    fig = go.Figure()
    all_vertices = np.empty((0,3), dtype=np.float64)

    for mesh in meshes:
        if mesh.faces.shape[0] == 0:
            continue
        vertices = np.array(mesh.vertices, dtype=np.float64)
        faces = np.array(mesh.faces, dtype=np.int64)
        x, y, z = vertices.T
        i, j, k = faces.T

        fig.add_trace(
            go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                opacity=0.7,
                color='lightblue',
                flatshading=True
            )
        )

        all_vertices = np.vstack([all_vertices, vertices])

    # Calcular rangs globals per eixos
    min_xyz = all_vertices.min(axis=0)
    max_xyz = all_vertices.max(axis=0)
    rang_max = max(max_xyz - min_xyz)
    center = (max_xyz + min_xyz)/2
    x_range = [center[0]-rang_max/2, center[0]+rang_max/2]
    y_range = [center[1]-rang_max/2, center[1]+rang_max/2]
    z_range = [center[2]-rang_max/2, center[2]+rang_max/2]

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', range=x_range),
            yaxis=dict(title='Y', range=y_range),
            zaxis=dict(title='Z', range=z_range),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1)
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    return fig
