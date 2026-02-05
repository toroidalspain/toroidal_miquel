import matplotlib.pyplot as plt
import numpy as np
import trimesh

def perfil_NACA(m, p, t, prec_perfil, alpha):

    # Vector de posició al llarg del perfil XY
    x = np.linspace(0, 1, prec_perfil)
    yt = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)

    # Camber i tangent
    yc = np.where(x <= p, (m / p**2)*(2*p*x - x**2),
                  (m / (1-p)**2)*((1-2*p)+2*p*x - x**2))
    theta = np.where(x <= p, (2*m / p**2)*(p - x),
                     (2*m/(1-p)**2)*(p - x))
    theta = np.arctan(theta)
    
    # Coordenades superior i inferior
    xu = x - yt*np.sin(theta)
    yu = yc + yt*np.cos(theta)
    xl = x + yt*np.sin(theta)
    yl = yc - yt*np.cos(theta)
    
    # Concatenar superior + inferior invertida
    perfil_x = np.concatenate((xu, xl[::-1]))
    perfil_y = np.concatenate((yu, yl[::-1]))
    
    # Centrar el perfil al seu centre
    perfil_x -= 0.5
    perfil_y -= np.mean(perfil_y)

    # Rotació alpha dins del pla XY (sobre Z)
    R_alpha = np.array([[np.cos(alpha), -np.sin(alpha)],
                        [np.sin(alpha),  np.cos(alpha)]])
    perfil_rot = R_alpha @ np.vstack((perfil_x, perfil_y))

    perfil_rot[0] -= np.mean(perfil_rot[0])
    perfil_rot[1] -= np.mean(perfil_rot[1])

    return perfil_rot

def corba(x0, x1, x_max, y_max, prec_corba):
    y_min = 0           # Valor dels mínims
    x_obj = (x_max-0.25*x0-0.25*x1)*2        # Posició del màxim objectiu (pot estar fora de [0,1])
    y_obj = y_max*2          # Valor del màxim objectiu

    # Parametre t (punts de la corba)
    t = np.linspace(0, 1, prec_corba)

    # Funció paramètrica per la corba
    # Usarem una forma quadràtica simple per fer la "paràbola-like" lliure
    # Corba tipus Bézier amb tres punts: mínim esquerra, màxim, mínim dreta
    x_cor = (1-t)**2 * x0 + 2*(1-t)*t*x_obj + t**2 * x1
    y_cor = (1-t)**2 * y_min + 2*(1-t)*t*y_obj + t**2 * y_min

    return x_cor, y_cor

def definir_mesh(X, Y, Z):

    n_slices, n_points = X.shape

    # 1. Aplanar els vèrtex
    vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # 2. Generar les cares connectant slices consecutives
    faces = []
    for i in range(n_slices - 1):
        for j in range(n_points - 1):
            # Índexs del quadrilàter
            v0 = i * n_points + j
            v1 = i * n_points + (j + 1)
            v2 = (i + 1) * n_points + (j + 1)
            v3 = (i + 1) * n_points + j
            # Dos triangles per quadrilàter
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])
        # opcional: tancar el perfil en circumferència
        v0 = i * n_points + (n_points - 1)
        v1 = i * n_points
        v2 = (i + 1) * n_points
        v3 = (i + 1) * n_points + (n_points - 1)
        faces.append([v0, v1, v2])
        faces.append([v0, v2, v3])

    faces = np.array(faces)

    # 3. Crear la malla de trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    return mesh

def rotate_3d_about_line(points, p0, v, theta):
    k = v / np.linalg.norm(v)
    p0 = np.array(p0)
    p_shifted = points - p0
    cross = np.cross(k, p_shifted)
    dot = np.dot(p_shifted, k)
    rotated = (
        p_shifted * np.cos(theta)
        + cross * np.sin(theta)
        + np.outer(dot, k) * (1 - np.cos(theta))
    )
    return rotated + p0

def generar_pala(scale_perfil, m, p, t_max, alpha, prec_perfil, x0, x1, x_max, y_max, prec_corba):

    alpha=np.deg2rad(-alpha)
    x_cor, y_cor = corba(x0, x1, x_max, y_max, prec_corba)
    # Inicialitzar arrays 3D
    X = np.ones((prec_corba, prec_perfil*2))
    Y = np.ones_like(X)
    Z = np.ones_like(X)

    # Desplaçament lateral (s'haurà de convertir a graus)
    d_lat = 0.5
    d = np.linspace(0, d_lat, prec_corba)
    theta=np.linspace(0, np.pi, prec_corba)

    for i in range(prec_corba):

        m_var = - m * 2 * (i/prec_corba - 0.5)

        perfil = perfil_NACA(m_var, p, t_max, prec_perfil, alpha)

        X[i] = perfil[0]*scale_perfil
        Y[i] = perfil[1]*scale_perfil
        Z[i] = 0

        # Convertim a punts Nx3
        points = np.vstack((X[i].ravel(), Y[i].ravel(), Z[i].ravel())).T
    
        # Definim línia de rotació
        p0 = np.array([0, 0, 0])
        v = np.array([np.cos(alpha), np.sin(alpha), 0])  # direcció

        # Rota tots els punts
        rotated = rotate_3d_about_line(points, p0, v, theta[i])

        # Reconstruïm les matrius
        X[i] = rotated[:, 0].reshape(X[i].shape)
        Y[i] = rotated[:, 1].reshape(Y[i].shape)
        Z[i] = rotated[:, 2].reshape(Z[i].shape)

        X[i] = X[i] + d[i]
        Y[i] = Y[i] - x_cor[i]
        Z[i] = Z[i] + y_cor[i] 

        mesh = definir_mesh(X,Y,Z)
        mesh.export("test.stl")

    return mesh

