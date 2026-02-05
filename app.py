import streamlit as st
from generador import generar_pala
from utils import trimesh_to_plotly  # Funció per convertir el mesh a plotly

st.set_page_config(page_title="Generador Pala 3D", layout="wide")
st.title("Generador de Pala 3D")

# --- Sidebar amb tots els paràmetres ---
st.sidebar.header("Paràmetres de la pala")

scale_perfil = st.sidebar.number_input("Scale Perfil", min_value=0.01, value=1.0)
m = st.sidebar.number_input("m", min_value=0.0, value=0.1)
p = st.sidebar.number_input("p", min_value=0.0, value=0.4)
t_max = st.sidebar.number_input("t_max", min_value=0.0, value=0.2)
alpha = st.sidebar.number_input("alpha (graus)", min_value=0.0, max_value=90.0, value=5.0)
prec_perfil = st.sidebar.number_input("prec_perfil", min_value=1, value=50)
x0 = st.sidebar.number_input("x0", value=0.0)
x1 = st.sidebar.number_input("x1", value=1.0)
x_max = st.sidebar.number_input("x_max", value=1.0)
y_max = st.sidebar.number_input("y_max", value=0.5)
prec_corba = st.sidebar.number_input("prec_corba", min_value=1, value=50)

# Botó per generar la pala
if st.sidebar.button("Generar pala 3D"):
    # Cridem la funció amb els paràmetres introduïts
    mesh_pala = generar_pala(
        scale_perfil,
        m,
        p,
        t_max,
        alpha,
        prec_perfil,
        x0,
        x1,
        x_max,
        y_max,
        prec_corba
    )

    # Convertim a plotly i mostrem
    fig = trimesh_to_plotly(mesh_pala)
    st.subheader("Visualització 3D de la pala")
    st.plotly_chart(fig, use_container_width=True)

    # --- Botó per descarregar com STL ---
    st.download_button(
        label="Descarregar pala com STL",
        data=mesh_pala.export(file_type="stl"),  # Exportem a STL
        file_name="pala_3D.stl",
        mime="application/sla"
    )
