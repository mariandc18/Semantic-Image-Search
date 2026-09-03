import streamlit as st
from pathlib import Path
from src.search import ImageSearch
from config import IMAGES_DIR
from PIL import Image

st.set_page_config(page_title="Image Similarity Search", layout="wide")

st.title("Image Similarity Search")


if 'search_engine' not in st.session_state:
    st.session_state.search_engine = ImageSearch()
    st.session_state.indexed = False

search_engine = st.session_state.search_engine

with st.sidebar:
    st.header("Configuración")
    
    if st.button("Indexar Imágenes"):
        with st.spinner("Indexando imágenes..."):
            search_engine.index_images()
            st.session_state.indexed = True
        st.success("Imágenes indexadas")
    
    top_k = st.slider("Resultados a mostrar", 1, 10, 5)
    
    if st.session_state.indexed:
        st.info("Listo")
    else:
        st.warning("Indexa las imágenes primero")

if not st.session_state.indexed:
    st.warning("Por favor, indexa tus imágenes usando el botón en la barra lateral.")
else:
    tab1, tab2 = st.tabs(["Buscar por Texto", "Buscar por Imagen"])
    
    with tab1:
        st.subheader("Búsqueda por texto")
        query = st.text_input("Describe lo que buscas:", placeholder="ej: cualquier cosa")
        
        if query:
            with st.spinner("Buscando..."):
                results = search_engine.search_by_text(query, top_k=top_k)
            
            if results:
                st.success(f"{len(results)} resultados")
                cols = st.columns(min(len(results), 3))
                
                for idx, result in enumerate(results):
                    with cols[idx % 3]:
                        img = Image.open(result['path'])
                        st.image(img, use_column_width=True)
                        st.caption(f"ID: {result['id']}\nSimilitud: {1-result['distance']:.2%}")
            else:
                st.info("No se encontraron resultados")
    
    with tab2:
        st.subheader("Búsqueda por imagen")
        uploaded_file = st.file_uploader("Sube una imagen", type=['jpg', 'jpeg', 'png', 'gif'])
        
        if uploaded_file:
            temp_path = Path("temp_upload.png")
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Tu imagen")
                st.image(uploaded_file, use_column_width=True)
            
            with col2:
                with st.spinner("Buscando imágenes similares..."):
                    results = search_engine.search_by_image(str(temp_path), top_k=top_k)
                
                if results:
                    st.success(f"{len(results)} resultados similares")
                    cols = st.columns(min(len(results), 2))
                    
                    for idx, result in enumerate(results):
                        with cols[idx % 2]:
                            img = Image.open(result['path'])
                            st.image(img, use_column_width=True)
                            st.caption(f"ID: {result['id']}\nSimilaridad: {1-result['distance']:.2%}")
                else:
                    st.info("No se encontraron imágenes similares")
            
            temp_path.unlink()