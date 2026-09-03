# Semantic Image Search

Motor de búsqueda semántica sobre colecciones de imágenes locales. Permite encontrar imágenes por descripción de texto o por similitud con otra imagen, usando embeddings de **CLIP** almacenados en **ChromaDB**.

---

## 📁 Estructura

```
├── src/
│   ├── embeddings.py     # CLIPEmbedder 
│   ├── db_manager.py     # ChromaManager 
│   └── search.py   # Pipeline principal
├── data/
│   ├── images/           # Imágenes a indexar
│   └── chroma_db/        # Base de datos vectorial
└── config.py             # Configuración 
```
