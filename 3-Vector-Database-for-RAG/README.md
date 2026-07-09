# Vector Database for RAG (Retrieval-Augmented Generation)

This repository contains educational materials, cheat sheets, and hands-on Jupyter notebooks focused on **Vector Databases**, **Similarity Search**, and their applications in **Retrieval-Augmented Generation (RAG)** systems. These materials are part of the Coursera course on Vector Databases for RAG.

---

## 📂 Repository Structure

The repository is organized into a single folder containing all course-related materials:

### 📔 Jupyter Notebooks (`Course_content/`)
* **[Similarity Search by Hand.ipynb](file:///c:/Users/dell/OneDrive/Documents/Coursera%20Courses/Vector-Database-for-RAG/Course_content/Similarity%20Search%20by%20Hand.ipynb)**
  * **Objective:** Learn the mathematical fundamentals behind vector similarity.
  * **Details:** Manually compute distance and similarity metrics (Euclidean/L2 Distance, Dot Product, Cosine Similarity/Distance) using `numpy`, `scipy`, and `PyTorch` to understand how embeddings are compared in high-dimensional spaces.
* **[similarity_search_with_chromadb.ipynb](file:///c:/Users/dell/OneDrive/Documents/Coursera%20Courses/Vector-Database-for-RAG/Course_content/similarity_search_with_chromadb.ipynb)**
  * **Objective:** Implement a basic vector database setup.
  * **Details:** Store grocery-related item texts, generate embeddings with Sentence Transformers (`all-MiniLM-L6-v2`), configure a collection using the Cosine distance space, and run nearest-neighbor similarity search queries.
* **[similarity_employeedata.ipynb](file:///c:/Users/dell/OneDrive/Documents/Coursera%20Courses/Vector-Database-for-RAG/Course_content/similarity_employeedata.ipynb)**
  * **Objective:** Build a practical semantic retrieval application.
  * **Details:** Insert comprehensive employee profiles (metadata & text) into ChromaDB, perform similarity search, and apply advanced metadata filters (using `$eq`, `$lt`, `$and`, `$or` logical operators) and full-text document filtering.

### 📝 Concept & Syntax Cheat Sheets (`Course_content/`)
* **[VectorDb_Chromadb_cheatsheet.md](file:///c:/Users/dell/OneDrive/Documents/Coursera%20Courses/Vector-Database-for-RAG/Course_content/VectorDb_Chromadb_cheatsheet.md)**
  * In-depth reference for distance metrics (L2, Cosine, Inner Product).
  * Explains vector indexing, specifically the **HNSW (Hierarchical Navigable Small World)** algorithm, and key configuration parameters.
  * Full syntax reference for ChromaDB operations including collection creation, CRUD operations, metadata filtering operators, and document content filtering.
* **[VectorDb_for_rag_cheatsheet.md](file:///c:/Users/dell/OneDrive/Documents/Coursera%20Courses/Vector-Database-for-RAG/Course_content/VectorDb_for_rag_cheatsheet.md)**
  * Explains Retrieval-Augmented Generation (RAG) framework, pipeline steps, and key pitfalls to avoid (e.g., mismatched embedding models, poor chunking).
  * Provides quick-start Python recipes for creating collections, adding, retrieving, updating, and deleting documents in ChromaDB.

---

## 🧠 Key Concepts & Learning Outcomes

### 1. Similarity & Distance Metrics
Understanding when to use each metric depending on your data distribution:
* **L2 Distance (Euclidean):** Measures straight-line distance. Highly sensitive to both vector magnitude and direction. Best for spatial or clustering tasks.
* **Dot Product (Inner Product):** Sensitive to direction and magnitude. Best when the length of the vector carries significance (e.g., popularity or confidence in recommendation systems).
* **Cosine Similarity/Distance:** Focuses entirely on vector orientation/angle rather than length. Ideal for high-dimensional, sparse text embeddings (NLP).

### 2. Hierarchical Navigable Small World (HNSW) Indexing
ChromaDB's primary approximate nearest neighbor (ANN) search graph structure:
* **`space`**: Distance metric used (default is `l2`, options: `cosine`, `ip`).
* **`ef_search` / `ef_construction`**: Parameters that trade off indexing speed/build time against query accuracy.
* **`max_neighbors` (M)**: Connective structure density controlling memory consumption and recall rate.

### 3. RAG Pipeline Framework
* **Document Chunking & Embedding:** Preparing text snippets and converting them into dense vectors.
* **Storage & Indexing:** Saving vectors along with metadata to enable filtering and fast retrieval.
* **Prompt Augmentation:** Feeding the retrieved context to an LLM to generate precise responses and prevent hallucinations.

---

## 🚀 Getting Started

### Prerequisites
To run the notebooks and Python scripts, ensure you have the following installed:
* Python 3.8+
* Jupyter Notebook / JupyterLab

### Installation
Clone the repository and install the required dependencies:
```bash
pip install numpy scipy torch chromadb sentence-transformers
```

### Running the Notebooks
Start the Jupyter Notebook server:
```bash
jupyter notebook
```
Navigate to the `Course_content/` folder and open any notebook to begin learning.

### Quick ChromaDB Example
Here is a quick snippet demonstrating how to initialize ChromaDB, configure it with `all-MiniLM-L6-v2` embeddings, and run a vector similarity query:

```python
import chromadb
from chromadb.utils import embedding_functions

# 1. Define embedding function
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# 2. Initialize in-memory Client
client = chromadb.Client()

# 3. Create a collection configured for Cosine similarity
collection = client.create_collection(
    name="quickstart_collection",
    configuration={
        "hnsw": {"space": "cosine"},
        "embedding_function": ef
    }
)

# 4. Add documents with IDs and optional Metadata
collection.add(
    documents=[
        "Retrieval-Augmented Generation optimizes LLM responses.",
        "Euclidean distance is sensitive to vector magnitudes.",
        "ChromaDB is an open-source AI-native vector database."
    ],
    metadatas=[
        {"topic": "RAG"},
        {"topic": "Math"},
        {"topic": "Database"}
    ],
    ids=["doc1", "doc2", "doc3"]
)

# 5. Query the collection
results = collection.query(
    query_texts=["How to improve LLM generation?"],
    n_results=1
)

print("Best Match ID:", results["ids"][0][0])
print("Best Match Text:", results["documents"][0][0])
print("Cosine Distance:", results["distances"][0][0])
```

---

## 👤 Author
* **Wojciech "Victor" Fulmyk** (Course Instructor)
