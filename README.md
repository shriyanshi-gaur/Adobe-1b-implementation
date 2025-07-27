# Persona-Driven Document Intelligence Pipeline

This project is a sophisticated pipeline designed to analyze a collection of PDF documents and extract the most relevant information based on a specific user persona and "job-to-be-done" (JTBD). It uses a combination of layout-aware machine learning for document chunking and modern language models for semantic relevance ranking.

## How It Works

The pipeline executes a multi-stage process to distill relevant information from a document corpus:

1.  **Input Loading**: The process begins by loading metadata from a `challenge1b_input.json` file, which defines the user `persona`, the `job_to_be_done`, and the list of documents to be analyzed.
2.  **Semantic Chunking**: Instead of arbitrarily splitting documents, the `DocumentProcessor` parses each PDF, analyzing features like font size, boldness, and spacing to identify headings. It uses a pre-trained classifier model from the `models_1a` directory to intelligently group text into meaningful sections based on the document's structure.
3.  **Similarity Ranking**: The `Analyzer` component combines the persona and JTBD into a single, focused query (e.g., "a foodie looking for unique dining experiences"). It then converts this query and all the extracted text chunks into vector embeddings using a `SentenceTransformer` model. By calculating the cosine similarity between the query vector and each chunk vector, the pipeline ranks every chunk from all documents by its relevance to the user's need.
4.  **Sub-Section Analysis**: The top 5 chunks with the highest similarity scores are selected for a deeper analysis. Each of these chunks is further broken down into individual paragraphs. The `Analyzer` then re-ranks these paragraphs against the original query to pinpoint the most salient sentences within the most relevant sections.
5.  **Output Generation**: The final results are saved to a `challenge1b_output.json` file. This file contains the list of the top-ranked sections, the refined text from the paragraph-level analysis, and metadata about the processing run.

## Features

  - **Intelligent PDF Parsing**: Extracts text along with structural and stylistic features from PDFs.
  - **ML-Powered Chunking**: Uses a classifier to identify document structure and create semantically coherent text chunks.
  - **Persona-Driven Analysis**: Tailors search and ranking to a specific user context by combining a persona and a task.
  - **Vector-Based Similarity**: Employs a `multi-qa-MiniLM-L6-cos-v1` Sentence Transformer model for accurate semantic ranking based on cosine similarity.
  - **Multi-Level Detail**: Provides both high-level section summaries and fine-grained paragraph-level analysis.
  - **Structured JSON Output**: Delivers clean, machine-readable output for easy integration with other systems.

## Project Structure

```
.
├── models_1a/
│   ├── heading_classifier.pkl   # Model for identifying headings in documents
│   └── label_mapping.pkl        # Label mapping for the classifier
└── src/
    ├── __main__.py              # Main entry point and pipeline orchestrator
    ├── document_processor.py    # Handles PDF parsing and semantic chunking
    ├── intelligent_analyzer.py  # Performs AI-powered ranking using sentence transformers
    ├── sub_analyzer.py          # Helper utility for paragraph splitting
    └── ...
```

## Requirements

  - Python 3.8+
  - PyMuPDF (`fitz`)
  - pandas
  - joblib
  - torch
  - sentence-transformers

You can install all necessary Python packages using the following command:

```bash
pip install -r requirements.txt
```

*(A `requirements.txt` file should be created with the packages listed above)*

You must also have the `models_1a` directory containing `heading_classifier.pkl` and `label_mapping.pkl` for the document chunking to work correctly.

## Usage

To run the analysis, execute the main module from the command line, providing the path to the document collection directory.

```bash
python -m src --collection_path /path/to/your/collection
```

The specified `collection_path` directory must contain:

1.  A `challenge1b_input.json` file with the `persona`, `job_to_be_done`, and `documents` keys.
2.  A subdirectory named `PDFs` containing all the PDF files listed in the input JSON.


## 🐳 Docker Setup

This setup ensures a consistent, offline, and CPU-only environment for evaluation. It involves three main steps.

-----

### 🧱 Step 2: Build the Docker Image

Next, navigate to the project's root directory (where the `Dockerfile` is located) in your terminal and run the build command. This packages all code, dependencies, and the downloaded model into a self-contained image.

The `--platform linux/amd64` flag ensures the image is built for the target evaluation environment.

```bash
docker build --platform linux/amd64 -t adobe_1b_implementation .
```

-----

### ▶️ Step 3: Run the Docker Container

Finally, run the container using the appropriate command for your system. This command mounts your local `Collection_1` folder into the container for processing and ensures no internet access is used during execution via `--network none`.

**For Windows (using PowerShell):**

```powershell
docker run --rm `
  -v "${PWD}\Challenge_1b\Collection_1:/app/input" `
  -v "${PWD}\Challenge_1b\Collection_1:/app/output" `
  --network none `
  adobe_1b_implementation
```

**For Mac/Linux (using Bash):**

```bash
docker run --rm \
  -v "$(pwd)/Challenge_1b/Collection_1:/app/input" \
  -v "$(pwd)/Challenge_1b/Collection_1:/app/output" \
  --network none \
  adobe_1b_implementation
```