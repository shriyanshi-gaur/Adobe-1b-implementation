# src/__main__.py
import argparse
import os
import json
from datetime import datetime

from .document_processor import DocumentProcessor
from .intelligent_analyzer import Analyzer
from .sub_analyzer import SubAnalyzer

def load_metadata(collection_path):
    metadata_path = os.path.join(collection_path, 'challenge1b_input.json')
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Metadata file not found: {metadata_path}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Round 1B: Persona-Driven Document Intelligence")
    parser.add_argument("--collection_path", type=str, required=True,
                        help="Path to the directory of the document collection.")
    args = parser.parse_args()

    metadata = load_metadata(args.collection_path)
    if not metadata: return

    # --- Configuration ---
    persona = metadata.get("persona", {}).get("role", "")
    jtbd = metadata.get("job_to_be_done", {}).get("task", "")
    documents = metadata.get("documents", [])
    BATCH_SIZE = 16 # A smaller batch size is safer for machines with 8GB RAM.

    print("🔄 Starting CPU-Optimized Parallel Pipeline...\n")

    # Initialize components
    doc_processor = DocumentProcessor(model_dir="models_1a")
    analyzer = Analyzer()
    sub_analyzer = SubAnalyzer(analyzer)

    # This step uses a ProcessPoolExecutor to parse documents in parallel on multiple CPU cores.
    print("🔬 Processing documents into semantic chunks (in parallel on CPU)...")
    all_chunks = doc_processor.get_all_chunks(args.collection_path, documents)
    print(f"📄 Found {len(all_chunks)} semantic chunks in total.")

    # This step uses batch processing on the CPU.
    print("\n🔬 Ranking chunks using batched cosine similarity on CPU...")
    ranked_chunks, _ = analyzer.rank_chunks_by_similarity(
        persona, jtbd, all_chunks, batch_size=BATCH_SIZE
    )
    print("✅ Chunks ranked with similarity scores.")

    print("\n🔬 Performing sub-section analysis on top 5 chunks...")
    extracted_sections_list = []
    subsection_analysis_list = []
    
    for chunk in ranked_chunks[:5]:
        extracted_sections_list.append({
            "document": chunk.get("document"), "section_title": chunk.get("section_title"),
            "importance_rank": chunk.get("importance_rank"), "page_number": chunk.get("page_number")
        })
        
        paragraphs = sub_analyzer._split_into_paragraphs(chunk.get("text", ""))
        sub_sections, _ = analyzer.rank_chunks_by_similarity(
            persona, jtbd, paragraphs, batch_size=BATCH_SIZE
        )
        for sub in sub_sections:
            subsection_analysis_list.append({
                "document": chunk.get("document"), "refined_text": sub.get("text"),
                "page_number": chunk.get("page_number")
            })
            
    print("✅ Sub-section analysis complete.\n")

    # Prepare and save the final output
    output_metadata = {
        "input_documents": [doc.get("filename") for doc in documents],
        "persona": persona, "job_to_be_done": jtbd, "processing_timestamp": datetime.now().isoformat()
    }
    output_data = {
        "metadata": output_metadata, "extracted_sections": extracted_sections_list,
        "subsection_analysis": subsection_analysis_list
    }
    output_json = json.dumps(output_data, indent=4)
    output_path = os.path.join(args.collection_path, "challenge1b_output.json")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_json)
    print(f"✅ Final output (CPU-Optimized) saved to: {output_path}")

if __name__ == "__main__":
    main()