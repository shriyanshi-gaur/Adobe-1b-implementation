# src/intelligent_analyzer.py
import os
import torch

# Explicitly disable GPU visibility to ensure CPU-only operation.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sentence_transformers import SentenceTransformer, util

class Analyzer:
    def __init__(self, model_name='multi-qa-MiniLM-L6-cos-v1'):
        """
        Initializes the SentenceTransformer model, forcing it to use the CPU.
        """
        model_path = os.path.join("models_1b", "multi-qa-MiniLM-L6-cos-v1")
        self.model = SentenceTransformer(model_path, device='cpu')


    def rank_chunks_by_similarity(self, persona: str, jtbd: str, items_to_rank: list, batch_size: int = 16) -> (list, torch.Tensor):
        """
        Ranks items based on cosine similarity using CPU-based batch processing.
        """
        if not items_to_rank:
            return [], None

        if isinstance(items_to_rank[0], str):
            chunks = [{"text": p} for p in items_to_rank]
        else:
            chunks = items_to_rank

        query = f"{persona} {jtbd}"
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # Create vector embeddings for each chunk of text using batch processing on the CPU.
        texts = [c.get('text', '') for c in chunks]
        text_embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=True
        )

        scores = util.pytorch_cos_sim(query_embedding, text_embeddings)[0]

        for i, chunk in enumerate(chunks):
            chunk["similarity_score"] = scores[i].item()

        sorted_chunks = sorted(chunks, key=lambda x: x["similarity_score"], reverse=True)

        for i, chunk in enumerate(sorted_chunks):
            chunk["importance_rank"] = i + 1
            
        return sorted_chunks, query_embedding