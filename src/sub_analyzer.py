# src/sub_analyzer.py
import re
from .intelligent_analyzer import Analyzer

class SubAnalyzer:
    def __init__(self, analyzer: Analyzer):
        """Initializes the SubAnalyzer."""
        self.analyzer = analyzer

    def _split_into_paragraphs(self, text: str) -> list[str]:
        """Splits a block of text into paragraphs."""
        paragraphs = [p.strip() for p in re.split(r'\n{1,}', text) if p.strip()]
        return paragraphs

    def analyze_chunk(self, main_chunk: dict, persona: str, jtbd: str, query_embedding) -> list[dict]:
        """
        Breaks a chunk into paragraphs and analyzes each one.
        """
        paragraphs = self._split_into_paragraphs(main_chunk.get("text", ""))
        if not paragraphs:
            return []

        # --- FIX: Use the 'simple' ranking method for paragraphs ---
        # This method doesn't require the 'documents' list and is perfect for this task.
        ranked_paragraphs, _ = self.analyzer.rank_chunks_by_relevance_simple(
            persona,
            jtbd,
            [{'text': p, 'section_title': ''} for p in paragraphs]
        )

        analyzed_sub_sections = []
        for p_data in ranked_paragraphs:
            analyzed_sub_sections.append({
                "sub_section_text": p_data['text'],
                "relevance_score": p_data['relevance_score'],
                "importance_rank": p_data['importance_rank'],
            })
            
        return analyzed_sub_sections