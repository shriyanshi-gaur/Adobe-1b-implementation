# src/document_processor.py
import os
import fitz  # PyMuPDF
import pandas as pd
import joblib
import re
from collections import defaultdict
import concurrent.futures

# --- Utility Functions (unchanged) ---
def is_title_case(text):
    words = str(text).split()
    if not words: return False
    return all(word.istitle() or (word.lower() in ['a', 'an', 'the', 'and', 'or', 'but', 'for', 'nor', 'on', 'at', 'to', 'from', 'by', 'of', 'in', 'with']) for word in words)

def is_uppercase(text):
    return str(text).isupper() and len(str(text).strip()) > 1

def ends_with_colon(text):
    return str(text).strip().endswith(":")

def starts_with_number(text):
    return bool(re.match(r"^\d+(\.\d+)*(\s+|$)", str(text).strip()))

def word_count(text):
    return len(str(text).split())

def has_bullet_prefix(text):
    return bool(re.match(r"^[•*—\-–•]+\s*", str(text).strip()))

def clean_extracted_text(text):
    text = str(text).strip()
    text = re.sub(r"^[•*—\-–•\uf0b7]+\s*", '', text)
    text = re.sub(r'\s*\(cid:\d+\)\s*', '', text)
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_conventional_heading(text):
    text = text.strip().lower()
    patterns = [
        r'^(appendix\s+[a-z\d]+[:.]?|section\s+\d+[:.]?|chapter\s+\d+[:.]?|part\s+[ivxlcdm]+[:.]?|\d+\.\s+)',
        r'^introduction[:.]?$', r'^conclusion[:.]?$', r'^summary[:.]?$', r'^abstract$', r'^references$'
    ]
    for pattern in patterns:
        if re.match(pattern, text): return 1
    return 0

class DocumentProcessor:
    def __init__(self, model_dir, confidence_threshold=0.70):
        print(f"📦 Loading feature-based heading classifier model from {model_dir}")
        try:
            self.model = joblib.load(os.path.join(model_dir, 'heading_classifier.pkl'))
            self.label_encoder = joblib.load(os.path.join(model_dir, 'label_mapping.pkl'))
        except FileNotFoundError as e:
            print(f"❌ Error: Model files not found in {model_dir}. Please ensure they exist.")
            raise e
            
        self.confidence_threshold = confidence_threshold
        self.feature_cols = [
            'font_size', 'is_bold', 'is_italic', 'y_pos_normalized', 'x_pos_normalized',
            'line_height', 'space_after_line', 'space_before_line',
            'normalized_space_after_line', 'normalized_space_before_line',
            'is_left_aligned', 'page', 'is_title_case', 'is_upper',
            'ends_colon', 'starts_number', 'word_count', 'has_bullet_prefix',
            'is_conventional_heading'
        ]

    def _parse_pdf_to_features(self, pdf_path):
        # This private method remains unchanged
        doc = fitz.open(pdf_path)
        all_lines_data = []
        for page_num, page in enumerate(doc, start=1):
            page_height = page.rect.height if page.rect.height > 0 else 1
            page_width = page.rect.width if page.rect.width > 0 else 1
            blocks = page.get_text("dict", sort=True).get("blocks", [])
            lines_raw = defaultdict(list)
            for block in blocks:
                if block["type"] != 0: continue
                for line_block in block.get("lines", []):
                    for span in line_block.get("spans", []):
                        if span['size'] < 6: continue
                        y_top_rounded = round(span["bbox"][1], 1)
                        lines_raw[y_top_rounded].append(span)
            current_page_lines_data = []
            for y_coord in sorted(lines_raw):
                sorted_spans = sorted(lines_raw[y_coord], key=lambda s: s["bbox"][0])
                full_line_text = " ".join(s["text"] for s in sorted_spans)
                cleaned_text = clean_extracted_text(full_line_text)
                if not cleaned_text.strip(): continue
                line_bbox = (min(s["bbox"][0] for s in sorted_spans), min(s["bbox"][1] for s in sorted_spans), max(s["bbox"][2] for s in sorted_spans), max(s["bbox"][3] for s in sorted_spans))
                font_sizes = [s["size"] for s in sorted_spans]
                is_bolds = [(s.get("flags", 0) & 2) != 0 for s in sorted_spans]
                is_italics = [(s.get("flags", 0) & 4) != 0 for s in sorted_spans]
                current_page_lines_data.append({"text": cleaned_text, "font_size": sum(font_sizes) / len(font_sizes) if font_sizes else 0, "is_bold": any(is_bolds), "is_italic": any(is_italics), "bbox": line_bbox, "page": page_num})
            if not current_page_lines_data: continue
            for i in range(len(current_page_lines_data)):
                line = current_page_lines_data[i]
                line_features = {'text': line['text'], 'page': line['page'], 'font_size': line['font_size'], 'is_bold': line['is_bold'], 'is_italic': line['is_italic'], 'y_pos_normalized': line['bbox'][1] / page_height, 'x_pos_normalized': line['bbox'][0] / page_width, 'line_height': line['bbox'][3] - line['bbox'][1], 'is_left_aligned': line['bbox'][0] < page_width * 0.1, 'is_title_case': is_title_case(line['text']), 'is_upper': is_uppercase(line['text']), 'ends_colon': ends_with_colon(line['text']), 'starts_number': starts_with_number(line['text']), 'word_count': word_count(line['text']), 'has_bullet_prefix': has_bullet_prefix(line['text']), 'is_conventional_heading': is_conventional_heading(line['text'])}
                space_before = line['bbox'][1] - current_page_lines_data[i-1]['bbox'][3] if i > 0 else 0
                space_after = current_page_lines_data[i+1]['bbox'][1] - line['bbox'][3] if i < len(current_page_lines_data) - 1 else 0
                line_features['space_before_line'] = space_before
                line_features['normalized_space_before_line'] = space_before / page_height
                line_features['space_after_line'] = space_after
                line_features['normalized_space_after_line'] = space_after / page_height
                all_lines_data.append(line_features)
        doc.close()
        return pd.DataFrame(all_lines_data)

    def _process_single_doc(self, doc_info):
        doc, collection_path = doc_info
        pdf_path = os.path.join(collection_path, 'PDFs', doc['filename'])
        doc_chunks = []
        try:
            df_lines = self._parse_pdf_to_features(pdf_path)
            if df_lines.empty:
                return doc_chunks

            X_predict = df_lines[self.feature_cols]
            preds_encoded = self.model.predict(X_predict)
            
            # --- FIX: Use the correct variable 'preds_encoded' for the transformation ---
            df_lines['label'] = self.label_encoder.inverse_transform(preds_encoded)
            
            current_chunk_text, current_chunk_title, current_chunk_page = [], "Introduction", 1
            
            for _, row in df_lines.iterrows():
                is_heading = row['label'].lower().startswith('h')
                
                if is_heading:
                    if current_chunk_text:
                        doc_chunks.append({'document': doc['filename'], 'page_number': current_chunk_page, 'section_title': current_chunk_title, 'text': " ".join(current_chunk_text).strip()})
                    current_chunk_title, current_chunk_page, current_chunk_text = row['text'], row['page'], []
                elif row['label'] == 'other':
                    current_chunk_text.append(row['text'])
            
            if current_chunk_text:
                doc_chunks.append({'document': doc['filename'], 'page_number': current_chunk_page, 'section_title': current_chunk_title, 'text': " ".join(current_chunk_text).strip()})
        except Exception as e:
            print(f"❌ Error processing {doc['filename']} in a worker process: {e}")
        
        return doc_chunks

    def get_all_chunks(self, collection_path, documents):
        all_chunks = []
        doc_infos = [(doc, collection_path) for doc in documents]

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(self._process_single_doc, doc_infos)
            for doc_chunks in results:
                all_chunks.extend(doc_chunks)

        return all_chunks