from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
model.save("models_1b/multi-qa-MiniLM-L6-cos-v1")
