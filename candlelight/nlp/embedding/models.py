"""Models available for embedding.
The heavy duty is lazy-loaded when needed.
"""
from .sbert import SBert

sbert_multilingual = SBert(
    model_name="paraphrase-multilingual-mpnet-base-v2",
    output_dim=768,
    max_seq_length=128,
    size_in_mb=970,
    languages=["en", "de", "fr", "es", "it", "nl", "pt", "ru", "tr"],
)

sbert_english = SBert(
    model_name="all-mpnet-base-v2",
    output_dim=768,
    max_seq_length=384,
    size_in_mb=420,
    languages=["en"],
)

sbert_english_light = SBert(
    model_name="all-MiniLM-L6-v2",
    output_dim=384,
    max_seq_length=256,
    size_in_mb=80,
    languages=["en"],
)
