import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from cls_models import SearchResult, DocumentSource


class BibleSource(DocumentSource):
    """Bible document source."""

    def __init__(
        self,
        embeddings_path: str,
        dataframe_path: str,
        models_cache_dir: str = "./models",
    ):

        self.name = "Bible"
        self.df = pd.read_csv(dataframe_path)
        self.model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")

        try:
            self.embeddings = np.load(embeddings_path)
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            self.verse_embeddings = self.model.encode(
                self.df["verse_txt"].tolist(), show_progress_bar=True
            )
            self.verse_embeddings = np.array(self.verse_embeddings).astype("float32")
            embeddings_file = "bible_verse_embeddings.npy"
            np.save(embeddings_file, self.verse_embeddings)
            print(f"✓ Embeddings saved to {embeddings_file}")

    def get_name(self) -> str:
        return self.name

    def get_embedding_model(self):
        return self.model

    def get_embeddings(self) -> np.ndarray:
        return self.embeddings

    def format_result(self, index: int) -> SearchResult:
        row = self.df.iloc[index]
        return SearchResult(
            text=row["verse_txt"],
            source=f"{row['book']} {row['chapter']}",
            reference=f"verse: {row['verse']}",
            relevance_score=0.0,  # To be filled in later
            positivity_score=0.0,  # To be filled in later
            metadata={
                "chapter_orig_idx": row["chapter_orig_idx"],
                "verse_orig_idx": row["verse_orig_idx"],
            },
        )


class CatechismSource(DocumentSource):
    """Catechism of the Catholic Church document source."""

    def __init__(
        self,
        embeddings_path: str,
        dataframe_path: str,
        models_cache_dir: str = "./models",
    ):
        import pandas as pd
        from sentence_transformers import SentenceTransformer

        self.name = "Catechism"
        self.df = pd.read_pickle(dataframe_path)
        self.embeddings = np.load(embeddings_path)
        self.model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")

    def get_name(self) -> str:
        return self.name

    def get_embedding_model(self):
        return self.model
