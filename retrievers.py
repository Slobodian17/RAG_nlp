import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from embeddings import generate_embeddings

class Retriever:
    """
    Abstract base class for retrievers.
    """
    def retrieve(self, query, top_k=5):
        raise NotImplementedError

class KeyWordRetriever(Retriever):
    """
    Keyword-based retriever using BM25.
    """

    def __init__(self, chunks):
        """
        Initialize the BM25 retriever with pre-tokenized chunks.

        Args:
            chunks (list): List of text chunks to index.
        """
        self.tokenized_chunks = [word_tokenize(chunk) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_chunks)
        self.chunks = chunks

    def retrieve(self, query, top_k=5):
        """
        Retrieve top-k chunks based on BM25 scores.

        Args:
            query (str): The search query.
            top_k (int): Number of top results to retrieve.

        Returns:
            list: List of (chunk, score) tuples sorted by relevance.
        """
        query_tokens = word_tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], scores[i]) for i in ranked_indices]


class SemanticRetriever(Retriever):
    """
    Semantic retriever using SentenceTransformers and FAISS.
    """

    def __init__(self, chunks, model_name='all-MiniLM-L6-v2', index_path="faiss_index"):
        """
        Initialize the semantic retriever with SentenceTransformers and FAISS.

        Args:
            chunks (list): List of text chunks.
            model_name (str): Model name for SentenceTransformers.
            index_path (str): Path to save/load the FAISS index.
        """
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        self.index_path = index_path
        self.index = self._create_faiss_index(chunks)

    def _create_faiss_index(self, chunks):
        """
        Create a FAISS index from text chunks.

        Args:
            chunks (list): List of text chunks.

        Returns:
            faiss.Index: Trained FAISS index.
        """

        embeddings = generate_embeddings(chunks)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        faiss.write_index(index, self.index_path)
        print(f"FAISS index saved to {self.index_path}")
        return index

    def retrieve(self, query, top_k=5):
        """
        Retrieve top-k chunks based on semantic similarity.

        Args:
            query (str): The search query.
            top_k (int): Number of top results to retrieve.

        Returns:
            list: List of (chunk, score) tuples sorted by relevance.
        """
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding, top_k)
        results = [(self.chunks[idx], 1 / (1 + distances[0][i])) for i, idx in enumerate(indices[0])]
        return results

class HybridRetriever(Retriever):
    """
    Hybrid retriever that combines results from keyword-based and semantic retrievers.
    """

    def __init__(self, keyword_retriever, semantic_retriever):
        """
        Initialize the HybridRetriever.

        Args:
            keyword_retriever (KeyWordRetriever): An instance of KeyWordRetriever.
            semantic_retriever (SemanticRetriever): An instance of SemanticRetriever.
        """
        self.keyword_retriever = keyword_retriever
        self.semantic_retriever = semantic_retriever

    def normalize_scores(self, scores):
        """
        Normalize a list of scores to a [0, 1] range.

        Args:
            scores (list): List of scores.

        Returns:
            list: Normalized scores.
        """
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [0.5] * len(scores)  # Avoid division by zero if all scores are the same
        return [(score - min_score) / (max_score - min_score) for score in scores]

    def retrieve(self, query, top_k=5):
        """
        Retrieve top-k chunks by combining keyword and semantic relevance.

        Args:
            query (str): The search query.
            top_k (int): Number of top results to retrieve.

        Returns:
            list: List of (chunk, combined_score) tuples sorted by combined relevance.
        """
        # Retrieve results from both retrievers
        keyword_results = self.keyword_retriever.retrieve(query, top_k=top_k)
        semantic_results = self.semantic_retriever.retrieve(query, top_k=top_k)

        # Extract chunks and scores from both retrievers
        keyword_chunks, keyword_scores = zip(*keyword_results) if keyword_results else ([], [])
        semantic_chunks, semantic_scores = zip(*semantic_results) if semantic_results else ([], [])

        # Normalize scores for both retrievers
        normalized_keyword_scores = self.normalize_scores(keyword_scores) if keyword_scores else []
        normalized_semantic_scores = self.normalize_scores(semantic_scores) if semantic_scores else []

        # Combine results by creating a mapping of chunk -> combined score
        score_map = {}

        # Add keyword scores to the map
        for chunk, score in zip(keyword_chunks, normalized_keyword_scores):
            score_map[chunk] = score_map.get(chunk, 0) + score

        # Add semantic scores to the map
        for chunk, score in zip(semantic_chunks, normalized_semantic_scores):
            score_map[chunk] = score_map.get(chunk, 0) + score

        # Sort the results by combined score
        sorted_results = sorted(score_map.items(), key=lambda x: x[1], reverse=True)

        # Return top-k results
        return sorted_results[:top_k]