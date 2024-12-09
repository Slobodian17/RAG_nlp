from FlagEmbedding import FlagReranker

class Reranker:
    def __init__(self, model_name='BAAI/bge-reranker-large'):
        self.model = FlagReranker(model_name, use_fp16=True)

    def rerank(self, query, retrieved_results, filter_num=1):
        """
        Rerank the retrieved results based on query relevance.

        Args:
            query (str): Query string.
            retrieved_results (list): List of chunks in text format.

        Returns:
            list: Reranked results as a list of chunks (sorted by relevance).
        """
        if not retrieved_results:
            return []

        input_pairs = [(query, chunk) for chunk in retrieved_results]
        scores = self.model.compute_score(input_pairs)

        reranked_results = sorted(
            zip(retrieved_results, scores),
            key=lambda x: x[1],
            reverse=True
        )

        reranked_chunks = [chunk for chunk, _ in reranked_results]
        reranked_chunks = reranked_chunks[:filter_num]
        return reranked_chunks
