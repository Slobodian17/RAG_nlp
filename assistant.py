import os
from litellm import completion

class Assistant:
    def __init__(self, retriever, reranker=None, model="groq/llama3-8b-8192", citation=None):
        """
        Initialize the Assistant.

        Args:
            retriever (Retriever): An instance of a Retriever class (KeyWordRetriever or SemanticRetriever).
            model (str): The name of the LLM model to use (default is "groq/llama3-8b-8192").
        """
        self.retriever = retriever
        self.model = model
        self.reranker = reranker
        self.citation = citation

    def simulate_llm_response(self, prompt, context, api_key):
        """
        Simulate an LLM response for demonstration purposes.

        Args:
            prompt (str): The prompt to send to the simulated LLM.
            context (str): The context to include in the prompt.
            api_key (str): The API key for Groq.

        Returns:
            str: The generated completion text.
        """
        os.environ['GROQ_API_KEY'] = api_key
        instruction = """

        Contextual AI Assistant

        You are an AI assistant designed to provide concise, accurate, and clear responses. Always adhere to the following principles:

        Core Principles:

        - Truthfulness: Prioritize accuracy. If unsure, acknowledge the limitation without guessing.
        - Contextual Understanding: Analyze the conversation history to understand the user's intent.
        - Clarity and Conciseness: Provide brief, direct answers without unnecessary elaboration.
        - Helpful Guidance: Offer practical suggestions when relevant, but keep it concise.
        - Error Handling: Acknowledge limitations and suggest alternatives when unable to answer.
        Important! Maximum length of your answer can be of 3-4 sentences.
        """


        response = completion(
            model=self.model,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "system", "content": context},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )

        generated_text = ""
        for chunk in response:
            generated_text += str(chunk["choices"][0]['delta']['content'])
        generated_text = generated_text[:-4]

        # max_line_length = 160
        # final_text = textwrap.fill(generated_text, width=max_line_length)
        return generated_text

    def handle_query(self, query, api_key, retriever_type="semantic", top_k=5, use_reranker=False):
        """
        Handle the user's query by retrieving relevant chunks and generating a simulated LLM response.

        Args:
            query (str): The user's query.
            retriever_type (str): Type of retriever to use ("semantic" or "keyword").
            top_k (int): Number of top results to retrieve.

        Returns:
            str: The generated response from the simulated LLM.
        """
        if retriever_type.lower() == "keyword":
            retrieved_chunks = [chunk for chunk, _ in self.retriever.retrieve(query, top_k=top_k)]
        elif retriever_type.lower() == "semantic":
            retrieved_chunks = [chunk for chunk, _ in self.retriever.retrieve(query, top_k=top_k)]
        elif retriever_type.lower() == "hybrid":
            retrieved_chunks = [chunk for chunk, _ in self.retriever.retrieve(query, top_k=top_k)]
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")

        if use_reranker and self.reranker:
            reranked_results = self.reranker.rerank(query, retrieved_chunks)
            citations = self.citation.search_citate(reranked_results)
            reranked_chunks = " ".join(reranked_results)
            return self.simulate_llm_response(query, reranked_chunks, api_key), reranked_chunks, citations

        citations = self.citation.search_citate(retrieved_chunks)
        retrieved_chunks_string = " ".join(retrieved_chunks)
        print(retrieved_chunks_string)
        return self.simulate_llm_response(query, retrieved_chunks_string, api_key), retrieved_chunks_string, citations
