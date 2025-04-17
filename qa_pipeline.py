import os
import json
import torch
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import AzureOpenAI
import faiss

class qa_pipeline:
    """
    A full Retrieval-Augmented Generation (RAG) pipeline using:
    - FAISS for vector search
    - SentenceTransformer for embedding
    - Cross-encoder for re-ranking
    - Azure OpenAI for answer generation
    """

    def __init__(self):
        """
        Initializes the RAG pipeline by loading:
        - SentenceTransformer encoder
        - Cross-encoder reranker (tokenizer + model)
        - Azure OpenAI LLM client
        """
        self.model = SentenceTransformer("BAAI/bge-base-en-v1.5")

        self.index = None  # FAISS index
        self.metadata = []  # List of metadata dictionaries (one per vector)

        self.reranker_model_name = "BAAI/bge-reranker-base"
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_name)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(self.reranker_model_name)

        self.client = AzureOpenAI(
            api_key="5UnXrfATc5KyXEVyxpjeJF9MInBazuoBMBkyHKEB1nARFKxuJGLtJQQJ99BCAC4f1cMXJ3w3AAABACOGNEeL",
            api_version="2024-08-01-preview",
            azure_endpoint="https://interviews3.openai.azure.com/"
        )

        print("✅ QA pipeline initialized.")

    def load(self, index_path="faiss_index.index", metadata_path="faiss_metadata.json"):
        """
        Loads a saved FAISS index and its associated metadata from disk.

        Args:
            index_path (str): Path to FAISS index file.
            metadata_path (str): Path to metadata JSON file.
        """
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def call_model(self, prompt):
        """
        Sends a prompt to the Azure OpenAI LLM and returns the response.

        Args:
            prompt (str): Formatted prompt for the model.

        Returns:
            str: The generated answer.
        """
        response = self.client.chat.completions.create(
            model="gpt-35-turbo",
            temperature=0.0,  # Makes answers deterministic
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ])
        return response.choices[0].message.content

    def build_rag_prompt(self, question: str, context: str) -> str:
        """
        Constructs a RAG-style prompt with clear instructions.

        Args:
            question (str): The user query.
            context (str): Retrieved chunks formatted as text.

        Returns:
            str: The full prompt to be fed into the LLM.
        """
        return f"""
        You are a helpful assistant. Use only the information from the context below to answer the user's question.

        Instructions:
        1. If the question is ambiguous, first explain the ambiguity clearly. Then, provide all possible answers based on the different interpretations using the provided context.
        2. If there is no information in the context that can answer the question, respond with exactly: "There is no answer to the question."

        Question:
        {question}

        Context:
        {context}
        """.strip()

    def ask_question(self, query, top_k=5, threshold=0.4):
        """
        Full RAG pipeline: retrieve, rerank, structure context, and generate an answer.

        Args:
            query (str): User's question.
            top_k (int): Number of top chunks to retrieve from FAISS.
            threshold (float): Minimum normalized score to keep a chunk after re-ranking.

        Returns:
            tuple: 
                - answer (str): Final model output
                - final_context (str): Full textual context provided to the LLM
                - final_ordered_sections (List[str]): Structured sections used in the context
                - grouped_chunks (Dict[Tuple[str, str], List[Tuple[int, str]]]): Chunks grouped by title/subheading
                - group_scores (Dict[Tuple[str, str], List[float]]): Reranker scores per group
                - final_metadatas (List[Dict]): Filtered metadata used for answering
                - final_scores (List[float]): Normalized reranker scores
        """
        # Step 1: Embed query and search FAISS
        query_embedding = self.model.encode([query]).astype("float32")
        distances, indices = self.index.search(query_embedding, top_k)

        # Step 2: Gather matched chunks and their metadata
        candidate_chunks = []
        candidate_metadatas = []
        for i in indices[0]:
            candidate_metadatas.append(self.metadata[i])
            candidate_chunks.append(self.metadata[i]['text'])

        # Step 3: Prepare text for reranker
        re_ranked_chunks = []
        for chunk, metadata in zip(candidate_chunks, candidate_metadatas):
            titles_text = f"{metadata['title']}\n {metadata['subheading']}"
            all_text = f"{titles_text}\n {chunk}"
            re_ranked_chunks.append(all_text)

        # Step 4: Run reranker and normalize scores
        rerank_inputs = [(query, chunk) for chunk in re_ranked_chunks]
        tokens = self.reranker_tokenizer(rerank_inputs, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            scores = self.reranker_model(**tokens).logits.squeeze()
            scores = (scores - scores.min()) / (scores.max() - scores.min())  # Normalize [0, 1]

        # Step 5: Filter low-score chunks
        filtered_chunks = []
        filtered_metadatas = []
        filtered_scores = []
        for score, chunk, metadata in zip(scores, candidate_chunks, candidate_metadatas):
            if score >= threshold:
                filtered_chunks.append(chunk)
                filtered_metadatas.append(metadata)
                filtered_scores.append(score.item())

        # Step 6: Sort by reranker score
        sorted_results = sorted(zip(filtered_chunks, filtered_metadatas, filtered_scores), key=lambda x: x[2], reverse=True)
        final_chunks = [x[0] for x in sorted_results]
        final_metadatas = [x[1] for x in sorted_results]
        final_scores = [x[2] for x in sorted_results]

        # Step 7: Group by title + subheading
        grouped_chunks = defaultdict(list)
        group_scores = defaultdict(list)
        for chunk, metadata, score in zip(final_chunks, final_metadatas, final_scores):
            key = (metadata['title'], metadata['subheading'])
            grouped_chunks[key].append((metadata['text_place'], chunk))
            group_scores[key].append(score)

        # Step 8: Sort chunks within each group
        for key in grouped_chunks:
            grouped_chunks[key].sort(key=lambda x: x[0])  # sort by position in section

        # Step 9: Order groups by max score
        group_ranks = sorted(group_scores.items(), key=lambda x: max(x[1]), reverse=True)
        ordered_keys = [key for key, _ in group_ranks]

        # Step 10: Build final context
        final_ordered_sections = []
        for key in reversed(ordered_keys):  # highest-score last = closest to LLM prompt
            title, subheading = key
            section_text = f"Title: {title}\nSubheading: {subheading}\n"
            section_text += "\n".join(chunk for _, chunk in grouped_chunks[key])
            final_ordered_sections.append(section_text)

        final_context = "\n\n---\n\n".join(final_ordered_sections)

        # Step 11: Generate answer from LLM
        prompt = self.build_rag_prompt(question=query, context=final_context)
        answer = self.call_model(prompt)

        # If no answer, clear context values
        if answer.startswith("There is no answer to the question."):
            final_context, final_ordered_sections, grouped_chunks, group_scores, final_metadatas, final_scores = [], [], [], [], [], []

        return answer, final_context, final_ordered_sections, grouped_chunks, group_scores, final_metadatas, final_scores
