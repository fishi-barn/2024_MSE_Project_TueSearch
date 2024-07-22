import numpy as np
import torch
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, util

from process import Process

class IRToolbox:
    def __init__(
        self, 
        documents: Dict[str, str], 
        doc_lengths: Dict[str, int], 
        doc_embeddings: Dict[str, np.ndarray],
        inverted_index: Dict[str, List[Tuple[str, int]]]
    ):
        """
        Class for searching documents based on a processed query.

        Args:
            documents (Dict[str, str]): Dictionary of document IDs and texts.
            doc_lengths (Dict[str, int]): Dictionary of document IDs and document lengths.
            doc_embeddings (Dict[str, np.ndarray]): Dictionary of document IDs and document embeddings.
            inverted_index (Dict[str, List[Tuple[str, int]]]): Inverted index mapping tokens to document IDs and counts.
        """
        self.documents = documents
        self.doc_lengths = doc_lengths
        self.doc_embeddings = doc_embeddings
        self.inverted_index = inverted_index
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.process = Process()

    #### Enhancement Methods ####

    def process_query(self, query: str) -> List[str]:
        """
        Process a search query similarly to how documents are processed.

        Args:
            query (str): Search query to be processed.

        Returns:
            List[str]: List of processed query tokens.
        """
        return self.process.process_text(query)

    def query_expansion(self, query_tokens: List[str]) -> List[str]:
        """
        Expand the query using semantically similar terms.

        Args:
            query_tokens (List[str]): List of tokens from the original query.

        Returns:
            List[str]: Expanded list of query tokens.
        """
        expanded_query = set(query_tokens)
        for token in query_tokens:
            similar_words = util.paraphrase_mining(self.model, [token], top_k=5)
            expanded_query.update([word for _, word, score in similar_words if score > 0.7])
        return list(expanded_query)

    def pseudo_relevance_feedback(self, initial_results: List[Tuple[str, float]], top_k: int = 5) -> List[str]:
        """
        Refine the query using pseudo relevance feedback from the top-k initial results.

        Args:
            initial_results (List[Tuple[str, float]]): Initial search results with document IDs and scores.
            top_k (int): Number of top documents to use for feedback.

        Returns:
            List[str]: Refined list of query tokens.
        """
        # Step 1: Select top-k documents
        top_k_documents = initial_results[:top_k]

        # add up to 3 random picks from rest of the documents
        n_random = min(3, len(initial_results) - top_k)
        if n_random > 0:
            random_picks = np.random.choice(len(initial_results[top_k:]), n_random, replace=False)
            for i in random_picks:
                top_k_documents.append(initial_results[top_k + i])
            
        # Step 2: Extract terms from top-k documents
        all_terms = []
        for doc_id, _ in top_k_documents:
            all_terms.extend(self.documents[doc_id])
        
        # Step 3: Calculate term frequencies
        term_frequencies = Counter(all_terms)
        
        # Step 4: Refine query
        # Here, we'll use the top-n most frequent terms as the refined query.
        # The number of terms to use for refining the query can be adjusted.
        top_n_terms = 25
        refined_query_terms = [term for term, freq in term_frequencies.most_common(top_n_terms)]
        
        return refined_query_terms

    #### Search Methods ####

    def tfidf_search(self, query: str) -> List[Tuple[str, float]]:
        """
        Perform a search using the TF-IDF method.

        Args:
            query (str): search query.

        Returns:
            List[Tuple[str, float]]: List of document IDs and their TF-IDF scores.
        """
        query_tokens = self.process_query(query)
        scores = defaultdict(float)
        query_term_count = Counter(query_tokens)
        for term, count in query_term_count.items():
            if term in self.inverted_index:
                doc_list = self.inverted_index[term]
                idf = np.log(1 + len(self.documents) / len(doc_list))
                for doc_id, term_freq in doc_list:
                    tf = term_freq / self.doc_lengths[doc_id]
                    scores[doc_id] += tf * idf * count
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def bm25_search(self, query: str, k1=1.5, b=0.75) -> List[Tuple[str, float]]:
        """
        Perform a search using the BM25 method.

        Args:
            query (str): Search query.
            k1 (float): BM25 term frequency saturation parameter.
            b (float): BM25 length normalization parameter.

        Returns:
            List[Tuple[str, float]]: List of document IDs and their BM25 scores.
        """
        query_tokens = self.process_query(query)
        scores = defaultdict(float)
        avg_doc_len = sum(self.doc_lengths.values()) / len(self.documents)
        query_term_count = Counter(query_tokens)
        for term, count in query_term_count.items():
            if term in self.inverted_index:
                doc_list = self.inverted_index[term]
                idf = np.log(1 + (len(self.documents) - len(doc_list) + 0.5) / (len(doc_list) + 0.5))
                for doc_id, term_freq in doc_list:
                    tf = term_freq / self.doc_lengths[doc_id]
                    score = idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (self.doc_lengths[doc_id] / avg_doc_len))))
                    scores[doc_id] += score * count
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def embedding_search(self, query: str) -> List[Tuple[str, float]]:
        """
        Perform a search using embeddings and cosine similarity.

        Args:
            query (str): Search query.

        Returns:
            List[Tuple[str, float]]: List of document IDs and their cosine similarity scores.
        """
        query_embedding = self.model.encode(query, convert_to_tensor=True, device='cpu').type(torch.float32)
        scores = {doc_id: util.pytorch_cos_sim(query_embedding, torch.tensor(doc_embedding).type(torch.float32)).item() for doc_id, doc_embedding in self.doc_embeddings.items()}
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
