import re
import json
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple
from sklearn.cluster import KMeans

from index import Index
from ir_tools import IRToolbox as IR

class TuebingenSearchEngine:
    def __init__(self):
        """
        Initialize the TuebingenSearchEngine with preprocessing tools and embeddings model.
        """
        # initialize index, loads documents and index data
        self.index = Index()
        self.index.load_index_from_file()

        self.documents = self.index.get_documents()
        self.inverted_index = self.index.get_inverted_index()
        self.doc_lengths = self.index.get_doc_lengths()
        self.doc_embeddings = self.index.get_doc_embeddings()  
        self.metadata = self.load_metadata_from_file()
        self.doc_topic_distributions = self.index.get_doc_topic_distributions()

        # initialize retrieval toolkit
        self.ir = IR(self.documents, self.doc_lengths, self.doc_embeddings, self.inverted_index)

    def get_documents(self):
        return self.documents
    
    def get_metadata(self):
        return self.metadata
      
    def load_metadata_from_file(self):
        """Load metadata from a json file."""
        with open('data/meta_info.json', 'r') as file:
            metadata = json.load(file)
        return metadata

    def search(self, query: str) -> List[Tuple[str, float]]:
        """
        Perform a search using the given query and return ranked results.

        Args:
            query (str): Search query.

        Returns:
            List[Tuple[str, float]]: List of document IDs and their scores.
        """
        # expand the query
        query_tokens = self.ir.process_query(query)
        expanded_query_tokens = self.ir.query_expansion(query_tokens)
        expanded_query = ''.join(expanded_query_tokens)

        # first retrieval using bm25 and tfidf
        tfidf_results = self.normalize_scores(self.ir.tfidf_search(expanded_query))
        bm25_results = self.normalize_scores(self.ir.bm25_search(expanded_query))

        # pseudo relevance feedback refines query
        combined_results = self.fusion([tfidf_results, bm25_results ])
        if len(combined_results) == 0: # fallback
            combined_results = tfidf_results + bm25_results
        feedback_tokens = self.ir.pseudo_relevance_feedback(combined_results)
        refined_query = ' '.join(feedback_tokens + expanded_query_tokens)
        
        # re-retrieval using bm25, tfidf and custom approaches
        search_methods = [self.ir.tfidf_search, self.ir.bm25_search, self.ir.embedding_search]
        refined_results = [self.normalize_scores(method(refined_query)) for method in search_methods]

        # currently we use TF-IDF, BM25 and embeddings for retrieval
        weights = [0.3, 0.4, 0.3]
        
        # fusion and diversification of the result
        combined_results = self.fusion(refined_results, weights)

        diversified_results = self.diversify_results(combined_results)
        return diversified_results

    def normalize_scores(self, scores: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Normalize scores to be between 0 and 1.

        Args:
            scores (List[Tuple[str, float]]): List of document IDs and their scores.

        Returns:
            List[Tuple[str, float]]: List of document IDs and their normalized scores.
        """
        # in case scores is empty return scores
        if not scores:
            return scores
        min_score = min(scores, key=lambda x: x[1])[1]
        max_score = max(scores, key=lambda x: x[1])[1]
        if max_score == min_score:
            return [(doc_id, 1) for doc_id, _ in scores]  # If all scores are the same, return 1 for all
        normalized_scores = [(doc_id, (score - min_score) / (max_score - min_score)) for doc_id, score in scores]
        return normalized_scores

    def fusion(self, results_list: List[List[Tuple[str, float]]], weights: List[float] = None) -> List[Tuple[str, float]]:
        """
        Combine scores from different retrieval methods using a weighted combination.

        Args:
            results_list (List[List[Tuple[str, float]]): List of search results from different methods.
            weights (List[float]): List of weights for each method. If None, equal weights are used.

        Returns:
            List[Tuple[str, float]]: Combined list of document IDs and their final scores.
        """
        # check if weights are provided and valid
        num_methods = len(results_list)
        default_weights = [1 / num_methods] * num_methods
        if weights is None or len(weights) != num_methods or not np.isclose(sum(weights), 1):
            weights = default_weights

        final_scores = defaultdict(float)
        
        for i, method_results in enumerate(results_list):
            method_weight = weights[i]
            for doc_id, score in method_results:
                final_scores[doc_id] += method_weight * score

        return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    def diversify_results(self, combined_results: List[Tuple[str, float]], lambda_div=0.5) -> List[Tuple[str, float]]:
        """
        Diversify the final results by combining clustering-based and topic modeling-based diversification.

        Args:
            combined_results (List[Tuple[str, float]]): Combined search results with document IDs and scores.
            lambda_div (float): Diversification parameter.

        Returns:
            List[Tuple[str, float]]: Diversified list of document IDs and their scores.
        """
        # Get results from both diversification methods
        clustering_results = self.cluster_based_diversification(combined_results, lambda_div)
        topic_modeling_results = self.topic_modeling_based_diversification(combined_results, lambda_div)
        
        # Combine and deduplicate results
        combined_diversified_results = self.combine_diversified_results(clustering_results, topic_modeling_results)
        return combined_diversified_results

    def cluster_based_diversification(self, combined_results: List[Tuple[str, float]], lambda_div=0.75) -> List[Tuple[str, float]]:
        """
        Diversify results using K-Means clustering.

        Args:
            combined_results (List[Tuple[str, float]]): Combined search results with document IDs and scores.
            lambda_div (float): Diversification parameter.

        Returns:
            List[Tuple[str, float]]: Diversified list of document IDs and their scores.
        """
        diversified_results = []
        selected_doc_ids = set()

        # Extract document embeddings for clustering
        doc_embeddings = np.array([self.doc_embeddings[doc_id] for doc_id, _ in combined_results])
        num_clusters = max(3, int(len(combined_results) * lambda_div))

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(doc_embeddings)
        clusters = defaultdict(list)
        for idx, label in enumerate(kmeans.labels_):
            clusters[label].append(combined_results[idx])

        # Interleave results from different clusters
        while len(diversified_results) < len(combined_results):
            for cluster in clusters.values():
                if cluster:
                    doc = cluster.pop(0)
                    if doc[0] not in selected_doc_ids:
                        diversified_results.append(doc)
                        selected_doc_ids.add(doc[0])
        
        return diversified_results[:100]  # Limit to top 100 results

    def topic_modeling_based_diversification(self, combined_results: List[Tuple[str, float]], lambda_div=0.5) -> List[Tuple[str, float]]:
        """
        Diversify search results based on pre-computed topic distributions.

        Args:
            combined_results (List[Tuple[str, float]]): A list of tuples where each tuple contains a document ID and its 
                                                        associated relevance score from the initial retrieval.
            lambda_div (float): A parameter to control the balance between relevance and diversity in the results. 
                                Higher values give more weight to diversity. Defaults to 0.5.

        Returns:
            List[Tuple[str, float]]: A diversified list of tuples containing document IDs and their associated relevance scores. 
                                    The list is limited to the top 100 diversified results.
        
        Note:
            This method leverages pre-computed document-topic distributions to ensure a diverse set of topics in the final 
            search results. The topic distributions are used to group documents into clusters, and results are interleaved 
            from different clusters to enhance diversity.
        """
        # return diversified_results[:100]  # Limit to top 100 results
        diversified_results = []
        selected_doc_ids = set()
        
        # Use pre-computed document-topic distributions
        doc_topic_distributions = np.array([self.doc_topic_distributions[doc_id] for doc_id, _ in combined_results])
        
        # Assign documents to topics based on the closest topic center
        num_topics = len(doc_topic_distributions[0])
        topic_clusters = defaultdict(list)
        
        for idx, topic_distribution in enumerate(doc_topic_distributions):
            dominant_topic = np.argmin(topic_distribution)  # closest cluster center
            topic_clusters[dominant_topic].append(combined_results[idx])
        
        # Interleave results from different topics
        while len(diversified_results) < len(combined_results):
            for cluster in topic_clusters.values():
                if cluster:
                    doc = cluster.pop(0)
                    if doc[0] not in selected_doc_ids:
                        diversified_results.append(doc)
                        selected_doc_ids.add(doc[0])
        
        return diversified_results[:100]  # Limit to top 100 results

    def combine_diversified_results(self, clustering_results: List[Tuple[str, float]], topic_modeling_results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Combine results from clustering-based and topic modeling-based diversification.

        Args:
            clustering_results (List[Tuple[str, float]]): Results from clustering-based diversification.
            topic_modeling_results (List[Tuple[str, float]]): Results from topic modeling-based diversification.

        Returns:
            List[Tuple[str, float]]: Combined and diversified list of document IDs and their scores.
        """
        # Define weights
        clustering_weight = 0.6
        topic_modeling_weight = 0.4

        # Create a dictionary to accumulate weighted scores
        combined_scores = {}

        # Apply weights to clustering results and add to the combined scores
        for doc_id, score in clustering_results:
            if doc_id in combined_scores:
                combined_scores[doc_id] += score * clustering_weight
            else:
                combined_scores[doc_id] = score * clustering_weight

        # Apply weights to topic modeling results and add to the combined scores
        for doc_id, score in topic_modeling_results:
            if doc_id in combined_scores:
                combined_scores[doc_id] += score * topic_modeling_weight
            else:
                combined_scores[doc_id] = score * topic_modeling_weight

        # Convert combined scores dictionary back to a sorted list of tuples
        sorted_combined_results = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)

        # Return the top 100 results
        return sorted_combined_results[:100]

    def search_from_file(self, file_path: str):
        """
        Loads a file of multiple queries and performs a search each.
        Returns a single tsv file with the ranking results for all queries.

        Args:
            file_path (str): Path to the file containing queries.
                File should be a tab separated tsv file with query_id and query text, e.g.:
                1   tübingen attractions
                2   food and drinks

        Note:
            The results are stored in a batch file named query_results.txt with tab separated columns:
            query_id    rank    url    score
            For Example:
            1    1    http://example.com    0.85
            1    2    http://example2.com    0.75
            2    1    http://example2.com    0.92
            2    2    http://example.com    0.85
        """
        queries = {}
        # open the file and read queries
        with open(file_path, 'r') as file:
            for line in file:
                query_id, query = line.split(maxsplit=1)
                queries[query_id] = query

        # perform search for each query and store results to a dictionary
        results = []
        for query_id, query in tqdm(queries.items(), desc="Processing queries"):
            query_result = self.search(query)
            for i, t in enumerate(query_result):
                results.append([query_id, i+1, self.metadata[t[0]]['url'], t[1]])

        # write results to a file as tab separated values
        with open('query_results.txt', 'w') as file:
            for result in results:
                file.write('\t'.join(map(str, result)) + '\n')
        print("queries processed and results saved to query_results.txt")
        