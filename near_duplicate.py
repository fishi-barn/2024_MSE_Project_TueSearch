import hashlib
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

"""
Find near-duplicate documents within a corpus using the SimHash algorithm.
"""

class SimHash:
    """
    A class to calculate SimHash values for documents.
    """

    def __init__(self, vector_size=64):
        """
        Initializes the SimHash object with a specified vector size for hash calculations.

        :param vector_size: The size of the hash vector. Default is 64 bits.
        """
        self.vector_size = vector_size

    def hash_word(self, word: str) -> int:
        """
        Hashes a single word using SHA-256 and returns a 64-bit integer.

        :param word: The word to hash.
        :return: A 64-bit hash of the word.
        """
        return int(hashlib.sha256(word.encode('utf-8')).hexdigest(), 16) % (1 << self.vector_size)

    def calculate_simhash(self, tokens: List[str], weights: Dict[str, int]) -> int:
        """
        Calculates the SimHash for a list of tokens.

        :param tokens: A list of tokens (words) from the document.
        :param weights: A dictionary mapping tokens to their weights.
        :return: The SimHash value as an integer.
        """
        vector = [0] * self.vector_size
        for token in tokens:
            weight = weights.get(token, 1)
            hash_value = self.hash_word(token)
            for i in range(self.vector_size):
                vector[i] += weight if hash_value & (1 << i) else -weight
        return sum(1 << i for i, v in enumerate(vector) if v > 0)


def get_hamming_distance(hash1: int, hash2: int) -> int:
    """Calculates the Hamming distance between two hashes."""
    x = hash1 ^ hash2
    dist = 0
    while x:
        dist += 1
        x &= x - 1
    return dist

def compute_similarity(hash1: int, hash2: int, vector_size: int) -> float:
    """Calculates the similarity between two hash values based on their Hamming distance."""
    return 1 - (get_hamming_distance(hash1, hash2) / vector_size)

def get_term_frequency(doc: List[str]) -> Dict[str, int]:
    """
    Calculates the term frequency for a document.

    :param doc: A list of tokens (words) from the document.
    :return: A dictionary mapping tokens to their frequency count.
    """
    word_counts = defaultdict(int)
    for word in doc:
        word_counts[word] += 1
    return word_counts

def merge_term_frequencies(TF1: Dict[str, int], TF2: Dict[str, int]) -> Dict[str, int]:
    """
    Merges two term frequency dictionaries.

    :param TF1: The first term frequency dictionary.
    :param TF2: The second term frequency dictionary.
    :return: A merged term frequency dictionary.
    """
    merged = Counter(TF1) + Counter(TF2)
    return dict(merged)

def get_term_weights(TF: Dict[str, int]) -> Dict[str, int]:
    """
    Calculates the weights for each term based on their frequency in the document.

    :param TF: The term frequency dictionary.
    :return: A dictionary mapping terms to their weights.
    """
    max_count = max(TF.values())
    return {term: max_count // count for term, count in TF.items()}

@dataclass
class DuplicateFinder:
    text_seperator: str
    simhasher: SimHash
    TF: Dict[str, int]
    duplicate_threshold: float
    pairwise_threshold: int
    target_hash: int = 0

    def update_TF(self, TF: Dict[str, int]):
        self.TF = TF

    def update_target_hash(self, target_hash: int):
        self.target_hash = target_hash

    def near_duplicate_found(self, target_hash: int, to_compare_hash: int) -> bool:
        similarity = compute_similarity(target_hash, to_compare_hash, self.simhasher.vector_size)
        # print(f"Similarity: {similarity} between {target_hash} and {to_compare_hash}")
        return similarity >= self.duplicate_threshold

    def pairwise_comparisons(self, target_hash: int, meta_data: Dict[str, Dict]) -> List[Tuple[str, int]]:
        """Simple pairwise comparison of the target SimHash with all documents in the corpus."""
        near_duplicates = []
        for docID in meta_data:
            hsh = meta_data[docID]["hash"]
            if self.near_duplicate_found(target_hash, hsh):
                near_duplicates.append((docID, hsh))
        return near_duplicates

    def bucket_comparisons(self, target_hash: int, meta_data: Dict[str, Dict]) -> List[Tuple[str, int]]:
        """
        Bucketing system where documents with similar SimHashes are grouped together.
        This reduces computation time for large corpus.
        """
        buckets = defaultdict(list)
        for docID in meta_data:
            hsh = meta_data[docID]["hash"]
            prefix = hsh >> (self.simhasher.vector_size - 6)
            buckets[prefix].append((docID, hsh))

        target_prefix = target_hash >> (self.simhasher.vector_size - 6)
        near_duplicates = []
        for doc in buckets.get(target_prefix, []):
            docID, doc_hash = doc
            if self.near_duplicate_found(target_hash, doc_hash):
                near_duplicates.append((docID, doc_hash))
        return near_duplicates

    def find_duplicates(self, target_doc: List[str], meta_data: Dict[str, Dict]) -> bool:
        """
        Finds near-duplicate documents in a corpus given a target document.

        :param target_doc: The target document as a list of tokens.
        :param corpus: Meta information as a dictionary mapping document IDs to their details.
        :return: A boolean indicating whether the target document is a near-duplicate in the corpus.
        """
        target_TF = get_term_frequency(target_doc)
        merged_term_frequencies = merge_term_frequencies(self.TF, target_TF)
        weights = get_term_weights(merged_term_frequencies)
        self.update_target_hash( self.simhasher.calculate_simhash(target_doc, weights) )

        if len(meta_data) < self.pairwise_threshold:
            near_duplicates = self.pairwise_comparisons(self.target_hash, meta_data)
        else:
            near_duplicates = self.bucket_comparisons(self.target_hash, meta_data)
        found_near_duplicates = len(near_duplicates) > 0
        # Update the corpus term frequency if no near-duplicates were found
        if not found_near_duplicates:
            self.update_TF(merged_term_frequencies)
        return found_near_duplicates
