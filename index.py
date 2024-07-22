import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
from typing import Dict
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

from process import Process

class Index:
    def __init__(self, documents: Dict[str, str] = None):
        """
        Class for indexing documents and computing embeddings.

        Args:
            documents (Dict[str, str], optional): Dictionary of document IDs and texts. Defaults to None.__bool__

        Note:
            If no documents are provided during initialization, they are loaded from data/processed_corpus.json.
        """
        self.documents = documents
        self.inverted_index = defaultdict(list)
        self.doc_lengths = defaultdict(int)
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.doc_embeddings = {}
        self.process = Process()
        self.doc_topic_distributions = {}

        # try to load documents from file if not provided
        if self.documents is None:
            self.load_documents_from_file()

    def index_documents(self, save_files: bool = True):
        """Index all added documents by creating an inverted index and computing embeddings."""
        # check if documents are loaded
        if self.documents is None:
            raise ValueError("No documents loaded.")

        for doc_id, processed_tokens in tqdm(self.documents.items(), desc="Indexing documents: "):
            token_counts = Counter(processed_tokens)
            self.doc_lengths[doc_id] = sum(token_counts.values())
            for token, count in token_counts.items():
                self.inverted_index[token].append((doc_id, count))
            text = ' '.join(processed_tokens)
            self.doc_embeddings[doc_id] = self.model.encode(text)
        print("Documents indexed.\nComputing topic distributions...")

        # Compute topic distributions using KMeans clustering
        self.compute_topic_distributions()

        if save_files:
            self.save_index_to_file()
            print("Saved index, document lengths, document embeddings, and topic distributions to files.")

    def compute_topic_distributions(self, num_topics: int = 100):
        """Compute topic distributions for all documents using LDA."""
        # Prepare the document texts for topic modeling
        docs = [' '.join(tokens) if isinstance(tokens, list) else tokens for tokens in self.documents.values()]
        vectorizer = CountVectorizer()
        doc_term_matrix = vectorizer.fit_transform(docs)
        
        # Fit the LDA model
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
        topic_distributions = lda.fit_transform(doc_term_matrix)
        
        # Assign topic distributions to documents
        for i, doc_id in enumerate(self.documents.keys()):
            self.doc_topic_distributions[doc_id] = topic_distributions[i]
        print("Topic distributions computed.")

    def load_documents_from_file(self):
        """Load documents from a json file with structure {doc_id: str}."""
        with open("data/processed_corpus.json", 'r', encoding='utf-8') as f:
            loaded_file = json.load(f)
            # decompress strings to lists of terms
            self.documents = {
                doc_id: loaded_file[doc_id].split("\n;\n") for doc_id in loaded_file
            }
        print("corpus is loaded!")

    def save_index_to_file(self, ):
        """
        Save the current index, doc_lengths and doc_embeddings to files.
        """
        index_meata = {
            "doc_lengths": self.doc_lengths,
            "doc_embeddings": {doc_id: emb.tolist() for doc_id, emb in self.doc_embeddings.items()},
            "doc_topic_distributions": {doc_id: dist.tolist() for doc_id, dist in self.doc_topic_distributions.items()}
        }
        with open("data/index.json", 'w', encoding='utf-8') as f:
            json.dump(dict(self.inverted_index), f, ensure_ascii=False, indent=4)
        with open("data/index_meta.json", 'w', encoding='utf-8') as f:
            json.dump(index_meata, f, ensure_ascii=False, indent=4)

    def load_index_from_file(self):
        """
        Load an index, doc_lengths and doc_embeddings from a file.
        """
        with open("data/index.json", 'r', encoding='utf-8') as f:
            index_data = json.load(f)
            self.inverted_index = defaultdict(list, {k: [(doc_id, count) for doc_id, count in v] for k, v in index_data.items()})
        with open("data/index_meta.json", 'r', encoding='utf-8') as f:
            index_data = json.load(f)
            self.doc_lengths = defaultdict(int, index_data["doc_lengths"])
            self.doc_embeddings = {doc_id: np.array(emb, dtype=np.float64) for doc_id, emb in index_data["doc_embeddings"].items()}
            self.doc_topic_distributions = {doc_id: np.array(dist, dtype=np.float64) for doc_id, dist in index_data["doc_topic_distributions"].items()}

    def get_inverted_index(self):
        """Return the inverted index."""
        return self.inverted_index

    def get_doc_lengths(self):
        """Return the document lengths."""
        return self.doc_lengths

    def get_doc_embeddings(self):
        """Return the document embeddings."""
        return self.doc_embeddings

    def get_documents(self):
        """Return the documents."""
        return self.documents

    def get_doc_topic_distributions(self):
        """Return the document topic distributions."""
        return self.doc_topic_distributions

