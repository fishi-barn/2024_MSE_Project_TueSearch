import string
from typing import List
import json
from spacy import load
import re

class Process:
    def __init__(self, model="en_core_web_sm", use_custon_entities=True):
        """Initialize the Document Processor with an NLP model and custom entities."""
        self.nlp = load(model, exclude=["senter"])
        self.phone_number_pattern = r'\+?\d[\d -]{8,14}\d'
        self.punctuation = string.punctuation + "–"
        if use_custon_entities:
            self.load_custom_entities("./data/custom_entities.json")

    def load_custom_entities(self, file_path):
        """Load custom entities from a file and add them to the entity ruler."""
        try:
            with open(file_path, 'r') as file:
                patterns = json.load(file)
            self.setup_entity_ruler(patterns)
        except IOError:
            print(f"Error opening or reading the file {file_path}")

    def setup_entity_ruler(self, patterns):
        """Add or update the entity ruler in the NLP pipeline with custom patterns."""
        if not self.nlp.has_pipe("entity_ruler"):
            entity_ruler = self.nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": True})
        else:
            entity_ruler = self.nlp.get_pipe("entity_ruler")
        entity_ruler.add_patterns(patterns)

    def get_phone_numbers(self, doc):
        """Extracts phone numbers from the document text using regex and removes them from the document."""
        phone_pattern = re.compile(self.phone_number_pattern)
        phone_numbers = phone_pattern.findall(doc)
        modified_doc_text = phone_pattern.sub("", doc)
        return phone_numbers, modified_doc_text

    def process_text(self, text: str) -> List[str]:
        """Process the document, filtering out unwanted tokens."""
        filtered_doc = []
        phones, mod_docs = self.get_phone_numbers(text)
        filtered_doc.extend(phones)

        # tokenize etc
        spacied_doc = self.nlp(mod_docs)
        for token in spacied_doc:
            if self._token_allowed(token):
                term = self._preprocess(token.lemma_)
                if len(term) > 1:
                    filtered_doc.append(term)
        # get entities
        for ent in spacied_doc.ents:
            if ent.label_ != "LINK" and "http" in ent.text:
                continue
            term = self._preprocess(ent.text)
            if len(term) > 1:
                filtered_doc.append(term)

        return filtered_doc

    def process_text_replacement_ents(self, text: str) -> List[str]:
        """Process the document, filtering out unwanted tokens, named entities replace tokens."""
        filtered_doc = []
        phones, mod_docs = self.get_phone_numbers(text)
        filtered_doc.extend(phones)

        spacied_doc = self.nlp(mod_docs)
        current_skip_end_idx = -1
        for token in spacied_doc:
            if token.idx < current_skip_end_idx: continue

            if token.ent_type_ and token.ent_type_ != "LINK" and "http" in token.text:
                current_skip_end_idx = token.idx + len(token.text)
            elif token.ent_iob_ == 'B':
                entity, entity_length = self._collect_full_entity(token, spacied_doc)
                term = self._preprocess(entity)
                if len(term) > 1: filtered_doc.append(term)
                current_skip_end_idx = token.idx + entity_length
                continue

            if self._token_allowed(token):
                term = self._preprocess(token.lemma_)
                if len(term) > 1:
                    filtered_doc.append(term)
        return filtered_doc

    def _collect_full_entity(self, start_token, doc):
        entity = [start_token.text]
        entity_length = len(start_token.text)
        for next_token in doc[start_token.i + 1:]:
            if next_token.ent_iob_ == 'I':
                entity.append(next_token.text)
                entity_length += len(next_token.text) + 1  # Adding 1 for the space
            else:
                break
        return ' '.join(entity), entity_length

    @staticmethod
    def _token_allowed(token):
        return bool(
            token
            and str(token).strip()
            and not token.is_stop
            and not token.is_punct
        )

    def _preprocess(self, word):
        clean = re.sub(rf"[{self.punctuation}\s]+", " ", word)
        return clean.strip().lower()