import json
from collections import defaultdict
from typing import Dict
from urllib.parse import urlparse
from loguru import logger

"""
This script provides classes for managing data stored in JSON files.
It is used in the crawling and scraping process to manage:
- Blacklisted URLs
- Visited URLs
- Metadata information for documents
- The corpus of processed documents
- Term frequencies across documents
- The frontier of URLs to be visited
"""

# Define file paths for various data stores
file_paths: Dict[str, str] = {
    "meta": "./data/meta_info.json",
    "processed_corpus": "./data/processed_corpus.json",
    "frontier": "./data/urls_frontier.json",
    "noVisit": "./data/urls_noVisit.json",
    "blacklist": "./data/domains_blacklisted.json",
    "TF": "./data/term_frequencies.json",
}


def load_json_file(file_path: str) -> Dict:
    """Load JSON data from a file, returning an empty dictionary if the file does not exist."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

def write_json_file(file_path: str, data: Dict) -> None:
    """Write JSON data to a file."""
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

class DataManager:
    """A base class for managing data stored in JSON files."""
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None
        self.load_data()

    def set_data(self, data: Dict) -> None:
        """Set the data to be managed."""
        self.data = data

    def load_data(self) -> None:
        """Load data from the file if not already loaded."""
        if self.data is None:
            self.data = defaultdict(lambda: None)
            loaded_data = load_json_file(self.file_path)
            self.data.update(loaded_data)

    def save_data(self) -> None:
        """Save the current data back to the file."""
        if self.data is not None:
            write_json_file(self.file_path, self.data)

    def get_data(self) -> Dict:
        """Ensure data is loaded and return it."""
        self.load_data()
        return self.data

    def update_data(self, key: str, value) -> None:
        """Update a specific key in the data and save."""
        self.load_data()
        if key in self.data:
            logger.warning(f"{self.__class__.__name__}: Duplicate key found {key}")
        else:
            self.data[key] = value


class Blacklisted(DataManager):
    """Manage URLs that are blacklisted."""
    def __init__(self):
        super().__init__(file_paths["blacklist"])

    def load_data(self) -> None:
        """Load data from the file if not already loaded."""
        if self.data is None:
            loaded_data = load_json_file(self.file_path)
            self.data = set(loaded_data)

    # NOTE: will never be used, blacklist is manually curated
    def update_data(self, url: str) -> None:
        return
    def save_data(self) -> None:
        return


class Visited(DataManager):
    """Manage URLs that have been visited or are blacklisted."""
    def __init__(self):
        super().__init__(file_paths["noVisit"])
        
    def url_exists(self, url: str) -> bool:
        """Check if a URL exists in the data."""
        self.load_data()
        base_url = urlparse(url).netloc
        return (base_url in self.data and url in self.data[base_url])

    def add_url(self, url: str, is_english: bool = True, skipped: bool = False) -> None:
        """Add a URL with info about it being in english or not."""
        # super().update_data(url, is_blacklisted)
        self.load_data()
        base_url = urlparse(url).netloc
        if base_url not in self.data:
            self.data[base_url] = {}
        if "eng" not in self.data[base_url]:
            self.data[base_url]["eng"] = False
        if is_english and not self.data[base_url]["eng"]:
            self.data[base_url]["eng"] = True

        self.data[base_url].update({url: skipped})

    def remove_url(self, url: str) -> None:
        """Remove a URL from the data."""
        self.load_data()
        base_url = urlparse(url).netloc
        if base_url in self.data and url in self.data[base_url]:
            del self.data[base_url][url]
            if not self.data[base_url]:
                del self.data[base_url]


class MetaInfo(DataManager):
    """Manage metadata information for documents."""
    def __init__(self):
        super().__init__(file_paths["meta"])

    def add_entry(self, doc_id: str, entry: Dict[str, str]) -> None:
        self.data[doc_id] = entry

class Corpus(DataManager):
    """Manage the corpus of processed documents."""
    def __init__(self):
        super().__init__(file_paths["processed_corpus"])

    def add_entry(self, doc_id: str, content: str) -> None:
        self.data[doc_id] = content


class TermFrequencies(DataManager):
    """Manage term frequencies across documents."""
    def __init__(self):
        super().__init__(file_paths["TF"])


class Frontier(DataManager):
    """Manage the frontier of URLs to be visited."""
    def __init__(self):
        super().__init__(file_paths["frontier"])
        self.prio_counter_not_mention = 0

    def not_mentioned(self):
        self.prio_counter_not_mention += 1

    def add_url(self, url: str, priority: int = 0, not_mentioned = False) -> None:
        """Add a URL to the frontier with an optional priority."""
        self.load_data()
        if not_mentioned:
            priority = self.prio_counter_not_mention

        base_url = urlparse(url).netloc
        if base_url in self.data["IN"]:
            self.data["IN"][base_url][url] = priority
        else:
            if base_url not in self.data["OUT"]:
                self.data["OUT"][base_url] = {}
            self.data["OUT"][base_url][url] = priority

    def remove_url(self, url: str) -> None:
        """Remove a URL from the frontier."""
        self.load_data()
        base_url = urlparse(url).netloc
        frontier_type = "IN"
        if base_url in self.data["OUT"]:
            frontier_type = "OUT"

        if base_url in self.data[frontier_type] and url in self.data[frontier_type][base_url]:
            del self.data[frontier_type][base_url][url]
            if not self.data[frontier_type][base_url]:
                del self.data[frontier_type][base_url]
        logger.info(f"{self.__class__.__name__}: Removing url {url}")
