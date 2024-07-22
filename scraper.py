import re
from dataclasses import dataclass, field
from typing import Dict, Set, List
from urllib.parse import urljoin, urlparse

# External modules
from near_duplicate import get_term_frequency, get_term_weights
from loguru import logger
from selectolax.lexbor import LexborHTMLParser as HTMLParser
from ftlangdetect import detect

# Own modules
from process import Process

# General utility functions
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = text.replace("'", "").replace("’", "")
    text = re.sub(r"\s*(p\.m\.)\s*", " pm", text)
    text = re.sub(r"\s*(a\.m\.)\s*", " am", text)
    return text

def txt_contains_term(text, terms):
    # use regex to find if the text contains the term
    if text:
        for term in terms:
            if re.search(term, text, re.IGNORECASE):
                return True
    return False

@dataclass
class ScraperConfig:
    """Define a configuration class for the scraper settings."""

    # Define dictionaries to specify which tags and attributes to check in the HTML.
    tags_to_check: Dict[str, list] =  field(default_factory=lambda: {
        "head": ["title", "meta"],
        "body": ["p", "h1", "h2", "h3", "h4", "h5", "h6", "th", "td", "li", "a", "blockquote"],
        "not_want": ["style", "script", "xmp", "iframe", "noembed", "noframes", "button", "src", "noscript", "form"],
        "text_deco" : ["span", "i", "b", "strong", "em", "u", "s", "strike"]
    })

    attributes_to_check: Dict[str, list] = field(default_factory=lambda: {
        "meta": ["description", "keywords"],
        "title": ["title"],
    })

    some_filetypes: List[str] = field(default_factory=lambda: [
        ".exe", ".xls", ".docx", ".doc", ".flv", ".img", ".ppt", ".iso", ".tz", ".eps", ".qcow2", ".deb", ".rpm",
        ".tzst", ".rar", ".lzo", ".lzma", ".odt", ".dmg", ".xlsx", ".mp4", ".swf", ".svg", ".gif", ".png", ".jpeg",
        ".tar", ".zst", ".mkv", ".7z", ".pkg", ".wmv", ".gz", ".apk", ".ps", ".msi", ".ai", ".zip", ".pptx", ".jpg",
        ".mp3", ".avi", ".mov", ".xz", ".php", ".pdf", ".lz", ".webm", ".bz2", ".tex", ".csv", ".json", ".xml" ,".log",
        ".rtf", ".txt", ".css", ".js", ".ts", ".py", ".java", ".cpp", ".c", ".h", ".hpp", ".cs", ".go", ".rb", ".apx"
    ])

    not_wanted_terms: List[str] = field(default_factory=lambda: [
        "imprint", "privacy", "cookie", "login", "download", "upload",
        "signin", "register", "checkout", "cart", "search", "edit", "script",
        "@", "readme", "terms", "conditions", "policy", "botc", "restricted",
        "email!", "newsletter", "subscribe", "unsubscribe", "eGuide", ":", ".it",
        ".fr", ".no", "/pdf/", "/src/", "/arxiv.org/format/", "/arxiv.org/html/",
        "/arxiv.org/ps/", "arxiv.org/show-email/"
    ])

    filter_html_struct: Dict[str, List[str]] = field(default_factory=lambda: {
        "section_removal": ["div#catlinks", "div[class=printfooter]", "ol[class=references]"],
        "specific_tag_removal": ["span[class=mw-whatlinkshere-tools]", "span[class=mw-editsection]"]
    })

    website_type: Dict[str, str] = field(default_factory=lambda: {
        "en.wikipedia.org": "div#bodyContent",
        "en.wikivoyage.org": "div#mw-content-text",
        "arxiv.org": "div[class=leftcolumn]"
    })

    # Define a separator for text data.
    text_seperator: str = "\n;\n" # TODO: if needed put into somehwere more "global"

class HTMLScraper:
    """Define the main HTML scraper class."""
    def __init__(self, config: ScraperConfig = None):
        """Initialize the scraper with configuration."""
        self.config = ScraperConfig() if not config else config
        self.processor = Process()

    def language_correct(self, text: List[str]) -> bool:
        """Check if the language of the page is in the searched for language."""
        # Join samples and detect language
        sample_text = " ".join(text)
        if sample_text:
            detected_language = detect(sample_text, low_memory=False)
            logger.warning(f"Language Detection: {detected_language}")
            return bool(
                "fr" not in detected_language["lang"]
                and (
                    "en" in detected_language["lang"]
                    or ("de" in detected_language["lang"] and detected_language["score"] < 0.8)
                )
            )
        # No language could be determined --> skip page
        return False

    def get_meta_info(self, html_parser: HTMLParser, html_url: str) -> Dict[str, str]:
        """Extract meta information from the HTML document."""
        meta_entry = {"url": html_url}

        for selector in self.config.tags_to_check["head"]:
            for attribute in self.config.attributes_to_check.get(selector, []):
                s = f'{selector}[name="{attribute}"]' if attribute != selector else selector
                meta_entry[attribute] = ""
                match = html_parser.head.css_first(s)
                if match:
                    output = match.attributes.get("content", "") if attribute != selector else match.text(strip=True)
                    if output:
                        meta_entry[attribute] = output.strip()
        return meta_entry

    def get_relevant_content(self, html_parser: HTMLParser, selector: str) -> str:
        """Extract relevant content from the HTML document."""
        main_node = html_parser.css_first(selector)
        if not main_node:
            return ""

        main_node.strip_tags(self.config.filter_html_struct["specific_tag_removal"])

        for section in self.config.filter_html_struct["section_removal"]:
            node = main_node.css_first(section)
            if node:
                node.decompose()

        return main_node.html

    def get_body_info(self, html_parser: HTMLParser, url: str) -> List[str]:
        """Extract text content from all div elements in the HTML document."""
        seen_div_nodes: Set = set()
        corpus_entry = []
        content_filtered_body: Set = set()
        parsed_url = urlparse(url)
        if self.config.website_type.get(parsed_url.netloc, False):
            html = self.get_relevant_content(html_parser, self.config.website_type[parsed_url.netloc])
            if html:
                html_parser = HTMLParser(html)
                divs = html_parser.css("div")
        else:
            main_body = html_parser.body.css_first("main")
            main_divs = set(main_body.css("div")) if main_body else set()
            all_divs_in_body = html_parser.body.css("div")
            for div in all_divs_in_body:
                if bool(
                    txt_contains_term(div.attributes.get("class"), ["content", "main"])
                    or txt_contains_term(div.attributes.get("id"), ["content", "main"])
                ):
                    content_filtered_body.add(div)

            divs = main_divs.union(content_filtered_body)

        for div in divs:
            if div in seen_div_nodes: continue
            context = {}
            for node in div.traverse(include_text=True):
                if node.tag == "div": seen_div_nodes.add(node)
                txt = node.text(deep=False).strip()
                if node.tag in self.config.tags_to_check["text_deco"]:
                    node.unwrap()
                if bool(
                    txt
                    and len(txt) > 1
                    and node.tag == "-text"
                    and node.parent.tag in self.config.tags_to_check["body"]
                    and not txt_contains_term(txt, ["cookie"])
                ):
                    context[clean_text(txt)] = None
            corpus_entry += context.keys()
        return corpus_entry

    def get_mention_of_tuebingen(self, html_parser):
        """Check if the page mentions Tübingen."""
        all_text = html_parser.body.text(separator=" ")
        all_text = clean_text(all_text).strip().lower()
        to_be_matched = ["tübingen", "tuebingen"]
        return txt_contains_term(all_text, to_be_matched)

    def get_urls(self, html_parser, html_url, visited, blacklisted):
        """Extract URLs from the HTML document."""
        def is_editor_or_app_url(url):
            # Compile patterns for query parameters and path segments indicating an editor or app interface
            query_param_patterns = [
                r"action=edit",
                r"mode=edit",
                r"start?"
            ]
            path_segment_patterns = [
                r"/edit/",
                r"/admin/",
                r"/dashboard/"
            ]
            # If no match is found, return False
            return bool(
                txt_contains_term(url, query_param_patterns)
                or txt_contains_term(url, path_segment_patterns)
            )
        def is_file(url):
            # Check if the URL is a file
            for filetype in self.config.some_filetypes:
                if url.endswith(filetype):
                    return True
            return False

        parsed_html_url = urlparse(html_url)

        # Specific rules/preprocessing for certain websites
        if self.config.website_type.get(parsed_html_url.netloc, False):
            html = self.get_relevant_content(html_parser, self.config.website_type[parsed_html_url.netloc])
            if html:
                html_parser = HTMLParser(html)
        html_base_url = f"{parsed_html_url.scheme}://{parsed_html_url.netloc}"

        valid_urls = set()
        path_urls = set()

        for node in html_parser.css("a[href]"):
            href = node.attributes.get("href")
            if href is None: continue
            # Check if its a internal page reference
            if href.startswith("#") or "#" in href: continue
            # Check if its a alternative language url
            hreflang = node.attributes.get("hreflang")
            if hreflang and "en" not in hreflang: continue
            # Check if its a relative hyperref
            curr_absolute_url = href
            if href.startswith("/"):
                curr_absolute_url = urljoin(html_base_url, href)

            curr_parsed_url = urlparse(curr_absolute_url)
            # Check if it is proper url structure
            if not curr_parsed_url.netloc or not curr_parsed_url.scheme: continue
            if curr_parsed_url.scheme not in ["http", "https"]: continue

            already_visited = (
                visited.get(curr_parsed_url.netloc)
                and curr_absolute_url in visited.get(curr_parsed_url.netloc)
            )
            if already_visited: continue

            # Check if the domain has an English version and if the new URL is of the same domain
            domain_has_english_version = (
                visited.get(parsed_html_url.netloc) and
                visited[parsed_html_url.netloc].get("eng") and
                parsed_html_url.netloc == curr_parsed_url.netloc
            )
            # Check if the path contains English keywords
            path_contains_english_keywords = txt_contains_term(parsed_html_url.path, ["/en/", "/english/", "/en-de/"])
            current_path_lacks_english_keywords = not txt_contains_term(curr_parsed_url.path, ["/en/", "/english/", "/en-de/"])
            # Skip if the current path does not contain English keywords
            if domain_has_english_version and path_contains_english_keywords and current_path_lacks_english_keywords:
                continue

            # Check if the current URL is valid based on several conditions
            if bool(
                not is_file(curr_parsed_url.path)
                and not is_editor_or_app_url(curr_parsed_url.path)
                and not txt_contains_term(curr_absolute_url, blacklisted)
                and not txt_contains_term(curr_parsed_url.path, self.config.not_wanted_terms)
                and not curr_parsed_url.path in path_urls
            ):
                valid_urls.add(curr_absolute_url)
                path_urls.add(curr_parsed_url.path)
        logger.success(f"Found {len(valid_urls)} new URLs.")
        return valid_urls

    def get_potential_english_version(self, html_parser, html_url):
        """Get a link to an english version of the website"""
        base_url = urlparse(html_url).netloc
        for node in html_parser.body.css("a[href]"):
            hreflang = node.attributes.get("hreflang", None)
            if hreflang and "en" in hreflang:
                href = node.attributes["href"]
                absolute_url = urljoin(base_url, href)
                return absolute_url
        return None

    def process_html(self, html, url, data_objects, dupeFinder, simHasher):
        """Main method to process HTML content."""
        data_frontier = data_objects["frontier"]
        data_noVisit = data_objects["noVisit"]
        data_blacklisted = data_objects["blacklist"]

        if html == "":
            logger.error(f"No html content found. Skipping page {url} and adding to visited URLs...")
            data_frontier.remove_url(url)
            data_noVisit.add_url(url, False, True)
            return

        if url in data_noVisit.get_data():
            logger.error(f"URL already visited. Skipping page {url}...")
            data_frontier.remove_url(url)
            return

        html_parser = HTMLParser(html)
        if html_parser.body is None:
            logger.error(f"No body content found. Skipping page {url} and adding to visited URLs...")
            data_frontier.remove_url(url)
            data_noVisit.add_url(url, False, True)
            return

        data_corpus = data_objects["corpus"]
        data_meta = data_objects["meta"]
        data_TF = data_objects["TF"]

        # --------------- Remove unwanted tags from the HTML document. --------------- #
        html_parser.root.strip_tags(self.config.tags_to_check["not_want"])

        # ----- Check the language of the page and proceed only if it's English. ----- #
        list_of_body_content = self.get_body_info(html_parser, url)
        is_in_english = self.language_correct(list_of_body_content)
        if not is_in_english:
            logger.warning("Skipping non-English page...")
            eng_ver = self.get_potential_english_version(html_parser, url)
            if eng_ver:
                if not data_noVisit.url_exists(eng_ver):
                    logger.success("Found new potential English version. Adding to frontier...")
                    data_frontier.add_url(eng_ver)
            data_noVisit.add_url(url, False, True)
            data_frontier.remove_url(url)
            return

        # ---------------------- Check if page mentions Tübingen --------------------- #
        page_mentions_tuebingen = self.get_mention_of_tuebingen(html_parser)
        if page_mentions_tuebingen:
        # ---------- Process the HTML content to extract corpus information. --------- #
            processed_body_data = self.processor.process_text( self.config.text_seperator.join(list_of_body_content) )
            current_meta = data_meta.get_data()
            current_corpus = data_corpus.get_data()
            if len(current_corpus) == 0:
                is_duplicate = False
                dupeFinder.update_TF( get_term_frequency(processed_body_data) )
                weights = get_term_weights( dupeFinder.TF )
                dupeFinder.update_target_hash( simHasher.calculate_simhash(processed_body_data, weights) )
            else:
                is_duplicate = dupeFinder.find_duplicates(processed_body_data, current_meta)

        # --------------- Check if current document is near-duplicate. --------------- #
            if is_duplicate:
                logger.warning(f"Near-duplicate found. Skipping {url} and adding to visited URLs...")
            else:
                target_hash = dupeFinder.target_hash
                current_TFs = dupeFinder.TF
                corpus_entry = self.config.text_seperator.join(processed_body_data)
                meta_data = self.get_meta_info(html_parser, url)
                meta_data["hash"] = target_hash
                new_urls = self.get_urls( html_parser, url, data_noVisit.get_data(), data_blacklisted.get_data() )

        # -------- Save new content to respective databases and write to file. -------
                last_docID = int( next(reversed(current_corpus)) ) if current_corpus else 0
                data_corpus.add_entry( last_docID + 1, corpus_entry )
                data_meta.add_entry( last_docID + 1, meta_data )
                data_TF.set_data(current_TFs)
                for new_url in new_urls:
                    data_frontier.add_url(new_url)

        else: # if no mention of tübingen then get urls
            logger.warning("No mention of Tübingen.")

        # ------------------- Successfully processed (visited) URL ------------------- #
        data_frontier.remove_url(url)
        data_noVisit.add_url(url, is_in_english, page_mentions_tuebingen)

