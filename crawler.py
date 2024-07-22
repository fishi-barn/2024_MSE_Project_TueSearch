from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Deque
import time

# External modules
import asyncio
import httpx
from loguru import logger

# Own modules
from scraper import HTMLScraper
from near_duplicate import SimHash, DuplicateFinder
import data
from cert import CertificateManager

# Global data objects used across the crawler system
data_objects = {
    "meta": data.MetaInfo(),
    "corpus": data.Corpus(),
    "TF": data.TermFrequencies(),
    "frontier": data.Frontier(),
    "noVisit": data.Visited(),
    "blacklist": data.Blacklisted()
}

# Initialize the components for handling near-duplicate detection and HTML scraping
simHasher = SimHash()
dupeFinder = DuplicateFinder("\n;\n", simHasher, data_objects["TF"].get_data(), 0.75, 500)
scraper = HTMLScraper()

# Certificate manager for handling SSL/TLS certificates
certManager = CertificateManager("./certs/cacert.pem", "./certs/adapt_cert.pem")

@dataclass
class CrawlerConfig:
    """Define a configuration class for the crawler settings."""
    frontier_file : str = "./data/urls_frontier.json"
    max_retries : int = 2
    concurrency_limit : int = 8
    rate_limit : int = 10
    fetch_cache_limit : int = 20
    cookies : Dict[str, str] = field(default_factory=lambda: {
        'gw-cookie-notice': '%5B%22default%22%5D'
    })
    headers : Dict[str, str] = field(default_factory=lambda: {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    # 'Accept-Encoding': 'gzip, deflate, br, zstd',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'cross-site',
    'Priority': 'u=1',
    })


class RateLimiter:
    """Implements token bucket rate limiting algorithm."""
    def __init__(self, tokens: int, refill_rate: float):
        self.capacity = tokens
        self.tokens = tokens
        self.refill_rate = refill_rate
        self.last_refill = time.monotonic()

    async def wait_for_token(self):
        """Asynchronously wait for a token to be available."""
        while self.tokens <= 0:
            await asyncio.sleep(self._time_until_refill())
            self.refill()
        self.tokens -= 1

    def refill(self):
        """Refill tokens based on the elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        refill_amount = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + refill_amount)
        self.last_refill = now

    def _time_until_refill(self):
        if self.tokens > 0:
            return 0
        next_refill_in = (1 - (self.tokens - self.capacity)) / self.refill_rate
        return max(0, next_refill_in - (time.monotonic() - self.last_refill))

class WebCrawler:
    """Main class for the web crawler."""
    def __init__(self, config: CrawlerConfig = None):
        self.config = config or CrawlerConfig()
        self.sem = asyncio.Semaphore(self.config.concurrency_limit)
        self.tasks: Deque[asyncio.Task] = deque()
        self.rate_limiter = RateLimiter(self.config.rate_limit, 1)
        self.cert_file = certManager.merged_cert_path
        self.cycler = 0


    async def fetch(self, client, url, retry_count=0):
        """Fetch a URL with rate limiting, retries, and error handling."""
        async with self.sem:
            await self.rate_limiter.wait_for_token()           # Wait for rate limiter token
            try:
                # Skip URLs with certain file types (double measure as it is also checked in scraper)
                if any(url.endswith(file_type) for file_type in scraper.config.some_filetypes):
                    logger.error(f"Skipping {url} due to file type")
                    return None
                else:
                    response = await client.get(url, timeout=4)
                    response.raise_for_status()
                    logger.info(f"Fetched {url} successfully")
                    return response.text

            # ------------------------------ ERROR HANDLING ------------------------------ #
            except asyncio.CancelledError:
                logger.debug(f"Fetch operation for {url} was cancelled.")
                raise
            except Exception as e:
                logger.warning(f"Fetching {url} failed with {e}")
                # Try again with certificate adaption
                if retry_count == 0:
                    certManager.combine_certs(url)
                if retry_count < self.config.max_retries:
                    logger.warning(f"Retrying {url} after {retry_count} attempts.")
                    return await self.fetch(client, url, retry_count+1)

                return await self.fetch_fail(url)

    async def fetch_fail(self, url: str) -> None:
        """Handle fetch failure by logging and updating data structures."""
        try:
            logger.error(f"FAILED FETCH: remove from frontier and add to visited {url}")
            data_objects['frontier'].remove_url(url)
            data_objects['noVisit'].add_url(url, False, True)
            return None
        except asyncio.CancelledError:
            raise

    async def crawl(self, client: httpx.AsyncClient, url: str):
        """Crawl a single URL."""
        html_content = await self.fetch(client, url)
        if html_content:
            logger.debug(f"Processing {url} ...")
            scraper.process_html(html_content, url, data_objects, dupeFinder, simHasher)

    def schedule_crawls(self, client, group=1):
        """
        Schedule crawl tasks based on the frontier using a domain cycler and "caching".
        - group 1: IN   --> used to be the original frontier urls but these have been crawled, so the datastructure now just works as a sectioner
        - group 2: OUT  --> urls that were not of the same domain as the original frontier urls
        - group 3: BOTH --> getting both IN and OUT urls
        """
        def task_getter(queues, split=False):
            len_entries = len(queues)
            limit = self.config.fetch_cache_limit
            if split:
                limit = limit//2
            bases = list(queues.keys())
            while( any(queues) ):
                self.cycler = self.cycler % len_entries
                current_dom = bases[self.cycler]
                if current_dom not in queues:
                    self.cycler += 1
                    continue
                queue = queues[current_dom]

                if not queue: # Emtpy queue
                    del queues[current_dom]
                    self.cycler += 1
                    continue

                # Stop scheduling new tasks if limit is reached
                stop_scheduling = len(self.tasks)%limit+1 >= limit/2 if split else len(self.tasks) >= limit
                if stop_scheduling:
                    logger.info(f"Limit of {limit} tasks reached for frontier {group_str[group]}")
                    return
                # Get the next URL from the queue
                curr_url = queue.popleft()

                # Schedule a new crawl task
                task = asyncio.create_task(self.crawl(client, curr_url))
                self.tasks.append(task)
                self.cycler += 1
        group_str = {
            1: "IN",
            2: "OUT",
            3: "BOTH"
        }
        frontier_id = group_str[group]
        frontier = data_objects["frontier"].get_data()
        # Create deque to iterate over
        if group == 3:
            if len(frontier["IN"]) == 0 and len(frontier["OUT"]) == 0: return False
            queues_IN =  {base : deque(urls) for base, urls in frontier["IN"].items()}
            queues_OUT =  {base : deque(urls) for base, urls in frontier["IN"].items()}
            task_getter(queues_IN, True)
            task_getter(queues_OUT, True)
        else:
            if len(frontier[frontier_id]) == 0: return False
            queues = {base : deque(urls) for base, urls in frontier[frontier_id].items()}
            task_getter(queues)
        return True

    async def run(self):
        """Main run loop for the crawler."""
        try:
            async with httpx.AsyncClient(headers=self.config.headers, follow_redirects=True, verify=self.cert_file) as client:
                while True:
                    any_left = self.schedule_crawls(client, 3)

                    # To handle crawler being stopped
                    try:
                        await asyncio.gather(*list(self.tasks))
                    except asyncio.CancelledError:
                        raise KeyboardInterrupt from None
                    self.tasks.clear()

                    logger.critical("----------------!SAVING! DONT ABORT!----------------")
                    if any_left:
                        for data_obj in data_objects.values():
                            data_obj.save_data()
                    else:
                        return
                    logger.success("Crawl cycle completed - saved data.")

        except KeyboardInterrupt:
            logger.debug("KeyboardInterrupt received, stopping crawler and performing cleanup...")
            self.tasks.clear()
            for data_obj in data_objects.values():
                data_obj.save_data()
            logger.success("Cleanup completed, exiting.")
        except Exception as e:
            logger.error(f"Crawler stopped due to an error: {e}")

if __name__ == "__main__":
    crawler = WebCrawler()
    asyncio.run(crawler.run())