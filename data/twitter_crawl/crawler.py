# Twitter API v2 crawler for custom multimodal propagation dataset.
#
# Features:
# 1. Search for known fake/real news articles by URL or keywords
# 2. Crawl retweet and reply trees for each source post
# 3. Extract text + images from each node in the propagation tree
# 4. Store as tree-structured JSON: {root, children, text, image_url, ...}
# 5. Label source posts using fact-checking sources (Snopes, Politifact)
#
# Reference: Twitter API v2 - tweet search + conversation_id lookup

import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import requests

logger = logging.getLogger(__name__)

TWITTER_API_BASE = "https://api.twitter.com/2"

# Known fake/real news URL patterns for fact-check labeling
FACT_CHECK_SOURCES = {
    "fake": [
        "snopes.com/fact-check",
        "politifact.com/factchecks",
    ],
    "real": [
        "reuters.com/fact-check",
        "apnews.com/hub/ap-fact-check",
        "factcheck.org",
    ],
}


class TwitterPropagationCrawler:
    """Crawls Twitter propagation trees for news articles.

    Uses Twitter API v2 with bearer token authentication to:
    - Search tweets containing news URLs
    - Build retweet/reply trees via conversation_id
    - Extract text and image attachments per node
    - Label source posts using known fact-checking URLs

    Usage:
        crawler = TwitterPropagationCrawler(bearer_token="...")
        tree = crawler.crawl_propagation_tree("1234567890")
        crawler.save_tree(tree, "output.json")
    """

    def __init__(
        self,
        bearer_token: str,
        output_dir: str = "./data/twitter_crawl/raw",
        rate_limit_delay: float = 1.0,
        max_depth: int = 5,
        max_children_per_node: int = 100,
    ):
        self.bearer_token = bearer_token
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_delay = rate_limit_delay
        self.max_depth = max_depth
        self.max_children_per_node = max_children_per_node
        self.headers = {"Authorization": f"Bearer {bearer_token}"}
        self._request_count = 0

    def _rate_limit_wait(self):
        """Simple rate limiting: wait between requests."""
        self._request_count += 1
        if self._request_count % 10 == 0:
            time.sleep(self.rate_limit_delay * 2)
        else:
            time.sleep(self.rate_limit_delay)

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make a GET request to Twitter API v2."""
        url = f"{TWITTER_API_BASE}/{endpoint}"
        try:
            resp = requests.get(url, headers=self.headers, params=params or {})
            self._rate_limit_wait()

            if resp.status_code == 429:
                reset_time = int(resp.headers.get("x-rate-limit-reset", 60))
                wait = max(reset_time - int(time.time()), 1)
                logger.warning(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                return self._get(endpoint, params)

            if resp.status_code != 200:
                logger.error(f"API error {resp.status_code}: {resp.text[:200]}")
                return None

            return resp.json()
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None

    def search_tweets(
        self,
        query: str,
        max_results: int = 100,
        tweet_fields: Optional[List[str]] = None,
        expansions: Optional[List[str]] = None,
        media_fields: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Search for tweets matching a query.

        Args:
            query: Twitter search query (e.g., "https://example.com/news is:retweet")
            max_results: Max tweets per request (10-100)
            tweet_fields: Additional tweet fields to return
            expansions: Expansions to include (e.g., "attachments.media_keys")
            media_fields: Media fields to return

        Returns:
            List of tweet dicts with requested fields
        """
        tweet_fields = tweet_fields or [
            "id", "text", "author_id", "conversation_id",
            "created_at", "public_metrics", "attachments", "referenced_tweets",
        ]
        expansions = expansions or [
            "attachments.media_keys", "referenced_tweets.id",
        ]
        media_fields = media_fields or [
            "media_key", "type", "url", "preview_image_url",
        ]

        all_tweets = []
        pagination_token = None

        while len(all_tweets) < max_results:
            params = {
                "query": query,
                "tweet.fields": ",".join(tweet_fields),
                "expansions": ",".join(expansions),
                "media.fields": ",".join(media_fields),
                "max_results": min(100, max_results - len(all_tweets)),
            }
            if pagination_token:
                params["pagination_token"] = pagination_token

            data = self._get("tweets/search/recent", params)
            if not data or "data" not in data:
                break

            all_tweets.extend(data["data"])
            pagination_token = data.get("meta", {}).get("next_token")
            if not pagination_token:
                break

        return all_tweets

    def get_tweet(self, tweet_id: str) -> Optional[Dict]:
        """Get a single tweet by ID with full fields."""
        params = {
            "tweet.fields": "id,text,author_id,conversation_id,created_at,"
                           "public_metrics,attachments,referenced_tweets",
            "expansions": "attachments.media_keys,referenced_tweets.id,"
                         "author_id",
            "media.fields": "media_key,type,url,preview_image_url",
            "user.fields": "id,username,name",
        }
        data = self._get(f"tweets/{tweet_id}", params)
        if data and "data" in data:
            return data["data"][0]
        return None

    def get_conversation(
        self, conversation_id: str, max_results: int = 500
    ) -> List[Dict]:
        """Get all tweets in a conversation thread."""
        query = f"conversation_id:{conversation_id}"
        return self.search_tweets(query, max_results=max_results)

    def get_retweeters(self, tweet_id: str, max_results: int = 100) -> List[Dict]:
        """Get users who retweeted a tweet."""
        all_users = []
        pagination_token = None

        while len(all_users) < max_results:
            params = {
                "max_results": min(100, max_results - len(all_users)),
                "user.fields": "id,username,name,public_metrics",
            }
            if pagination_token:
                params["pagination_token"] = pagination_token

            data = self._get(f"tweets/{tweet_id}/retweeted_by", params)
            if not data or "data" not in data:
                break

            all_users.extend(data["data"])
            pagination_token = data.get("meta", {}).get("next_token")
            if not pagination_token:
                break

        return all_users

    def _extract_media_urls(self, includes: Dict, media_keys: List[str]) -> List[str]:
        """Extract image URLs from includes media objects."""
        media_map = {m["media_key"]: m for m in includes.get("media", [])}
        urls = []
        for key in media_keys:
            if key in media_map:
                m = media_map[key]
                if m.get("type") == "photo":
                    url = m.get("url") or m.get("preview_image_url")
                    if url:
                        urls.append(url)
        return urls

    def _build_tree_recursive(
        self,
        tweet_id: str,
        tweets_by_parent: Dict[str, List[Dict]],
        includes: Dict,
        depth: int = 0,
    ) -> Optional[Dict]:
        """Recursively build propagation tree from tweet data."""
        if depth > self.max_depth:
            return None

        node = {
            "tweet_id": tweet_id,
            "depth": depth,
            "text": "",
            "image_urls": [],
            "author_id": "",
            "metrics": {},
            "children": [],
        }

        children = tweets_by_parent.get(tweet_id, [])[:self.max_children_per_node]
        for child in children:
            child_node = {
                "tweet_id": child["id"],
                "depth": depth + 1,
                "text": child.get("text", ""),
                "image_urls": [],
                "author_id": child.get("author_id", ""),
                "metrics": child.get("public_metrics", {}),
                "children": [],
            }

            media_keys = (
                child.get("attachments", {}).get("media_keys", [])
            )
            if media_keys:
                child_node["image_urls"] = self._extract_media_urls(
                    includes, media_keys
                )

            sub_children = tweets_by_parent.get(child["id"], [])
            if sub_children:
                child_node["children"] = [
                    sub for sub in sub_children[:self.max_children_per_node]
                ]

            node["children"].append(child_node)

        return node

    def crawl_propagation_tree(self, source_tweet_id: str) -> Optional[Dict]:
        """Crawl the full propagation tree for a source tweet.

        Builds the tree by:
        1. Fetching the source tweet
        2. Getting all tweets in the conversation
        3. Grouping tweets by parent (reply-to or retweet-of)
        4. Building a recursive tree structure

        Returns:
            Tree-structured dict with root, children, text, image_urls, etc.
            None if the tweet doesn't exist or can't be crawled.
        """
        source = self.get_tweet(source_tweet_id)
        if not source:
            logger.error(f"Could not fetch source tweet {source_tweet_id}")
            return None

        conversation_id = source.get("conversation_id", source_tweet_id)
        all_tweets = self.get_conversation(conversation_id)

        if not all_tweets:
            logger.warning(f"No tweets found in conversation {conversation_id}")
            return None

        # Get includes for media
        params = {
            "ids": ",".join(t["id"] for t in all_tweets[:100]),
            "tweet.fields": "attachments,referenced_tweets",
            "expansions": "attachments.media_keys,referenced_tweets.id",
            "media.fields": "media_key,type,url,preview_image_url",
        }
        includes_data = self._get("tweets", params)
        includes = includes_data.get("includes", {}) if includes_data else {}

        # Build parent->children mapping
        tweets_by_parent: Dict[str, List[Dict]] = {}
        retweeted_ids = set()

        for tweet in all_tweets:
            refs = tweet.get("referenced_tweets", [])
            for ref in refs:
                if ref["type"] == "retweeted":
                    tweets_by_parent.setdefault(ref["id"], []).append(tweet)
                    retweeted_ids.add(ref["id"])
                elif ref["type"] == "replied_to":
                    tweets_by_parent.setdefault(ref["id"], []).append(tweet)

        # Also group by conversation for non-referenced tweets
        for tweet in all_tweets:
            parent = tweet.get("conversation_id")
            if parent and parent != tweet["id"]:
                if parent not in retweeted_ids:
                    tweets_by_parent.setdefault(parent, []).append(tweet)

        # Build source node
        source_media = (
            source.get("attachments", {}).get("media_keys", [])
        )
        source_images = (
            self._extract_media_urls(includes, source_media)
            if source_media else []
        )

        root_node = {
            "tweet_id": source["id"],
            "depth": 0,
            "text": source.get("text", ""),
            "image_urls": source_images,
            "author_id": source.get("author_id", ""),
            "metrics": source.get("public_metrics", {}),
            "children": [],
        }

        children = tweets_by_parent.get(source["id"], [])
        for child in children[:self.max_children_per_node]:
            child_media = (
                child.get("attachments", {}).get("media_keys", [])
            )
            child_images = (
                self._extract_media_urls(includes, child_media)
                if child_media else []
            )

            child_node = {
                "tweet_id": child["id"],
                "depth": 1,
                "text": child.get("text", ""),
                "image_urls": child_images,
                "author_id": child.get("author_id", ""),
                "metrics": child.get("public_metrics", {}),
                "children": [],
            }

            sub_children = tweets_by_parent.get(child["id"], [])
            for sub in sub_children[:self.max_children_per_node]:
                sub_media = (
                    sub.get("attachments", {}).get("media_keys", [])
                )
                sub_images = (
                    self._extract_media_urls(includes, sub_media)
                    if sub_media else []
                )
                child_node["children"].append({
                    "tweet_id": sub["id"],
                    "depth": 2,
                    "text": sub.get("text", ""),
                    "image_urls": sub_images,
                    "author_id": sub.get("author_id", ""),
                    "metrics": sub.get("public_metrics", {}),
                    "children": [],
                })

            root_node["children"].append(child_node)

        logger.info(
            f"Crawled tree for {source_tweet_id}: "
            f"{len(all_tweets)} tweets, depth={self.max_depth}"
        )
        return root_node

    def label_from_fact_check(self, source_url: str) -> Optional[int]:
        """Label a news article URL using fact-checking sources.

        Queries known fact-checking sites to determine if an article
        has been rated as fake or real.

        Returns:
            0 = real, 1 = fake, None = unknown
        """
        # Check if URL appears on known fact-check pages
        for category, domains in FACT_CHECK_SOURCES.items():
            for domain in domains:
                if domain in source_url:
                    return 1 if category == "fake" else 0

        # Try to search fact-check sites for the URL
        try:
            from urllib.parse import quote
            query = quote(source_url)

            # Check Snopes
            snopes_url = f"https://www.snopes.com/fact-check/?s={query}"
            resp = requests.get(snopes_url, timeout=10)
            if resp.status_code == 200:
                text = resp.text.lower()
                if "false" in text or "fake" in text:
                    return 1
                if "true" in text or "real" in text or "correct" in text:
                    return 0

            # Check Politifact
            politifact_url = (
                f"https://www.politifact.com/factchecks/?q={query}"
            )
            resp = requests.get(politifact_url, timeout=10)
            if resp.status_code == 200:
                text = resp.text.lower()
                if "false" in text or "pants on fire" in text:
                    return 1
                if "true" in text or "mostly true" in text:
                    return 0

        except requests.RequestException:
            pass

        return None

    def label_from_database(
        self, tweet_id: str, database_path: str = None
    ) -> Optional[int]:
        """Label using a local fact-check database (JSON file).

        Database format:
        [
            {"tweet_id": "123", "url": "https://...", "label": 1},
            {"tweet_id": "456", "url": "https://...", "label": 0},
        ]
        """
        if database_path is None:
            database_path = str(self.output_dir / "fact_check_database.json")

        db_path = Path(database_path)
        if not db_path.exists():
            return None

        with open(db_path) as f:
            database = json.load(f)

        for entry in database:
            if entry.get("tweet_id") == tweet_id:
                return entry["label"]
            if entry.get("url"):
                result = self.label_from_fact_check(entry["url"])
                if result is not None:
                    return result

        return None

    def save_tree(self, tree: Dict, filename: str):
        """Save propagation tree to JSON file."""
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(tree, f, indent=2, default=str)
        logger.info(f"Saved tree to {output_path}")

    def crawl_and_label(
        self,
        source_tweet_id: str,
        source_url: Optional[str] = None,
    ) -> Tuple[Optional[Dict], Optional[int]]:
        """Crawl a propagation tree and attempt to label it.

        Returns:
            (tree, label) tuple where label is 0/1/None
        """
        tree = self.crawl_propagation_tree(source_tweet_id)
        if tree is None:
            return None, None

        label = self.label_from_database(source_tweet_id)
        if label is None and source_url:
            label = self.label_from_fact_check(source_url)

        if label is not None:
            tree["label"] = label
            tree["labeled_by"] = "fact_check"

        filename = f"tree_{source_tweet_id}.json"
        self.save_tree(tree, filename)

        return tree, label

    def batch_crawl(
        self,
        tweet_ids: List[str],
        urls: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[str, Optional[Dict], Optional[int]]]:
        """Crawl multiple propagation trees.

        Args:
            tweet_ids: List of source tweet IDs to crawl
            urls: Optional dict mapping tweet_id -> source_url for labeling

        Returns:
            List of (tweet_id, tree, label) tuples
        """
        urls = urls or {}
        results = []

        for tweet_id in tweet_ids:
            logger.info(f"Crawling {tweet_id} ({len(results)+1}/{len(tweet_ids)})")
            source_url = urls.get(tweet_id)
            tree, label = self.crawl_and_label(tweet_id, source_url)
            results.append((tweet_id, tree, label))

        # Summary
        crawled = sum(1 for _, t, _ in results if t is not None)
        labeled = sum(1 for _, _, l in results if l is not None)
        logger.info(
            f"Batch complete: {crawled}/{len(tweet_ids)} crawled, "
            f"{labeled} labeled"
        )

        return results
