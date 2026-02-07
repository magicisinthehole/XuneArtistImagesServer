from flask import Flask, send_file, Response, abort, make_response, request
from io import BytesIO
import cv2
import numpy as np
import requests
import html
from pathlib import Path
from datetime import datetime, timedelta, date
import os
import os.path
from time import sleep, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import re
import musicbrainzngs
import pylast

# Configuration - API keys from environment variables (no defaults - must be set)
FANART_TV_API_KEY = os.environ.get('FANART_TV_API_KEY')
GENIUS_ACCESS_TOKEN = os.environ.get('GENIUS_ACCESS_TOKEN')
LASTFM_API_KEY = os.environ.get('LASTFM_API_KEY')
LASTFM_API_SECRET = os.environ.get('LASTFM_API_SECRET')
MUSICBRAINZ_APP_NAME = "Zune artist images recreation server"
MUSICBRAINZ_VERSION = "2.0"
MUSICBRAINZ_CONTACT = os.environ.get('MUSICBRAINZ_CONTACT')

cwd = os.getcwd()
app = Flask(__name__)

# Rate limiting
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["120 per minute"],
    storage_uri="memory://",
)

# MBID validation (standard UUID format used by MusicBrainz)
MBID_PATTERN = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
    re.IGNORECASE
)
MAX_IMAGE_WIDTH = 3840


def validate_mbid(mbid):
    """Validate that mbid is a properly formatted UUID."""
    return bool(MBID_PATTERN.match(mbid))

# Set up MusicBrainz user agent
musicbrainzngs.set_useragent(MUSICBRAINZ_APP_NAME, MUSICBRAINZ_VERSION, MUSICBRAINZ_CONTACT)

# Set up Last.fm network for biography fallback
lastfm_network = pylast.LastFMNetwork(api_key=LASTFM_API_KEY, api_secret=LASTFM_API_SECRET)

# HTTP headers for API requests
headers = {
    'User-Agent': f'{MUSICBRAINZ_APP_NAME}/{MUSICBRAINZ_VERSION}',
    'Api-User-Agent': f'{MUSICBRAINZ_APP_NAME}/{MUSICBRAINZ_VERSION} ({MUSICBRAINZ_CONTACT})'
}

# Reusable session for connection pooling
http_session = requests.Session()
http_session.headers.update(headers)


# Per-service rate limiting for external APIs
class ServiceThrottle:
    """Thread-safe token bucket rate limiter per service."""
    def __init__(self, max_per_second):
        self.min_interval = 1.0 / max_per_second
        self.lock = threading.Lock()
        self.last_call = 0.0

    def wait(self):
        """Block until it's safe to make the next request."""
        with self.lock:
            now = time()
            elapsed = now - self.last_call
            if elapsed < self.min_interval:
                sleep(self.min_interval - elapsed)
            self.last_call = time()


_service_throttles = {
    'theaudiodb.com': ServiceThrottle(2),      # 2 req/sec (free tier)
    'ws.audioscrobbler.com': ServiceThrottle(5),  # Last.fm: 5 req/sec
    'api.genius.com': ServiceThrottle(5),       # Genius: ~5 req/sec (conservative)
}


def throttled_get(url, **kwargs):
    """Rate-limited wrapper around http_session.get()."""
    from urllib.parse import urlparse
    hostname = urlparse(url).hostname
    throttle = _service_throttles.get(hostname)
    if throttle:
        throttle.wait()
    return http_session.get(url, **kwargs)


# Lock to prevent concurrent image fetches for the same artist
_fetch_locks = {}
_fetch_locks_lock = threading.Lock()


def get_artist_lock(artist_dir):
    """Get or create a lock for a specific artist directory."""
    with _fetch_locks_lock:
        if artist_dir not in _fetch_locks:
            _fetch_locks[artist_dir] = threading.Lock()
        return _fetch_locks[artist_dir]


def sanitize_artist_name(artist_name):
    """Sanitize artist name for use as directory name."""
    safe = str(artist_name).replace("/", "-")
    safe = safe.replace("..", "")
    safe = safe.strip(". ")
    return safe if safe else "unknown"


def save_xml_response(artist, filename, xml_data):
    """Save XML response to disk for debugging."""
    safe_artist = sanitize_artist_name(artist)
    xml_dir = Path("/artists") / safe_artist / "xml"
    try:
        xml_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Unable to create XML directory for {safe_artist}: {e}")
        return

    try:
        (xml_dir / filename).write_text(xml_data, encoding="utf-8")
    except Exception as e:
        print(f"Unable to write XML response for {safe_artist}: {e}")


# Disk cache management - LRU eviction at 50GB
import shutil

CACHE_MAX_BYTES = 50 * 1024 * 1024 * 1024  # 50GB
CACHE_CHECK_INTERVAL = 3600  # Check once per hour
_last_cache_check = 0
_cache_check_lock = threading.Lock()


def touch_artist(artist_dir):
    """Mark artist as recently used by updating directory mtime."""
    try:
        os.utime(f"/artists/{artist_dir}")
    except Exception:
        pass


def check_cache_size():
    """Evict least recently used artists if disk cache exceeds limit."""
    global _last_cache_check
    now = time()
    if now - _last_cache_check < CACHE_CHECK_INTERVAL:
        return
    with _cache_check_lock:
        if now - _last_cache_check < CACHE_CHECK_INTERVAL:
            return
        _last_cache_check = now

    artists_path = Path("/artists")
    if not artists_path.exists():
        return

    entries = []
    total_size = 0
    for entry in artists_path.iterdir():
        if not entry.is_dir():
            continue
        mtime = entry.stat().st_mtime
        dir_size = sum(f.stat().st_size for f in entry.rglob('*') if f.is_file())
        entries.append((entry, mtime, dir_size))
        total_size += dir_size

    if total_size <= CACHE_MAX_BYTES:
        return

    # Sort oldest first, evict until under limit
    entries.sort(key=lambda x: x[1])
    for entry, mtime, dir_size in entries:
        if total_size <= CACHE_MAX_BYTES:
            break
        print(f"Cache eviction: removing {entry.name} (last used: {datetime.fromtimestamp(mtime).date()})")
        try:
            shutil.rmtree(entry)
            total_size -= dir_size
        except Exception as e:
            print(f"Error evicting {entry.name}: {e}")


# Cache for artist info to avoid repeated API calls
_artist_info_cache = {}

# Persistent mapping of MBID prefix -> full MBID (for image endpoint resolution)
_mbid_prefix_map = {}
MBID_MAP_FILE = "/artists/mbid_map.txt"


def load_mbid_map():
    """Load MBID prefix mapping from disk."""
    global _mbid_prefix_map
    try:
        if os.path.exists(MBID_MAP_FILE):
            with open(MBID_MAP_FILE, "r") as f:
                for line in f:
                    line = line.strip()
                    if ":" in line:
                        prefix, full_mbid = line.split(":", 1)
                        _mbid_prefix_map[prefix] = full_mbid
            print(f"Loaded {len(_mbid_prefix_map)} MBID mappings")
    except Exception as e:
        print(f"Error loading MBID map: {e}")


def save_mbid_mapping(mbid):
    """Save MBID prefix -> full MBID mapping to disk."""
    prefix = mbid[:8]
    if prefix not in _mbid_prefix_map:
        _mbid_prefix_map[prefix] = mbid
        try:
            with open(MBID_MAP_FILE, "a") as f:
                f.write(f"{prefix}:{mbid}\n")
        except Exception as e:
            print(f"Error saving MBID mapping: {e}")


def resolve_mbid_from_prefix(prefix):
    """Resolve full MBID from prefix using cache or mapping file."""
    # Check in-memory cache first
    for cached_mbid in _artist_info_cache:
        if cached_mbid.startswith(prefix):
            return cached_mbid

    # Check persistent mapping
    if prefix in _mbid_prefix_map:
        return _mbid_prefix_map[prefix]

    return None


def get_artist_info(mbid):
    """Get artist name and external links from MusicBrainz."""
    if mbid in _artist_info_cache:
        return _artist_info_cache[mbid]

    result = musicbrainzngs.get_artist_by_id(mbid, includes=['url-rels'])
    artist = result['artist']

    # Build URLs dict from relations
    urls = {}
    for rel in artist.get('url-relation-list', []):
        rel_type = rel.get('type', '').lower()
        urls[rel_type] = rel.get('target', '')

    info = {
        'name': artist['name'],
        'sort_name': artist.get('sort-name', artist['name']),
        'mbid': mbid,
        'urls': urls
    }

    _artist_info_cache[mbid] = info
    save_mbid_mapping(mbid)  # Save prefix -> full MBID mapping
    return info


def get_wiki_title_from_urls(urls):
    """Extract Wikipedia article title from MusicBrainz URL relations."""
    # Option 1: Direct Wikipedia URL
    if 'wikipedia' in urls:
        wiki_url = urls['wikipedia']
        if 'en.wikipedia.org/wiki/' in wiki_url:
            return wiki_url.split('/wiki/')[-1]

    # Option 2: Via Wikidata
    if 'wikidata' in urls:
        wikidata_url = urls['wikidata']
        qid = wikidata_url.split('/')[-1]
        try:
            wd_response = http_session.get(
                f'https://www.wikidata.org/wiki/Special:EntityData/{qid}.json',
                timeout=10
            )
            if wd_response.ok:
                data = wd_response.json()
                sitelinks = data.get('entities', {}).get(qid, {}).get('sitelinks', {})
                if 'enwiki' in sitelinks:
                    return sitelinks['enwiki']['title'].replace(' ', '_')
        except Exception as e:
            print(f"Error fetching Wikidata: {e}")

    return None


def clean_lastfm_bio(bio_text):
    """Clean and format Last.fm biography text, limiting to 4 paragraphs."""
    if not bio_text:
        return None, None

    # Remove "Read more on Last.fm" link and similar
    bio_text = re.sub(r'<a href="https?://www\.last\.fm[^"]*"[^>]*>Read more on Last\.fm[^<]*</a>\.?', '', bio_text)
    bio_text = re.sub(r'Read more on Last\.fm\.?', '', bio_text)

    # Remove Last.fm internal markup like [artist]Name[/artist] or broken versions like [)artist]
    bio_text = re.sub(r'\[\/?artist\]', '', bio_text)
    bio_text = re.sub(r'\[\)artist\]', '', bio_text)
    bio_text = re.sub(r'\[artist\][^\[]*\[/artist\]', lambda m: m.group(0).replace('[artist]', '').replace('[/artist]', ''), bio_text)

    # Strip existing HTML tags to get plain text
    plain_text = re.sub(r'<[^>]+>', '', bio_text)

    # Normalize whitespace and newlines
    plain_text = re.sub(r'\r\n', '\n', plain_text)
    plain_text = re.sub(r'\r', '\n', plain_text)

    # Split into paragraphs (by double newline or single newline with blank)
    paragraphs = re.split(r'\n\s*\n|\n{2,}', plain_text)

    # Clean each paragraph
    cleaned = []
    for p in paragraphs:
        p = p.strip()
        # Skip empty paragraphs or very short ones (likely artifacts)
        if p and len(p) > 20:
            # Fix common unicode issues
            p = p.replace('\u2019', "'")  # Right single quote
            p = p.replace('\u2018', "'")  # Left single quote
            p = p.replace('\u201c', '"')  # Left double quote
            p = p.replace('\u201d', '"')  # Right double quote
            p = p.replace('\u2014', '-')  # Em dash
            p = p.replace('\u2013', '-')  # En dash
            p = p.replace('\u2026', '...')  # Ellipsis
            p = p.replace('\xa0', ' ')  # Non-breaking space
            cleaned.append(p)

    # Limit to 4 paragraphs
    cleaned = cleaned[:4]

    if not cleaned:
        return None, None

    plain = '\n\n'.join(cleaned)
    # Don't escape here - let the bio endpoint do it once for XML embedding
    html_content = ''.join(f'<p>{p}</p>' for p in cleaned)

    return plain, html_content


def get_lastfm_biography(mbid, artist_name):
    """Fetch artist biography from Last.fm (original source)."""
    lastfm_throttle = _service_throttles['ws.audioscrobbler.com']
    try:
        bio = None

        # Try by MBID first
        try:
            lastfm_throttle.wait()
            artist = lastfm_network.get_artist_by_mbid(mbid)
            lastfm_throttle.wait()
            bio = artist.get_bio_content()
        except pylast.WSError:
            pass

        # Fallback: try by artist name
        if not bio:
            try:
                lastfm_throttle.wait()
                artist = lastfm_network.get_artist(artist_name)
                lastfm_throttle.wait()
                bio = artist.get_bio_content()
            except pylast.WSError:
                pass

        if bio:
            plain, html_content = clean_lastfm_bio(bio)
            if html_content:
                print(f"Last.fm bio found for {artist_name}")
                return {
                    'extract': plain,
                    'extract_html': html_content,
                    'source': 'lastfm'
                }

    except Exception as e:
        print(f"Last.fm bio error for {artist_name}: {e}")

    return None


def get_biography(mbid):
    """Fetch artist biography from Last.fm (primary) or Wikipedia (fallback)."""
    artist_info = get_artist_info(mbid)
    artist_name = artist_info['name']

    # 1. Try Last.fm first (original source)
    lastfm_bio = get_lastfm_biography(mbid, artist_name)
    if lastfm_bio and lastfm_bio.get('extract'):
        return lastfm_bio

    # 2. Fallback to Wikipedia
    try:
        wiki_title = get_wiki_title_from_urls(artist_info['urls'])

        if not wiki_title:
            return None

        wiki_response = http_session.get(
            f'https://en.wikipedia.org/api/rest_v1/page/summary/{wiki_title}',
            timeout=10
        )
        if wiki_response.ok:
            data = wiki_response.json()
            print(f"Wikipedia bio found for {artist_name}")
            return {
                'extract': data.get('extract', ''),
                'extract_html': data.get('extract_html', ''),
                'thumbnail': data.get('thumbnail', {}).get('source'),
                'originalimage': data.get('originalimage', {}).get('source'),
                'description': data.get('description', ''),
                'source_url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                'source': 'wikipedia'
            }
    except Exception as e:
        print(f"Error fetching Wikipedia biography for {mbid}: {e}")

    return None


def get_wikipedia_images(wiki_title, max_images=8):
    """Fetch multiple images from Wikipedia media-list endpoint."""
    images = []

    try:
        response = http_session.get(
            f'https://en.wikipedia.org/api/rest_v1/page/media-list/{wiki_title}',
            timeout=10
        )
        if response.ok:
            data = response.json()
            for item in data.get('items', []):
                if len(images) >= max_images:
                    break

                if item.get('type') != 'image':
                    continue

                title = (item.get('title') or '').lower()

                # Skip icons, logos, flags, timelines
                if any(skip in title for skip in ['icon', 'logo', 'flag', 'timeline', 'map']):
                    continue

                # Get image URL - try original first, then srcset
                source = item.get('original', {}).get('source', '')

                if not source:
                    # Try srcset - get largest available
                    srcset = item.get('srcset', [])
                    if srcset:
                        # srcset is sorted by size, last is largest
                        source = srcset[-1].get('src', '')

                # Convert thumbnail URL to full-resolution original
                # Thumbnail: /wikipedia/commons/thumb/c/c2/File.jpg/640px-File.jpg
                # Original:  /wikipedia/commons/c/c2/File.jpg
                if '/thumb/' in source and source.count('/') > 6:
                    parts = source.split('/thumb/')
                    if len(parts) == 2:
                        base = parts[0]
                        rest = parts[1]
                        # Remove the size suffix (e.g., /640px-File.jpg -> /File.jpg)
                        rest_parts = rest.rsplit('/', 1)
                        if len(rest_parts) == 2:
                            source = base + '/' + rest_parts[0]

                if not source:
                    continue

                # Ensure full URL
                if source.startswith('//'):
                    source = 'https:' + source

                # Skip SVGs, GIFs, timelines, and non-image files
                source_lower = source.lower()
                if any(skip in source_lower for skip in ['.svg', '.gif', 'timeline', '/timeline/']):
                    continue

                # Only accept jpg/jpeg/png
                if not any(ext in source_lower for ext in ['.jpg', '.jpeg', '.png']):
                    continue

                images.append(source)
                print(f"Wikipedia image found: {title[:50]}")

    except Exception as e:
        print(f"Error fetching Wikipedia images: {e}")

    return images


def mbid_to_image_uuid(mbid, image_num):
    """Generate a deterministic image UUID from MBID and image number."""
    # Use first 8 chars of MBID + image number for uniqueness
    return f"{mbid[:8]}-{image_num:04d}-0000-0000-000000000000"


def _fetch_fanart(mbid, seen_urls):
    """Fetch from fanart.tv. Returns (thumb_candidates, other_images, fanart_data)."""
    thumb_candidates = []
    other_images = []
    if not FANART_TV_API_KEY:
        return thumb_candidates, other_images
    try:
        fanart_url = f'https://webservice.fanart.tv/v3.2/music/{mbid}?api_key={FANART_TV_API_KEY}'
        response = http_session.get(fanart_url, timeout=10)
        if response.ok:
            fanart_data = response.json()
            for img in fanart_data.get('artistthumb', []):
                url = img['url']
                if url not in seen_urls:
                    seen_urls.add(url)
                    thumb_candidates.append((url, 'fanart-thumb'))
            for img_type in ['artistbackground', 'artist4kbackground']:
                for img in fanart_data.get(img_type, []):
                    url = img['url']
                    if url not in seen_urls:
                        seen_urls.add(url)
                        other_images.append((url, f'fanart-{img_type}'))
            print(f"fanart.tv returned {len(thumb_candidates) + len(other_images)} images for {mbid}")
    except Exception as e:
        print(f"fanart.tv error for {mbid}: {e}")
    return thumb_candidates, other_images


def _fetch_deezer(artist_name, seen_urls):
    """Fetch from Deezer. Returns list of (url, source) thumb candidates."""
    results = []
    try:
        deezer_url = f'https://api.deezer.com/search/artist?q={requests.utils.quote(artist_name)}'
        response = http_session.get(deezer_url, timeout=10)
        if response.ok:
            data = response.json()
            artists_data = data.get('data', [])
            if artists_data:
                url = artists_data[0].get('picture_xl')
                if url and 'user' not in url and url not in seen_urls:
                    seen_urls.add(url)
                    results.append((url, 'deezer'))
    except Exception as e:
        print(f"Deezer error: {e}")
    return results


def _fetch_genius(artist_name, seen_urls):
    """Fetch from Genius. Returns list of (url, source) thumb candidates."""
    results = []
    if not GENIUS_ACCESS_TOKEN:
        return results
    try:
        genius_headers = {
            'Authorization': f'Bearer {GENIUS_ACCESS_TOKEN}',
            'User-Agent': f'{MUSICBRAINZ_APP_NAME}/{MUSICBRAINZ_VERSION}'
        }
        genius_url = f'https://api.genius.com/search?q={requests.utils.quote(artist_name)}'
        response = throttled_get(genius_url, headers=genius_headers, timeout=10)
        if response.ok:
            data = response.json()
            for hit in data.get('response', {}).get('hits', []):
                primary_artist = hit.get('result', {}).get('primary_artist', {})
                if primary_artist.get('name', '').lower() == artist_name.lower():
                    img_url = primary_artist.get('image_url')
                    if img_url and 'default_avatar' not in img_url and img_url not in seen_urls:
                        seen_urls.add(img_url)
                        results.append((img_url, 'genius'))
                    break
    except Exception as e:
        print(f"Genius error: {e}")
    return results


def _fetch_audiodb(mbid, seen_urls):
    """Fetch from TheAudioDB. Returns list of (url, source)."""
    results = []
    try:
        audiodb_url = f'https://theaudiodb.com/api/v1/json/2/artist-mb.php?i={mbid}'
        response = throttled_get(audiodb_url, timeout=10)
        if response.ok:
            data = response.json()
            artists = data.get('artists')
            if artists:
                artist = artists[0]
                for field in ['strArtistFanart', 'strArtistFanart2', 'strArtistFanart3',
                              'strArtistFanart4', 'strArtistThumb', 'strArtistWideThumb',
                              'strArtistCutout']:
                    url = artist.get(field)
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        results.append((url, f'audiodb-{field}'))
                print(f"TheAudioDB returned {len(results)} images for {mbid}")
    except Exception as e:
        print(f"TheAudioDB error for {mbid}: {e}")
    return results


def _fetch_wikipedia(artist_info, seen_urls):
    """Fetch from Wikipedia. Returns list of (url, source)."""
    results = []
    try:
        wiki_title = get_wiki_title_from_urls(artist_info['urls'])
        if wiki_title:
            wiki_images = get_wikipedia_images(wiki_title, max_images=10)
            for url in wiki_images:
                if url not in seen_urls:
                    seen_urls.add(url)
                    results.append((url, 'wikipedia'))
            if results:
                print(f"Wikipedia returned {len(results)} images for {artist_info['mbid']}")
    except Exception as e:
        print(f"Wikipedia images error: {e}")
    return results


MIN_IMAGE_BYTES = 10 * 1024       # 10KB - skip tiny placeholders
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10MB per image


def _download_image(url_source):
    """Download a single image and compute its metrics."""
    url, source = url_source
    try:
        response = http_session.get(url, timeout=10, stream=True)
        if response.ok:
            content_length = response.headers.get('Content-Length')
            if content_length:
                size = int(content_length)
                if size > MAX_IMAGE_BYTES:
                    print(f"Skipping oversized image from {source}: {size} bytes")
                    response.close()
                    return None
                if size < MIN_IMAGE_BYTES:
                    print(f"Skipping tiny image from {source}: {size} bytes")
                    response.close()
                    return None
            chunks = []
            total = 0
            for chunk in response.iter_content(chunk_size=65536):
                total += len(chunk)
                if total > MAX_IMAGE_BYTES:
                    print(f"Image exceeded 10MB during download from {source}")
                    response.close()
                    return None
                chunks.append(chunk)
            data = b''.join(chunks)
            if len(data) < MIN_IMAGE_BYTES:
                print(f"Skipping tiny image from {source}: {len(data)} bytes")
                return None
            metrics = ImageMetrics(data)
            return (url, source, metrics)
    except Exception as e:
        print(f"Error downloading from {source}: {e}")
    return None


def get_images(mbid, artist_dir):
    """Fetch artist images from multiple sources with smart deduplication.

    Priority order for thumb (slot 0):
    1. fanart.tv artistthumb
    2. Deezer (single image, good for thumb)
    3. Genius (single image, good for thumb)
    Then for remaining slots:
    4. fanart.tv artistbackground/artist4kbackground
    5. TheAudioDB
    6. Wikipedia
    """
    seen_urls = set()  # Thread-safe for reads; each fetcher only adds unique URLs
    artist_info = get_artist_info(mbid)
    artist_name = artist_info['name']

    # Parallel URL collection from all sources
    with ThreadPoolExecutor(max_workers=5) as executor:
        fanart_future = executor.submit(_fetch_fanart, mbid, seen_urls)
        deezer_future = executor.submit(_fetch_deezer, artist_name, seen_urls)
        genius_future = executor.submit(_fetch_genius, artist_name, seen_urls)
        audiodb_future = executor.submit(_fetch_audiodb, mbid, seen_urls)
        wiki_future = executor.submit(_fetch_wikipedia, artist_info, seen_urls)

        fanart_thumbs, fanart_other = fanart_future.result()
        deezer_results = deezer_future.result()
        genius_results = genius_future.result()
        audiodb_results = audiodb_future.result()
        wiki_results = wiki_future.result()

    # Assemble in priority order: thumb candidates first, then gallery images
    thumb_candidates = fanart_thumbs + deezer_results + genius_results
    other_images = fanart_other + audiodb_results + wiki_results
    images = thumb_candidates + other_images
    print(f"Total URLs collected: {len(images)} ({len(thumb_candidates)} thumb candidates)")

    # Download images in parallel with early termination
    downloaded = []
    enough = threading.Event()

    def download_with_cancel(url_source):
        if enough.is_set():
            return None
        result = _download_image(url_source)
        if result and len(downloaded) >= 16:
            enough.set()
        return result

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = []
        for img in images:
            if enough.is_set():
                break
            futures.append(executor.submit(download_with_cancel, img))
        for future in as_completed(futures):
            result = future.result()
            if result:
                downloaded.append(result)
                if len(downloaded) >= 16:
                    enough.set()

    print(f"Downloaded {len(downloaded)} images")

    # 7. Deduplicate and select best images (sequential for correctness)
    first_image_data = None
    metrics_cache = []  # List of ImageMetrics for kept images
    img_slot = 0

    for url, source, metrics in downloaded:
        if img_slot >= 8:
            break

        if not metrics.hash:
            continue

        # Check for duplicates
        dup_idx = find_duplicate_index(metrics, metrics_cache)

        if dup_idx >= 0:
            # Found a duplicate - check if new one is higher resolution
            if metrics.resolution > metrics_cache[dup_idx].resolution:
                print(f"Replacing slot {dup_idx} with higher res from {source} ({metrics.resolution} > {metrics_cache[dup_idx].resolution})")
                metrics_cache[dup_idx] = metrics
            else:
                print(f"Skipping duplicate from {source}: {url[:50]}...")
            continue

        # New unique image
        metrics_cache.append(metrics)
        img_slot += 1

    # 8. Find the most portrait-oriented image for device background (slot 7)
    if len(metrics_cache) >= 8:
        # Find index of most portrait image (highest ratio, but not slot 0 which is thumb)
        best_portrait_idx = 1
        best_ratio = metrics_cache[1].aspect_ratio if len(metrics_cache) > 1 else 0
        for i in range(2, len(metrics_cache)):
            if metrics_cache[i].aspect_ratio > best_ratio:
                best_ratio = metrics_cache[i].aspect_ratio
                best_portrait_idx = i

        # If the best portrait isn't already at slot 7, swap it there
        if best_portrait_idx != 7 and best_ratio > 0.8:
            print(f"Moving image {best_portrait_idx} (ratio {best_ratio:.2f}) to slot 7 for device background")
            metrics_cache[7], metrics_cache[best_portrait_idx] = metrics_cache[best_portrait_idx], metrics_cache[7]

    # 9. Write all images to disk
    for i, metrics in enumerate(metrics_cache):
        write_images(artist_dir, i, metrics.data)
        print(f"Saved image {i} for {artist_dir} (res: {metrics.resolution}, aspect: {metrics.aspect_ratio:.2f})")
        if i == 0:
            first_image_data = metrics.data

    # 10. If we have fewer than 6 numbered images (1-6), use thumb as an additional numbered image
    # Note: image 7 is reserved for deviceBackgroundImage
    existing_numbered = sum(1 for n in range(1, 7) if os.path.isfile(f'/artists/{artist_dir}/{n}.jpg'))
    if existing_numbered < 6 and first_image_data:
        # Find the next available slot (1-6 only)
        for slot in range(1, 7):
            if not os.path.isfile(f'/artists/{artist_dir}/{slot}.jpg'):
                # Save the thumb image also as this numbered image (original size)
                path = f'/artists/{artist_dir}/{slot}.jpg'
                with open(path, 'wb') as f:
                    f.write(first_image_data)
                print(f"Copied thumb to {slot}.jpg for {artist_dir}")
                break


class ImageMetrics:
    """Container for all computed image metrics (single decode)."""
    __slots__ = ['data', 'resolution', 'aspect_ratio', 'hash', '_histogram']

    def __init__(self, image_data):
        self.data = image_data
        self._histogram = None

        # Decode once for grayscale metrics
        nparr = np.frombuffer(image_data, np.uint8)
        img_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if img_gray is None:
            self.resolution = 0
            self.aspect_ratio = 0
            self.hash = None
            return

        height, width = img_gray.shape[:2]
        self.resolution = height * width
        self.aspect_ratio = height / width if width > 0 else 0

        # Compute perceptual hash as 64-bit integer (fast XOR comparison)
        resized = cv2.resize(img_gray, (8, 8), interpolation=cv2.INTER_AREA)
        avg = resized.mean()
        self.hash = 0
        for i, pixel in enumerate(resized.flatten()):
            if pixel > avg:
                self.hash |= (1 << i)

    @property
    def histogram(self):
        """Lazy histogram computation - only computed when first accessed."""
        if self._histogram is None and self.data is not None:
            nparr = np.frombuffer(self.data, np.uint8)
            img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_color is not None:
                hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
                cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                self._histogram = hist
        return self._histogram


def histogram_similarity(hist1, hist2):
    """Compare two histograms. Returns 0-1 where 1 is identical."""
    if hist1 is None or hist2 is None:
        return 0
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def find_duplicate_index(new_metrics, existing_metrics_list, hash_threshold=10, hist_threshold=0.85):
    """Check if image is similar to any existing image using hash, then histogram (lazy).
    Returns index of duplicate or -1 if no match."""
    if new_metrics.hash is None:
        return -1

    for i, existing in enumerate(existing_metrics_list):
        # Check perceptual hash first (fast integer XOR + popcount)
        if existing.hash is not None:
            dist = bin(new_metrics.hash ^ existing.hash).count('1')
            if dist < hash_threshold:
                return i

        # Check histogram only if hash didn't match (lazy, slower)
        if existing.histogram is not None:
            similarity = histogram_similarity(new_metrics.histogram, existing.histogram)
            if similarity > hist_threshold:
                print(f"  Histogram match: {similarity:.2f} similarity")
                return i

    return -1


def resize_image_to_width(img, target_width):
    """Resize image to target width maintaining aspect ratio. Returns resized image or original if smaller."""
    original_height, original_width = img.shape[:2]

    # Don't upscale - if image is smaller than target, return original
    if original_width <= target_width:
        return img

    aspect_ratio = target_width / original_width
    new_height = int(original_height * aspect_ratio)
    return cv2.resize(img, (target_width, new_height), interpolation=cv2.INTER_AREA)


def cropThumb(image):
    """Create 160x120 thumbnail from image."""
    img = cv2.imread(image)
    if img is None:
        print(f"Error: Could not read image {image}")
        return

    original_height, original_width = img.shape[:2]
    new_width = 160
    aspect_ratio = new_width / original_width
    new_height = int(original_height * aspect_ratio)

    cropped_image = cv2.resize(img, (new_width, new_height))

    # Crop center 160x120
    x_start = int((cropped_image.shape[1] / 2) - 80)
    y_start = 0
    x_end = int((cropped_image.shape[1] / 2) + 80)
    y_end = min(120, cropped_image.shape[0])

    cropped_img = cropped_image[y_start:y_end, x_start:x_end]
    cv2.imwrite(image, cropped_img)


def write_images(artist_dir, img_num, data):
    """Write image data to disk at original resolution, re-encoded as 90% quality JPEG."""
    # Decode the image data
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: Could not decode image for slot {img_num}")
        return

    # Re-encode at 90% JPEG quality (saves space, consistent format)
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    if img_num == 0:
        # Thumb: save at full resolution, resize on-demand when serving
        path = f'/artists/{artist_dir}/thumb.jpg'
        with open(path, 'wb') as f:
            f.write(buffer.tobytes())
    else:
        # Store original resolution - resizing happens on-demand when serving
        path = f'/artists/{artist_dir}/{img_num}.jpg'
        with open(path, 'wb') as f:
            f.write(buffer.tobytes())


@app.route("/v3.0/en-US/music/artist/<mbid>", strict_slashes=False)
def overview(mbid):
    """Artist overview endpoint."""
    if not validate_mbid(mbid):
        return abort(400)
    try:
        artist_info = get_artist_info(mbid)
    except Exception as e:
        print(f"Error getting artist info for {mbid}: {e}")
        return abort(404)

    artist_name = artist_info['name']
    artist_dir = sanitize_artist_name(artist_name)
    touch_artist(artist_dir)
    check_cache_size()
    img_id = mbid_to_image_uuid(mbid, 10)  # 10 for thumbnail reference

    # Create artist directory if needed
    try:
        Path(f"/artists/{artist_dir}").mkdir(exist_ok=True)
    except PermissionError:
        print(f"Permission denied: Unable to create dir.")
        return "500 Internal Server Error", 500
    except Exception as e:
        print(f"An error occurred: {e}")
        return "500 Internal Server Error", 500

    # Monthly image refresh: if artist dir is older than 30 days, clear images
    artist_path = Path(f"/artists/{artist_dir}")
    try:
        dir_mtime = datetime.fromtimestamp(artist_path.stat().st_mtime).date()
        if (date.today() - timedelta(30)) >= dir_mtime:
            for i in range(1, 8):
                img_path = f"/artists/{artist_dir}/{i}.jpg"
                if os.path.isfile(img_path):
                    os.remove(img_path)
                thumb_path = f"/artists/{artist_dir}/thumb.jpg"
                if os.path.isfile(thumb_path):
                    os.remove(thumb_path)
    except FileNotFoundError:
        pass

    # Auto-fetch images in background if not cached yet
    if not os.path.isfile(f"/artists/{artist_dir}/thumb.jpg"):
        def _bg_fetch(m, d):
            lock = get_artist_lock(d)
            with lock:
                if not os.path.isfile(f"/artists/{d}/thumb.jpg"):
                    try:
                        get_images(m, d)
                    except Exception as e:
                        print(f"Background image fetch error for {d}: {e}")
        threading.Thread(target=_bg_fetch, args=(mbid, artist_dir), daemon=True).start()

    xml_data = f'''<?xml version="1.0" encoding="utf-8"?><a:entry xmlns:a="http://www.w3.org/2005/Atom" xmlns:os="http://a9.com/-/spec/opensearch/1.1/" xmlns="http://schemas.zune.net/catalog/music/2007/10"><a:link rel="zune://artist/biography" type="application/atom+xml" href="/v3.0/en-US/music/artist/{mbid}/biography" /><a:link rel="self" type="application/atom+xml" href="/v3.0/en-US/music/artist/{mbid}" /><a:updated>1900-01-01T00:00:00.000000Z</a:updated><a:title type="text">{html.escape(artist_name)}</a:title><a:id>urn:uuid:{mbid}</a:id><sortName>{html.escape(artist_info['sort_name'])}</sortName><playRank>0</playRank><playCount>0</playCount><favoriteCount>0</favoriteCount><sendCount>0</sendCount><isDisabled>False</isDisabled><startDate>1900-01-01T00:00:00Z</startDate><image><id>urn:uuid:{img_id}</id></image><a:author><a:name>Microsoft Corporation</a:name></a:author></a:entry>''' + "\n\n\n\n\n\n\n "

    save_xml_response(artist_name, f"{mbid}_overview.xml", xml_data)

    return Response(xml_data, mimetype='application/xml', headers={
        'Content-Type': 'application/xml',
        'Cache-Control': 'max-age=86400',
        'Connection': 'keep-alive',
        'Keep-Alive': 'timeout=150000, max=10',
        'Expires': 'Sun, 19 Apr 2071 10:00:00 GMT',
        'Access-Control-Allow-Origin': '*'
    })


@app.route("/v3.0/en-US/music/artist/<mbid>/deviceBackgroundImage", strict_slashes=False)
def backgroundImage(mbid):
    """Return background image for device. Supports optional width query param."""
    if not validate_mbid(mbid):
        return abort(400)
    response_headers = {
        'Content-Type': 'image/jpeg',
        'Cache-Control': 'max-age=86400',
        'Connection': 'keep-alive',
        'Keep-Alive': 'timeout=150000, max=10',
        'Expires': 'Sun, 19 Apr 2071 10:00:00 GMT',
        'Access-Control-Allow-Origin': '*'
    }

    try:
        artist_info = get_artist_info(mbid)
    except Exception:
        return abort(404)

    artist_dir = sanitize_artist_name(artist_info['name'])
    touch_artist(artist_dir)
    target_file = f"/artists/{artist_dir}/7.jpg"

    # Wait for image to be available (background fetch may be in progress)
    start_time = time()
    while not os.path.isfile(target_file):
        sleep(0.5)
        if (time() - start_time) > 10:
            return abort(404)

    # Check for query parameters
    target_width = request.args.get('width', type=int)
    if target_width is not None:
        target_width = min(target_width, MAX_IMAGE_WIDTH)
    full_resolution = request.args.get('full', '').lower() == 'true'

    # If full=true, return original resolution
    if full_resolution:
        response = make_response(send_file(target_file, mimetype="image/jpeg"))
        response.headers = response_headers
        return response

    if target_width and target_width > 0:
        # Resize on-demand
        img = cv2.imread(target_file)
        if img is None:
            print(f"Failed to read image for resize: {target_file}")
            return abort(500)
        resized = resize_image_to_width(img, target_width)
        _, buffer = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
        response = make_response(send_file(BytesIO(buffer.tobytes()), mimetype="image/jpeg"))
        response.headers = response_headers
        return response

    # No width param - return original
    response = make_response(send_file(target_file, mimetype="image/jpeg"))
    response.headers = response_headers
    return response


@app.route("/v3.0/en-US/image/<imgID>/", strict_slashes=False)
def getImg(imgID):
    """Return image by UUID. Supports optional width query param for resizing."""
    response_headers = {
        'Content-Type': 'image/jpeg',
        'Cache-Control': 'max-age=86400',
        'Connection': 'keep-alive',
        'Keep-Alive': 'timeout=150000, max=10',
        'Expires': 'Sun, 19 Apr 2071 10:00:00 GMT',
        'Access-Control-Allow-Origin': '*'
    }

    # Parse UUID to get MBID prefix and image number
    parts = imgID.split("-")
    if len(parts) < 2:
        return abort(404)

    mbid_prefix = parts[0]
    try:
        num = int(parts[1])
    except ValueError:
        return abort(404)

    # Resolve full MBID from prefix
    mbid = resolve_mbid_from_prefix(mbid_prefix)
    if not mbid:
        print(f"Could not resolve MBID from prefix: {mbid_prefix}")
        return abort(404)

    try:
        artist_info = get_artist_info(mbid)
    except Exception:
        return abort(404)

    artist_dir = sanitize_artist_name(artist_info['name'])
    touch_artist(artist_dir)

    if num not in range(1, 8) and num != 10:
        return abort(404)

    # Wait for image to be available (background fetch may be in progress)
    target_file = f"/artists/{artist_dir}/thumb.jpg" if num == 10 else f"/artists/{artist_dir}/{num}.jpg"

    start_time = time()
    while not os.path.isfile(target_file):
        sleep(0.5)
        if (time() - start_time) > 10:
            return abort(404)

    # Check for query parameters
    target_width = request.args.get('width', type=int)
    if target_width is not None:
        target_width = min(target_width, MAX_IMAGE_WIDTH)
    full_resolution = request.args.get('full', '').lower() == 'true'

    # If full=true, return original resolution regardless of image type
    if full_resolution:
        response = make_response(send_file(target_file, mimetype="image/jpeg"))
        response.headers = response_headers
        return response

    # For thumb (num == 10), default to 160x120 crop for device if no width specified
    if num == 10 and not target_width:
        # Serve 160x120 cropped thumb for device (legacy behavior)
        img = cv2.imread(target_file)
        if img is None:
            print(f"Failed to read thumb for crop: {target_file}")
            return abort(500)

        # Resize to 160 width maintaining aspect ratio, then crop center 160x120
        original_height, original_width = img.shape[:2]
        new_width = 160
        aspect_ratio = new_width / original_width
        new_height = int(original_height * aspect_ratio)
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Crop center 160x120
        x_start = int((resized.shape[1] / 2) - 80)
        y_start = 0
        x_end = int((resized.shape[1] / 2) + 80)
        y_end = min(120, resized.shape[0])
        cropped = resized[y_start:y_end, x_start:x_end]

        _, buffer = cv2.imencode('.jpg', cropped, [cv2.IMWRITE_JPEG_QUALITY, 90])
        response = make_response(send_file(BytesIO(buffer.tobytes()), mimetype="image/jpeg"))
        response.headers = response_headers
        return response

    if target_width and target_width > 0:
        # Resize on-demand to specified width
        img = cv2.imread(target_file)
        if img is None:
            print(f"Failed to read image for resize: {target_file}")
            return abort(500)
        resized = resize_image_to_width(img, target_width)
        _, buffer = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
        response = make_response(send_file(BytesIO(buffer.tobytes()), mimetype="image/jpeg"))
        response.headers = response_headers
        return response

    # No width param - return original (full resolution)
    response = make_response(send_file(target_file, mimetype="image/jpeg"))
    response.headers = response_headers
    return response


@app.route("/v3.0/en-US/music/artist/<mbid>/albums", strict_slashes=False)
def albums(mbid):
    """Albums endpoint (stub)."""
    if not validate_mbid(mbid):
        return abort(400)
    try:
        artist_info = get_artist_info(mbid)
        return artist_info['name']
    except Exception:
        return abort(404)


@app.route("/v3.0/en-US/music/artist/<mbid>/similarArtists", strict_slashes=False)
def similar(mbid):
    """Similar artists endpoint (stub)."""
    if not validate_mbid(mbid):
        return abort(400)
    try:
        artist_info = get_artist_info(mbid)
        return artist_info['name']
    except Exception:
        return abort(404)


@app.route("/v3.0/en-US/music/artist/<mbid>/tracks", strict_slashes=False)
def tracks(mbid):
    """Tracks endpoint (stub)."""
    if not validate_mbid(mbid):
        return abort(400)
    try:
        artist_info = get_artist_info(mbid)
        return artist_info['name']
    except Exception:
        return abort(404)


@app.route("/v3.0/en-US/music/artist/<mbid>/biography", strict_slashes=False)
def bio(mbid):
    """Artist biography endpoint."""
    if not validate_mbid(mbid):
        return abort(400)
    try:
        artist_info = get_artist_info(mbid)
    except Exception:
        return abort(404)

    artist_name = artist_info['name']
    bio_data = get_biography(mbid)

    if bio_data and bio_data.get('extract_html'):
        bio_content = bio_data['extract_html']
        # Add attribution based on source
        if bio_data.get('source') == 'wikipedia':
            attribution = f'<p><small>Biography from <a href="{bio_data.get("source_url", "")}">Wikipedia</a> (CC BY-SA)</small></p>'
            bio_content = bio_content + attribution
    else:
        bio_content = f"<p>No biography available for {html.escape(artist_name)}.</p>"

    xml_data = f'''<?xml version="1.0" encoding="utf-8"?><a:entry xmlns:a="http://www.w3.org/2005/Atom" xmlns:os="http://a9.com/-/spec/opensearch/1.1/" xmlns="http://schemas.zune.net/catalog/music/2007/10"><a:link rel="self" type="application/atom+xml" href="/v3.0/en-US/music/artist/{mbid}/biography" /><a:updated>1900-01-01T00:00:00.0000000Z</a:updated><a:title type="text">{html.escape(artist_name)}</a:title><a:id>tag:catalog.zune.net,1900-01-01:/music/artist/{mbid}/biography</a:id><a:content type="html">{html.escape(bio_content)}</a:content><a:author><a:name>Microsoft Corporation</a:name></a:author></a:entry>'''

    save_xml_response(artist_name, f"{mbid}_biography.xml", xml_data)

    return Response(xml_data, mimetype='application/xml', headers={
        'Content-Type': 'application/xml',
        'Cache-Control': 'max-age=86400',
        'Connection': 'keep-alive',
        'Keep-Alive': 'timeout=150000, max=10',
        'Expires': 'Sun, 19 Apr 2071 10:00:00 GMT',
        'Access-Control-Allow-Origin': '*'
    })


@app.route("/v3.0/en-US/music/artist/<mbid>/images", strict_slashes=False)
@limiter.limit("30 per minute")
def images(mbid):
    """Artist images list endpoint."""
    if not validate_mbid(mbid):
        return abort(400)
    try:
        artist_info = get_artist_info(mbid)
    except Exception:
        return abort(404)

    artist_name = artist_info['name']
    artist_dir = sanitize_artist_name(artist_name)
    touch_artist(artist_dir)

    # Ensure images are fetched before returning (with lock to prevent concurrent fetches)
    artist_lock = get_artist_lock(artist_dir)
    with artist_lock:
        if not os.path.isfile(f"/artists/{artist_dir}/thumb.jpg"):
            try:
                Path(f"/artists/{artist_dir}").mkdir(exist_ok=True)
            except Exception:
                pass
            get_images(mbid, artist_dir)

    # Only generate entries for images that actually exist (1-6 only, 7 is for deviceBackground)
    entries = []
    for i in range(1, 7):
        if os.path.isfile(f"/artists/{artist_dir}/{i}.jpg"):
            img_id = mbid_to_image_uuid(mbid, i)
            entries.append(f'''<a:entry><a:updated>1900-01-01T00:00:00.000000Z</a:updated><a:title type="text">List Of Items</a:title><a:id>urn:uuid:{img_id}</a:id><instances><imageInstance><id>urn:uuid:{img_id}</id><url>http://art.zune.net/1/{img_id}/504/image.jpg</url><format>jpg</format><width>1000</width><height>1000</height></imageInstance></instances></a:entry>''')

    xml_data = f'''<?xml version="1.0" encoding="utf-8"?><a:feed xmlns:a="http://www.w3.org/2005/Atom" xmlns:os="http://a9.com/-/spec/opensearch/1.1/" xmlns="http://schemas.zune.net/catalog/music/2007/10"><a:link rel="self" type="application/atom+xml" href="/v3.0/en-US/music/artist/{mbid}/images" /><a:updated>1900-01-01T00:00:00.000000Z</a:updated><a:title type="text">List Of Items</a:title><a:id>tag:catalog.zune.net,1966-09-20:/music/artist/{mbid}/images</a:id>{''.join(entries)}<a:author><a:name>Microsoft Corporation</a:name></a:author></a:feed>'''

    save_xml_response(artist_name, f"{mbid}_images.xml", xml_data)

    return Response(xml_data, mimetype='application/xml', headers={
        'Content-Type': 'application/xml',
        'Cache-Control': 'max-age=86400',
        'Connection': 'keep-alive',
        'Keep-Alive': 'timeout=150000, max=10',
        'Expires': 'Sun, 19 Apr 2071 10:00:00 GMT',
        'Access-Control-Allow-Origin': '*'
    })


@app.route("/healthz")
@limiter.exempt
def healthz():
    """Health check endpoint for Docker/Traefik."""
    return "ok", 200


def init_app():
    """Initialize application - create artists directory and load mappings."""
    # Validate required environment variables
    required_vars = {
        'FANART_TV_API_KEY': FANART_TV_API_KEY,
        'LASTFM_API_KEY': LASTFM_API_KEY,
        'LASTFM_API_SECRET': LASTFM_API_SECRET,
        'MUSICBRAINZ_CONTACT': MUSICBRAINZ_CONTACT,
    }
    missing = [name for name, val in required_vars.items() if not val]
    if missing:
        print(f"FATAL: Missing required environment variables: {', '.join(missing)}")
        print("Set these variables before starting the server.")
        exit(1)

    if not GENIUS_ACCESS_TOKEN:
        print("WARNING: GENIUS_ACCESS_TOKEN not set. Genius image source disabled.")

    print("Checking if /artists/ exists")
    try:
        Path("/artists").mkdir(exist_ok=True)
        print("Directory /artists/ ready.")
    except PermissionError:
        print("Permission denied: Unable to create directory /artists/")
        exit()
    except Exception as e:
        print(f"An error occurred: {e}")
        exit()

    # Load MBID prefix mappings from disk
    load_mbid_map()


if __name__ == '__main__':
    init_app()
    app.run(host="127.0.0.2", port=80)
else:
    init_app()
