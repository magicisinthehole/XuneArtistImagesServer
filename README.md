# XuneArtistImagesServer

Artist image and biography server for [Xune](https://github.com/magicisinthehole/Xune). Serves artist metadata over the Zune catalog XML format, fetching images from multiple sources and caching them to disk.

Forked from [ZuneArtistImages](https://github.com/spidersandmoths/ZuneArtistImages), which fetched images from Discogs with Google Images Search as a fallback, and served biographies from Last.fm.

This fork replaces the image sources, rewrites the image pipeline with parallel fetching and perceptual hash deduplication, stores images at original resolution with on-demand resizing, adds input validation and rate limiting, and supports containerized deployment.

### Changes from upstream

**Image sources** -- Discogs and Google Images Search replaced with:

1. fanart.tv (artist thumbnails, backgrounds, 4K backgrounds)
2. Deezer (artist portraits)
3. Genius (artist photos)
4. TheAudioDB (multiple image types)
5. Wikipedia (article images)

All sources are fetched in parallel. Near-duplicate images are filtered using perceptual hashing and histogram comparison.

**Image storage** -- Images are stored at original resolution instead of being downsized to 480px on save. Resizing is done on-demand when served, via `width` and `full` query parameters.

**Biography** -- Still sourced from Last.fm (primary), with Wikipedia added as a fallback. Last.fm markup is cleaned (HTML stripped, paragraphs limited, Unicode normalized).

**Image IDs** -- Replaced Discogs-based image UUIDs with MBID-prefix-based IDs, removing the Discogs dependency from image URL resolution.

**Infrastructure** -- Added MBID input validation, artist name sanitization, per-IP rate limiting, per-service API throttling, connection pooling, 50 GB LRU disk cache eviction, health check endpoint, environment variable configuration, and Docker support.

## API

All endpoints follow the Zune catalog URL scheme. Artist lookup is by MusicBrainz ID.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v3.0/en-US/music/artist/<mbid>` | Artist overview (XML) |
| GET | `/v3.0/en-US/music/artist/<mbid>/images` | Image list (XML) |
| GET | `/v3.0/en-US/music/artist/<mbid>/biography` | Biography (XML) |
| GET | `/v3.0/en-US/music/artist/<mbid>/deviceBackgroundImage` | Background image (JPEG) |
| GET | `/v3.0/en-US/image/<imageId>/` | Image by ID (JPEG) |
| GET | `/v3.0/en-US/music/artist/<mbid>/albums` | Albums (stub) |
| GET | `/v3.0/en-US/music/artist/<mbid>/similarArtists` | Similar artists (stub) |
| GET | `/v3.0/en-US/music/artist/<mbid>/tracks` | Tracks (stub) |
| GET | `/healthz` | Health check |

### Query Parameters

Image endpoints accept the following optional parameters:

- `width` (int) -- resize to specified width in pixels, max 3840
- `full=true` -- return original resolution

## Setup

### Requirements

- Docker (recommended) or Python 3.12+
- API keys for: fanart.tv, Last.fm, MusicBrainz (contact email)
- Optional: Genius API token

### Environment Variables

Copy `.env.example` to `.env` and fill in your keys. See the file for registration links.

| Variable | Required | Description |
|----------|----------|-------------|
| `FANART_TV_API_KEY` | Yes | fanart.tv API key |
| `LASTFM_API_KEY` | Yes | Last.fm API key |
| `LASTFM_API_SECRET` | Yes | Last.fm API secret |
| `MUSICBRAINZ_CONTACT` | Yes | Contact email for MusicBrainz API |
| `GENIUS_ACCESS_TOKEN` | No | Genius API token |
| `GUNICORN_WORKERS` | No | Worker processes (default: 4) |
| `GUNICORN_THREADS` | No | Threads per worker (default: 8) |
| `LOG_LEVEL` | No | Log level (default: info) |

### Docker

```sh
docker build -t zuneapi .
docker run -d \
  --env-file .env \
  -v /path/to/cache:/artists \
  -p 8000:8000 \
  zuneapi
```

To match a specific host UID/GID for the volume mount:

```sh
docker build --build-arg APP_UID=568 --build-arg APP_GID=568 -t zuneapi .
```

### Without Docker

```sh
pip install -r requirements.txt
gunicorn main:app
```

Configuration is read from `gunicorn.conf.py` automatically.

## Image Storage

Images are cached to the `/artists` directory (or `artists/` relative to the working directory outside Docker). Each artist gets a subdirectory containing:

- `thumb.jpg` -- primary artist image
- `1.jpg` through `6.jpg` -- gallery images
- `7.jpg` -- device background image
- `xml/` -- cached XML responses

The cache is bounded at 50 GB with LRU eviction. Images refresh automatically after 30 days.

## Rate Limiting

- 120 requests/minute per IP (all endpoints)
- 30 requests/minute per IP (`/images` endpoint)
- Per-service throttling for upstream APIs (TheAudioDB, Last.fm, Genius)

## License

See upstream repository.
