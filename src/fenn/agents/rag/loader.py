from pathlib import Path


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".json", ".py", ".js", ".ts", ".html", ".css", ".yaml", ".yml", ".toml", ".csv"}


def load_documents(source):
    """
    Load text content from a source and return it as a list of strings.

    Supported sources:
    - File path (.txt, .md, .pdf)
    - Folder path (recursively loads all supported files)
    - YouTube URL (fetches transcript)
    - Wikipedia URL (fetches article text)
    - Any other HTTP/HTTPS URL (fetches and strips HTML)

    Parameters
    ----------
    source : str or Path
        The source to load from.

    Returns
    -------
    list of str
        List of non-empty document strings.

    Raises
    ------
    FileNotFoundError
        If a local path does not exist.
    ValueError
        If a URL returns content that is too short or blocked.
    ImportError
        If a required optional dependency is not installed.
    """
    source = str(source)

    # ── URL routing ───────────────────────────────────────
    if "youtube.com" in source or "youtu.be" in source:
        return [_load_youtube(source)]

    if source.startswith("http://") or source.startswith("https://"):
        return [_load_url(source)]

    # ── Local file / folder ───────────────────────────────
    path = Path(source).resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"[cofone] path not found: {path}\n"
            f"Check that the file or folder exists and the path is correct."
        )

    docs = []
    if path.is_file():
        doc = _read_file(path)
        if doc:
            docs.append(doc)
    elif path.is_dir():
        found = list(path.rglob("*"))
        supported = [f for f in found if f.is_file() and f.suffix in SUPPORTED_EXTENSIONS]
        if not supported:
            print(f"[cofone] warning: no supported files found in {path}")
            print(f"[cofone] supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}")
        for f in supported:
            doc = _read_file(f)
            if doc:
                docs.append(doc)

    return [d for d in docs if d]


def _read_file(path):
    """Read a single file. Returns None on error (logged to stdout)."""
    try:
        if path.suffix == ".pdf":
            return _read_pdf(path)
        return path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[cofone] read error {path.name}: {e}")
        return None


def _read_pdf(path):
    """
    Extract text from a PDF file using pypdf.
    Requires: pip install "cofone[pdf]"  or  pip install pypdf
    """
    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        pages  = [p.extract_text() or "" for p in reader.pages]
        text   = "\n".join(pages).strip()
        if not text:
            print(f"[cofone] warning: PDF '{path.name}' returned no text (may be scanned/image-based)")
        return text or None
    except ImportError:
        raise ImportError(
            "[cofone] pypdf not installed.\n"
            "Run: pip install \"cofone[pdf]\"  or  pip install pypdf"
        )


def _load_url(url):
    """
    Fetch a web page and return its visible text content.
    Requires: pip install httpx beautifulsoup4  (included in cofone core)

    For Wikipedia URLs, delegates to _load_wikipedia() for cleaner output.
    """
    try:
        import httpx
        from bs4 import BeautifulSoup

        if "wikipedia.org" in url:
            return _load_wikipedia(url)

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = httpx.get(url, timeout=15, follow_redirects=True, headers=headers)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)

        if len(text) < 100:
            raise ValueError(
                f"[cofone] page too short or blocked: {url}\n"
                f"The server may require authentication or block bots."
            )
        return text

    except ImportError:
        raise ImportError(
            "[cofone] httpx or beautifulsoup4 not installed.\n"
            "Run: pip install httpx beautifulsoup4"
        )


def _load_wikipedia(url):
    """
    Fetch a Wikipedia article's full text using the wikipedia package.
    Automatically detects language from the URL subdomain.
    Requires: pip install "cofone[web]"  or  pip install wikipedia
    """
    try:
        import wikipedia
        import re

        # Detect language from subdomain (it.wikipedia, fr.wikipedia, etc.)
        lang_match = re.search(r"https?://([a-z]+)\.wikipedia", url)
        lang = lang_match.group(1) if lang_match else "en"
        wikipedia.set_lang(lang)

        # Extract article title from URL slug
        slug = url.rstrip("/").split("/wiki/")[-1]
        slug = slug.replace("_", " ")
        slug = re.sub(r"%[0-9A-Fa-f]{2}", lambda m: bytes.fromhex(m.group()[1:]).decode("utf-8", errors="replace"), slug)

        page = wikipedia.page(slug, auto_suggest=False)
        return page.content

    except ImportError:
        raise ImportError(
            "[cofone] wikipedia package not installed.\n"
            "Run: pip install \"cofone[web]\"  or  pip install wikipedia"
        )
    except Exception as e:
        raise ValueError(f"[cofone] Wikipedia error for '{url}': {e}")


def _load_youtube(url):
    """
    Fetch a YouTube video's transcript/subtitles.
    Language priority: English first, then Italian, then any available.
    Auto-generated captions are supported.
    Requires: pip install "cofone[web]"  or  pip install youtube-transcript-api
    Supports youtube-transcript-api >= 0.6.0 and >= 0.7.0 (dual fallback).
    """
    try:
        import re
        from youtube_transcript_api import YouTubeTranscriptApi

        video_id = re.search(r"(?:v=|youtu\.be/)([^&\n?#]+)", url)
        if not video_id:
            raise ValueError(
                f"[cofone] could not extract video ID from URL: {url}\n"
                f"Expected format: https://www.youtube.com/watch?v=VIDEO_ID"
            )

        vid = video_id.group(1)

        try:
            # API >= 0.7.0
            ytt = YouTubeTranscriptApi()
            transcript_list = ytt.list(vid)
            transcript = transcript_list.find_transcript(["en", "it"]).fetch()
            return " ".join(
                t.get("text", str(t)) if isinstance(t, dict) else str(t)
                for t in transcript
            )
        except Exception:
            # Fallback: API < 0.7.0
            transcript = (
                YouTubeTranscriptApi
                .list_transcripts(vid)
                .find_transcript(["en", "it"])
                .fetch()
            )
            return " ".join(t["text"] for t in transcript)

    except ImportError:
        raise ImportError(
            "[cofone] youtube-transcript-api not installed.\n"
            "Run: pip install \"cofone[web]\"  or  pip install youtube-transcript-api"
        )