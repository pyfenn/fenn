import re

def chunk_text(text, mode="smart", size=500, overlap=50):
    """
    Split a document string into chunks for indexing.

    Parameters
    ----------
    text : str
        The full document text to split.
    mode : str
        Chunking strategy. One of:
        - "smart"      : paragraphs first, then sentences if paragraph > 600 chars (default)
        - "paragraphs" : split only on blank lines
        - "sentences"  : split on sentence boundaries, group up to ~500 chars
        - "fixed"      : fixed-length slices with overlap
    size : int
        Chunk size in characters. Used only in "fixed" mode. Default: 500.
    overlap : int
        Overlap between consecutive chunks in characters. Used only in "fixed" mode. Default: 50.

    Returns
    -------
    list of str
        Non-empty text chunks.
    """
    if mode == "smart":
        return _chunk_smart(text)
    elif mode == "sentences":
        return _chunk_sentences(text)
    elif mode == "paragraphs":
        return _chunk_paragraphs(text)
    elif mode == "fixed":
        return _chunk_fixed(text, size, overlap)
    else:
        raise ValueError(
            f"[cofone] unknown chunk_mode '{mode}'.\n"
            f"Valid options: 'smart', 'paragraphs', 'sentences', 'fixed'."
        )


def _chunk_smart(text):
    """
    Smart chunking: paragraph-first with sentence-level fallback.

    Algorithm:
    1. Split into paragraphs on blank lines.
    2. For each paragraph: if len > 600 chars, further split into sentences.
    3. Return all non-empty chunks.
    """
    paragraphs = _chunk_paragraphs(text)
    chunks = []
    for para in paragraphs:
        if len(para) > 600:
            chunks.extend(_chunk_sentences(para))
        else:
            chunks.append(para)
    return [c for c in chunks if c.strip()]


def _chunk_paragraphs(text):
    """Split on blank lines. Each paragraph is one chunk."""
    blocks = re.split(r"\n\s*\n", text)
    return [b.strip() for b in blocks if b.strip()]


def _chunk_sentences(text):
    """
    Split on sentence boundaries (. ! ?) and group sentences
    into chunks of up to ~500 characters.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) < 500:
            current += " " + s
        else:
            if current:
                chunks.append(current.strip())
            current = s
    if current:
        chunks.append(current.strip())
    return chunks


def _chunk_fixed(text, size, overlap):
    """
    Split into fixed-length slices of `size` characters,
    with `overlap` characters shared between consecutive chunks.
    Overlap prevents losing context at chunk boundaries.
    """
    if overlap >= size:
        raise ValueError(
            f"[cofone] overlap ({overlap}) must be smaller than size ({size})."
        )
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks