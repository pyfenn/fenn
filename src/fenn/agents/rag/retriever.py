import json
import os
from pathlib import Path
from collections import defaultdict
from .chunker import chunk_text

try:
    import transformers
    transformers.logging.set_verbosity_error()
except ImportError:
    pass


EMBEDDING_ENV_KEYS = {
    "openai":      "OPENAI_API_KEY",
    "gemini":      "GEMINI_API_KEY",
    "cohere":      "COHERE_API_KEY",
    "mistral":     "MISTRAL_API_KEY",
    "openrouter":  "OPENROUTER_API_KEY",
    "voyage":      "VOYAGE_API_KEY",
    "jina":        "JINA_API_KEY",
    "nvidia":      "NVIDIA_API_KEY",
    "together":    "TOGETHER_API_KEY",
    "ollama":      None,
    "local":       None,
}

EMBEDDING_URLS = {
    "openai":      "https://api.openai.com/v1",
    "gemini":      "https://generativelanguage.googleapis.com/v1beta/openai",
    "mistral":     "https://api.mistral.ai/v1",
    "openrouter":  "https://openrouter.ai/api/v1",
    "nvidia":      "https://integrate.api.nvidia.com/v1",
    "together":    "https://api.together.xyz/v1",
    "ollama":      "http://localhost:11434/v1",
}

EMBEDDING_BATCH_SIZES = {
    "cohere":      96,
    "openai":      2048,
    "gemini":      100,
    "mistral":     512,
    "openrouter":  512,
    "nvidia":      256,
    "together":    512,
    "jina":        128,
    "voyage":      128,
    "ollama":      64,
    "local":       512,
}

LOCAL_EMBEDDING_PROVIDERS = {"local", "ollama"}


class Retriever:
    def __init__(
        self,
        use_faiss=False,
        embedding_provider="local",
        embedding_model="all-MiniLM-L6-v2",
        embedding_api_key=None,
        chunk_mode="smart",
        persist_path=None,
    ):
        self.chunks              = []
        self.use_faiss           = use_faiss
        self.embedding_provider  = embedding_provider
        self.embedding_model     = embedding_model
        self.embedding_api_key   = embedding_api_key
        self.chunk_mode          = chunk_mode
        self.persist_path        = Path(persist_path) if persist_path else None
        self._index              = defaultdict(list)
        self._faiss_index        = None
        self._local_embedder     = None
        self._loaded_from_disk   = False  # FIX: track disk load

    def _resolve_embedding_key(self):
        if self.embedding_api_key:
            return self.embedding_api_key
        if self.embedding_provider in LOCAL_EMBEDDING_PROVIDERS:
            return None
        env = EMBEDDING_ENV_KEYS.get(self.embedding_provider)
        if env:
            key = os.environ.get(env)
            if not key:
                raise ValueError(
                    f"[cofone] embedding API key not found for '{self.embedding_provider}'.\n"
                    f"Set {env} in your .env file and call load_dotenv(),\n"
                    f"or pass embedding_api_key='...' directly to RAG()."
                )
            return key
        return None

    def _embed(self, texts, input_type="search_document"):
        p = self.embedding_provider
        if p == "local":
            return self._embed_local(texts)
        if p in ("openai", "gemini", "mistral", "openrouter", "nvidia", "together"):
            return self._embed_openai_compat(texts)
        if p == "ollama":
            return self._embed_ollama(texts)
        if p == "cohere":
            return self._embed_cohere(texts, input_type=input_type)
        if p == "voyage":
            return self._embed_voyage(texts)
        if p == "jina":
            return self._embed_jina(texts)

    def _embed_local(self, texts):
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            if self._local_embedder is None:
                self._local_embedder = SentenceTransformer(self.embedding_model)
            batch_size = EMBEDDING_BATCH_SIZES["local"]
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                all_embeddings.append(
                    self._local_embedder.encode(batch, convert_to_numpy=True).astype("float32")
                )
            import numpy as np
            return np.concatenate(all_embeddings, axis=0)
        except ImportError:
            raise ImportError(
                "[cofone] sentence-transformers not installed.\n"
                "Run: pip install \"cofone[faiss]\"  or  pip install sentence-transformers"
            )

    def _embed_openai_compat(self, texts):
        try:
            from openai import OpenAI
            import numpy as np
            base_url   = EMBEDDING_URLS.get(self.embedding_provider, EMBEDDING_URLS["openai"])
            api_key    = self._resolve_embedding_key()
            client     = OpenAI(api_key=api_key or "local", base_url=base_url)
            batch_size = EMBEDDING_BATCH_SIZES.get(self.embedding_provider, 512)
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch    = texts[i:i + batch_size]
                response = client.embeddings.create(model=self.embedding_model, input=batch)
                all_embeddings.extend([d.embedding for d in response.data])
            return np.array(all_embeddings, dtype="float32")
        except ImportError:
            raise ImportError(
                "[cofone] 'openai' package not found.\n"
                "Run: pip install openai"
            )

    def _embed_ollama(self, texts):
        try:
            from openai import OpenAI
            import numpy as np
            client     = OpenAI(api_key="ollama", base_url=EMBEDDING_URLS["ollama"])
            batch_size = EMBEDDING_BATCH_SIZES["ollama"]
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch    = texts[i:i + batch_size]
                response = client.embeddings.create(model=self.embedding_model, input=batch)
                all_embeddings.extend([d.embedding for d in response.data])
            return np.array(all_embeddings, dtype="float32")
        except ImportError:
            raise ImportError(
                "[cofone] 'openai' package not found.\n"
                "Run: pip install openai"
            )

    def _embed_cohere(self, texts, input_type="search_document"):
        try:
            import cohere
            import numpy as np
            client     = cohere.Client(self._resolve_embedding_key())
            batch_size = EMBEDDING_BATCH_SIZES["cohere"]
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch    = texts[i:i + batch_size]
                response = client.embed(
                    texts=batch,
                    model=self.embedding_model,
                    input_type=input_type,
                )
                all_embeddings.extend(response.embeddings)
            return np.array(all_embeddings, dtype="float32")
        except ImportError:
            raise ImportError("[cofone] 'cohere' package not found.\nRun: pip install cohere")

    def _embed_voyage(self, texts):
        try:
            import voyageai
            import numpy as np
            client     = voyageai.Client(api_key=self._resolve_embedding_key())
            batch_size = EMBEDDING_BATCH_SIZES["voyage"]
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch    = texts[i:i + batch_size]
                response = client.embed(batch, model=self.embedding_model)
                all_embeddings.extend(response.embeddings)
            return np.array(all_embeddings, dtype="float32")
        except ImportError:
            raise ImportError(
                "[cofone] 'voyageai' package not found.\n"
                "Run: pip install voyageai"
            )

    def _embed_jina(self, texts):
        try:
            import httpx
            import numpy as np
            headers    = {
                "Authorization": f"Bearer {self._resolve_embedding_key()}",
                "Content-Type":  "application/json",
            }
            batch_size = EMBEDDING_BATCH_SIZES["jina"]
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                resp  = httpx.post(
                    "https://api.jina.ai/v1/embeddings",
                    json={"model": self.embedding_model, "input": batch},
                    headers=headers,
                    timeout=30,
                )
                resp.raise_for_status()
                all_embeddings.extend([d["embedding"] for d in resp.json()["data"]])
            return np.array(all_embeddings, dtype="float32")
        except ImportError:
            raise ImportError(
                "[cofone] 'httpx' package not found.\n"
                "Run: pip install httpx"
            )

    # ── Index ──────────────────────────────────────────────────────────────────

    def index(self, docs):
        # FIX: load from disk only once (first call), then accumulate normally
        if self.persist_path and not self._loaded_from_disk:
            if self._load_from_disk():
                self._loaded_from_disk = True
                return

        for doc in docs:
            for chunk in chunk_text(doc, mode=self.chunk_mode):
                self.chunks.append(chunk)

        if self.use_faiss:
            self._build_faiss()
            if self.persist_path:
                self._save_to_disk()
        else:
            self._build_bm25()

    def _build_bm25(self):
        for idx, chunk in enumerate(self.chunks):
            for word in set(chunk.lower().split()):
                self._index[word].append(idx)

    def _build_faiss(self):
        try:
            import faiss
            import numpy as np
        except ImportError:
            raise ImportError("[cofone] faiss-cpu not installed.\nRun: pip install \"cofone[faiss]\"")
        vectors = self._embed(self.chunks)
        faiss.normalize_L2(vectors)
        dim = vectors.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dim)
        self._faiss_index.add(vectors)

    def _save_to_disk(self):
        import faiss
        self.persist_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._faiss_index, str(self.persist_path / "index.faiss"))
        (self.persist_path / "chunks.json").write_text(
            json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[cofone] index saved to {self.persist_path} ({len(self.chunks)} chunks)")

    def _load_from_disk(self):
        index_file  = self.persist_path / "index.faiss"
        chunks_file = self.persist_path / "chunks.json"
        if not index_file.exists() or not chunks_file.exists():
            return False
        try:
            import faiss
            self._faiss_index = faiss.read_index(str(index_file))
            self.chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
            print(f"[cofone] index loaded from {self.persist_path} ({len(self.chunks)} chunks)")
            return True
        except Exception as e:
            print(f"[cofone] cache load failed ({e}), rebuilding index...")
            return False

    # ── Query ──────────────────────────────────────────────────────────────────

    def query(self, text, top_k=5):
        if self.use_faiss:
            return self._query_faiss(text, top_k)
        return self._query_bm25(text, top_k)

    def _query_faiss(self, text, top_k):
        import faiss
        import numpy as np
        vec = self._embed([text], input_type="search_query")
        faiss.normalize_L2(vec)
        _, indices = self._faiss_index.search(vec, min(top_k, len(self.chunks)))
        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]

    def _query_bm25(self, text, top_k):
        scores = defaultdict(int)
        for word in text.lower().split():
            for idx in self._index.get(word, []):
                scores[idx] += 1
        if scores:
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return [self.chunks[i] for i, _ in ranked[:top_k]]
        return self.chunks[:top_k]