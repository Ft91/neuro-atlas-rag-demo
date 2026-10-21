"""Wire existing retrieval/storage and provider adapters; never reindex."""
from hashlib import sha256
import json
from pathlib import Path
from ai_processor.openai_client import OpenAIClient
from ai_processor.providers import create_provider
from rag.indexing import SearchIndex
from rag.pipeline import RAGPipeline
from rag.cache import SQLiteCache
from rag.retrieval import HybridRetriever, SearchSettings


def build_pipeline(root, settings, *, provider=None, allow_paid=False, mode="hybrid"):
    root = Path(root)
    injected_provider = provider is not None
    if (settings.embedding_provider, settings.embedding_model) != ("openai", "text-embedding-3-small"):
        raise ValueError("Текущий индекс требует openai / text-embedding-3-small")
    if provider is None:
        if not allow_paid:
            raise ValueError("Реальный LLM-вызов требует явного разрешения")
        provider = create_provider(settings)
    index = SearchIndex(root / "data/atlas_search.sqlite3")
    documents, records, revision = index.read()
    if len(documents) != 49 or len(records) != 20:
        raise ValueError("Разрешены только 49 документов из 20 статей")
    vector = None
    if mode != "fts":
        from storage.vector_db import VectorDatabase
        vector = VectorDatabase(str(root / "data/chroma_openai"), "atlas_openai_" + revision[:16],
                                embedding_model=settings.embedding_model, embedding_dimensions=1536)
        # Get only: a missing collection must not be silently created.
        vector.collection = vector.client.get_collection(vector.collection_name, embedding_function=None)
        if set(vector.collection.get()["ids"]) != {d["document_id"] for d in documents}:
            raise ValueError("Неполный векторный индекс")
        cases_path = root / "data/retrieval_cases_10.json"
        cases_hash = sha256(cases_path.read_bytes()).hexdigest()
        cache_path = root / ("data/query_vectors_" + cases_hash[:16] + ".json")
        saved_queries = {}
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            cases = json.loads(cases_path.read_text(encoding="utf-8"))
            if (cached["model"], cached["dimensions"], cached["cases_sha256"]) != (settings.embedding_model, 1536, cases_hash):
                raise ValueError("Несовместимые сохранённые векторы вопросов")
            vector._validate_embeddings(cached["vectors"], len(cases))
            saved_queries = {c["query"]: v for c,v in zip(cases, cached["vectors"])}

        def query_embeddings(texts):
            if all(t in saved_queries for t in texts):
                return [saved_queries[t] for t in texts]
            if not allow_paid:
                raise ValueError("Нет сохранённого вектора вопроса; платный вызов не разрешён")
            if not settings.openai_api_key:
                raise ValueError("Для нового эмбеддинга требуется OPENAI_API_KEY")
            # Reuse the existing OpenAI client; GigaChat credentials are never used here.
            own_client = not isinstance(provider, OpenAIClient)
            client = (OpenAIClient(settings.openai_api_key).client if own_client else provider.client)
            try:
                response = client.embeddings.create(model=settings.embedding_model, input=texts,
                                                     dimensions=1536, encoding_format="float")
                data = sorted(response.data, key=lambda x: x.index)
                if [x.index for x in data] != list(range(len(texts))):
                    raise ValueError("Embedding response order mismatch")
                return vector._validate_embeddings([x.embedding for x in data], len(texts))
            except Exception:
                raise RuntimeError("Ошибка OpenAI при получении эмбеддинга вопроса; подробности скрыты") from None
            finally:
                if own_client:
                    client.close()
        vector.embedding_function = query_embeddings
    cache = SQLiteCache(root / "data/rag_cache.sqlite3") if settings.cache_enabled and not injected_provider else None
    identity = {"provider": settings.llm_provider,
                "model": settings.openai_model if settings.llm_provider == "openai" else settings.gigachat_model,
                "embedding_provider": settings.embedding_provider, "embedding_model": settings.embedding_model}
    return RAGPipeline(root, HybridRetriever(index, vector,
        settings=SearchSettings(settings.search_top_k, settings.hybrid_search)), provider,
        cache=cache, cache_identity=identity)
