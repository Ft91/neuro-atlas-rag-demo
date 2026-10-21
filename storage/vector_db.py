"""
Векторная база данных на основе ChromaDB.
Инкапсулирует логику работы с векторным хранилищем.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Optional
import openai
import logging
import math

logger = logging.getLogger(__name__)


class VectorDatabase:
    """Обертка для работы с ChromaDB."""
    
    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
        openai_api_key: str = None,
        embedding_function=None,
        embedding_model: str = "text-embedding-3-small",
        embedding_dimensions: int = 1536
    ):
        """
        Инициализирует векторную БД.
        
        Args:
            persist_directory: Директория хранения данных
            collection_name: Имя коллекции
            openai_api_key: API ключ OpenAI для эмбеддингов
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Инициализируем OpenAI
        if openai_api_key:
            openai.api_key = openai_api_key
        self._has_api_key = bool(openai_api_key)
        self.embedding_function = embedding_function
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.openai = openai
        
        # Инициализируем ChromaDB клиент с персистентностью
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False
            )
        )
        
        # Коллекция
        self.collection = None
        
        logger.info(f"VectorDB инициализирован: {persist_directory}")
    
    def get_or_create_collection(self, metadata=None):
        """Получает или создает коллекцию."""
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata=metadata or {"description": "RAG документы"},
                embedding_function=None
            )
            logger.info(f"Коллекция '{self.collection_name}' готова")
            return self.collection
        except Exception as e:
            logger.error(f"Ошибка создания коллекции: {e}")
            raise
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: List[Dict],
        ids: Optional[List[str]] = None
    ):
        """
        Добавляет документы в коллекцию.
        
        Args:
            texts: Список текстов
            metadatas: Список метаданных
            ids: Список ID документов
        """
        if not self.collection:
            self.get_or_create_collection()
        
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        
        try:
            # Создаем эмбеддинги через OpenAI
            logger.info(f"Создание эмбеддингов для {len(texts)} документов...")
            embeddings = self._create_embeddings(texts)
            
            # Добавляем в коллекцию
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Добавлено {len(texts)} документов")
        except Exception as e:
            logger.error(f"Ошибка добавления документов: {e}")
            raise
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict] = None
    ) -> Dict:
        """
        Выполняет семантический поиск.
        
        Args:
            query: Поисковый запрос
            n_results: Количество результатов
            where: Фильтр по метаданным
            
        Returns:
            Результаты поиска
        """
        if not self.collection:
            self.get_or_create_collection()
        
        try:
            # Создаем эмбеддинг для запроса
            query_embedding = self._create_embeddings([query])[0]
            
            # Выполняем поиск
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where
            )
            
            return results
        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")
            raise
    
    def _create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Создает эмбеддинги через OpenAI.
        
        Args:
            texts: Список текстов
            
        Returns:
            Список векторов эмбеддингов
        """
        if self.embedding_function is not None:
            return self._validate_embeddings(self.embedding_function(texts), len(texts))
        if not self._has_api_key:
            raise ValueError("OpenAI key required unless an explicit test embedding function is supplied")
        embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = self.openai.embeddings.create(
                    input=batch,
                    model=self.embedding_model
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Ошибка создания эмбеддингов: {e}")
                raise
        
        return self._validate_embeddings(embeddings, len(texts))
    
    def get_stats(self) -> Dict:
        """
        Получает статистику коллекции.
        
        Returns:
            Словарь со статистикой
        """
        if not self.collection:
            self.get_or_create_collection()
        
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "document_count": count
            }
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return {"error": str(e)}
    
    def delete_collection(self):
        """Удаляет коллекцию."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Коллекция '{self.collection_name}' удалена")
        except Exception as e:
            logger.error(f"Ошибка удаления коллекции: {e}")


    def _validate_embeddings(self, embeddings, expected_count):
        if len(embeddings) != expected_count:
            raise ValueError("Embedding response count mismatch")
        if any(len(v) != self.embedding_dimensions or
               any(not math.isfinite(float(x)) for x in v) for v in embeddings):
            raise ValueError("Embedding dimensions/values mismatch")
        return embeddings

    def upsert_documents(self, texts, metadatas, ids):
        """Only embed missing/changed rows; stable IDs make retries idempotent."""
        if len(texts) != len(metadatas) or len(texts) != len(ids) or len(set(ids)) != len(ids):
            raise ValueError("Invalid vector document batch")
        if self.collection is None:
            self.get_or_create_collection()
        if not ids:
            return 0
        old = self.collection.get(ids=ids, include=["documents", "metadatas"])
        existing = {i: (t, m) for i, t, m in zip(old["ids"], old["documents"], old["metadatas"])}
        changed = [n for n, i in enumerate(ids) if existing.get(i) != (texts[n], metadatas[n])]
        if changed:
            content = [texts[n] for n in changed]
            self.collection.upsert(
                ids=[ids[n] for n in changed], documents=content,
                metadatas=[metadatas[n] for n in changed],
                embeddings=self._create_embeddings(content))
        return len(changed)
