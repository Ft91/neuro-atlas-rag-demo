"""
Загрузчик документов.
Обрабатывает различные форматы документов и подготавливает их для индексации.
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple
import re
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Загрузчик и обработчик документов."""
    
    @staticmethod
    def load_txt(file_path: str) -> str:
        """
        Загружает текстовый файл.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Очищенный текст
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Очистка текста
            content = re.sub(r' +', ' ', content)
            content = re.sub(r'\n{3,}', '\n\n', content)
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            return '\n'.join(lines)
        
        except Exception as e:
            logger.error(f"Ошибка загрузки TXT файла {file_path}: {e}")
            raise
    
    @staticmethod
    def load_html(file_path: str) -> str:
        """
        Загружает HTML файл и извлекает текст.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Очищенный текст из HTML
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Парсим HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Удаляем script и style теги
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Извлекаем текст
            text = soup.get_text()
            
            # Очистка
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)
            text = re.sub(r' +', ' ', text)
            text = re.sub(r'\n{3,}', '\n\n', text)
            
            return text.strip()
        
        except Exception as e:
            logger.error(f"Ошибка загрузки HTML файла {file_path}: {e}")
            raise
    
    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = 500,
        overlap: int = 100
    ) -> List[str]:
        """
        Разбивает текст на чанки с умным разделением.
        
        Args:
            text: Исходный текст
            chunk_size: Размер чанка
            overlap: Перекрытие между чанками
            
        Returns:
            Список чанков
        """
        if len(text) <= chunk_size:
            return [text]
        
        # Разделители по приоритету
        separators = ['\n\n', '\n', '. ', '! ', '? ', '; ', ', ', ' ']
        
        def split_recursive(text: str, seps: List[str]) -> List[str]:
            """Рекурсивное разбиение по разделителям."""
            if not text or len(text) <= chunk_size:
                return [text] if text else []
            
            for sep in seps:
                if sep in text:
                    parts = text.split(sep)
                    result = []
                    current = ""
                    
                    for part in parts:
                        part_with_sep = part + sep if part != parts[-1] else part
                        
                        if len(current) + len(part_with_sep) <= chunk_size:
                            current += part_with_sep
                        else:
                            if current:
                                result.append(current.strip())
                            
                            if len(part_with_sep) > chunk_size:
                                next_seps = seps[seps.index(sep) + 1:] if seps.index(sep) + 1 < len(seps) else []
                                if next_seps:
                                    result.extend(split_recursive(part_with_sep, next_seps))
                                else:
                                    # Простое разбиение
                                    for i in range(0, len(part_with_sep), chunk_size - overlap):
                                        result.append(part_with_sep[i:i + chunk_size])
                                current = ""
                            else:
                                current = part_with_sep
                    
                    if current:
                        result.append(current.strip())
                    
                    return result
            
            # Простое разбиение если нет разделителей
            result = []
            for i in range(0, len(text), chunk_size - overlap):
                result.append(text[i:i + chunk_size])
            return result
        
        chunks = split_recursive(text, separators)
        return [c for c in chunks if c.strip()]
    
    @staticmethod
    def create_chunks_with_metadata(
        text: str,
        chunk_size: int,
        overlap: int,
        source: str,
        doc_type: str
    ) -> List[Dict]:
        """
        Создает чанки с метаданными.
        
        Args:
            text: Исходный текст
            chunk_size: Размер чанка
            overlap: Перекрытие
            source: Источник (имя файла)
            doc_type: Тип документа
            
        Returns:
            Список чанков с метаданными
        """
        chunks = DocumentLoader.chunk_text(text, chunk_size, overlap)
        
        chunks_with_metadata = []
        for i, chunk in enumerate(chunks):
            chunks_with_metadata.append({
                "text": chunk,
                "metadata": {
                    "source": source,
                    "type": doc_type,
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "char_count": len(chunk)
                }
            })
        
        return chunks_with_metadata
    
    @staticmethod
    def load_document(file_path: str) -> Tuple[str, str]:
        """
        Загружает документ в зависимости от типа.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Кортеж (текст, тип_документа)
        """
        file_ext = Path(file_path).suffix.lower()
        
        logger.info(f"Загрузка файла: {file_path}")
        
        if file_ext == '.txt':
            text = DocumentLoader.load_txt(file_path)
            return text, 'txt'
        elif file_ext in ['.html', '.htm']:
            text = DocumentLoader.load_html(file_path)
            return text, 'html'
        else:
            raise ValueError(f"Неподдерживаемый формат: {file_ext}")

