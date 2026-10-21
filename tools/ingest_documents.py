"""
Скрипт для загрузки документов в векторную БД.
Обрабатывает файлы из папки data/ и индексирует их.
"""

import sys
import os
from pathlib import Path
from typing import List
import argparse
import logging

# Добавляем родительскую директорию в путь для импорта
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Settings
from storage import VectorDatabase, DocumentLoader
from utils import setup_logging


def process_documents(
    file_paths: List[str],
    chunk_size: int,
    chunk_overlap: int
) -> tuple:
    """
    Обрабатывает список документов.
    
    Args:
        file_paths: Список путей к файлам
        chunk_size: Размер чанка
        chunk_overlap: Перекрытие
        
    Returns:
        Кортеж (тексты, метаданные, идентификаторы)
    """
    all_chunks = []
    all_metadatas = []
    all_ids = []
    
    doc_counter = 0
    
    for file_path in file_paths:
        try:
            print(f"Обработка: {file_path}")
            
            # Загружаем документ
            text, doc_type = DocumentLoader.load_document(file_path)
            print(f"  Загружено: {len(text)} символов")
            
            # Разбиваем на чанки
            chunks_with_meta = DocumentLoader.create_chunks_with_metadata(
                text=text,
                chunk_size=chunk_size,
                overlap=chunk_overlap,
                source=Path(file_path).name,
                doc_type=doc_type
            )
            
            print(f"  Создано чанков: {len(chunks_with_meta)}")
            
            # Добавляем в общий список
            for chunk_data in chunks_with_meta:
                all_chunks.append(chunk_data['text'])
                all_metadatas.append(chunk_data['metadata'])
                
                chunk_id = f"doc_{doc_counter}_chunk_{chunk_data['metadata']['chunk_id']}"
                all_ids.append(chunk_id)
            
            doc_counter += 1
            print(f"  ✓ Успешно обработан\n")
        
        except Exception as e:
            print(f"  ✗ Ошибка: {e}\n")
            continue
    
    return all_chunks, all_metadatas, all_ids


def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(
        description="Загрузка документов в векторную БД"
    )
    parser.add_argument(
        '--files',
        nargs='+',
        help='Список файлов (по умолчанию все из data/)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=500,
        help='Размер чанка (по умолчанию: 500)'
    )
    parser.add_argument(
        '--overlap',
        type=int,
        default=100,
        help='Перекрытие (по умолчанию: 100)'
    )
    
    args = parser.parse_args()
    
    # Настройка логирования
    setup_logging(level="INFO")
    logger = logging.getLogger(__name__)
    
    print("=" * 60)
    print("ЗАГРУЗКА ДОКУМЕНТОВ В ВЕКТОРНУЮ БД")
    print("=" * 60)
    
    # Определяем файлы для обработки
    if args.files:
        file_paths = args.files
    else:
        # Берем все файлы из agent/data/
        data_dir = Path(__file__).parent.parent / 'data'
        if not data_dir.exists():
            print(f"✗ Директория {data_dir} не найдена!")
            print(f"Создайте папку agent/data/ и поместите туда документы")
            sys.exit(1)
        
        file_paths = [
            str(f) for f in data_dir.iterdir()
            if f.suffix.lower() in ['.txt', '.html', '.htm']
        ]
    
    if not file_paths:
        print("✗ Не найдено файлов для обработки!")
        sys.exit(1)
    
    print(f"Файлов для обработки: {len(file_paths)}")
    print(f"Размер чанка: {args.chunk_size}")
    print(f"Перекрытие: {args.overlap}")
    print("=" * 60)
    print()
    
    # Обрабатываем документы
    texts, metadatas, ids = process_documents(
        file_paths=file_paths,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap
    )
    
    if not texts:
        print("✗ Не удалось обработать ни одного документа!")
        sys.exit(1)
    
    print(f"Всего чанков: {len(texts)}\n")
    
    # Загружаем настройки
    try:
        settings = Settings.from_env()
    except ValueError as e:
        print(f"✗ Ошибка настроек: {e}")
        print("\nУстановите OPENAI_API_KEY в переменных окружения или .env файле")
        sys.exit(1)
    
    # Инициализируем векторную БД
    print("=" * 60)
    print("ИНИЦИАЛИЗАЦИЯ ВЕКТОРНОЙ БД")
    print("=" * 60)
    
    vector_db = VectorDatabase(
        persist_directory=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection,
        openai_api_key=settings.openai_api_key
    )
    
    vector_db.get_or_create_collection()
    
    # Добавляем документы
    print("\n" + "=" * 60)
    print("ДОБАВЛЕНИЕ ДОКУМЕНТОВ")
    print("=" * 60)
    
    try:
        vector_db.add_documents(
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print("\n✓ Все документы успешно добавлены!")
        
        # Выводим статистику
        stats = vector_db.get_stats()
        print("\n" + "=" * 60)
        print("СТАТИСТИКА")
        print("=" * 60)
        print(f"Коллекция: {stats['name']}")
        print(f"Документов: {stats['document_count']}")
        print("=" * 60)
        
        print("\n✅ ЗАГРУЗКА ЗАВЕРШЕНА!")
        print("\nТеперь можно запустить бота:")
        print("  python main.py")
    
    except Exception as e:
        logger.error(f"Ошибка добавления документов: {e}", exc_info=True)
        print(f"\n✗ Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

