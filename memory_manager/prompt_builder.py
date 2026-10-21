"""
Построитель промптов.
Формирует готовые промпты для ИИ на основе контекста и истории.
"""

from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Построитель промптов для ИИ."""
    
    def __init__(
        self,
        system_prompt: str = None,
        max_context_length: int = 4000
    ):
        """
        Инициализирует построитель промптов.
        
        Args:
            system_prompt: Системный промпт
            max_context_length: Максимальная длина контекста
        """
        self.system_prompt = system_prompt
        self.max_context_length = max_context_length
        
        logger.info("PromptBuilder инициализирован")
    
    def build_context_from_documents(
        self,
        documents: List[Dict]
    ) -> str:
        """
        Формирует контекст из найденных документов.
        
        Args:
            documents: Список документов с метаданными
            
        Returns:
            Отформатированный контекст
        """
        if not documents:
            return "Контекст отсутствует."
        
        context_parts = []
        total_length = 0
        
        for i, doc in enumerate(documents, 1):
            text = doc.get('text', '')
            source = doc.get('source', 'unknown')
            relevance = doc.get('relevance', 0)
            
            # Проверяем, не превысим ли лимит
            doc_text = f"Документ {i} (Источник: {source}, Релевантность: {relevance:.2f}):\n{text}\n"
            
            if total_length + len(doc_text) > self.max_context_length:
                logger.warning(
                    f"Достигнут лимит контекста. "
                    f"Использовано {i-1} из {len(documents)} документов"
                )
                break
            
            context_parts.append(doc_text)
            total_length += len(doc_text)
        
        return "\n---\n".join(context_parts)
    
    def build_conversation_context(
        self,
        history: List[Dict],
        max_messages: int = 10
    ) -> List[Dict]:
        """
        Формирует контекст истории диалога.
        
        Args:
            history: История сообщений
            max_messages: Максимальное количество сообщений
            
        Returns:
            Отфильтрованная история для контекста
        """
        if not history:
            return []
        
        # Берем последние N сообщений
        recent_history = history[-max_messages:]
        
        return recent_history
    
    def build_messages_for_ai(
        self,
        query: str,
        context_documents: List[Dict],
        conversation_history: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Формирует полный список сообщений для ИИ.
        
        Args:
            query: Текущий запрос пользователя
            context_documents: Документы из базы знаний
            conversation_history: История диалога
            system_prompt: Системный промпт (переопределяет дефолтный)
            
        Returns:
            Список сообщений в формате OpenAI
        """
        messages = []
        
        # Системный промпт
        sys_prompt = system_prompt or self.system_prompt
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        
        # История диалога (если есть)
        if conversation_history:
            filtered_history = self.build_conversation_context(conversation_history)
            messages.extend(filtered_history)
        
        # Формируем контекст из документов
        context = self.build_context_from_documents(context_documents)
        
        # Текущий запрос с контекстом
        user_message = f"""Контекст из базы знаний:

{context}

---

Вопрос пользователя: {query}

Ответь на вопрос, используя информацию из предоставленного контекста."""
        
        messages.append({"role": "user", "content": user_message})
        
        logger.info(f"Сформирован промпт: {len(messages)} сообщений")
        
        return messages
    
    def summarize_context(self, text: str, max_length: int = 1000) -> str:
        """
        Сокращает контекст до заданной длины.
        
        Args:
            text: Исходный текст
            max_length: Максимальная длина
            
        Returns:
            Сокращенный текст
        """
        if len(text) <= max_length:
            return text
        
        # Простое обрезание с многоточием
        return text[:max_length - 3] + "..."

