import sqlite3
import numpy as np
from typing import List, Tuple, Optional
import faiss  # Для ускорения поиска ближайших соседей


class EmbeddingDatabase:
    def __init__(self, db_name: str = 'embeddings.db'):
        self.db_name = db_name
        self.dimension = 512  # Размерность эмбеддингов (можно изменить)
        self.index = faiss.IndexFlatIP(self.dimension)  # Индекс FAISS для косинусного сходства
        self.create_database()

    def create_database(self):
        """Создает базу данных и таблицу, если они не существуют."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        # Создаем таблицу заново
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
        ''')

        conn.commit()
        conn.close()

    def insert_embedding(self, name: str, embedding: np.ndarray):
        """
        Вставляет эмбеддинг в базу данных.

        :param name: Имя студента.
        :param embedding: Эмбеддинг (вектор).
        """
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        # Нормализуем эмбеддинг
        normalized_embedding = embedding / np.linalg.norm(embedding)

        # Преобразуем нормализованный эмбеддинг в байты для хранения в BLOB
        embedding_bytes = normalized_embedding.tobytes()

        cursor.execute('''
            INSERT INTO embeddings (name, embedding) VALUES (?, ?)
        ''', (name, embedding_bytes))

        conn.commit()
        conn.close()

        # Добавляем эмбеддинг в индекс FAISS
        self.index.add(np.array([normalized_embedding]).astype(np.float32))

    def find_nearest_embeddings(self, query_embedding: np.ndarray, top_k: int = 10, similarity_threshold: float = 0.6) -> \
    List[Tuple[int, str, float]]:
        """
        Находит ближайшие эмбеддинги по косинусному расстоянию.

        :param query_embedding: Эмбеддинг для поиска.
        :param top_k: Количество ближайших результатов.
        :param similarity_threshold: Порог сходства (от 0 до 1).
        :return: Список кортежей (id, name, similarity).
        """
        # Нормализуем запрос
        query_embedding_normalized = query_embedding / np.linalg.norm(query_embedding)

        # Получаем все эмбеддинги из базы данных
        all_embeddings = self.get_all_embeddings()

        # Если в базе данных нет эмбеддингов, возвращаем пустой список
        if not all_embeddings:
            return []

        # Ищем ближайшие соседи с помощью FAISS
        D, I = self.index.search(np.array([query_embedding_normalized]).astype(np.float32), top_k)

        # Получаем результаты
        nearest_embeddings = []
        for i in range(top_k):
            # Проверяем, что индекс находится в пределах списка all_embeddings
            if I[0][i] < len(all_embeddings):
                embedding_id, name, _ = all_embeddings[I[0][i]]
                similarity = D[0][i]
                if similarity >= similarity_threshold:  # Фильтруем по порогу сходства
                    nearest_embeddings.append((embedding_id, name, similarity))
            else:
                # Если индекс выходит за пределы, пропускаем его
                continue

        return nearest_embeddings

    def get_all_embeddings(self) -> List[Tuple[int, str, np.ndarray]]:
        """
        Возвращает все эмбеддинги из базы данных.

        :return: Список кортежей (id, name, embedding).
        """
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, name, embedding FROM embeddings
        ''')

        results = cursor.fetchall()
        conn.close()

        embeddings = []
        for result in results:
            embedding_id, name, embedding_bytes = result
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            embeddings.append((embedding_id, name, embedding))

        return embeddings