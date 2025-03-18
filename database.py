class FaceDatabase:
    def __init__(self, db_path='faces.db'):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._initialize_db()

    def _initialize_db(self):
        """Создаёт таблицу, если она не существует."""
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS faces
                               (id INTEGER PRIMARY KEY, name TEXT, embedding BLOB)''')
        self.conn.commit()

    def save_embedding(self, embedding, name=None):
        """Сохраняет эмбеддинг в базу данных."""
        self.cursor.execute("INSERT INTO faces (name, embedding) VALUES (?, ?)",
                            (name, embedding.tobytes()))
        self.conn.commit()

    def find_embedding(self, new_embedding, threshold=0.6):
        """Ищет наиболее похожий эмбеддинг в базе данных."""
        self.cursor.execute("SELECT id, name, embedding FROM faces")
        rows = self.cursor.fetchall()

        best_id = -1
        best_name = ""
        max_similarity = -1

        for row in rows:
            face_id = row[0]
            name = row[1]  # Может быть None
            stored_embedding = np.frombuffer(row[2], dtype=np.float32)
            similarity = self._cosine_similarity(new_embedding, stored_embedding)

            if similarity > max_similarity and similarity > threshold:
                max_similarity = similarity
                best_id = face_id
                best_name = name

        return best_id, best_name

    def _cosine_similarity(self, embedding1, embedding2):
        """Вычисляет косинусное сходство между двумя эмбеддингами."""
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def close(self):
        """Закрывает соединение с базой данных."""
        self.conn.close()