import cv2
from model import FaceModel
from database import EmbeddingDatabase
import numpy as np
from mtcnn import MTCNN
import uuid  # Для генерации уникального ID


class VideoCapture:
    def __init__(self):
        """
        Инициализация модели и базы данных.
        """
        self.face_model = FaceModel()
        self.face_db = EmbeddingDatabase()
        self.detector = MTCNN()

    def start(self):
        """
        Запускает захват видео с камеры.
        """
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Детекция лица с использованием MTCNN
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.detector.detect_faces(rgb_frame)

            if faces and faces[0]['confidence'] > 0.98:
                for face in faces:
                    x, y, width, height = face['box']
                    # Корректируем координаты, чтобы они не выходили за пределы кадра
                    x = max(0, x)
                    y = max(0, y)
                    width = min(width, frame.shape[1] - x)
                    height = min(height, frame.shape[0] - y)

                    # Вырезаем область лица
                    face_img = frame[y:y + height, x:x + width]

                    # Преобразуем изображение лица в формат, подходящий для модели
                    face_img_resized = cv2.resize(face_img, (160, 160))  # Resize to 160x160 (Facenet input size)
                    face_img_normalized = cv2.normalize(face_img_resized, None, alpha=0, beta=255,
                                                        norm_type=cv2.NORM_MINMAX)

                    # Извлечение эмбеддинга из изображения лица
                    embedding = self.face_model.extract_embedding(face_img_normalized)
                    if embedding is not None and len(embedding) == 128:
                        # Поиск ближайших соседей
                        nearest_embeddings = self.face_db.find_nearest_embeddings(np.array(embedding), top_k=10,
                                                                                  similarity_threshold=0.4)

                        if nearest_embeddings:
                            predicted_id, predicted_name, similarity = nearest_embeddings[0]
                            # Отображение результата над рамкой
                            text = f"Name: {predicted_name}, Similarity: {similarity:.2f}"
                            cv2.putText(frame, text, (x, y - 10),  # Позиция текста над рамкой
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        else:
                            # Если человек неизвестен, сохраняем его в базу данных с именем, равным ID
                            unknown_id = str(uuid.uuid4())  # Генерация уникального ID
                            self.face_db.insert_embedding(unknown_id, np.array(embedding))
                            text = f"Unknown (ID: {unknown_id})"
                            cv2.putText(frame, text, (x, y - 10),  # Позиция текста над рамкой
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                    # Рисуем прямоугольник вокруг лица
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

            # Отображение кадра
            cv2.imshow('Frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()