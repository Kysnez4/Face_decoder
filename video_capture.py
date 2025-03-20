
import cv2
from model import FaceModel
from database import EmbeddingDatabase
import numpy as np
from mtcnn import MTCNN
import uuid
from collections import defaultdict, deque, Counter

class Face:
    def __init__(self, face_id, name, shape):
        self.face_id = face_id
        self.name = name
        self.shape = shape
        self.buffer = deque(maxlen=20)  # Буфер для хранения последних распознаваний
        self.head_tracker = None  # Трекер для головы
        self.head_bbox = None  # Область головы
        self.last_known_name = name  # Храним последнее известное имя

    def update_shape(self, new_shape):
        self.shape = new_shape

    def add_recognition(self, face_id, name, similarity):
        self.buffer.append((face_id, name, similarity))
        self.last_known_name = name # Обновляем последнее известное имя

    def get_stable_id_and_name(self):
        if not self.buffer:
            return None, None

        # Голосование: выбираем наиболее часто встречающийся ID и имя в буфере
        id_counter = Counter([rec[0] for rec in self.buffer])
        name_counter = Counter([rec[1] for rec in self.buffer])

        most_common_id = id_counter.most_common(1)[0][0]
        most_common_name = name_counter.most_common(1)[0][0]

        return most_common_id, most_common_name

class VideoCapture:
    def __init__(self, detection_interval=5):
        """
        Инициализация модели, базы данных и интервала детекции.
        detection_interval:  Выполнять распознавание лиц только каждые detection_interval кадров.
        """
        self.face_model = FaceModel()
        self.face_db = EmbeddingDatabase()
        self.detector = MTCNN()
        self.face_buffer = defaultdict(list)  # Буфер для хранения эмбеддингов лиц
        self.buffer_size = 50  # Размер буфера для каждого лица
        self.recognition_history = defaultdict(lambda: deque(maxlen=40))  # История распознаваний для каждого лица
        self.faces = {}  # Словарь для хранения объектов Face
        self.detection_interval = detection_interval # Интервал детекции
        self.frame_count = 0 # Счетчик кадров

    def align_face(self, image):
        """
        Выравнивает лицо по ключевым точкам.
        """
        faces = self.detector.detect_faces(image)

        if faces:
            # Получаем ключевые точки лица
            keypoints = faces[0]['keypoints']
            left_eye = keypoints['left_eye']
            right_eye = keypoints['right_eye']

            # Вычисляем угол поворота
            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dY, dX)) - 180

            # Поворачиваем изображение
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            aligned_face = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

            return aligned_face
        return image

    def save_multiple_embeddings(self, name, face_img):
        """
        Сохраняет несколько эмбеддингов для одного лица (frontal view, повороты на 90 градусов).
        """
        # Выравниваем лицо
        aligned_face = self.align_face(face_img)

        # Извлекаем эмбеддинг для frontal view
        frontal_embedding = self.face_model.extract_embedding(aligned_face)
        if frontal_embedding is not None:
            self.face_db.insert_embedding(name, frontal_embedding)

        # Извлекаем эмбеддинги для повернутых лиц
        rotated_left = cv2.rotate(aligned_face, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rotated_right = cv2.rotate(aligned_face, cv2.ROTATE_90_CLOCKWISE)

        left_embedding = self.face_model.extract_embedding(rotated_left)
        right_embedding = self.face_model.extract_embedding(rotated_right)

        if left_embedding is not None:
            self.face_db.insert_embedding(name, left_embedding)
        if right_embedding is not None:
            self.face_db.insert_embedding(name, right_embedding)

    def start(self):
        """
        Запускает захват видео с камеры.
        """
        cap = cv2.VideoCapture(1)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1

            # Детекция лица выполняется только через заданный интервал
            if self.frame_count % self.detection_interval == 0:
                # Детекция лица с использованием MTCNN
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = self.detector.detect_faces(rgb_frame)

                if faces and faces[0]['confidence'] > 0.99:
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
                        face_img_resized = cv2.resize(face_img, (160, 160), interpolation=cv2.INTER_AREA)  # Resize to 160x160 (Facenet input size)
                        face_img_normalized = cv2.normalize(face_img_resized, None, alpha=0, beta=255,
                                                            norm_type=cv2.NORM_MINMAX)

                        # Извлечение эмбеддинга из изображения лица
                        embedding = self.face_model.extract_embedding(face_img_normalized)
                        if embedding is not None and len(embedding) == 512:
                            # Поиск ближайших соседей
                            nearest_embeddings = self.face_db.find_nearest_embeddings(np.array(embedding), top_k=25,
                                                                                      similarity_threshold=0.6)

                            if nearest_embeddings:
                                predicted_id, predicted_name, similarity = nearest_embeddings[0]

                                # Если лицо уже есть в словаре, обновляем его
                                if predicted_id in self.faces:
                                    face_obj = self.faces[predicted_id]
                                    face_obj.update_shape(embedding)
                                    face_obj.add_recognition(predicted_id, predicted_name, similarity)
                                else:
                                    # Создаем новый объект Face
                                    face_obj = Face(predicted_id, predicted_name, embedding)
                                    face_obj.add_recognition(predicted_id, predicted_name, similarity)
                                    self.faces[predicted_id] = face_obj

                                # Получаем стабильные ID и имя
                                stable_id, stable_name = face_obj.get_stable_id_and_name()

                                if stable_id and stable_name:
                                    text = f"Name: {stable_name}, Similarity: {similarity:.2f}"
                                    cv2.putText(frame, text, (x, y - 10),  # Позиция текста над рамкой
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                                # Инициализация трекера для головы
                                if face_obj.head_tracker is None:
                                    face_obj.head_tracker = cv2.TrackerKCF_create()
                                    face_obj.head_bbox = (x, y, width, height)
                                    face_obj.head_tracker.init(frame, face_obj.head_bbox)

                            else:
                                # Если человек неизвестен, проверяем, есть ли он в базе данных
                                nearest_embeddings = self.face_db.find_nearest_embeddings(np.array(embedding), top_k=25,
                                                                                          similarity_threshold=0.7)
                                if nearest_embeddings:
                                    # Если лицо найдено в базе данных, используем существующий ID
                                    predicted_id, predicted_name, similarity = nearest_embeddings[0]
                                    text = f"Name: {predicted_name}, Similarity: {similarity:.2f}"

                                else:
                                    # Если лицо не найдено, создаем новый ID
                                    unknown_id = str(uuid.uuid4())  # Генерация уникального ID
                                    self.save_multiple_embeddings(unknown_id,
                                                                  face_img_normalized)  # Сохраняем несколько эмбеддингов


            # Отслеживание головы выполняется на каждом кадре, независимо от интервала детекции
            for face_id, face_obj in self.faces.items():
                if face_obj.head_tracker is not None:
                    success, bbox = face_obj.head_tracker.update(frame)
                    if success:
                        # Рисуем прямоугольник вокруг головы
                        x, y, w, h = map(int, bbox)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        face_obj.head_bbox = (x, y, w, h)

                        # Выводим имя над трекером
                        text = face_obj.last_known_name
                        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Синий цвет

                    else:
                        # Если трекер потерял голову, сбрасываем его
                        face_obj.head_tracker = None

            # Отображение кадра
            cv2.imshow('Frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()