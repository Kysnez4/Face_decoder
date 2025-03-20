import os
from mtcnn import MTCNN  # MTCNN для детекции лиц
from database import EmbeddingDatabase  # Класс для работы с базой данных

from deepface import DeepFace
import cv2
import numpy as np

class FaceModel:
    def __init__(self, model_name="ArcFace"):
        self.model_name = model_name
        self.model = DeepFace.build_model(model_name)  # Загружаем модель

    def extract_embedding(self, img):
        """Извлекает эмбеддинг из изображения с использованием DeepFace."""
        try:
            # Используем DeepFace для извлечения эмбеддинга
            result = DeepFace.represent(
                img_path=img,
                model_name=self.model_name,
                detector_backend='skip',
                enforce_detection=False
            )

            if result:
                embedding = result[0]["embedding"]
                return np.array(embedding)
            else:
                raise ValueError("Не удалось извлечь эмбеддинг!")
        except Exception as e:
            raise ValueError(f"Ошибка при извлечении эмбеддинга: {e}")


# Инициализация MTCNN, модели и базы данных
detector = MTCNN()
face_model = FaceModel()
embedding_db = EmbeddingDatabase()

# Путь к папке с фотографиями студентов
students_folder = "students"


# Функция для обработки фотографий студента
def process_student_photos(student_name, photos_folder):
    """
    Обрабатывает фотографии студента, находит лица с помощью MTCNN и добавляет их эмбеддинги в базу данных.

    :param student_name: Имя студента (название папки).
    :param photos_folder: Путь к папке с фотографиями студента.
    """
    # Получаем список всех файлов в папке
    photo_files = [f for f in os.listdir(photos_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for photo_file in photo_files:
        # Полный путь к фотографии
        photo_path = os.path.join(photos_folder, photo_file)

        # Загружаем изображение
        with open(photo_path, 'rb') as f:
            image_data = np.frombuffer(f.read(), dtype=np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        if image is None:
            print(f"Ошибка загрузки изображения: {photo_path}")
            continue

        # Преобразуем изображение в RGB (MTCNN ожидает RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Детекция лиц с помощью MTCNN
        faces = detector.detect_faces(rgb_image)

        if not faces:
            print(f"Лица не найдены на фотографии: {photo_path}")
            continue

        # Обрабатываем каждое обнаруженное лицо
        for i, face in enumerate(faces):
            # Получаем координаты лица
            x, y, width, height = face['box']

            # Корректируем координаты, чтобы они не выходили за пределы изображения
            x = max(0, x)
            y = max(0, y)
            width = min(width, image.shape[1] - x)
            height = min(height, image.shape[0] - y)

            # Вырезаем область лица
            face_img = image[y:y + height, x:x + width]

            # Преобразуем изображение лица в формат, подходящий для модели
            resized_face = cv2.resize(face_img, (160, 160))  # Пример для FaceNet (160x160)
            normalized_face = cv2.normalize(resized_face, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

            # Извлекаем эмбеддинг
            embedding = face_model.extract_embedding(normalized_face)
            if embedding is not None and len(embedding) == 512:  # Проверяем, что эмбеддинг корректен
                # Сохраняем эмбеддинг в базу данных с именем студента
                embedding_db.insert_embedding(student_name, np.array(embedding))
                print(f"Эмбеддинг для {student_name} ({photo_file}, лицо {i + 1}) добавлен в базу данных.")
            else:
                print(f"Не удалось извлечь эмбеддинг для {student_name} ({photo_file}, лицо {i + 1}).")


# Основной цикл для обработки всех студентов
for student_name in os.listdir(students_folder):
    # Декодируем имя студента (если необходимо)
    student_name = student_name.encode('utf-8').decode('utf-8')  # Убедимся, что имя корректно декодировано

    student_folder = os.path.join(students_folder, student_name)

    # Проверяем, что это папка
    if os.path.isdir(student_folder):
        print(f"Обработка студента: {student_name}")
        process_student_photos(student_name, student_folder)
    else:
        print(f"Пропуск: {student_folder} не является папкой.")

print("Все эмбеддинги успешно добавлены в базу данных.")