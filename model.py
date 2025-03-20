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
