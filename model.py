import insightface
import cv2

class FaceModel:
    def __init__(self):
        self.model = insightface.app.FaceAnalysis()
        self.model.prepare(ctx_id=0, det_size=(640, 640))

    def extract_embedding(self, image_path):
        """Извлекает эмбеддинг из изображения."""
        img = cv2.imread(image_path)
        faces = self.model.get(img)
        if len(faces) == 0:
            raise ValueError("Лицо не обнаружено на изображении!")
        return faces[0].embedding  # Возвращаем эмбеддинг первого лица

    def extract_embedding_from_frame(self, frame):
        """Извлекает эмбеддинг из кадра видео."""
        faces = self.model.get(frame)
        if len(faces) == 0:
            return None
        return faces[0].embedding