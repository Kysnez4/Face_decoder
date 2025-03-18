import cv2
from model import FaceModel
from database import FaceDatabase

class VideoCapture:
    def __init__(self):
        self.face_model = FaceModel()
        self.face_db = FaceDatabase()

    def start(self):
        """Запускает захват видео с камеры."""
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Извлечение эмбеддинга из кадра
            embedding = self.face_model.extract_embedding_from_frame(frame)
            if embedding is not None:
                # Поиск ID в базе данных
                predicted_id, predicted_name = self.face_db.find_embedding(embedding)

                # Отображение результата
                if predicted_id != -1:
                    cv2.putText(frame, f"ID: {predicted_id}, Name: {predicted_name}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Unknown", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Отображение кадра
            cv2.imshow('Frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.face_db.close()