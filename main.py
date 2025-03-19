import mediapipe as mp
import numpy as np
import cv2
from PIL import Image

def apply_face_filter(camera_index=0, nose_path='', star_path='', detection_confidence=0.5):
    """
    Запускает видеопоток, накладывает изображения на лицо с использованием MediaPipe FaceMesh.
    
    :param camera_index: индекс камеры (по умолчанию 0)
    :param nose_path: путь к изображению носа
    :param star_path: путь к изображению звезды (для глаз)
    :param detection_confidence: минимальная уверенность для детекции лица (0.0 - 1.0)
    """
    cap = cv2.VideoCapture(camera_index)
    
    # Инициализация FaceMesh
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=detection_confidence
    )
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        results = mp_face_mesh.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Обработка лицевых точек
        nose, left_eye, right_eye = None, None, None
        left_face, right_face = None, None
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmark_dict = {i: lm for i, lm in enumerate(face_landmarks.landmark)}
                nose = landmark_dict.get(4)
                left_eye = (landmark_dict.get(133), landmark_dict.get(159), landmark_dict.get(145), landmark_dict.get(33))
                right_eye = (landmark_dict.get(362), landmark_dict.get(386), landmark_dict.get(374), landmark_dict.get(263))
                left_face = landmark_dict.get(254)
                right_face = landmark_dict.get(454)
        
        # Проверка наличия ключевых точек
        if nose and left_eye and right_eye and left_face and right_face:
            nose_d = 800 * (right_face.x - left_face.x)
            width_left_eye = 650 * (left_eye[0].x - left_eye[3].x)
            width_right_eye = 650 * (right_eye[3].x - right_eye[0].x)
            
            nx, ny = int(nose.x * frame.shape[1] - nose_d / 2), int(nose.y * frame.shape[0] - nose_d / 2)
            lx, ly = int(left_eye[0].x * frame.shape[1] - width_left_eye / 2), int((left_eye[1].y + left_eye[2].y) * frame.shape[0] / 2 - width_left_eye / 2)
            rx, ry = int(right_eye[0].x * frame.shape[1] - width_right_eye / 2), int((right_eye[1].y + right_eye[2].y) * frame.shape[0] / 2 - width_right_eye / 2)
            
            # Загрузка изображений
            image_nose = cv2.resize(cv2.imread(nose_path, cv2.IMREAD_UNCHANGED), (int(nose_d), int(nose_d)))
            image_star = cv2.resize(cv2.imread(star_path, cv2.IMREAD_UNCHANGED), (int(width_left_eye), int(width_left_eye)))
            
            # Преобразование в PIL
            img = Image.fromarray(frame)
            img.paste(Image.fromarray(cv2.cvtColor(image_nose, cv2.COLOR_RGBA2BGRA)), (nx, ny), mask=Image.fromarray(image_nose[:, :, 3]))
            img.paste(Image.fromarray(cv2.cvtColor(image_star, cv2.COLOR_RGBA2BGRA)), (lx, ly), mask=Image.fromarray(image_star[:, :, 3]))
            img.paste(Image.fromarray(cv2.cvtColor(image_star, cv2.COLOR_RGBA2BGRA)), (rx, ry), mask=Image.fromarray(image_star[:, :, 3]))

            frame = np.array(img)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('Face Filter', frame)

        if cv2.waitKey(5) & 0xFF == 27:  # Выход по ESC
            break

    cap.release()
    cv2.destroyAllWindows()

# Пример вызова функции
apply_face_filter(1, r"C:\Users\User\Desktop\ML_CV\mediapaichik\pngwing.com.png", 
                     r"C:\Users\User\Desktop\ML_CV\mediapaichik\pngegg (2).png")
