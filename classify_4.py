import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
import pickle
import warnings
warnings.filterwarnings("ignore")

# Tải mô hình đã huấn luyện
with open("body_language.pkl", "rb") as f:
    model = pickle.load(f)

# Khởi tạo mô hình Holistic của Mediapipe
mp_holistic = mp.solutions.holistic
cap = cv2.VideoCapture(0)

# Biến theo dõi class trước đó
previous_class = None

# Khởi tạo mô hình Holistic
with mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        # Chuyển đổi màu sắc để xử lý
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Xử lý hình ảnh với Mediapipe
        results = holistic.process(image)

        # Chuyển hình ảnh lại để hiển thị
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Thực hiện dự đoán
        try:
            pose = results.pose_landmarks.landmark
            pose_row = list(
                np.array(
                    [
                        [landmark.x, landmark.y, landmark.z, landmark.visibility]
                        for landmark in pose
                    ]
                ).flatten()
            )

            face = results.face_landmarks.landmark if results.face_landmarks else []
            face_row = list(
                np.array(
                    [
                        [landmark.x, landmark.y, landmark.z, landmark.visibility]
                        for landmark in face
                    ]
                ).flatten()
            )

            # Kết hợp các hàng tọa độ
            row = pose_row + face_row
            X = pd.DataFrame([row])

            # Dự đoán hành vi
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            print(body_language_class, body_language_prob)

            # Hiển thị CLASS
            cv2.putText(
                image,
                "CLASS",
                (95, 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                body_language_class.split(" ")[0],  # Hiển thị tên class đầu tiên
                (90, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Hiển thị PROBABILITY
            cv2.putText(
                image,
                "PROB",
                (15, 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                str(round(body_language_prob[np.argmax(body_language_prob)], 2)),
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        except Exception as e:
            print(f"Error: {e}")

        # Hiển thị hình ảnh webcam với kết quả
        cv2.imshow("Webcam Feed", image)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
        # Kiểm tra nếu cửa sổ đã bị đóng (bằng cách nhấn nút 'X')
        if cv2.getWindowProperty("Webcam Feed", cv2.WND_PROP_VISIBLE) < 1:
            break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
