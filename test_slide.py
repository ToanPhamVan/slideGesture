import asyncio
import websockets
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
import pickle
import time

# Tải mô hình đã huấn luyện
mp_holistic = mp.solutions.holistic
cap = cv2.VideoCapture(0)

with open("body_language.pkl", "rb") as f:
    model = pickle.load(f)
    
ws_clients = set()

async def handler(websocket):
    ws_clients.add(websocket)
    try:
        # Gửi thông báo xác nhận kết nối thành công
        await websocket.send("connected")
        while True:
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Xử lý hình ảnh
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    try:
                        pose = results.pose_landmarks.landmark
                        pose_row = list(
                            np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten()
                        )

                        face = results.face_landmarks.landmark if results.face_landmarks else []
                        face_row = list(
                            np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten()
                        )

                        row = pose_row + face_row
                        X = pd.DataFrame([row])
                        # Dự đoán cử chỉ
                        body_language_class = model.predict(X)[0]
                        data = f"{body_language_class}"
                        await websocket.send(data)
                        await asyncio.sleep(2)
                    except Exception as e:
                        pass  # Không in lỗi nếu có ngoại lệ

    except websockets.ConnectionClosed:
        pass  # Không in lỗi khi kết nối WebSocket bị đóng
    finally:
        ws_clients.remove(websocket)

async def main():
    try:
        print("WebSocket server starting...")
        async with websockets.serve(handler, "localhost", 8765):
            await asyncio.Future()  # Giữ server hoạt động liên tục
    except KeyboardInterrupt:
        pass  # Không in gì khi ngừng chương trình bằng Ctrl+C

# Chạy server
asyncio.run(main())
