import cv2
import os
import time
import tkinter as tk
from tkinter import messagebox

# Đặt lại giá trị mặc định
cap = None
recording = False
out = None
video_filename = ""

# Tạo thư mục để lưu video nếu chưa có
output_folder = "recorded_videos"
os.makedirs(output_folder, exist_ok=True)

# Đọc danh sách tên class từ file text
def load_classes_from_file(filename):
    classes = []
    try:
        with open(filename, 'r') as file:
            classes = [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        messagebox.showerror("Error", f"File '{filename}' not found.")
    return classes

# Hàm bắt đầu quay video
def start_recording():
    global cap, recording, out, video_filename
    
    # Kiểm tra xem đã có camera chưa, nếu chưa thì khởi tạo lại
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    # Lấy tên class từ dropdown
    class_name = class_name_var.get().strip()
    if not class_name:
        messagebox.showerror("Error", "Please select a class.")
        return

    # Tạo tên file video với timestamp và tên class
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    video_filename = os.path.join(output_folder, f"{class_name}_{timestamp}.avi")
    
    # Cài đặt thông số video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = 20.0  # Số khung hình mỗi giây
    
    # Định nghĩa codec và tạo đối tượng VideoWriter
    out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
    recording = True
    messagebox.showinfo("Recording", f"Recording started for class '{class_name}'. Press Stop to end.")

# Hàm dừng quay video
def stop_recording():
    global recording, out, cap
    recording = False
    if out:
        out.release()  # Giải phóng VideoWriter để dừng lưu video
    if cap is not None:
        cap.release()  # Giải phóng camera để dừng quay
    cv2.destroyAllWindows()  # Đóng mọi cửa sổ OpenCV
    messagebox.showinfo("Recording", f"Recording saved: {video_filename}")

# Tạo giao diện Tkinter
root = tk.Tk()
root.title("Video Recording Form")

# Nhãn và dropdown chọn class
tk.Label(root, text="Class Name:").grid(row=0, column=0, padx=10, pady=10)

# Đọc tên class từ file text
class_file = "classes.txt"  # Đường dẫn đến file .txt chứa danh sách class
class_names = load_classes_from_file(class_file)

class_name_var = tk.StringVar(value=class_names[0] if class_names else "")

# Dropdown chọn class
class_dropdown = tk.OptionMenu(root, class_name_var, *class_names)
class_dropdown.grid(row=0, column=1, padx=10, pady=10)

# Nút Start và Stop
start_button = tk.Button(root, text="Start Recording", command=start_recording)
start_button.grid(row=1, column=0, padx=10, pady=10)

stop_button = tk.Button(root, text="Stop Recording", command=stop_recording)
stop_button.grid(row=1, column=1, padx=10, pady=10)

# Hàm ghi lại video
def record_video():
    global cap, recording, out
    if recording and cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            out.write(frame)  # Ghi lại frame vào video
            cv2.imshow("Recording", frame)  # Hiển thị frame trong cửa sổ OpenCV
    
    root.after(10, record_video)  # Gọi lại hàm sau 10ms để ghi video tiếp

# Bắt đầu vòng lặp ghi video
record_video()

# Chạy vòng lặp giao diện Tkinter
root.mainloop()

# Giải phóng tài nguyên khi chương trình kết thúc
if cap is not None:
    cap.release()
cv2.destroyAllWindows()
