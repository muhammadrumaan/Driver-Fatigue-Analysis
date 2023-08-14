import tkinter as tk
from tkinter import filedialog,messagebox
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

class DrowsinessDetectorUI:
    def __init__(self,img_width, img_height):
        self.root = tk.Tk()
        self.root.title("Drowsiness Detector")
        self.root.configure(bg="white")  # Set background color
        
        self.loaded_model = load_model("C:/Users/rumaa/OneDrive/Documents/drowsiness_detection_model_new.h5")
        self.img_width = img_width
        self.img_height = img_height
        
        title_label = tk.Label(self.root, text="Drowsiness Detector", font=("Arial", 24, "bold"), fg="#3897F0", bg="white")
        title_label.pack(pady=20)
        
        button_frame = tk.Frame(self.root, bg="white")
        button_frame.pack()

        self.upload_button = tk.Button(button_frame, text="Upload Image", command=self.upload_image, font=("Arial", 14), bg="#3897F0", fg="white", borderwidth=0)
        self.upload_button.grid(row=0, column=0, padx=10)

        self.monitor_button = tk.Button(button_frame, text="Start Monitoring", command=self.start_monitoring, font=("Arial", 14), bg="#3897F0", fg="white", borderwidth=0)
        self.monitor_button.grid(row=0, column=1, padx=10)
        
        
        self.result_label = tk.Label(self.root, text="", font=("Arial", 16, "bold"), fg="#E74C3C", bg="white")
        self.result_label.pack(pady=20)

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.analyze_image(file_path)

    def start_monitoring(self):
        cap = cv2.VideoCapture(0)
        drowsiness_detected = False
        no_drowsiness_frames = 0  # Counter for consecutive frames with no drowsiness

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            drowsiness_detected_in_current_frame = False

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)

                for (ex, ey, ew, eh) in eyes:
                    eye_region = roi_gray[ey:ey+eh, ex:ex+ew]
                    eye_array = cv2.resize(eye_region, (100, 100))
                    eye_array = cv2.cvtColor(eye_array, cv2.COLOR_GRAY2RGB)
                    eye_array = eye_array.astype('float') / 255.0
                    eye_array = np.expand_dims(eye_array, axis=0)

                    prediction = self.loaded_model.predict(eye_array)
                    prediction_value = prediction[0]

                    if prediction_value < 0.5:
                        self.show_warning()
                        drowsiness_detected_in_current_frame = True
                        drowsiness_detected = True
                        no_drowsiness_frames = 0
                    else:
                        no_drowsiness_frames += 1

            if not drowsiness_detected_in_current_frame:
                no_drowsiness_frames += 1

            cv2.imshow("Monitoring", frame)

            if no_drowsiness_frames >= 10:  # Adjust this threshold as needed
                drowsiness_detected = False

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if not drowsiness_detected:
            print("No drowsiness detected.")
            
            cap.release()
            cv2.destroyAllWindows()
            if not drowsiness_detected:
                print("No drowsiness detected.")

    def analyze_image(self, file_path):
        image = tf.keras.preprocessing.image.load_img(file_path, target_size=(img_width, img_height))
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)

        prediction = self.loaded_model.predict(image_array)
        prediction_value = prediction[0]
        if prediction_value < 0.5:
            result = "Closed eyes - Person is likely sleepy."
        else:
            result = "Opened eyes - Person is likely awake."

        self.show_result(result)

    def show_warning(self):
        print("Drowsiness detected!")

    def show_result(self, result):
        messagebox.showinfo("Drowsiness Detection Result", result)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    face_cascade = cv2.CascadeClassifier('C:/Users/rumaa/OneDrive/Documents/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('C:/Users/rumaa/OneDrive/Documents/haarcascade_eye.xml')

    img_width, img_height = 100, 100  # Set image dimensions
    batch_size = 32
    epochs = 10
    app = DrowsinessDetectorUI(img_width, img_height)
    app.run()
