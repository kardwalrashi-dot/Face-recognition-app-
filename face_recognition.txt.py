import cv2
import face_recognition
import os
from datetime import datetime

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.lang import Builder
from kivy.properties import StringProperty

KV = '''
<MainUI>:
    orientation: 'vertical'

    # Header
    Label:
        text: "Face Recognition App"
        size_hint_y: 0.1
        font_size: '22sp'
        bold: True

    # Camera Screen
    Image:
        id: cam
        size_hint_y: 0.6

    # Status Text
    Label:
        id: status
        text: root.status_text
        size_hint_y: 0.1
        font_size: '18sp'

    # Buttons
    BoxLayout:
        size_hint_y: 0.2
        spacing: 10
        padding: 10

        Button:
            text: "Start"
            on_press: root.start_camera()

        Button:
            text: "Stop"
            on_press: root.stop_camera()

        Button:
            text: "Retry"
            on_press: root.retry()
'''

class MainUI(BoxLayout):
    status_text = StringProperty("Status: Idle")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.capture = cv2.VideoCapture(0)
        self.running = False
        self.paused = False

        # Load known faces
        self.known_encodings = []
        self.known_names = []

        path = "known_faces"
        if os.path.exists(path):
            for file in os.listdir(path):
                img = face_recognition.load_image_file(f"{path}/{file}")
                enc = face_recognition.face_encodings(img)
                if len(enc) > 0:
                    self.known_encodings.append(enc[0])
                    self.known_names.append(file.split(".")[0])

        self.unknown_path = "unknown_faces"
        os.makedirs(self.unknown_path, exist_ok=True)

        Clock.schedule_interval(self.update, 1.0/30.0)

    def start_camera(self):
        self.running = True
        self.paused = False
        self.status_text = "Status: Running"

    def stop_camera(self):
        self.running = False
        self.status_text = "Status: Stopped"

    def retry(self):
        self.paused = False
        self.status_text = "Status: Running"

    def update(self, dt):
        if not self.running or self.paused:
            return

        ret, frame = self.capture.read()
        if not ret:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, faces)

        for (top, right, bottom, left), face_encoding in zip(faces, encodings):

            matches = face_recognition.compare_faces(self.known_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                index = matches.index(True)
                name = self.known_names[index]

                self.status_text = f"Match Found: {name}"
                self.paused = True

            else:
                self.status_text = "Unknown Face"

                face_img = frame[top:bottom, left:right]
                filename = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
                cv2.imwrite(f"{self.unknown_path}/{filename}", face_img)

            # Draw box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show frame in UI
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

        self.ids.cam.texture = texture

class FaceApp(App):
    def build(self):
        Builder.load_string(KV)
        return MainUI()

FaceApp().run()