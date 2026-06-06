import cv2

class WitchLightRecognizer:
    def __init__(self):
        # Load OpenCV's built-in, highly optimized frontal face detector xml blueprint
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            print("[CRITICAL] Failed to load native face cascade xml tracking blueprint.")

    def run_stream(self):
        # Bind directly to default system webcam hardware bus
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            print("[CRITICAL] Camera hardware bus initialization failed.")
            return

        print("[SYSTEM ACTIVE] Pure OpenCV matrix stream deployed. Press 'q' to disconnect.")

        while True:
            # Capture individual frame matrix
            ret, frame = video_capture.read()
            if not ret:
                print("[ERROR] Frame dropped by system hardware bus.")
                break

            # Core Optimization: Map matrix to grayscale for fast pixel analysis
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Spatial Object Localization via sliding window processing
            faces = self.face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30)
            )

            # Render geometric boundary tracking overlays onto the live frame array
            for (x, y, w, h) in faces:
                # Render tracking box rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Render matching metadata tag banner above tracking box
                cv2.putText(
                    frame, 
                    "Operator Tracking Active", 
                    (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 255, 0), 
                    2
                )

            # Paint processed array out to the screen window context
            cv2.imshow("The Witch Light - Native Identity Engine", frame)

            # Keyboard interception layer hook
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Termination sequence caught. Flushing memory structures.")
                break

        # Post-loop cleaning routines: release arrays and sever hardware locks
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    engine = WitchLightRecognizer()
    engine.run_stream()