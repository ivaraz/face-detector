import cv2

face_ref = cv2.CascadeClassifier("face_ref.xml")
camera = cv2.VideoCapture(0)

def face_detector(frame):
    optimized_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_ref.detectMultiScale(optimized_frame, scaleFactor=1.1, minNeighbors=3)
    return faces
def drawer_box(frame):
    for x, y, w, h in face_detector(frame):
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)

def close_windows():
    camera.release()
    cv2.destroyAllWindows()
    exit()

def main():
    while True:
        result,frame = camera.read()
        if result is False:
            break
        drawer_box(frame)
        cv2.imshow("Ivar AI", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            close_windows()

if __name__ == '__main__':
    main()