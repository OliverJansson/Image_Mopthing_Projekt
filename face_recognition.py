
# Dependices
import cv2 as cv

def detect_face(image:int, debug:bool=False):
    face_alg_path = "haarcascade_frontalface_default.xml"
    haar_cascade = cv.CascadeClassifier(face_alg_path)

    image_flatt = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    # Detect face
    det_face = haar_cascade.detectMultiScale(image_flatt, scaleFactor=1.05, 
                                             minNeighbors=5, minSize=(50,50),  
                                             flags=cv.CASCADE_SCALE_IMAGE)

    if debug: #Showes found faces
        print(f"Faces detected: {len(det_face)}")
        for x, y, w, h in det_face:
            face_cropped = image[y:y+h, x:x+w]
            cv.namedWindow("Face")
            cv.imshow("Face", face_cropped)
            cv.waitKey(0)
            cv.destroyAllWindows()
            return
        
    return det_face

if __name__ == "__main__":

    from pathlib import Path
    root_dir = Path(__file__).resolve().parent

    for files in (root_dir / "Images").iterdir():
        image = cv.imread(files)
        detect_face(image, debug=True)
