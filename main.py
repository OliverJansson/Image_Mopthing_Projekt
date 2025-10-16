# Standard libs
import cv2 as cv
import argparse
import numpy as np

# My libs
import face_recognition as fc

def pixelate(image:int, pixel_size:int = 30):

    det_faces = fc.detect_face(image)

    if len(det_faces) == 0:
        return image, True
    
    new_image = image.copy()

    for x, y, width, hight in det_faces:
        #Resize the image with interpolation
        resized_image = cv.resize(image[y:y+hight, x:x+width], (width // pixel_size, hight // pixel_size), 
                               interpolation=cv.INTER_LINEAR)
        resized_image = cv.resize(resized_image, (width, hight), interpolation=cv.INTER_NEAREST)
        new_image[y:y+hight, x:x+width] = resized_image

    return new_image, False

### BackBone ###
def pixelate_circul(image, pixel_size):
    
    det_faces = fc.detect_face(image)

    if len(det_faces) == 0:
        return image, True
    
    new_image = image.copy()

    for x, y, width, hight in det_faces:
        #Resize the image with interpolation
        resized_image = cv.resize(image[y:y+hight, x:x+width], (width // pixel_size, hight // pixel_size), 
                               interpolation=cv.INTER_LINEAR)
    
      
        h_s, w_s, _ = resized_image.shape
        radius = pixel_size // 2

        crop = np.ones((h_s*pixel_size,w_s*pixel_size,3))*np.mean(resized_image)
        
        for y_c in range(h_s):
            for x_c in range(w_s):
                color = tuple(int(c) for c in resized_image[y_c, x_c])
                center_x = x_c * pixel_size + radius
                center_y = y_c * pixel_size + radius
                cv.circle(crop, (center_x, center_y), 
                          radius, color, -1, lineType=cv.LINE_AA)
        
        new_image[y:y+h_s*pixel_size, x:x+w_s*pixel_size] = crop

    return new_image, False


def callback(kernal_size:int):
    pass


def pixel_face(filename:str, scale:float=0.25):

    image = cv.imread(filename)
    h,w,_=image.shape
    print(f"Hight: {h}px, Width: {w}px")
    if h > 2500:
        h = int(h*scale)
        w = int(w*scale)
    if h < 400:
        h = int(h/scale)
        w = int(w/scale)

    image = cv.resize(image, (w, h))
    
    windowName = "Face Pixelated"
    cv.namedWindow(windowName)
    cv.createTrackbar("Size", windowName, 10, 50, callback)

    while True:
        if cv.waitKey(1) == ord("q"):
            break

        kernal_size = max(1, cv.getTrackbarPos("Size", windowName))
        new_image, flag = pixelate(image, pixel_size=kernal_size)

        cv.imshow(windowName, new_image)

    cv.destroyAllWindows()
    return


def pixel_face_video():
    cap = cv.VideoCapture(0)

    windowName = "Face Pixelated"
    cv.namedWindow(windowName)
    cv.createTrackbar("Size", windowName, 10, 50, callback)
    while True:
        if cv.waitKey(1) == ord("q"):
            break
        
        kernal_size = max(2, cv.getTrackbarPos("Size", windowName))

        ret, frame = cap.read()
        #new_frame, flag = pixelate(frame,   pixel_size=kernal_size)
        new_frame, flag = pixelate_circul(frame,   pixel_size=kernal_size)
        cv.imshow(windowName, new_frame)

    cap.release()
    cv.destroyAllWindows() 
    return

########## MAIN ###########

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Morph") 
    parser.add_argument("-v", "--video", action='store_true', help="Uses the camera")
    parser.add_argument("--debug", action='store_true', help="Debugs, loops Image dir")
    parser.add_argument("--path", default="Images/Image1.jpg", type=str, help="FilePath to the image") 
    parser.add_argument("-s", "--scale", type=float, help="Percentage to scale the image") 
    args = parser.parse_args()
    
 
    if args.video:
        pixel_face_video()

    elif args.debug:
        from pathlib import Path
        root_dir = Path(__file__).resolve().parent

        for files in (root_dir / "Images").iterdir():
            pixel_face(files)

    else:
        if args.scale:
            pixel_face(args.path, scale=args.scale)
        else:
            pixel_face(args.path)
