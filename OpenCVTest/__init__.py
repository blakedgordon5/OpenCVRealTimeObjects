import cv2 as cv
import numpy as np


def main():
    cap = cv.VideoCapture(0)

    while(True):
        _, frame = cap.read()

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 10, param1=50, param2=30, minRadius=0, maxRadius=25)
        if circles is not None :
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # draw the outer circle
                cv.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)


        cv.imshow('OpenCV Find Squares', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()