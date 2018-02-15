import cv2 as cv
import numpy as np


def main():
    cap = cv.VideoCapture(0)

    while(True):
        _, frame = cap.read()

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
        _, contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            epsilon = 0.015 * cv.arcLength(cnt,True)
            approx = cv.approxPolyDP(cnt,epsilon,True)

            if len(approx) == 4 and cv.contourArea(cnt) > 100:
                rect = cv.minAreaRect(cnt)
                box = cv.boxPoints(rect)
                box = np.int0(box)
                cv.drawContours(frame, [box], 0, (0, 255, 0), 2)


        cv.imshow('OpenCV Find Squares', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()