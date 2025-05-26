import cv2
import numpy as np

# window_width = 
size=(700, 350)
size2 = (1400, 700)

def main():
    # Initialize video capture for 4 streams
    cap1 = cv2.VideoCapture("/dev/video0")  
    while True:
        # Read frames from each capture
        ret1, frame1 = cap1.read()
        cv2.imshow('Video Streams', frame1)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video captures and close windows
    cap1.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
