import cv2
import numpy as np

# window_width = 
size=(683,320)

def main():
    # Initialize video capture for 4 streams
    cap2 = cv2.VideoCapture("rtsp://admin:admin1234@192.168.1.252:554/cam/realmonitor?channel=1&subtype=0")

    while True:
        # Read frames from each capture
        ret1, frame1 = cap2.read()

        # Resize frames to ensure they fit in the display window
        frame1 = cv2.resize(frame1, size)
        frame1=cv2.rotate(frame1,cv2.ROTATE_180)

        # Show the grid of frames
        cv2.imshow('252', frame1)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video captures and close windows
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
