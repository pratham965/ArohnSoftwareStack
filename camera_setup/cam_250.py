import cv2
import numpy as np

# window_width = 
size=(683,290)

def main():
    # Initialize video capture for 4 streams
    cap1 = cv2.VideoCapture("rtsp://admin:Teaminferno@192.168.1.250:554/cam/realmonitor?channel=1&subtype=0")

    while True:
        # Read frames from each capture
        ret1, frame1 = cap1.read()

        # Resize frames to ensure they fit in the display window
        frame1 = cv2.resize(frame1, size)

        # Show the grid of frames
        cv2.imshow('250', frame1)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video captures and close windows
    cap1.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
