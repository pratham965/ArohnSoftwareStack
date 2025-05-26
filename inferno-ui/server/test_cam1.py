import cv2

# Open the default camera (usually the first USB camera, index 0)
cap = cv2.VideoCapture('rtsp://admin:ABCdef123@192.168.1.64:554/cam/realmonitor?channel=1&subtype=0')

if not cap.isOpened():
    print("Error: Could not open video device")
else:
    # Set camera resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Display the resulting frame
        frame = cv2.resize(frame, (780, 480))

        cv2.imshow('cam64', frame)


        # Press 'q' to quit the video display
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
