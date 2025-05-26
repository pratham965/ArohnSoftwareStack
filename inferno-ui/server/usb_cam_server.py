from flask import Flask, Response
import time
import cv2

app = Flask(__name__)

# Replace with your RTSP camera URL
camera_urls = {
    '1' : 0,
    # '2' : 1
}


def generate_frames(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        raise Exception("Unable to open RTSP stream")

    fps_limit = 10  # Limit to 10 frames per second
    start_time = time.time()

    try:
        while True:
            success, frame = cap.read()

            if not success:
                print("Failed to capture frame. Releasing camera.")
                break

            # Limit FPS
            if time.time() - start_time < 1.0 / fps_limit:
                continue

            start_time = time.time()

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Failed to encode frame. Releasing camera.")
                break
            frame = buffer.tobytes()

            # Yield the output as a multipart data stream (for MJPEG format)
            yield (b'--frame\r\n'

                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        # Release the camera when done
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released.")

    

@app.route('/video_feed/<cam_id>')
def video_feed(cam_id):

    rtsp_url = camera_urls.get(cam_id)

    return Response(generate_frames(rtsp_url), mimetype='multipart/x-mixed-replace; boundary=frame')

    

@app.route('/')
def index():
    return '''
    <html>
        <body>
            <h1>RTSP Stream</h1>
            <img src="/video_feed/1">
        </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)