import rospy
import cv2
import numpy as np
from ultralytics import YOLO
from sensor_msgs.msg import Image,Joy
from cv_bridge import CvBridge
from arrowFunc import *
import serial
import time
import serial
import serial.tools.list_ports


class SerialPortChecker():
    def _init_(self, baud_rate, timeout):
        self.baud_rate = baud_rate
        self.timeout = timeout

    def list_serial_ports(self):
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports if "USB" in port.description]

    def find_port(self, keyword):
        for port in self.list_serial_ports():
            with serial.Serial(port, self.baud_rate, timeout=self.timeout) as ser:
                print(f"Checking {port}...")
                ser.flushInput()  # Clear input buffer
                ser.flushOutput()  # Clear output buffer

                # Read data during the timeout period
                i=0
                while i<10:
                    data = ser.readline().decode('utf-8').strip()
                    # print(data)
                    if data.lower()[0] == keyword:
                        print(f"SENSOR detected on {port}!")
                        
                    i+=1
                    return port
        print("No port found.")
        exit()

BAUD_RATE=9600
# SERIAL_PORT=SerialPortChecker(BAUD_RATE, 2).find_port("a")
SERIAL_PORT='/dev/ttyUSB1'

speed = 1.0

model = YOLO(r"/home/pratham/Downloads/8s15epoch.pt")
cone_model = YOLO(r"/home/pratham/Downloads/cone2.pt")

rospy.init_node('kinect_yolo_node', anonymous=True)
pub = rospy.Publisher("/joy", Joy,queue_size=10)

bridge = CvBridge()
cached_results = None
update_interval = 5
frame_count = 0
depth_frame = None
color_frame = None
msg = Joy()
msg.axes = [0.0,0.0]
window_center = 0
x_center = 0
turnin = False
arrow_number = 1
file = open("gps.txt","w")
file.close()
LR_counter = [0,0]
outer_offset = 400 # phle 300 thi 
inner_offset = 100
isCone = False



def crop_yolo_bbox(image, bbox):
    # Extract coordinates from bbox (assuming it has xyxy format)
    if bbox:
        x1, y1, x2, y2 = map(int, bbox.xyxy[0])  # Get coordinates in pixel format
        cropped_image = image[y1:y2, x1:x2] # Crop the region based on pixel coordinates
        # Optionally return both cropped image and YOLO-normalized bbox
        return cropped_image
    return image

def read_gps_data(diff):
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
            while True:
                global arrow_number
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').split(',')
                    gps_data = line[1:3]
                    gps_line = f"Latitude: {gps_data[0]} , Longitude: {gps_data[1]}"
                    print(gps_line)
                    file = open('gps.txt','a')
                    dir = "LEFT" if diff<0 else "RIGHT"
                    file.write(f"\n################# ARROW {arrow_number} : {dir} #################\n")
                    file.write(f"{gps_line}\n")
                    file.close()
                    arrow_number+=1
                    return
                time.sleep(0.01)

    except serial.SerialException as e:
        print(f"Serial connection error: {e}")
    except KeyboardInterrupt:
        print("Exiting...")
    except Exception as e:
        print(f"Unexpected error: {e}")  

def read_imu_data():
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
            print("Connected to MPU6050. Reading IMU data...")
            current_angle = 0  # Initial angle
            last_time = time.time()
            while True:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore')
                    line = line.split(',')
                    imu_data = line[-1]
                    gz = float(imu_data)
                    if gz:
                        current_time = time.time()
                        delta_time = current_time - last_time
                        last_time = current_time

                        if abs(gz) > 1:  
                            current_angle += gz * delta_time
                        print(f"Current Angle: {current_angle:.2f}°")

                        if abs(current_angle) >= 89:
                            print("90-degree turn detected!")
                            return "90-degree turn detected!"
                time.sleep(0.01)

    except serial.SerialException as e:
        print(f"Serial connection error: {e}")
    except KeyboardInterrupt:
        print("Exiting...")
    except Exception as e:
        print(f"Unexpected error: {e}") 

import os 
class FSM():
    def __init__(self):
        self.state="search"
        self.turned = False
        self.search_frames = 0
        msg.axes = [0.0,0.0]
        pub.publish(msg)

    def search(self):
        self.state = "search"
        self.turned = False
        self.search_frames += 1
        print(self.search_frames)
        if(self.search_frames >20):
            cached_results = cone_model.predict(color_frame, conf=0.3,verbose=False)
            max_box=max_box_calculation(cached_results)
            if max_box:
                global isCone
                print("manik isCone ko true kr raha hu. Ange ka dekh lio")
                isCone = True
            else:
                if not isCone:
                    self.search_frames=0    
                
        msg.axes = [0.0,0.0]
        pub.publish(msg)

    def approach(self):
        self.state="approach"
        global turnin
        outer_box_left = window_center - outer_offset
        outer_box_right = window_center + outer_offset
        if (x_center>outer_box_right or x_center<outer_box_left) or turnin==True:
            align(outer_box_left,outer_box_right)  
        else:
            msg.axes=[0.0,speed]
            pub.publish(msg)
        self.turned = False
        if not isCone:
            self.search_frames = 0



    def stop(self):
        self.state="stop"
        if not isCone:
            self.search_frames = 0
        msg.axes = [0.0,0.0]
        pub.publish(msg)
        if isCone:
            print("ander iscone")
            os._exit(404)
        global LR_counter
        should_rotate = True
        if(not self.turned ):
            print("STOPPING FOR 10s")
            read_gps_data(LR_counter[0] - LR_counter[1])
            wait()
            print(LR_counter)

            if LR_counter[0] - LR_counter[1]>0:
                print("LEFT")
                msg.axes = [speed,0.0]
            elif LR_counter[1] - LR_counter[0]>0:
                print("RIGHT")
                msg.axes = [-speed,0.0]
            elif LR_counter[1]==LR_counter[0]:
                print('Going Back')
                msg.axes = [0.0,-speed]
                should_rotate=False    

            pub.publish(msg)
            if should_rotate:
                read_imu_data()
                self.turned = True
                msg.axes = [0.0,0.0]
                pub.publish(msg)
            else:
                time.sleep(3)  #Let rover go back a bit to get clear view of arrow for inference
            LR_counter = [0,0]
        


    def get_state(self):
        return self.state
    
    def state_selector(self,found,depth_value,area,max_box):
        distance=1500
        
        if (depth_value==None) and found:
            self.approach()
        elif depth_value == None and not found:
            self.search()
        elif depth_value > distance:
            self.approach()
        elif area and depth_value <= distance:
            if area >=color_frame.shape[1]*color_frame.shape[0]*0.03:
                print("turn wala stoping....")
                self.stop()

        

def depth_callback(msg):
    global depth_frame
    depth_frame = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough') 
    depth_frame = np.array(depth_frame, dtype=np.uint16)  # Ensure depth data is in the correct format  

def depth_calculation(color_frame,x1,y1,x2,y2): 
    color_to_depth_x = depth_frame.shape[1] / color_frame.shape[1]
    color_to_depth_y = depth_frame.shape[0] / color_frame.shape[0]

    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    depth_x = int(center_x * color_to_depth_x)
    depth_y = int(center_y * color_to_depth_y)

    if 0 <= depth_x < depth_frame.shape[1] and 0 <= depth_y < depth_frame.shape[0]:
        depth_value = depth_frame[depth_y, depth_x]
    if depth_value :
        return depth_value
    else:
        return None

def max_box_calculation(cached_results):
    max_box=None
    maximum = -1
    for result in cached_results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            diagonal = ((x2 - x1)*2 + (y2 - y1)*2)*0.5
            if diagonal > maximum:
                max_box = box
                maximum = diagonal
    return max_box

def boundig_box_maker(max_box,color_frame):
    depth_value=None
    x1, y1, x2, y2 = map(int, max_box.xyxy[0])
    global x_center
    x_center=(x1+x2)//2
    confidence = max_box.conf[0]
    depth_value=depth_calculation(color_frame,x1,y1,x2,y2)
    cv2.rectangle(color_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(color_frame,(x_center,(y1+y2)//2),1,(255,0,0),2)
    depth_text = f'Depth: {depth_value} mm' if depth_value else 'Depth: N/A'
    text = f' {confidence:.2f}, {depth_text}'
    cv2.putText(color_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return depth_value

start_time = time.time()


def color_callback(msg):
    global cached_results, frame_count, depth_frame
    depth_value=None
    max_box = None
    found=None
    width,height,area=None,None,None
    global color_frame
    color_frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    global window_center
    window_center = color_frame.shape[1]//2
    if frame_count % update_interval == 0: # prediction harr update interval ke baad hi hoga (to skip frams)
        if  isCone :
            cached_results = cone_model.predict(color_frame, conf=0.3,verbose=False)
        else:    
            cached_results = model.predict(color_frame, conf=0.3,verbose=False)
        
    

    if cached_results:

        max_box=max_box_calculation(cached_results)
        
        if max_box:
            found=True
            depth_value=boundig_box_maker(max_box,color_frame)

            x_min, y_min, x_max, y_max = map(int, max_box.xyxy[0])
    
    # Calculate width and height
            width = x_max - x_min
            height = y_max - y_min
            area=height*width

            if frame_count % update_interval == 0 and depth_value:
                #AMAN ki bakchodi (vote according to prediction value)
                global LR_counter
                dir = arrow_detect(color_frame)
                if(dir!=None):
                    LR_counter[dir]+=1

            # Ensure width and height are positive
            if width < 0 or height < 0:
                raise ValueError("Invalid bounding box coordinates!")

    # cv2.line(color_frame,(window_center - outer_offset, 0),(window_center - outer_offset, color_frame.shape[0]),(255,0,0),2)
    # cv2.line(color_frame,(window_center + outer_offset, 0),(window_center + outer_offset, color_frame.shape[0]),(255,0,0),2)


    # cv2.line(color_frame,(window_center - inner_offset, 0),(window_center - inner_offset, color_frame.shape[0]),(0,255,0),2)
    # cv2.line(color_frame,(window_center + inner_offset, 0),(window_center + inner_offset, color_frame.shape[0]),(0,255,0),2)
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 0:
        fps = frame_count / elapsed_time
    else:
        fps = 0
    #cv2.putText(color_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 

    robot.state_selector(found,depth_value,area,max_box)

    color_frame_small = cv2.resize(color_frame, (960, 540), interpolation=cv2.INTER_NEAREST)
    # cv2.imshow('YOLO Kinect Stream', color_frame_small)
    

    print(f"depth=>{depth_value} : found=>{found} : isCone={isCone} : state=>{robot.get_state()} : LR_counter=>{LR_counter} : area : {area} ")

  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        rospy.signal_shutdown('User Exit')
        cv2.destroyAllWindows()

def wait():
    sec=5
    print("fuck bc")
    for i in range(sec):
        print(f'{sec-i} seconds left')
        time.sleep(1)

def align(outer_box_left, outer_box_right):
    global turnin
    diff = x_center - window_center
    turn_dir = -1 if diff>0 else 1
    msg.axes = [turn_dir*speed,0.0]
    pub.publish(msg)
    if (x_center>outer_box_left+inner_offset and x_center <outer_box_right-inner_offset):
        turnin=False
        msg.axes = [0.0,0.0]
        pub.publish(msg)
    else:
        turnin = True


rospy.Subscriber('/kinect2/hd/image_color', Image, color_callback)
rospy.Subscriber('/kinect2/sd/image_depth', Image, depth_callback)
robot = FSM()
rospy.spin()
