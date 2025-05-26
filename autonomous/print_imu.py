import serial
import time
import serial.tools.list_ports
# import ctrl
# Configuration
  # Replace with your port
BAUD_RATE=115200
class SerialPortChecker():
    def __init__(self, baud_rate, timeout):
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
                data = ser.readline().decode('utf-8').strip()
                if keyword.lower() in data.lower():
                    print(f"'{keyword}' detected on {port}!")
                    return port
        print("No port found.")
        exit()


SERIAL_PORT=SerialPortChecker(BAUD_RATE, 2).find_port("auto")
# SERIAL_PORT='/dev/ttyUSB0'

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
                    print(line)
                    #print("aa")
                    gps_data = line[1:3]
                    imu_data = line[-1]
                   
                    print(f"Latitude: {gps_data[0]} , Longitude: {gps_data[-1]}")
                
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

def parse_imu_data(line):
    """
    Parses the IMU data line into a dictionary with gyroscope and accelerometer values.
    """
    try:
        parts = line.split(" Accel ")
        gyro_part = ''.join(parts[0].split()[2:5])
        gx, gy, gz = map(float, gyro_part.split(","))
        return gz
    except Exception as e:
        print(f"Error parsing IMU data: {e}")
        return None

if __name__ == "__main__":
    read_imu_data()
