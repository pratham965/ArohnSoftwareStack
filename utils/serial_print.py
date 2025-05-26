import serial
import serial.tools.list_ports
# import ctrl
# Configuration
  # Replace with your port
BAUD_RATE=115200
usb_num = int(input('enter the usb number to check : '))
SERIAL_PORT=f'/dev/ttyUSB{usb_num}'
print(SERIAL_PORT)
def read_imu_data():
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
            print("Reading Data")
            while True:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore')
                    print(line)
            
                    

    except serial.SerialException as e:
        print(f"Serial connection error: {e}")
    except KeyboardInterrupt:
        print("Exiting...")
    except Exception as e:
        print(f"Unexpected error: {e}")



if __name__ == "__main__":
    read_imu_data()
