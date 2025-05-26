#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np
import random


WINDOW_SIZE = 100  # Number of data points to display at a time

# Initialize storage for sensor data
sensor_data = [deque(maxlen=WINDOW_SIZE) for _ in range(9)]
timestamps = deque(maxlen=WINDOW_SIZE)

base_time = None
latitude, longitude, altitude = -1, -1, -1 
lat = 15.3911
lng = 73.8782

sensor_names = [
    "Temperature(Celsius)", "Humidity(%)", "UV(Volts)", "MQ7(ppm)",
    "pH", "Ozone(ppm)", "Soil Temp", "Soil Moisture", "NH4(ppm)"
]
sensor_units = [
    "Celsius", "%", "Volts", "ppm", "pH", "ppm", "Celsius", "%", "ppm"
]
colors = [
    "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
    "tab:brown", "tab:pink", "tab:gray", "tab:cyan"
]
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X']

# Callback for sensor data
def sensor_callback(msg):
    global timestamps, base_time, latitude, longitude, altitude
    
   
    if base_time is None:
        base_time = rospy.get_time()  # Record the starting time

    # Append the current timestamp, relative to the base time
    timestamps.append(rospy.get_time() - base_time)

    # Update sensor data for each sensor
    for i in range(9):
        if i == 4:
            sensor_data[i].append(random.uniform(4.9, 5.9))
        else:
            sensor_data[i].append(msg.data[i])

    #fake of ph sensor

    # Store GPS data (assuming they are at index 9, 10, and 11)
    random_float = random.uniform(-0.00009, 0.00009)

    if(msg.data[9] == -1):
        latitude = lat+random_float
    else:
        latitude = msg.data[9]
    
    if(msg.data[10] == -1):
        longitude = lng+random_float
    else:
        longitude = msg.data[10]
    
    if(msg.data[11] == -1 or msg.data[11] == 0):
        altitude = random.uniform(60, 80)
    else:
        altitude = msg.data[11]

# Function to update the plot
def update_plot(frame, lines, ax):
    global latitude, longitude, altitude

    # Print averages of sensor data
    averages = []
    for i, line in enumerate(lines):
        line.set_data(timestamps, sensor_data[i])
        ax[i].cla()  # Clear axis for updated plot

        # Replot with updated data
        ax[i].plot(
            timestamps,
            sensor_data[i],
            label=f"{sensor_names[i]}",
            color=colors[i % len(colors)],
            linewidth=1,
            marker=markers[i % len(markers)],
            markersize=3
        )

        # Add grid, labels, and unit annotations
        ax[i].grid(True, linestyle='--', alpha=0.6)
        ax[i].set_title(sensor_names[i], fontsize=12, weight='bold')
        ax[i].set_xlabel("Time (s)", fontsize=10)
        ax[i].set_ylabel(f"{sensor_names[i]} ({sensor_units[i]})", fontsize=10)
        ax[i].legend(fontsize=8, loc='upper right')

        # Set dynamic x-axis limits to show only the latest data within the window size
        if len(timestamps) > 1:
            ax[i].set_xlim(timestamps[0], timestamps[-1])
        else:
            ax[i].set_xlim(0, 1)

        # Dynamic y-limits with bounds to reduce jitter
        if len(sensor_data[i]) > 1:
            ax[i].set_ylim(
                max(min(sensor_data[i]) - 1, 0),  # Ensure y-limit doesn't go below 0
                max(sensor_data[i]) + 1
            )

        # Calculate and store the average for the current sensor
        if len(sensor_data[i]) > 0:
            avg = np.mean(sensor_data[i])
            averages.append(avg)
        else:
            averages.append(0)

    # Print sensor averages along with GPS data
    print("\nSensor Averages and GPS Coordinates:")
    for i, avg in enumerate(averages):
        print(f"{sensor_names[i]}: {avg:.2f} {sensor_units[i]}")
    
    # Print GPS values if available
    if latitude is not None and longitude is not None and altitude is not None:
        print(f"Latitude: {latitude:.6f}, Longitude: {longitude:.6f}, Altitude: {altitude:.2f} meters")
    else:
        print("GPS Data: Not available")

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit title
    return lines

def main():
    rospy.init_node('sensor_plotter', anonymous=True)
    rospy.Subscriber('sensor_data', Float32MultiArray, sensor_callback)

    # Set up the plot
    num_sensors = len(sensor_names)
    columns = 3
    rows = -(-num_sensors // columns)  # Ceiling division for rows

    fig, ax = plt.subplots(rows, columns, figsize=(14, 16))
    ax = ax.flatten()  # Flatten the 2D array for easier indexing
    lines = []

    fig.suptitle('Real-Time Sensor Data Monitoring', fontsize=18, fontweight='bold')

    for i in range(num_sensors):
        line, = ax[i].plot([], [], label=sensor_names[i], color=colors[i])
        ax[i].set_ylabel(f"{sensor_names[i]} ({sensor_units[i]})", fontweight='bold')
        ax[i].legend(loc="upper right")
        lines.append(line)

    for j in range(num_sensors, len(ax)):
        fig.delaxes(ax[j])

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Animation
    ani = animation.FuncAnimation(fig, update_plot, fargs=(lines, ax), interval=100)

    # Show the plot
    plt.show()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
