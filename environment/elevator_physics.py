import math
import random

MAX_SPEED = 2.5 # 2.5 m/s
ACCELERATION = 1.0 # 1.0 m/s²
FLOOR_HEIGHT = 3.5 # 3.5 m
DOOR_DELAY = 1 # 1 s
DOOR_TIME = 2 # 2 s
DEPARTURE_DELAY = 2 # 2 s

def calculate_travel_time(from_floor, to_floor):
    time_to_max_speed = MAX_SPEED / ACCELERATION
    ramp_distance = 0.5 * ACCELERATION * time_to_max_speed ** 2
    total_distance = abs(from_floor - to_floor) * FLOOR_HEIGHT
    constant_distance = total_distance - ramp_distance * 2

    if abs(from_floor - to_floor) >= 2:
        total_time = time_to_max_speed * 2 + constant_distance / MAX_SPEED
    else:
        t = math.sqrt(total_distance / ACCELERATION)
        total_time = t * 2

    return total_time

def calculate_stop_time():
    boarding_time = random.randint(3, 6)
    total_time = DOOR_DELAY + DOOR_TIME * 2 + DEPARTURE_DELAY + boarding_time
    return total_time, boarding_time