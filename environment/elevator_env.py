import random
import numpy as np
import gymnasium as gym
from environment.building import Building
from environment.building import Moving
from environment.traffic_patterns import (spawn_passengers)

action_map = {
    0: Moving.idle,
    1: Moving.up,
    2: Moving.down
}

class ElevatorEnv(gym.Env):
    def __init__(self):
        elevator_low = np.tile([0, -1, 0], 4)
        elevator_high = np.tile([19, 1, 20], 4)

        floor_low = np.tile([0, 0, 0, 0], 20)  # waitingUp, waitingDown, waitingSince, waitingSince
        floor_high = np.tile([1, 1, 10000, 10000], 20)

        low = np.concatenate([elevator_low, floor_low])
        high = np.concatenate([elevator_high, floor_high])

        self.action_space = gym.spaces.MultiDiscrete([3, 3, 3, 3])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.building = Building(20, 4)
        self.seconds_counter = 0
        self.time_of_day = 5
        self.current_step = 0
        self.max_steps = 10000

    def _get_observation(self):
        obs = []

        for elevator in self.building.elevators:
            obs.append(elevator.currentFloor)  # 0-19
            obs.append(elevator.moving.value)  # -1, 0, oder 1
            obs.append(len(elevator.targetFloors))

        for floor in self.building.floors:
            obs.append(floor.waitingUp)
            obs.append(floor.waitingDown)
            obs.append(floor.waitingUpSince)
            obs.append(floor.waitingDownSince)

        return np.array(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.building = Building(20, 4)
        self.current_step = 0
        self.seconds_counter = 0
        self.time_of_day = 5
        return self._get_observation(), {}

    def step(self, action):
        waiting_times = []
        reward = 0
        pickups = 0
        dropoffs = 0
        self.current_step += 1
        self.seconds_counter += 1

        if self.seconds_counter == 3600:
            self.seconds_counter = 0
            self.time_of_day += 1

        if self.time_of_day == 24:
            self.time_of_day = 0

        # statt 0.3 pro Step pro Floor:
        spawn_passengers(self.building, self.time_of_day, probability=0.02)

        for i, elevator in enumerate(self.building.elevators):
            elevator.moving = action_map[action[i]]
            elevator.currentFloor = max(0, min(19, elevator.currentFloor + elevator.moving.value))

        for floor in self.building.floors:
            if floor.waitingUp:
                floor.waitingUpSince += 1
                waiting_times.append(floor.waitingUpSince)
            if floor.waitingDown:
                floor.waitingDownSince += 1
                waiting_times.append(floor.waitingDownSince)

            for elevator in self.building.elevators:
                if elevator.currentFloor == floor.number:
                    if elevator.moving.value >= 0 and floor.waitingUp:
                        target = random.randint(floor.number + 1, len(self.building.floors) - 1)
                        elevator.targetFloors.append(target)
                        floor.waitingUp = False
                        floor.waitingUpSince = 0
                        pickups += 1
                    elif elevator.moving.value <= 0 and floor.waitingDown:
                        target = random.randint(0, floor.number - 1)
                        elevator.targetFloors.append(target)
                        floor.waitingDown = False
                        floor.waitingDownSince = 0
                        pickups += 1

        for elevator in self.building.elevators:
            if elevator.targetFloors:
                next_target = elevator.targetFloors[0]
                if (elevator.moving.value > 0 and elevator.currentFloor < next_target) or \
                        (elevator.moving.value < 0 and elevator.currentFloor > next_target):
                    reward += 0.01  # kleiner Bonus pro Step in richtiger Richtung

                count = elevator.targetFloors.count(elevator.currentFloor)

                if count > 0:
                    dropoffs += count
                    elevator.targetFloors = [
                        f for f in elevator.targetFloors
                        if f != elevator.currentFloor
                    ]

        avg_wait = 0
        if len(waiting_times) > 0:
            avg_wait = sum(waiting_times) / len(waiting_times)
            reward += dropoffs - min(avg_wait, 100) / 100
        else:
            reward += dropoffs

        observation = self._get_observation()

        terminated = self.current_step > 100 and not any(
            floor.waitingUp or floor.waitingDown
            for floor in self.building.floors
        ) and not any(
            elevator.targetFloors
            for elevator in self.building.elevators
        )

        truncated = self.current_step >= self.max_steps

        info = {
            "pickups": pickups,
            "dropoffs": dropoffs,
            "avg_wait": avg_wait,
            "truncated": truncated,
            "reward": reward
        }

        return observation, reward, terminated, truncated, info