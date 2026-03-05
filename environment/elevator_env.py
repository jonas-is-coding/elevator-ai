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
        # Für 4 Aufzüge: je [currentFloor, direction] = 8 Werte
        elevator_low = np.tile([0, -1], 4)  # → [0,-1, 0,-1, 0,-1, 0,-1]
        elevator_high = np.tile([19, 1], 4)  # → [19,1, 19,1, 19,1, 19,1]

        # Für 20 Stockwerke: je [waitingUp, waitingDown] = 40 Werte
        floor_low = np.tile([0, 0], 20)
        floor_high = np.tile([1, 1], 20)

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

        for floor in self.building.floors:
            obs.append(floor.waitingUp)
            obs.append(floor.waitingDown)

        return np.array(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.building = Building(20, 4)
        self.current_step = 0
        self.seconds_counter = 0
        self.time_of_day = 5
        return self._get_observation(), {}

    def step(self, action):
        waiting_times = []
        self.current_step += 1
        self.seconds_counter += 1

        if self.seconds_counter == 3600:
            self.seconds_counter = 0
            self.time_of_day += 1

        if self.time_of_day == 24:
            self.time_of_day = 0

        # statt 0.3 pro Step pro Floor:
        spawn_passengers(self.building, self.time_of_day, probability=0.001)

        for i, elevator in enumerate(self.building.elevators):
            elevator.moving = action_map[action[i]]
            elevator.currentFloor += elevator.moving.value

        for floor in self.building.floors:
            if floor.waitingUp:
                floor.waitingUpSince += 1
                waiting_times.append(floor.waitingUpSince)
            if floor.waitingDown:
                floor.waitingDownSince += 1
                waiting_times.append(floor.waitingDownSince)

            for elevator in self.building.elevators:
                if elevator.currentFloor == floor.number:
                    if elevator.moving.value > 0:
                        floor.waitingUp = False
                        floor.waitingUpSince = 0
                    if elevator.moving.value < 0:
                        floor.waitingDown = False
                        floor.waitingDownSince = 0

        reward = 0

        reward = 0
        if len(waiting_times) > 0:
            avg_wait = sum(waiting_times) / len(waiting_times)
            reward = -min(avg_wait, 100) / 100  # zwischen -1 und 0

        observation = self._get_observation()

        terminated = self.current_step > 100 and not any(
            floor.waitingUp or floor.waitingDown
            for floor in self.building.floors
        )

        truncated = self.current_step >= self.max_steps

        info = {} # Vielleicht später ändern

        return observation, reward, terminated, truncated, info