from enum import Enum

class Moving(Enum):
    down = -1
    up = 1
    idle = 0

class Elevator:
    def __init__(self, number: int, moving: Moving, currentFloor: int, targetFloors: list[int]):
        self.number = number
        self.moving = moving
        self.currentFloor = currentFloor
        self.targetFloors = targetFloors

class Floor:
    def __init__(self, number: int, waitingUp: bool, waitingDown: bool):
        self.number = number
        self.waitingUp = waitingUp
        self.waitingDown = waitingDown
        self.waitingUpSince = 0
        self.waitingDownSince = 0

class Building:
    def __init__(self, num_floors: int, num_elevators: int):
        self.floors = [Floor(i, False, False) for i in range(num_floors)]
        self.elevators = [Elevator(i, Moving.idle, 0, []) for i in range(num_elevators)]