import random

def spawn_passengers(building, time_of_day, probability = 0.3):

    morning = time_of_day > 6 and time_of_day < 10
    evening = time_of_day > 16 and time_of_day < 20

    for floor in building.floors:
        going_up = random.choice([True, False])
        floor_probability = probability

        if morning:
            going_up = random.random() < 0.8
            if floor.number == 0:
                floor_probability = 0.8
        elif evening:
            going_up = random.random() > 0.8
            if floor.number > 0:
                floor_probability = 0.4

        if random.random() < floor_probability:
            if floor.number == len(building.floors) - 1:
                floor.waitingDown = True
            elif floor.number == 0:
                floor.waitingUp = True
            else:
                if going_up:
                    floor.waitingUp = True
                else:
                    floor.waitingDown = True