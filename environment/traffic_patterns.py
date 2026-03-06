import random

def spawn_passengers(building, time_of_day, probability):

    morning = time_of_day > 6 and time_of_day < 10
    evening = time_of_day > 16 and time_of_day < 20

    for floor in building.floors:
        going_up = random.choice([True, False])
        floor_probability = probability

        if morning:
            if floor.number == 0:
                going_up = True  # alle vom Erdgeschoss fahren hoch
                floor_probability = probability * 3
            else:
                floor_probability = probability * 0.3  # obere Stockwerke fast leer morgens
        elif evening:
            going_up = random.random() > 0.8
            if floor.number > 0:
                floor_probability = probability * 2  # 2x mehr als normal

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