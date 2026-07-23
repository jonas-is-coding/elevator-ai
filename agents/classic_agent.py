def scan_action(building):
    actions = {e.number: 0 for e in building.elevators}
    committed = set()  # Aufzüge, die diesen Step nicht neu vergeben werden dürfen

    # Priorität 1: Aufzüge mit Fahrgästen an Bord fahren zu ihrem nächsten Ziel
    for elevator in building.elevators:
        if elevator.targetFloors:
            next_target = elevator.targetFloors[0]
            if elevator.currentFloor < next_target:
                actions[elevator.number] = 1
            elif elevator.currentFloor > next_target:
                actions[elevator.number] = 2
            committed.add(elevator.number)

    # Priorität 2: Aufzüge, die bereits einem Ruf zugewiesen sind, bleiben dabei,
    # bis der Ruf tatsächlich bedient wurde - keine Neuvergabe nur weil gerade
    # ein näherer Ruf auftaucht
    for elevator in building.elevators:
        if elevator.number in committed:
            continue
        if elevator.assignedCall is not None:
            floor_number, direction = elevator.assignedCall
            floor = building.floors[floor_number]
            still_waiting = floor.waitingUp if direction == "up" else floor.waitingDown

            if not still_waiting:
                elevator.assignedCall = None  # Ruf wurde inzwischen bedient
            else:
                committed.add(elevator.number)
                if elevator.currentFloor < floor_number:
                    actions[elevator.number] = 1
                elif elevator.currentFloor > floor_number:
                    actions[elevator.number] = 2
                # sonst: idle stehen bleiben, damit der Pickup in diesem Step
                # überhaupt registriert werden kann (Bewegung wird VOR dem
                # Pickup-Check angewendet - wer wegfährt, verpasst ihn)

    # Priorität 3: freie Aufzüge auf die am längsten wartenden Rufe verteilen
    # (nicht nach Stockwerksnummer, sonst werden niedrige Stockwerke bevorzugt)
    calls = []
    for floor in building.floors:
        if floor.waitingUp:
            calls.append((floor.waitingUpSince, floor.number, "up"))
        if floor.waitingDown:
            calls.append((floor.waitingDownSince, floor.number, "down"))
    calls.sort(key=lambda c: c[0], reverse=True)

    for _, floor_number, direction in calls:
        free_elevators = [e for e in building.elevators if e.number not in committed]
        if not free_elevators:
            break

        closest = min(free_elevators, key=lambda e: abs(e.currentFloor - floor_number))
        committed.add(closest.number)
        closest.assignedCall = (floor_number, direction)

        if closest.currentFloor < floor_number:
            actions[closest.number] = 1
        elif closest.currentFloor > floor_number:
            actions[closest.number] = 2
        # sonst: idle, siehe Kommentar oben - sofort wegfahren würde den
        # Pickup in diesem Step verhindern

    return list(actions.values())
