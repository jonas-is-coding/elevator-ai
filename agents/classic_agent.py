def scan_action(building):
    actions = {e.number: 0 for e in building.elevators}
    assigned_elevators = set()  # merkt sich welche Aufzüge schon vergeben sind

    for floor in building.floors:
        if floor.waitingUp or floor.waitingDown:
            # nur noch freie Aufzüge berücksichtigen
            free_elevators = [e for e in building.elevators if e.number not in assigned_elevators]

            if not free_elevators:
                break

            closest = min(free_elevators, key=lambda e: abs(e.currentFloor - floor.number))
            assigned_elevators.add(closest.number)  # als vergeben markieren

            if closest.currentFloor < floor.number:
                actions[closest.number] = 1
            elif closest.currentFloor > floor.number:
                actions[closest.number] = 2
            else:
                # Aufzug ist bereits da – fahre in Richtung des Wartenden
                if floor.waitingUp:
                    actions[closest.number] = 1
                else:
                    actions[closest.number] = 2

    return list(actions.values())