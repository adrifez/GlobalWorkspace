def execute_action(direction, position):
    x = position[0]
    y = position[1]

    if direction == "U":
        y = y+1
    elif direction == "R":
        x = x+1
    elif direction == "D":
        y = y-1
    else:
        x = x-1

    return [x,y]

def opposite_direction(direction):
    opposite_direction = ""

    if direction == "U":
        opposite_direction = "D"
    elif direction == "R":
        opposite_direction = "L"
    elif direction == "D":
        opposite_direction = "U"
    else:
        opposite_direction = "R"

    return opposite_direction
