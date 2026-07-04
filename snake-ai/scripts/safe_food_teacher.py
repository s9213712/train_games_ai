from collections import deque


ACTIONS = {
    0: (-1, 0),
    1: (0, -1),
    2: (0, 1),
    3: (1, 0),
}


def generate_even_board_cycle(board_size):
    if board_size % 2 != 0:
        raise ValueError("Hamiltonian cycle requires an even board size.")

    path = [(0, col) for col in range(board_size)]
    for row in range(1, board_size):
        cols = range(board_size - 1, 0, -1) if row % 2 else range(1, board_size)
        for col in cols:
            path.append((row, col))
    for row in range(board_size - 1, 0, -1):
        path.append((row, 0))
    return path


def hamiltonian_action(game):
    cycle = generate_even_board_cycle(game.board_size)
    cycle_index = {cell: index for index, cell in enumerate(cycle)}
    head = game.snake[0]
    nxt = cycle[(cycle_index[head] + 1) % len(cycle)]
    return action_between(head, nxt)


def action_between(current, nxt):
    delta = (nxt[0] - current[0], nxt[1] - current[1])
    for action, action_delta in ACTIONS.items():
        if delta == action_delta:
            return action
    raise ValueError(f"Cells are not adjacent: {current} -> {nxt}")


def next_snake_state(snake, food, board_size, action):
    row_delta, col_delta = ACTIONS[action]
    head = snake[0]
    new_head = (head[0] + row_delta, head[1] + col_delta)
    if not in_bounds(new_head, board_size):
        return None, False

    ate = new_head == food
    blocked = set(snake if ate else snake[:-1])
    if new_head in blocked:
        return None, False

    new_snake = [new_head] + list(snake if ate else snake[:-1])
    return new_snake, ate


def in_bounds(cell, board_size):
    row, col = cell
    return 0 <= row < board_size and 0 <= col < board_size


def neighbors(cell, board_size):
    row, col = cell
    for row_delta, col_delta in ACTIONS.values():
        nxt = (row + row_delta, col + col_delta)
        if in_bounds(nxt, board_size):
            yield nxt


def shortest_path(start, goal, blocked, board_size):
    if start == goal:
        return [start]

    queue = deque([start])
    previous = {start: None}
    while queue:
        cell = queue.popleft()
        for nxt in neighbors(cell, board_size):
            if nxt in previous or nxt in blocked:
                continue
            previous[nxt] = cell
            if nxt == goal:
                path = [nxt]
                while previous[path[-1]] is not None:
                    path.append(previous[path[-1]])
                return list(reversed(path))
            queue.append(nxt)
    return None


def reachable_count(start, blocked, board_size):
    queue = deque([start])
    seen = {start}
    while queue:
        cell = queue.popleft()
        for nxt in neighbors(cell, board_size):
            if nxt in seen or nxt in blocked:
                continue
            seen.add(nxt)
            queue.append(nxt)
    return len(seen)


def tail_reachable(snake, board_size):
    if len(snake) < 2:
        return True
    head = snake[0]
    tail = snake[-1]
    blocked = set(snake[1:-1])
    return shortest_path(head, tail, blocked, board_size) is not None


def simulate_path_to_food(snake, food, board_size, path):
    current_snake = list(snake)
    for nxt in path[1:]:
        action = action_between(current_snake[0], nxt)
        current_snake, _ = next_snake_state(current_snake, food, board_size, action)
        if current_snake is None:
            return None
    return current_snake


def safe_food_action(game):
    board_size = game.board_size
    snake = list(game.snake)
    food = game.food
    head = snake[0]

    blocked = set(snake[:-1])
    path = shortest_path(head, food, blocked, board_size)
    if path and len(path) > 1:
        candidate = action_between(head, path[1])
        simulated = simulate_path_to_food(snake, food, board_size, path)
        if simulated and tail_reachable(simulated, board_size):
            return candidate

    valid_actions = []
    for action in ACTIONS:
        simulated, _ = next_snake_state(snake, food, board_size, action)
        if simulated is None:
            continue
        free_after_action = board_size * board_size - len(simulated)
        reachable = reachable_count(simulated[0], set(simulated[1:-1]), board_size)
        if tail_reachable(simulated, board_size) and reachable >= min(free_after_action + 1, len(snake)):
            distance = abs(simulated[0][0] - food[0]) + abs(simulated[0][1] - food[1])
            valid_actions.append((distance, -reachable, action))

    if valid_actions:
        valid_actions.sort()
        return valid_actions[0][2]

    try:
        return hamiltonian_action(game)
    except ValueError:
        for action in ACTIONS:
            simulated, _ = next_snake_state(snake, food, board_size, action)
            if simulated is not None:
                return action
    return 0
