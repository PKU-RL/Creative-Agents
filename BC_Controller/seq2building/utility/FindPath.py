import numpy as np
from collections import deque

def is_valid_move(x, y, z, grid):
    return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and 0 <= z < grid.shape[2] and grid[x, y, z] == 1

def find_path(start, end, grid):
    directions = [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0),(1,0,1),(1,0,-1),(1,1,0),(1,-1,0),(-1,0,1),(-1,0,-1),(-1,1,0),(-1,-1,0),]

    start = (start[0], start[1], start[2])  # Adjust starting point based on the given conditions
    end = (end[0], end[1], end[2])

    visited = set()
    queue = deque([(start, [start])])

    while queue:
        current, path = queue.popleft()
        if current == end:
            return path

        if current in visited:
            continue

        visited.add(current)

        for dx, dy, dz in directions:
            new_position = (current[0] + dx, current[1] + dy, current[2] + dz)
            if is_valid_move(*new_position, grid):
                queue.append((new_position, path + [new_position]))

    return None

# map_size = (32, 32, 32)
# map_data = np.zeros(map_size)
# map_data[0, 0, 0] = 1  # 起点
# map_data[30, 30, 30] = 1  # 终点

# start_point = (0, 0, 0)
# end_point = (30, 30, 30)
# path = find_path(start_point, end_point, map_data)
# if path:
#     print("存在路径:", path)
# else:
#     print("无法找到路径")