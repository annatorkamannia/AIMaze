import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Directions for moving in the maze (up, down, left, right)
directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

# Function to generate a random maze using Depth-First Search (DFS)
def generate_maze(width, height):
    maze = np.ones((height, width), dtype=int)
    start = (random.randint(0, height - 1), random.randint(0, width - 1))
    stack = [start]
    maze[start[0], start[1]] = 0

    while stack:
        current_cell = stack[-1]
        x, y = current_cell
        neighbors = []
        for direction in directions:
            nx, ny = x + direction[0] * 2, y + direction[1] * 2
            if 0 <= nx < height and 0 <= ny < width and maze[nx, ny] == 1:
                neighbors.append((nx, ny))

        if neighbors:
            next_cell = random.choice(neighbors)
            stack.append(next_cell)
            mx, my = (x + next_cell[0]) // 2, (y + next_cell[1]) // 2
            maze[mx, my] = 0
            maze[next_cell[0], next_cell[1]] = 0
        else:
            stack.pop()

    return maze

# BFS Solver with Learning Rate Simulation
def bfs_maze_solver(maze, start, end):
    rows, cols = len(maze), len(maze[0])
    queue = deque([(start, [start])])
    visited = set()
    visited.add(start)

    exploration_steps = 0  # Track exploration steps
    learning_rates = []    # To store learning rate data

    while queue:
        current_pos, path = queue.popleft()
        x, y = current_pos
        exploration_steps += 1

        # Log the learning rate progress as 1/steps (simulating learning rate)
        learning_rate = 1 / exploration_steps
        learning_rates.append(learning_rate)

        if (x, y) == end:
            return path, learning_rates  # Return solution path and learning rate data

        for direction in directions:
            nx, ny = x + direction[0], y + direction[1]
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 0 and (nx, ny) not in visited:
                queue.append(((nx, ny), path + [(nx, ny)]))
                visited.add((nx, ny))

    return None, learning_rates  # No solution found, but return learning rates

# Function to plot maze, solution, and learning rate
def plot_solution_and_learning_rate(maze, path, start, end, learning_rates):
    # Create the figure for maze and learning rate
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the maze and the solution path
    maze_array = np.array(maze)
    ax1.imshow(maze_array, cmap='binary', origin='upper')
    
    if path:
        path = np.array(path)
        ax1.plot(path[:, 1], path[:, 0], color='red', linewidth=2, label="Solution Path")
    
    ax1.scatter(start[1], start[0], color='green', s=100, label="Start")  # Start point
    ax1.scatter(end[1], end[0], color='blue', s=100, label="End")  # End point
    ax1.set_xticks(np.arange(-0.5, len(maze[0]), 1))
    ax1.set_yticks(np.arange(-0.5, len(maze), 1))
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.grid(True)
    ax1.legend(loc="upper right")
    ax1.set_title("Maze Solution")

    # Plot the learning rate over steps
    ax2.plot(learning_rates, color='purple')
    ax2.set_title("Learning Rate Over Time (Steps)")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Learning Rate (1/Steps)")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# Generate a random maze
width, height = 21, 21
generated_maze = generate_maze(width, height)

# Define start and end points
start = (0, 0)
end = (height - 1, width - 1)

# Ensure start and end points are not walls
if generated_maze[start[0], start[1]] == 1 or generated_maze[end[0], end[1]] == 1:
    print("Start or End point is in a wall. Please regenerate the maze.")
else:
    # Solve the maze and track learning rate
    solution, learning_rates = bfs_maze_solver(generated_maze, start, end)
    
    if solution is None:
        print("No solution found for this maze.")
    else:
        # Plot the maze solution and learning rate plot
        plot_solution_and_learning_rate(generated_maze, solution, start, end, learning_rates)