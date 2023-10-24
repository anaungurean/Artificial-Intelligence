import copy
import time
import queue

class Puzzle:
    def __init__(self, matrix): #subpunctul1
        self.matrix = matrix
        self.last_moved = None  

    def __lt__(self, other):
        return True

    def get_empty_position(self):
        for i, row in enumerate(self.matrix):
            if 0 in row:
                return i, row.index(0)
        return None

    def get_neighbors(self, x, y):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                neighbors.append((nx, ny))

        return neighbors

    def print_puzzle(self):
        for row in self.matrix:
            row_representation = []

            for item in row:
                if item == 0:
                    row_representation.append('_')
                else:
                    row_representation.append(str(item))

            row_string = ' '.join(row_representation)

            print(row_string)

    def print_last_moved(self):
        print(self.last_moved)

    def is_final(self):
        expected_values = list(range(1, 9))
        flat_matrix = []

        for row in self.matrix:
            for item in row:
                if item != 0:
                    flat_matrix.append(item)

        return flat_matrix == expected_values

    def can_move(self, x, y, direction): #subpunctul3
        if (x, y) == self.last_moved:
            return False

        dx, dy = 0, 0
        if direction == "up":
            dx, dy = -1, 0
        elif direction == "down":
            dx, dy = 1, 0
        elif direction == "left":
            dx, dy = 0, -1
        elif direction == "right":
            dx, dy = 0, 1

        if not (0 <= x + dx < 3 and 0 <= y + dy < 3):
            return False

        if self.matrix[x + dx][y + dy] != 0:
            return False

        return True

    def move(self, x, y, direction): #subpunctul3
        if not self.can_move(x, y, direction):
            return None

        dx, dy = 0, 0
        if direction == "up":
            dx, dy = -1, 0
        elif direction == "down":
            dx, dy = 1, 0
        elif direction == "left":
            dx, dy = 0, -1
        elif direction == "right":
            dx, dy = 0, 1

        self.matrix[x + dx][y + dy], self.matrix[x][y] = self.matrix[x][y], self.matrix[x + dx][y + dy]

        self.last_moved = (x + dx, y + dy)
        return self

def initialize_puzzle(instance): #subpunctul2
    return Puzzle(instance)

def is_final(state):
    return state.is_final()

def depth_limited_DFS(state, depth, visited):
    if is_final(state):
        return state

    if depth == 0:
        return None

    visited.add(str(state.matrix))
    empty_x, empty_y = state.get_empty_position()
    neighbors_positions = state.get_neighbors(empty_x, empty_y)

    for nx, ny in neighbors_positions:
        if state.can_move(nx, ny, get_direction_from_positions(nx, ny, empty_x, empty_y)):
            new_puzzle = copy.deepcopy(state)
            new_puzzle.move(nx, ny, get_direction_from_positions(nx, ny, empty_x, empty_y))

            if str(new_puzzle.matrix) not in visited:
                res = depth_limited_DFS(new_puzzle, depth - 1, visited)
                if res is not None:
                    return res

    return None


def get_direction_from_positions(ex, ey, nx, ny): #subpunctul4
    if nx == ex - 1 and ny == ey:
        return "up"
    if nx == ex + 1 and ny == ey:
        return "down"
    if nx == ex and ny == ey - 1:
        return "left"
    if nx == ex and ny == ey + 1:
        return "right"
    return None

def IDDFS(init_state, max_depth):

    for depth in range(max_depth + 1):
        visited = set()
        result = depth_limited_DFS(init_state, depth, visited)

        if result is not None:
            print("Soluție găsită:")
            result.print_puzzle()
            print(f"Numărul total de mutări:{depth}")
            return 0

    print("Nicio soluție găsită.")
    return None


def greedy(puzzle, heuristic_name):

    puzzle.heuristic_value = average_heuristic_value(puzzle, heuristic_name)
    pq = queue.PriorityQueue()
    initial_depth = 0
    pq.put((puzzle.heuristic_value, initial_depth, puzzle))

    visited = set()
    visited.add(str(puzzle.matrix))

    while not pq.empty():
        avg, depth, puzzle = pq.get()

        if is_final(puzzle):
            print("Soluție găsită:")
            puzzle.print_puzzle()
            print(f"Numărul total de mutări: {depth}")
            return 1

        empty_x, empty_y = puzzle.get_empty_position()
        neighbors_positions = puzzle.get_neighbors(empty_x, empty_y)

        for nx, ny in neighbors_positions:
            if puzzle.can_move(nx, ny, get_direction_from_positions(nx, ny, empty_x, empty_y)):
                new_puzzle = copy.deepcopy(puzzle)
                new_puzzle.move(nx, ny, get_direction_from_positions(nx, ny, empty_x, empty_y))

                if str(new_puzzle.matrix) not in visited:
                    new_puzzle.heuristic_value = average_heuristic_value(new_puzzle, heuristic_name)
                    pq.put((new_puzzle.heuristic_value, depth + 1, new_puzzle))
                    visited.add(str(new_puzzle.matrix))

    print("Nicio soluție găsită.")
    return 0



def get_manhattan_distance(p, q):

    distance = 0
    for p_i, q_i in zip(p, q):
        distance += abs(p_i - q_i)

    return distance


def get_hamming_distance(p, q):

    distance = 0
    for p_i, q_i in zip(p, q):
        if p_i != q_i:
            distance += 1

    return distance


def get_chebyshev_distance(p, q):

    distance = 0
    for p_i, q_i in zip(p, q):
        distance = max(distance, abs(p_i - q_i))

    return distance


def average_heuristic_value(puzzle, heuristic_name):
    sume = 0
    k = 0
    empty_x, empty_y = puzzle.get_empty_position()
    neighbors_positions = puzzle.get_neighbors(empty_x, empty_y)
    for nx, ny in neighbors_positions:
        if puzzle.can_move(nx, ny, get_direction_from_positions(nx, ny, empty_x, empty_y)):
            k += 1
            if heuristic_name == "manhattan":
                sume += get_manhattan_distance((nx,ny),(empty_x,empty_y))
            elif heuristic_name == "hamming":
                sume += get_hamming_distance((nx,ny),(empty_x,empty_y))
            elif heuristic_name == "chebyshev":
                sume += get_chebyshev_distance((nx, ny), (empty_x, empty_y))
            else:
                return -1
    return sume/k



def testare():
    instance_1 = [[8, 6, 7], [2, 5, 4], [0, 3, 1]]
    instance_2 = [[2, 5, 3], [1, 0, 6], [4, 7, 8]]
    instance_3 =[[2, 7, 5], [0, 8, 4], [3, 1, 6]]
    for i in range(1,4):

        print(f"____________Instanța{i}______________" )
        if i == 1:
            initial_state = instance_1
        elif i == 2:
            initial_state = instance_2
        else:
            initial_state = instance_3

        puzzle = initialize_puzzle(initial_state)

        print(f"Pentru instanta:{initial_state}")
        print()


        print("IDDFS:")
        start_time = time.time()
        IDDFS(puzzle,50)
        end_time = time.time()
        print(f"Timpul de execuție: {round(end_time - start_time,2)} secunde")
        print()

        print("Greedy cu euristica Manhattan: ")
        start_time = time.time()
        greedy(puzzle,"manhattan")
        end_time = time.time()
        print(f"Timpul de execuție: {round(end_time - start_time,2)} secunde")
        print()


        print("Greedy cu euristica Hamming: ")
        start_time = time.time()
        greedy(puzzle, "hamming")
        end_time = time.time()
        print(f"Timpul de execuție: {round(end_time - start_time,2)} secunde")
        print()


        print("Greedy cu euristica Chebyshev: ")
        start_time = time.time()
        greedy(puzzle, "chebyshev")
        end_time = time.time()
        print(f"Timpul de execuție: {round(end_time - start_time,2)} secunde")
        print()




testare()




'''
instance = [[8, 6, 0], [5, 4, 7], [2, 3, 1]]
puzzle = initialize_puzzle(instance)
puzzle.print_puzzle()

# 2. Testing moving a piece
print("\nMoving 6 right:")
puzzle.move(0, 1, "right")
puzzle.print_puzzle()

# 3. Testing moving the same piece again
print("\nMoving 6 left:")
puzzle.move(0, 2, "left")
puzzle.print_puzzle()

# 3. Testing moving the same piece outside the grid
print("\nMoving 6 up:")
puzzle.move(0, 2, "up")
puzzle.print_puzzle()

# 4. Testing moving the same piece with a not empty space
print("\nMoving 6 down:")
puzzle.move(0, 2, "down")
puzzle.print_puzzle()


instance1 = [[1, 2, 0], [5, 4, 5], [6, 7, 8]]
puzzle = initialize_puzzle(instance1)
print()
puzzle.print_puzzle()
print(f"Is this a final state : {puzzle.is_final()}")

instance1 = [[1, 2, 0], [3, 5, 4], [6, 7, 8]]
puzzle = initialize_puzzle(instance1)
print()
puzzle.print_puzzle()
print(f"Is this a final state : {puzzle.is_final()}")

print()
"""
initial_state = [
    [8, 6, 7],
    [2, 5, 4],
    [0, 3, 1]
]

initial_state = [
    [2, 5, 3],
    [1, 0, 6],
    [4, 7, 8]
]
"""

initial_state = [
    [2, 7, 5],
    [0, 8, 4],
    [3, 1, 6]
]

puzzle = initialize_puzzle(initial_state)

solution = IDDFS(puzzle, 50)

if solution:
    print("Soluție găsită:")
    solution.print_puzzle()
else:
    print("Nu s-a găsit nicio soluție în adâncimea specificată.")
'''
















