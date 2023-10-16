import copy
class Puzzle:
    def __init__(self, matrix): #subpunctul1
        self.matrix = matrix
        self.last_moved = None  # Poziția ultimei celule mutate

    def get_empty_position(self):
        for i, row in enumerate(self.matrix):
            if 0 in row:
                return i, row.index(0)
        return None

    def get_neighbors(self, x, y):
        # direcțiile de mișcare: sus, jos, stânga, dreapta
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        neighbors = [(x+dx, y+dy) for dx, dy in directions if 0 <= x+dx < 3 and 0 <= y+dy < 3]
        return neighbors

    def print_puzzle(self):
        for row in self.matrix:
            print(' '.join(['_' if x == 0 else str(x) for x in row]))

    def print_last_moved(self):
        print(self.last_moved)

    def is_final(self): #subpunctul2
        flat_matrix = [item for row in self.matrix for item in row if item != 0]  # ignorăm celula goală
        return flat_matrix == [1, 2, 3, 4, 5, 6, 7, 8]

    def can_move(self, x, y, direction): #subpunctul3
        if (x, y) == self.last_moved: #După mutarea unei celule, ea nu mai poate fi mutată din nou decât după ce unul din vecinii săi a fost mutat
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

        # Verificăm dacă mișcarea este în interiorul grilei
        if not (0 <= x + dx < 3 and 0 <= y + dy < 3):
            return False

        # Verificăm dacă celula în care se face mișcarea este goală
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

        # Facem schimbul între celula selectată și celula goală
        self.matrix[x + dx][y + dy], self.matrix[x][y] = self.matrix[x][y], self.matrix[x + dx][y + dy]

        # Actualizăm celula care a fost mișcată ultima dată
        self.last_moved = (x + dx, y + dy)
        return self

    def get_valid_neighbors(self, x, y):
        directions = ["up", "down", "left", "right"]
        valid_moves = []
        for direction in directions:
            if self.can_move(x, y, direction):
                new_puzzle = Puzzle([row.copy() for row in self.matrix])
                new_puzzle.move(x, y, direction)
                valid_moves.append(new_puzzle)
        return valid_moves
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
    for neighbor in state.get_valid_neighbors(empty_x, empty_y):
        if str(neighbor.matrix) not in visited:
            res = depth_limited_DFS(neighbor, depth-1, visited)
            if res is not None:
                return res
    return None

def IDDFS(init_state, max_depth):
    for depth in range(max_depth + 1):
        visited = set()
        sol = depth_limited_DFS(init_state, depth, visited)
        if sol is not None:
            return sol
    return None


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


instance1 = [[1, 2, 0], [3, 4, 5], [6, 7, 8]]
puzzle = initialize_puzzle(instance1)
print()
puzzle.print_puzzle()
print(f"Is this a final state : {puzzle.is_final()}")

instance1 = [[1, 2, 0], [3, 5, 4], [6, 7, 8]]
puzzle = initialize_puzzle(instance1)
print()
puzzle.print_puzzle()
print(f"Is this a final state : {puzzle.is_final()}")


"""
initial_instance = [[8, 6, 0], [5, 4, 7], [2, 3, 1]]
puzzle = initialize_puzzle(initial_instance)
solution = IDDFS(puzzle, 5)
if solution:
    solution.print_puzzle()
else:
    print("No solution found within the given depth.")
"""
















