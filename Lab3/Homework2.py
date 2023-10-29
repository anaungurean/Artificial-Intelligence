class CSP:

    def __init__(self, variables, domains, constraints):
        self.variables = variables
        self.domains = domains
        self.constraints = constraints
        self.solution = None

sudoku_init = [
    [8, 4, 0, 0, 5, 0, -1, 0, 0],
    [3, 0, 0, 6, 0, 8, 0, 4, 0],
    [0, 0, -1, 4, 0, 9, 0, 0, -1],
    [0, 2, 3, 0, -1, 0, 9, 8, 0],
    [1, 0, 0, -1, 0, -1, 0, 0, 4],
    [0, 9, 8, 0, -1, 0, 1, 6, 0],
    [-1, 0, 0, 5, 0, 3, -1, 0, 0],
    [0, 3, 0, 1, 0, 6, 0, 0, 7],
    [0, 0, 0, 0, 2, 0, 0, 1, 3]
]

variables = []
for i in range(0,9):
    for j in range(0,9):
        variables.append((i,j)) #toate pozitiile din sudoku

print(f"Mulțimea de variabile:{variables}")

domains = {} #dictionar cu cheie: (i,j) si valoarea : domeniul de valori pe care-l poate lua
for i in range(0,9):
    for j in range(0,9):
        if sudoku_init[i][j] == -1: #trebuie nr par
            domains[(i, j)] = {2, 4, 6, 8}
        elif sudoku_init[i][j] == 0:
            domains[(i, j)] = {1, 2, 3, 4, 5, 6, 7, 8, 9}
        else:
            domains[(i, j)] = {sudoku_init[i][j]} #are deja valoare

print(f"Domenii pentru fiecare valoare:{domains}")


def find_start_region(x):
    if x in (0,1,2):
        return 0
    elif x in (3,4,5):
        return 3
    else:
        return 6


def add_constraint(var):
    constraints[var] = set() #evitam duplicatele
    for j in range(0,9): #constrangere pe linie
        if j != var[1]: #sa nu fie aceeasi pozitie
            constraints[var].add((var[0], j))

    for i in range(0,9):
        if i != var[0]:
            constraints[var].add((i,var[1]))

    start_region_i = find_start_region(var[0]) #constrangeri pt regiuni
    start_region_j = find_start_region(var[1])
    for i in (start_region_i, start_region_i+2, 1):
        for j in (start_region_j, start_region_j+2, 1):
            if (i, j) != var:
                constraints[var].add((i,j))


constraints = {} #dictionar cu cheie: (i,j) si valoarea : lista de constangeri, pozitile cu care nu poate fi egal
for i in range(0, 9):
    for j in range(0, 9):
        add_constraint((i, j)) #20 de fiecare ar trebui

print(f"Constrangerile pentru fiecare pozitie: {constraints}")

csp = CSP(variables, domains, constraints) #initializare 







