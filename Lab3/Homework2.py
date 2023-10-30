class CSP:

    def __init__(self, variables, domains, constraints):
        self.variables = variables
        self.domains = domains
        self.constraints = constraints
        self.solution = None

    def find_solution(self):
        assignment = {}
        self.solution = self.backtrack_with_fc(assignment, self.domains)
        return self.solution

    def backtrack_with_fc(self, assignment, domains):
        if self.is_complete(assignment):
            return assignment

        var = self.next_unassigned_variables(assignment, domains)
        for value in domains[var].copy():
            if self.is_consistent(assignment, var, value):
                new_assignment = assignment.copy()
                new_assignment[var] = value
                new_domains = self.update_domains_FC(domains.copy(), var, value)
                if not any(len(new_domains[v]) == 0 for v in new_domains):
                    res = self.backtrack_with_fc(new_assignment, new_domains)
                    if res is not None:
                        return res
        return None

    def update_domains_FC(self, domains, var, value):
        new_domains = domains

        for j in range(9):
            if j != var[1] and value in new_domains[(var[0], j)]:
                new_domains[(var[0], j)].remove(value)

        for i in range(9):
            if i != var[0] and value in new_domains[(i, var[1])]:
                new_domains[(i, var[1])].remove(value)

        start_region_i, start_region_j = find_start_region(var[0]), find_start_region(var[1])

        for i in range(start_region_i, start_region_i + 3):
            for j in range(start_region_j, start_region_j + 3):
                if (i, j) != var and value in new_domains[(i, j)]:
                    new_domains[(i, j)].remove(value)

        return new_domains

    def is_consistent(self, assignment, var, value): #respecta toate constrangerile
        for constraint in self.constraints[var]:
            if constraint in assignment and assignment[constraint] == value:
                return False
        return True

    def next_unassigned_variables(self, assignment, domains):
        unassigned_vars = [var for var in self.variables if var not in assignment]
        return min(unassigned_vars, key=lambda var: len(domains[var])) #sper ca aici trb MRV aplicat

    def is_complete(self, assignment): #fiecare variabila are asignata o valoare, adica am terminat cautarea
        return len(assignment) == len(self.variables)


def find_start_region(x):
    if x in (0,1,2):
        return 0
    elif x in (3,4,5):
        return 3
    else:
        return 6



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
    constraints[var] = set() #evităm duplicatele
    for j in range(0,9): #constrângere pe linie
        if j != var[1]: #să nu fie aceeași poziție
            constraints[var].add((var[0], j))

    for i in range(0,9):
        if i != var[0]:
            constraints[var].add((i,var[1]))

    start_region_i = find_start_region(var[0]) #constrângeri pt regiuni
    start_region_j = find_start_region(var[1])
    for i in (start_region_i, start_region_i+2):
        for j in (start_region_j, start_region_j+2):
            if (i, j) != var:
                constraints[var].add((i,j))



constraints = {} #dictionar cu cheie: (i,j) si valoarea : lista de constangeri, pozitile cu care nu poate fi egal
for i in range(0, 9):
    for j in range(0, 9):
        add_constraint((i, j)) #20 de fiecare ar trebui

print(f"Constrangerile pentru fiecare pozitie: {constraints}")
print()

csp = CSP(variables, domains, constraints)
sol = csp.find_solution()


def print_sudoku(table):
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("- " * 11)
        for j in range(9):
            if j % 3 == 0 and j != 0:
                print("|", end=" ")
            print(table[i][j], end=" ")
        print()


print("Tabla de sudoku inițială:")
print_sudoku(sudoku_init)
print()

if sol is not None:
     print("Tabla de sudoku finală:")
     # print(sol) avem doar asignarile trebuie sa le punem in forma de matrice
     solution = [[0 for i in range(9)] for j in range(9)]
     for (i, j) in sol:
         solution[i][j] = sol[(i, j)]
     print_sudoku(solution)
else:
    print("No solution found")





