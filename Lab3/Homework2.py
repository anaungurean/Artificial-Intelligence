import queue
import time
class CSP:

    def __init__(self, variables, domains, constraints):
        self.variables = variables
        self.domains = domains
        self.constraints = constraints
        self.solution_fc = None
        self.solution_ac = None

    def find_solution_fc(self):
        assignment = {}
        start_time = time.time()
        self.solution_fc = self.backtrack_with_fc(assignment, self.domains)
        end_time = time.time()
        print(f"Timpul pentru metoda Forward Checking: {end_time - start_time} sec")
        return self.solution_fc

    def find_solution_ac(self):
        assignment = {}
        start_time = time.time()
        self.solution_ac = self.backtrack_with_ac(assignment, self.domains)
        end_time = time.time()
        print(f"Timpul pentru metoda Arc Consistency: {end_time - start_time} sec")
        return self.solution_ac

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

    def is_consistent(self, assignment, var, value):
        for constraint_var in self.constraints[var]:
            if constraint_var in assignment and assignment[constraint_var] == value:
                return False
        return True

    def next_unassigned_variables(self, assignment, domains):
        unassigned_vars = []

        for var in self.variables:
            if var not in assignment:
                unassigned_vars.append(var)

        def number_of_possible_values(variable):
            return len(domains[variable])

        variable_with_minimum_possible_values = min(unassigned_vars, key=number_of_possible_values)

        return variable_with_minimum_possible_values

    def is_complete(self, assignment):
        return len(assignment) == len(self.variables)

    def arc_consistency(self, domains):
        q = queue.Queue()
        for X in self.variables:
            for Y in self.constraints[X]:
                q.put((X, Y))

        while not q.empty():
            (X, Y) = q.get()
            ok = True
            for x in domains[X].copy():
                if not any(y for y in domains[Y] if self.is_consistent({}, Y, y)):
                    ok = False
                    domains[X].remove(x)

            if not ok:
                for Z in self.constraints[X]:
                    q.put((Z, X))
        return domains

    def backtrack_with_ac(self, assignment, domains):
        if self.is_complete(assignment):
            return assignment

        var = self.next_unassigned_variables(assignment, domains)

        for value in domains[var].copy():
            if self.is_consistent(assignment, var, value):
                new_assignment = assignment.copy()
                new_assignment[var] = value
                reduced_domains = self.arc_consistency(domains.copy())
                if not any(len(reduced_domains[v]) == 0 for v in reduced_domains):
                    res = self.backtrack_with_ac(new_assignment, reduced_domains)
                    if res is not None:
                        return res
        return None


def find_start_region(x):
    if x in (0,1,2):
        return 0
    elif x in (3,4,5):
        return 3
    else:
        return 6


# sudoku_init = [
#     [0, -1, -1, 5, 0, -1, 0, -1, 0],
#     [7, -1, 0, -1, 0, 0, -1, 3, 8],
#     [9, -1, 3, 7, 4, -1, 5, 0, -1],
#     [0, 0, 0, -1, 7, 4, 0, 2, -1],
#     [-1, 8, 0, 0, 6, 2, 0, 7, 0],
#     [-1, 0, 2, 0, 0, 0, -1, -1, 1],
#     [8, 0, -1, 4, -1, 0, 0, 0, 0],
#     [0, 1, 0, 0, -1, 9, -1, -1, 4],
#     [-1, 0, 4, 6, 1, 0, -1, 5, 0]
# ]

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
        variables.append((i,j))

print(f"Mulțimea de variabile:{variables}")

domains = {}
for i in range(0,9):
    for j in range(0,9):
        if sudoku_init[i][j] == -1:
            domains[(i, j)] = {2, 4, 6, 8}
        elif sudoku_init[i][j] == 0:
            domains[(i, j)] = {1, 2, 3, 4, 5, 6, 7, 8, 9}
        else:
            domains[(i, j)] = {sudoku_init[i][j]}

print(f"Domenii pentru fiecare valoare:{domains}")


def find_start_region(x):
    if x in (0, 1, 2):
        return 0
    elif x in (3, 4, 5):
        return 3
    else:
        return 6


def add_constraint(var):
    constraints[var] = set()
    for j in range(0,9):
        if j != var[1]:
            constraints[var].add((var[0], j))

    for i in range(0,9):
        if i != var[0]:
            constraints[var].add((i,var[1]))

    start_region_i = find_start_region(var[0])
    start_region_j = find_start_region(var[1])
    for i in range(start_region_i, start_region_i+3):
        for j in range(start_region_j, start_region_j+3):
            if (i, j) != var:
                constraints[var].add((i,j))



constraints = {}
for i in range(0, 9):
    for j in range(0, 9):
        add_constraint((i, j))

print(f"Constrangerile pentru fiecare pozitie: {constraints}")
print()

csp_fc = CSP(variables, domains, constraints)
csp_ac = CSP(variables, domains, constraints)
sol_fc = csp_fc.find_solution_fc()
sol_ac = csp_ac.find_solution_ac()


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

if sol_fc is not None:
     print("Tabla de sudoku finală:")
     solution = [[0 for i in range(9)] for j in range(9)]
     for (i, j) in sol_fc:
         solution[i][j] = sol_fc[(i, j)]
     print_sudoku(solution)
else:
    print("No solution found")


if sol_ac is not None:
     print("\nTabla de sudoku finală:")
     solution = [[0 for i in range(9)] for j in range(9)]
     for (i, j) in sol_ac:
         solution[i][j] = sol_ac[(i, j)]
     print_sudoku(solution)
else:
    print("No solution found")





