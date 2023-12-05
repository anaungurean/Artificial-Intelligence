import random
import matplotlib.pyplot as plt
import numpy as np

class QLearningAgent:
    def __init__(self, num_rows, num_cols, initial_state, goal_state, wind, learning_rate, discount_factor, num_episodes ): #a
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.wind = wind
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.num_episodes = num_episodes
        self.q_table = np.zeros((self.num_rows, self.num_cols, 4))

    def take_action(self, state, action):
        action_deltas = [(1, 0), (-1,0), (0,-1), (0,1)]  # jos, sus, stanga, dreapta
        delta = action_deltas[action]

        wind_effect = self.wind[state[1]] if 0 <= state[1] < len(self.wind) else 0
        delta = (delta[0] - wind_effect, delta[1])

        next_state = (max(0, min(state[0] + delta[0], self.num_rows - 1)),
                      max(0, min(state[1] + delta[1], self.num_cols - 1)))

        return next_state

    def q_learning(self, epsilon=0.1):

        for episode in range(self.num_episodes):
            current_state = self.initial_state

            while current_state != self.goal_state:

                if random.uniform(0, 1) < epsilon:
                    action = random.choice([0, 1, 2, 3])
                else:
                    action = np.argmax(self.q_table[current_state])

                next_state = self.take_action(current_state, action)
                reward = -1
                if next_state == self.goal_state:
                    reward = 999_999.999

                best_next_action = np.argmax(self.q_table[next_state])
                self.q_table[current_state][action] += self.learning_rate * (
                        reward + self.discount_factor * self.q_table[next_state][best_next_action]
                        - self.q_table[current_state][action])

                current_state = next_state


        current_state = self.initial_state

    def get_optimal_path(self):
        current_state = self.initial_state
        path = [current_state]

        while current_state != self.goal_state:
            action = np.argmax(self.q_table[current_state])
            next_state = self.take_action(current_state, action)
            path.append(next_state)
            current_state = next_state

        return path

    def display_policy(self):
        print("Determined Policy:")
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                state = (row, col)
                action = np.argmax(self.q_table[state])
                if state == self.goal_state:
                    print(" G ", end="")
                elif state == self.initial_state:
                    print(" I ", end="")
                else:
                    if action == 0:
                        print(" ↓ ", end="")
                    elif action == 1:
                        print(" ↑ ", end="")
                    elif action == 2:
                        print(" ← ", end="")
                    elif action == 3:
                        print(" → ", end="")
            print()

    def display_q_values(self):
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                state = (row, col)
                q_values = self.q_table[state]
                print(f"State: {state}, Q-values: {q_values}")


if __name__ == '__main__':

    num_rows = 7
    num_cols = 10
    initial_state = (3, 0)
    goal_state = (3, 7)
    wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    learning_rate = 0.1
    discount_factor = 0.9
    num_episodes = 200

    agent = QLearningAgent(num_rows, num_cols, initial_state, goal_state, wind, learning_rate, discount_factor, num_episodes)
    actions = [0, 1, 2, 3]  # 0: jos, 1: sus, 2: stanga, 3: dreapta
    agent.q_learning()
    agent.display_policy()



    # for action in actions:
    #     next_state = agent.take_action(initial_state, action)
    #     print(f"Action: {action}, Next State: {next_state}")
    # optimal_path = agent.get_optimal_path()
    # print("Optimal Path:")
    # for state in optimal_path:
    #     print(state)
    # agent.display_q_values()


