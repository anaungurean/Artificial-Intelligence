import random

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

        # Inițializarea tabelului Q cu valori de 0
        self.q_table = np.zeros((self.num_rows, self.num_cols, 4)) # 4 = numărul de acțiuni posibile

    def take_action(self, state, action):
        action_deltas = [(1, 0), (-1,0), (0,-1), (0,1)]  # jos, sus, stanga, dreapta
        delta = action_deltas[action]  # aplicăm modificarea pentru acțiunea dată

        wind_effect = self.wind[state[1]] if 0 <= state[1] < len(self.wind) else 0
        delta = (delta[0] - wind_effect, delta[1])

        # Asigurăm că agentul nu iese din grid
        next_state = (max(0, min(state[0] + delta[0], self.num_rows - 1)),
                      max(0, min(state[1] + delta[1], self.num_cols - 1)))

        return next_state

    def q_learning(self, epsilon=0.1):
        for episode in range(self.num_episodes):
            current_state = self.initial_state
            total_reward = 0  # Initialize total_reward for the current episode

            while current_state != self.goal_state:
                # Epsilon-greedy strategy for action selection
                if random.uniform(0, 1) < epsilon:
                    action = random.choice([0, 1, 2, 3])  # Explore by choosing a random action
                else:
                    action = np.argmax(self.q_table[current_state])  # Exploit by choosing the best action

                next_state = self.take_action(current_state, action)
                reward = -1
                if next_state == self.goal_state:
                    reward = 100  # Use a large but finite reward for reaching the goal

                best_next_action = np.argmax(self.q_table[next_state])
                self.q_table[current_state][action] += self.learning_rate * (
                        reward + self.discount_factor * self.q_table[next_state][best_next_action]
                        - self.q_table[current_state][action])

                total_reward += reward  # Accumulate the reward for the current episode
                current_state = next_state

            # Print total reward obtained in each episode
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")

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


if __name__ == '__main__':

    num_rows = 7
    num_cols = 10
    initial_state = (3, 0)
    goal_state = (3, 7)
    wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    learning_rate = 0.1
    discount_factor = 0.9
    num_episodes = 200

    # Inițializarea agentului
    agent = QLearningAgent(num_rows, num_cols, initial_state, goal_state, wind, learning_rate, discount_factor, num_episodes)

    # Testarea funcției take_action
    actions = [0, 1, 2, 3]  # 0: jos, 1: sus, 2: stanga, 3: dreapta
    agent.q_learning()

    for action in actions:
        next_state = agent.take_action(initial_state, action)
        print(f"Action: {action}, Next State: {next_state}")

    # Obținerea și afișarea drumului optim
    optimal_path = agent.get_optimal_path()
    print("Optimal Path:")
    for state in optimal_path:
        print(state)

