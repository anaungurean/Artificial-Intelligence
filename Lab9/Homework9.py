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

        wind_effect = self.wind[state[0]] if 0 <= state[0] < len(self.wind) else 0
        delta = (delta[0] - wind_effect, delta[1])

        # Asigurăm că agentul nu iese din grid
        next_state = (max(0, min(state[0] + delta[0], self.num_rows - 1)),
                      max(0, min(state[1] + delta[1], self.num_cols - 1)))

        return next_state

    def q_learning(self):
        for episode in range(self.num_episodes):
            current_state = self.initial_state

            while current_state != self.goal_state:
                action = np.argmax(self.q_table[current_state])
                next_state = self.take_action(current_state, action)

                reward = -1
                if next_state == self.goal_state:
                    reward = float('inf')

                best_next_action = np.argmax(self.q_table[next_state])
                self.q_table[current_state][action] += self.learning_rate * (
                        reward + self.discount_factor * self.q_table[next_state][best_next_action]
                        - self.q_table[current_state][action])
                current_state = next_state
            current_state = self.initial_state

    def get_optimal_policy(self):
        optimal_policy = np.zeros((self.num_rows, self.num_cols), dtype=int)

        for row in range(self.num_rows):
            for col in range(self.num_cols):
                optimal_policy[row, col] = np.argmax(self.q_table[row, col, :])

        return optimal_policy

if __name__ == '__main__':

    num_rows = 7
    num_cols = 10
    initial_state = (3, 0)
    goal_state = (3, 7)
    wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    learning_rate = 0.1
    discount_factor = 0.9
    num_episodes = 1000

    # Inițializarea agentului
    agent = QLearningAgent(num_rows, num_cols, initial_state, goal_state, wind, learning_rate, discount_factor, num_episodes)

    # Testarea funcției take_action
    actions = [0, 1, 2, 3]  # 0: jos, 1: sus, 2: stanga, 3: dreapta

    for action in actions:
        next_state = agent.take_action(initial_state, action)
        print(f"Action: {action}, Next State: {next_state}")

    # Training the Q-learning agent
    agent.q_learning()
    optimal_policy = agent.get_optimal_policy()
    print("Optimal Policy:")
    print(optimal_policy)



