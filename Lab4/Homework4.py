import numpy as np

class NumberScramble:
    def __init__(self): #initial state
        self.available_numbers = list(range(1,10))
        self.player_moves = {'Player': [], 'AI': []}
        self.current_player = 'Player'
        self.triplets_sum_15 = [
            [2,7,6], [9,5,1], [4,3,8],
            [2,9,4], [7,5,3], [6,1,8],
            [2,5,8], [6,5,4]
        ]

    def display_state_representation(self): #state representation
        print("\nNumere disponibile: " + ' '.join(map(str, self.available_numbers)))
        print("Mutări Player: " + ' '.join(map(str, self.player_moves['Player'])))
        print("Mutări AI: " + ' '.join(map(str, self.player_moves['AI'])))

    def is_winner(self,player):
        moves = self.player_moves
        for triplet in self.triplets_sum_15:
            if all(value in moves[player] for value in triplet):
                return True
        return False

    def is_draw(self):
        return len(self.available_numbers) == 0

    def is_game_over(self): #final states
        return self.is_winner('Player') or self.is_winner('AI') or self.is_draw()

    def get_winner(self):
        if self.is_draw():
            return 'Draw'
        elif self.is_winner('Player'):
            return 'Player'
        return 'AI'

    def validate_move(self, number): #validations
        return number in self.available_numbers

    def make_move(self, number, player): #transitions
        if self.validate_move(number):
            self.player_moves[player].append(number)
            self.available_numbers.remove(number)
            self.current_player = 'AI' if player == 'Player' else 'Player'
            return True
        return False

    def switch_player(self):
        self.current_player = 'AI' if self.current_player == 'Player' else 'Player'

def run_game():
    game = NumberScramble()
    while not game.is_game_over():
        if game.current_player == 'Player':
            game.display_state_representation()
            valid_entry = False
            while not valid_entry:
                try:
                    player_move = int(input("Choose a number (1-9): "))
                    if player_move in range(1,10):
                        valid_entry = True
                    else:
                        print("Invalid number. Please choose a number between 1 and 9.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            if not game.make_move(player_move, 'Player'):
                print("Invalid move, number is not available. Try again.")
                continue
        else:
            ai_move = np.random.choice(game.available_numbers)
            print("\nAI's turn.")
            print("AI chooses: " + str(ai_move))
            game.make_move(ai_move, 'AI')
    game.switch_player()

    winner = game.get_winner()
    if winner == 'Draw':
        print("\nIt's a draw!")
    else:
        print("\nThe winner is: " + winner)



if __name__ == "__main__":
    run_game()












