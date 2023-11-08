import numpy as np

class NumberScramble:
    def __init__(self):
        self.available_numbers = list(range(1,10))
        self.player_moves = {'Player': [], 'AI': []}
        self.current_player = 'Player'
        self.triplets_sum_15 = [
            [2,7,6], [9,5,1], [4,3,8],
            [2,9,4], [7,5,3], [6,1,8],
            [2,5,8], [6,5,4]
        ]

    def display_state_representation(self):
        print("\nAvailable numbers: " + ' '.join(map(str, self.available_numbers)))
        print("Player's moves: " + ' '.join(map(str, self.player_moves['Player'])))
        print("AI moves: " + ' '.join(map(str, self.player_moves['AI'])))

    def is_winner(self, player):
        moves = self.player_moves
        for triplet in self.triplets_sum_15:
            if all(value in moves[player] for value in triplet):
                return True
        return False

    def is_draw(self):
        return len(self.available_numbers) == 0

    def is_game_over(self):
        return self.is_winner('Player') or self.is_winner('AI') or self.is_draw()

    def get_winner(self):
        if self.is_draw():
            return 'Draw'
        elif self.is_winner('Player'):
            return 'Player'
        return 'AI'

    def validate_move(self, number):
        return number in self.available_numbers

    def make_move(self, number, player):
        if self.validate_move(number):
            self.player_moves[player].append(number)
            self.available_numbers.remove(number)
            self.switch_player()
            return True
        return False

    def switch_player(self):
        self.current_player = 'AI' if self.current_player == 'Player' else 'Player'

    def heuristic(self):
        ai_score = 0
        player_score = 0
        for triplet in self.triplets_sum_15:
            if all(num in self.available_numbers or num in self.player_moves['AI'] for num in triplet):
                ai_score += 1
            if all(num in self.available_numbers or num in self.player_moves['Player'] for num in triplet):
                player_score += 1
        if self.current_player == 'AI':
            return ai_score - player_score
        else:
            return player_score-ai_score

    def minmax(self, depth, is_max_player):
        if self.is_game_over():
            if self.is_winner('AI'):
                return 10
            elif self.is_winner('Player'):
                return -10
            else:
                return 0

        if depth == 0:
            return self.heuristic()

        if is_max_player:
            best_score = -np.inf
            for number in self.available_numbers:
                self.make_move(number,'AI')
                score = self.minmax(depth - 1, False)
                self.undo_move(number, 'AI')
                best_score = max(best_score, score)
            return best_score
        else:
            best_score = np.inf
            for number in self.available_numbers:
                self.make_move(number, 'Player')
                score = self.minmax(depth - 1, True)
                self.undo_move(number, 'Player')
                best_score = min(best_score, score)
            return best_score

    def undo_move(self, number, player):
        self.player_moves[player].remove(number)
        self.available_numbers.append(number)
        self.available_numbers.sort()

    def find_best_move_for_AI(self):
        best_score = -np.inf
        best_move = None
        for number in self.available_numbers:
            self.make_move(number, 'AI')
            score = self.minmax(6,False)
            self.undo_move(number,'AI')
            if score >= best_score:
                best_score = score
                best_move = number
        return best_move




def run_game():
    game = NumberScramble()
    while not game.is_game_over():
        if game.current_player == 'Player':
            game.display_state_representation()
            valid_entry = False
            while not valid_entry:
                try:
                    player_move = int(input("Choose a number (1-9): "))
                    if player_move in range(1, 10):
                        valid_entry = True
                    else:
                        print("Invalid number. Please choose a number between 1 and 9.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            if not game.make_move(player_move, 'Player'):
                print("Invalid move, number is not available. Try again.")
                continue
        else:
            ai_move = game.find_best_move_for_AI()
            if ai_move is not None:
                print("\nAI's turn.")
                print("AI chooses: " + str(ai_move))
                game.make_move(ai_move, 'AI')
            else:
                print("AI passes its turn (no valid moves).")
    game.switch_player()

    winner = game.get_winner()
    if winner == 'Draw':
        print("\nIt's a draw!")
    else:
        print("\nThe winner is: " + winner)


if __name__ == "__main__":
    run_game()