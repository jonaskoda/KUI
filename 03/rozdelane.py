import copy
import math
from game_board import GameBoard
class MyPlayer():
    '''Template Docstring for MyPlayer, look at the TODOs''' # TODO a short description of your player

    def __init__(self, my_color,opponent_color, board_size=8):
        self.name = 'username' #TODO: fill in your username
        self.my_color = my_color
        self.opponent_color = opponent_color
        self.board_size = board_size
        self.p1_color = 0
        self.p2_color = 1
        self.empty_color = -1

    def move(self,board):
     
        choose_list = []
        best_value = 0
        best_move = None

        possible_moves = self.get_all_valid_moves(board)
        for coords in possible_moves:
            board_copy = copy.deepcopy(board)
            move_value = self.minimax(board_copy, 1, False)
        choose_list.append([move_value, coords])
        
        for i in choose_list:
            if i[0] > best_value:
                best_value = i[0]
                best_move = i[1]
        
        return best_move

    def __is_correct_move(self, move, board):
        dx = [-1, -1, -1, 0, 1, 1, 1, 0]
        dy = [-1, 0, 1, 1, 1, 0, -1, -1]
        for i in range(len(dx)):
            if self.__confirm_direction(move, dx[i], dy[i], board)[0]:
                return True, 
        return False

    def __confirm_direction(self, move, dx, dy, board):
        posx = move[0]+dx
        posy = move[1]+dy
        opp_stones_inverted = 0
        if (posx >= 0) and (posx < self.board_size) and (posy >= 0) and (posy < self.board_size):
            if board[posx][posy] == self.opponent_color:
                opp_stones_inverted += 1
                while (posx >= 0) and (posx <= (self.board_size-1)) and (posy >= 0) and (posy <= (self.board_size-1)):
                    posx += dx
                    posy += dy
                    if (posx >= 0) and (posx < self.board_size) and (posy >= 0) and (posy < self.board_size):
                        if board[posx][posy] == -1:
                            return False, 0
                        if board[posx][posy] == self.my_color:
                            return True, opp_stones_inverted
                    opp_stones_inverted += 1

        return False, 0

    def get_all_valid_moves(self, board):
        valid_moves = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                if (board[x][y] == -1) and self.__is_correct_move([x, y], board):
                    valid_moves.append( (x, y) )

        if len(valid_moves) <= 0:
            print('No possible move!')
            return None
        return valid_moves
    
    def minimax(self, board, depth, maximizing):
        if (depth == 0 or self.get_all_valid_moves(board) == None):
            score = GameBoard().get_score()
            if maximizing:
                if self.my_color == self.p1_color:
                    score = score[0]
                else:
                    score = score[1] 
            else: # minimizing
                if self.my_color == self.p1_color:
                    score = score[1]
                else:
                    score = score[0] 
            return score
        
        if maximizing:
            max_alpha = -math.inf
            list_of_moves = self.get_all_valid_moves(board)
            for i in list_of_moves:
                board_copy = copy.deepcopy(board)
                updated_board = self.play_move(i, self.my_color, board_copy)
                eval = self.minimax(updated_board, depth-1, False)
                max_alpha = max(max_alpha, eval)            # function for picking the larger value
            return max_alpha

        else: #minimizing
            min_beta = +math.inf
            list_of_moves = self.get_all_valid_moves(board)  #have to add that I want valid moves for opponent
            for i in list_of_moves:
                board_copy = copy.deepcopy(board)
                updated_board = self.play_move(i, self.opponent_color, board_copy)
                eval = self.minimax(updated_board, depth-1, True)
                min_beta = min(min_beta, eval)              # function for picking the smaller value
            return min_beta
