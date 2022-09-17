import copy
import math
from game_board import GameBoard

class MyPlayer():
    '''Template Docstring for MyPlayer, look at the TODOs''' # TODO a short description of your player

    def __init__(self, my_color,opponent_color, board_size=8):
        self.name = 'Potato' 
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

        possible_moves = self.get_all_valid_moves(board, self.my_color)
        for coords in possible_moves:
            board_copy = copy.deepcopy(board)
            move_value = self.minimax(board_copy, 1, False, self.opponent_color)
        choose_list.append([move_value, coords])
        
        for i in choose_list:
            if i[0] > best_value:
                best_value = i[0]
                best_move = i[1]
        
        return best_move

    def __is_correct_move(self, move, players_color, board):
        dx = [-1, -1, -1, 0, 1, 1, 1, 0]
        dy = [-1, 0, 1, 1, 1, 0, -1, -1]
        for i in range(len(dx)):
            if self.confirm_direction(move, dx[i], dy[i], players_color, board):
                return True, 
        return False
    '''
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
    '''
    def confirm_direction(self,move,dx,dy,players_color, board):
        '''
        Looks into dirextion [dx,dy] to find if the move in this dirrection is correct.
        It means that first stone in the direction is oponents and last stone is players.
        :param move: position where the move is made [x,y]
        :param dx: x direction of the search
        :param dy: y direction of the search
        :param player: player that made the move
        :return: True if move in this direction is correct
        '''
        if players_color == self.p1_color:
            opponents_color = self.p2_color
        else:
            opponents_color = self.p1_color

        posx = move[0]+dx
        posy = move[1]+dy
        if (posx>=0) and (posx<self.board_size) and (posy>=0) and (posy<self.board_size):
            if board[posx][posy] == opponents_color:
                while (posx>=0) and (posx<self.board_size) and (posy>=0) and (posy<self.board_size):
                    posx += dx
                    posy += dy
                    if (posx>=0) and (posx<self.board_size) and (posy>=0) and (posy<self.board_size):
                        if board[posx][posy] == self.empty_color:
                            return False
                        if board[posx][posy] == players_color:
                            return True

        return False
    
    def play_move(self,move,players_color, board):
        '''
        :param move: position where the move is made [x,y]
        :param player: player that made the move
        :param board: board that the move is made on
        '''

        board[move[0]][move[1]] = players_color
        dx = [-1,-1,-1,0,1,1,1,0]
        dy = [-1,0,1,1,1,0,-1,-1]
        for i in range(len(dx)):
            if self.confirm_direction(move,dx[i],dy[i],players_color, board):
                board = self.change_stones_in_direction(move,dx[i],dy[i],players_color, board)

        return board
    
    def change_stones_in_direction(self,move,dx,dy,players_color, board):
        posx = move[0]+dx
        posy = move[1]+dy
        while (not(board[posx][posy] == players_color)):
            board[posx][posy] = players_color
            posx += dx
            posy += dy
        return board

    def get_all_valid_moves(self, board, players_color):
        valid_moves = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                if (board[x][y] == -1) and self.__is_correct_move([x, y], players_color, board):
                    valid_moves.append( (x, y) )

        if len(valid_moves) <= 0:
            print('No possible move!')
            return None
        return valid_moves
    
    def get_score(self, board, color):
        stones = [0 , 0]
        for x in range(self.board_size):
            for y in range(self.board_size):
                if board[x][y] == self.p1_color:
                    stones[0] += self.my_color
                if board[x][y] == self.p2_color:
                    stones[1] += 1
        if color == self.p1_color:
            return stones[0]
        else:
            return stones[1]

    def minimax(self, board, depth, maximizing, players_color):
        if (depth == 0 or self.get_all_valid_moves(board, players_color) == None):
            if maximizing:
                score = self.get_score(board, self.my_color)
            else:
                score = self.get_score(board, self.opponent_color)
            return score
        if maximizing:
            max_alpha = -math.inf
            list_of_moves = self.get_all_valid_moves(board, self.my_color)
            for i in list_of_moves:
                board_copy = copy.deepcopy(board)
                updated_board = self.play_move(i, self.my_color, board_copy)
                eval = self.minimax(updated_board, depth-1, False, self.opponent_color)
                max_alpha = max(max_alpha, eval)            # function for picking the larger value
            return max_alpha

        else: #minimizing
            min_beta = +math.inf
            list_of_moves = self.get_all_valid_moves(board, self.opponent_color)  #have to add that I want valid moves for opponent
            for i in list_of_moves:
                board_copy = copy.deepcopy(board)
                updated_board = self.play_move(i, self.opponent_color, board_copy)
                eval = self.minimax(updated_board, depth-1, True, self.my_color)
                min_beta = min(min_beta, eval)              # function for picking the smaller value
            return min_beta


