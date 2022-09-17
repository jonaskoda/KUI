import copy
import math
from board_creator import evaluated_board

class MyPlayer():
    '''Used: Minimax with alpha-beta pruning, board with evaluated positions ''' 

    def __init__(self, my_color,opponent_color, board_size=8):
        self.name = 'Potato' 
        self.my_color = my_color
        self.opponent_color = opponent_color
        self.board_size = board_size
        self.p1_color = 0
        self.p2_color = 1
        self.empty_color = -1
        create_eval_board = evaluated_board(board_size)               # based on size of the board create a board
        self.eval_board = create_eval_board.choose_board()            # with evaluated coords for use as a heuristic

    def move(self,board):
        
        choose_list = []

        possible_moves = self.get_all_valid_moves(self.my_color,board)
        for coords in possible_moves:
            heurist_value = self.get_eval_value(coords)
            updated_board = self.make_move(coords, self.my_color, board)
    
            move_value = self.minimax(updated_board, 3, False, self.my_color, -math.inf, math.inf)
            final_value = move_value + (heurist_value * 3)
            choose_list.append([final_value, coords])
        
        best_move = self.get_best_val(choose_list)
        
        return best_move

    def get_best_val(self, choose_list):                          # choose the biggest value
        best_value = -math.inf
        for i in choose_list:
            if i[0] > best_value:
                best_value = i[0]
                best_move = i[1]
        
        return best_move

    def __is_correct_move(self, move, players_color, board):      # all of the functions up to minimax come 
        dx = [-1, -1, -1, 0, 1, 1, 1, 0]                          # from game_board and are slightly modified 
        dy = [-1, 0, 1, 1, 1, 0, -1, -1]
        for i in range(len(dx)):
            if self.confirm_direction(move, dx[i], dy[i], players_color, board):
                return True, 
        return False

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

    def get_all_valid_moves(self, players_color, board):
        valid_moves = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                if (board[x][y] == -1) and self.__is_correct_move([x, y], players_color, board):
                    valid_moves.append( (x, y) )

        if len(valid_moves) <= 0:
            #print('No possible move!')
            return None
        return valid_moves
    
    def get_score(self, board, color):
        stones = [0 , 0]
        for x in range(self.board_size):
            for y in range(self.board_size):
                if board[x][y] == self.p1_color:
                    stones[0] += 1
                if board[x][y] == self.p2_color:
                    stones[1] += 1
        if color == self.p1_color:
            return stones[0]
        else:
            return stones[1]

    def minimax(self, board, depth, maximizing, players_color, alpha, beta):       # (heuristic)
        if (depth == 0 or self.get_all_valid_moves(players_color, board) == None): # base case for termination
            if maximizing:
                point_score = self.get_score(board, self.my_color)                 # return score for this player 
            else:
                point_score = self.get_score(board, self.opponent_color)           # return score for opponent
            return point_score

        if maximizing:                                                             # predicting players move
            max_score = -math.inf
            list_of_moves = self.get_all_valid_moves(players_color, board)
            for i in list_of_moves:
                heurist_value = self.get_eval_value(i)
                updated_board = self.make_move(i, self.my_color, board)            # function for returning heuristic value based on position

                eval = self.minimax(updated_board, depth-1, False, self.opponent_color, alpha, beta) # recursive call for minimizing
                eval += (heurist_value * 3)
                max_score = max(max_score, eval)                                   # function for picking the larger value
                alpha = max(eval, alpha)
                if alpha >= beta:                                                  # alpha-beta pruning, stop based on enough info
                    break
            return max_score

        else:                                                                      # minimizing (predicting opponents move)                         
            min_score = +math.inf
            list_of_moves = self.get_all_valid_moves(players_color, board) 
            for i in list_of_moves:
                heurist_value = self.get_eval_value(i)                             
                updated_board = self.make_move(i, self.opponent_color, board)

                eval = self.minimax(updated_board, depth-1, True, self.my_color, alpha, beta)  # recursive call for maximizing
                eval += (heurist_value * 3)                                        
                min_score = min(min_score, eval)                                   # function for picking the smaller value
                beta = min(eval, beta)
                if alpha >= beta:
                    break
            return min_score

    def make_move(self, coords, players_color, board):
        board_copy = copy.deepcopy(board)
        updated_board = self.play_move(coords, players_color, board_copy)

        return updated_board
        
    def get_eval_value(self, coords):               # (heuristic) return value based on position on the board 
        row = coords[0]                            
        col = coords[1]

        heur_value = self.eval_board[row][col] 

        return heur_value



