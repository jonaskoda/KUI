class evaluated_board:
    ''' evaluated_board creates a board with each position evaluated with certain value
        in order to provide a player with an option to create heuristics based on value 
        at a specific position '''

    def __init__(self, board_size):
        self.board_size = board_size

    def choose_board(self):
        if self.board_size == 6:
            return self.board6()
        elif self.board_size == 8:
            return self.board8()
        elif self.board_size == 10:
            return self.board10()

    def board6(self):
        board = [[ 4,-3, 2, 2,-3, 4],
                 [-3,-4,-1,-1,-4,-3],
                 [ 2,-1, 1, 1,-1, 2],
                 [ 2,-1, 1, 1,-1, 2],
                 [-3,-4,-1,-1,-4,-3],
                 [ 4,-3, 2, 2,-3, 4]]

        return board

    def board8(self):
        board = [[ 4,-3, 2, 2, 2, 2,-3, 4],
                 [-3,-4,-1,-1,-1,-1,-4,-3],
                 [ 2,-1, 1, 0, 0, 1,-1, 2],
                 [ 2,-1, 0, 1, 1, 0,-1, 2],
                 [ 2,-1, 0, 1, 1, 0,-1, 2],
                 [ 2,-1, 1, 0, 0, 1,-1, 2],
                 [-3,-4,-1,-1,-1,-1,-4,-3],
                 [ 4,-3, 2, 2, 2, 2,-3, 4]]

        return board

    def board10(self):
        board = [[ 4,-3, 2, 2, 2, 2, 2, 2,-3, 4],
                 [-3,-4,-1,-1,-1,-1,-1,-1,-4,-3],
                 [ 2,-1, 1, 1, 0, 0, 1, 1,-1, 2],
                 [ 2,-1, 0, 0, 1, 1, 0, 0,-1, 2],
                 [ 2,-1, 0, 1, 1, 1, 1, 0,-1, 2],
                 [ 2,-1, 0, 1, 1, 1, 1, 0,-1, 2],
                 [ 2,-1, 0, 0, 1, 1, 0, 0,-1, 2],
                 [ 2,-1, 1, 1, 0, 0, 1, 1,-1, 2],
                 [-3,-4,-1,-1,-1,-1,-1,-1,-4,-3],
                 [ 4,-3, 2, 2, 2, 2, 2, 2,-3, 4]]

        return board
