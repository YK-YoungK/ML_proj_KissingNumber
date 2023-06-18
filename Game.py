import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import itertools

import random
from utils import *

args_game = dotdict({
    'augumentation': 10,
})

dim4_sol = torch.tensor([[1,1,0,0],[1,-1,0,0],[-1,1,0,0],[-1,-1,0,0],
                         [1,0,1,0],[1,0,-1,0],[-1,0,1,0],[-1,0,-1,0],
                         [1,0,0,1],[1,0,0,-1],[-1,0,0,1],[-1,0,0,-1],
                         [0,1,1,0],[0,1,-1,0],[0,-1,1,0],[0,-1,-1,0],
                         [0,1,0,1],[0,1,0,-1],[0,-1,0,1],[0,-1,0,-1],
                         [0,0,1,1],[0,0,1,-1],[0,0,-1,1],[0,0,-1,-1]])

dim5_sol = torch.tensor([[1,1,0,0,0],[1,-1,0,0,0],[-1,1,0,0,0],[-1,-1,0,0,0],
                         [1,0,1,0,0],[1,0,-1,0,0],[-1,0,1,0,0],[-1,0,-1,0,0],
                         [1,0,0,1,0],[1,0,0,-1,0],[-1,0,0,1,0],[-1,0,0,-1,0],
                         [1,0,0,0,1],[1,0,0,0,-1],[-1,0,0,0,1],[-1,0,0,0,-1],
                         [0,1,1,0,0],[0,1,-1,0,0],[0,-1,1,0,0],[0,-1,-1,0,0],
                         [0,1,0,1,0],[0,1,0,-1,0],[0,-1,0,1,0],[0,-1,0,-1,0],
                         [0,1,0,0,1],[0,1,0,0,-1],[0,-1,0,0,1],[0,-1,0,0,-1],
                         [0,0,1,1,0],[0,0,1,-1,0],[0,0,-1,1,0],[0,0,-1,-1,0],
                         [0,0,1,0,1],[0,0,1,0,-1],[0,0,-1,0,1],[0,0,-1,0,-1],
                         [0,0,0,1,1],[0,0,0,1,-1],[0,0,0,-1,1],[0,0,0,-1,-1]])

def generate_inverse_mappings(d, order):
    inverse_mapping = [0] * d
    for i, val in enumerate(order):
        inverse_mapping[val] = i
    #print(inverse_mapping)
    return inverse_mapping

def get_permuted_tensors(tensor1, tensor2):
    """Generate all possible tensors obtained by permuting the axes of A"""
    d = len(tensor1.shape)
    orders = list(itertools.permutations(range(d)))
    orders = orders[1:]     # delete "identity" permutation
    random.shuffle(orders)

    tensors = []
    
    # add origin data
    p1 = tensor1.reshape(-1).tolist()
    tensors.append((tensor2, p1))

    augu_size = min(args_game.augumentation, len(orders))
    for i in range(augu_size):
        permutation1 = tensor1.permute(orders[i]).reshape(-1).tolist()
        permutation2 = torch.index_select(tensor2, 1, torch.tensor(generate_inverse_mappings(d, orders[i])))
        tensors.append((permutation2, permutation1))
    
    return tensors



class Game():
    """
    The Kissing Number Game class.
    """
    def __init__(self, dim, boundary, upper_bound, print_result=1):
        '''
        Input:
            dim: Dimension for kissing number
            boundary: a positive int, we want to search in points with each dimension in [-boundary, boundary] (int)
            upper_bound: The upper bound for the kissing number problem
        '''
        self.dim, self.boundary, self.upper_bound, self.print_result = dim, boundary, upper_bound, print_result
        self.best = 0
        self.best_board = torch.zeros(size=(upper_bound, dim))

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        startBoard = torch.zeros(size = (self.upper_bound, self.dim))
        
        # Utilizing results in low-dimensional space
        if self.dim == 5:
            startBoard[:24, :4] = dim4_sol
        if self.dim == 6:
            startBoard[:40, :5] = dim5_sol
        
        return startBoard

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return (self.upper_bound, self.dim)

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        actionSize = (2 * self.boundary + 1) ** self.dim
        # NOTE: for simplicity, we let (0,0,...,0) to be possible but illegal action.
        return actionSize

    def getNextState(self, board, action):
        """
        Input:
            board: current board
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
        """
        maxdim, _ = torch.max(torch.abs(board), dim=1)
        _, minplace = torch.min(maxdim, dim = 0)
        minplace = minplace.item()
        nextBoard = board.clone()
        nextBoard[minplace, :] = self.getAction(action)
        return nextBoard

    def getPointNum(self, board):
        maxdim, _ = torch.max(torch.abs(board), dim=1)
        _, minplace = torch.min(maxdim, dim=0)
        pointnum = minplace.item()
        return pointnum
    
    def checkAngle(self, v1, v2):
        '''
        Input: v1, v2 are two vectors.
        Return: 1 if feasible, otherwise 0
        NOTE: If v1=v2, should return 0
        '''
        inner_prod = torch.dot(v1, v2)
        if inner_prod <= 0:
            return 1
        squarenorm1 = torch.dot(v1, v1)
        squarenorm2 = torch.dot(v2, v2)
        if 4 * (inner_prod ** 2) <= squarenorm1 * squarenorm2:
            return 1
        return 0
    
    def getAction(self, x):
        tmp = 1
        action = torch.zeros(size=(self.dim,))
        for i in range(self.dim - 1, -1, -1):
            action[i] = ((x // tmp) % (2 * self.boundary + 1)) - self.boundary
            tmp = tmp * (2 * self.boundary + 1)
        return action

    def getValidMoves(self, board):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        len = self.getActionSize()
        validMoves = torch.ones(size = (len,))
        pointnum = self.getPointNum(board)

        for i in range(len):
            action = self.getAction(i)
            
            if i == len // 2:
                validMoves[i] = 0
                continue
            for j in range(pointnum):
                if self.checkAngle(action, board[j]) == 0:
                    validMoves[i] = 0
                    break
        
        return validMoves


    def getGameEnded(self, board):
        """
        Input:
            board: current board

        Returns:
            r: 0 if game has not ended. 
            If the game ended, return the reward.
               
        """
        valid = self.getValidMoves(board)
        validnum = torch.sum(valid).item()
        if validnum == 0:
            pointnum = self.getPointNum(board)
            if pointnum > self.best:
                self.best = pointnum
                self.best_board = torch.clone(board)
                if self.print_result:
                    print("End of kissing instance, num of points:", pointnum, flush=True)
                    print(board, flush=True)
                    print(" ", flush=True)
            return pointnum
        else:
            return 0

    def getCanonicalForm(self, board, player):
        return board

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        point_num = self.getPointNum(board)
        if point_num <= self.dim:
            return [(board,pi)]
        new_shape = tuple([2 * self.boundary + 1] * self.dim)
        pi_tensor = torch.tensor(pi).view(*new_shape)
        return get_permuted_tensors(pi_tensor, board)

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        # convert the tensor to a nested Python list
        board_list = board.tolist()
        # convert the nested list to a string
        string = str(board_list)
        return string
