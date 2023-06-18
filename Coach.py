import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from MCTS import MCTS
from Game import Game

import multiprocessing
import copy
from utils import *
from itertools import chain
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

log = logging.getLogger(__name__)


def executeEpisode_parallel(nnet, args):
    """
    This function executes one episode of self-play, starting with player 1.
    As the game is played, each turn is added as a training example to
    trainExamples. The game is played till the game ends. After the game
    ends, the outcome of the game is used to assign values to each example
    in trainExamples.

    It uses a temp=1 if episodeStep < tempThreshold, and thereafter
    uses temp=0.

    Returns:
        trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                        pi is the MCTS informed policy vector, v is +1 if
                        the player eventually won the game, else -1.
    """
    trainExamples = []
    game = Game(args.dim, args.boundary, args.upper_bound, print_result=0)
    board = game.getInitBoard()
    curPlayer = 1
    episodeStep = 0
    mcts = MCTS(game, nnet, args)  # reset search tree

    while True:
        episodeStep += 1
        canonicalBoard = game.getCanonicalForm(board, curPlayer)
        temp = int(episodeStep < args.tempThreshold)

        pi = mcts.getActionProb(canonicalBoard, temp=temp)
        sym = game.getSymmetries(canonicalBoard, pi)
        for b, p in sym:
            trainExamples.append([b, curPlayer, p, None])

        action = np.random.choice(len(pi), p=pi)
        board = game.getNextState(board, action)

        r = game.getGameEnded(board)

        if r != 0:
            log.info('Finish one episode')
            return [(x[0], x[2], r * ((-1) ** (x[1] != curPlayer))) for x in trainExamples], game.best, game.best_board

def executeEpisode_parallel_pack(nnet, args):
    torch.cuda.init()
    trainExamples = []
    best_num = 0
    best_board = torch.zeros(size=(args.upper_bound, args.dim))
    for i in range(1):
        # run episode
        tmpexample, tmpbest, tmpboard = executeEpisode_parallel(nnet, args)
        trainExamples.extend(tmpexample)
        if tmpbest > best_num:
            best_num = tmpbest
            best_board = torch.clone(tmpboard)
    return trainExamples, best_num, best_board
        

class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board = self.game.getNextState(board, action)

            r = self.game.getGameEnded(board)

            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
            if not self.skipFirstSelfPlay or i > 1:
                num_process = 8

                # Create a multiprocessing pool with the desired number of processes
                pool = multiprocessing.Pool(processes=num_process)
                results = pool.starmap(executeEpisode_parallel_pack, [(self.nnet, self.args) for i in range(num_process)])
    
                # Close the pool to indicate that no more tasks will be submitted
                pool.close()
                # Wait for all processes to complete
                pool.join()

                results_trainexamples, results_bestnum, results_bestboard = zip(*results)
                train_examples = list(chain.from_iterable(results_trainexamples))
                
                # Update best result
                np_bestnum = np.array(results_bestnum)
                best_index = np.argmax(np_bestnum)
                best_num = np_bestnum[best_index]

                if best_num > self.game.best:
                    print("New result found, num of points:", best_num, flush=True)
                    print(results_bestboard[best_index], flush=True)
                    print(" ", flush=True)
                    self.game.best = best_num
                    self.game.best_board = torch.clone(results_bestboard[best_index])

                # Create a deque and add the results to it
                iterationTrainExamples = deque(train_examples)

            # shuffle examples before training
            trainExamples = []
            trainExamples.extend(iterationTrainExamples)
            shuffle(trainExamples)
            self.nnet.train(trainExamples)

            log.info('NEW MODEL')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
