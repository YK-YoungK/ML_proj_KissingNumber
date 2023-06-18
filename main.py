import logging

import coloredlogs

from Coach import Coach
from Game import Game
from NeuralNet import NeuralNet as nn
from utils import *
import multiprocessing as mp

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 200,
    'numEps': 20,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     #
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 30,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         #
    'cpuct1': 1.25,
    'cpuct2': 10000,

    'dim': 6, 
    'boundary': 2, 
    'upper_bound': 79,          # NOTE: Need to be [strictly] greater than known upper bound.

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})




def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(args.dim, args.boundary, args.upper_bound)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
