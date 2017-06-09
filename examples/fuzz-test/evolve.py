"""
This example produces networks that can remember a fixed-length sequence of bits. It is
intentionally very (overly?) simplistic just to show the usage of the NEAT library. However,
if you come up with a more interesting or impressive example, please submit a pull request!

This example also demonstrates the use of a custom activation function.
"""

from __future__ import division, print_function

from bitstring import BitArray, Bits

import math
import os
import sys
import random
import multiprocessing

import neat

import subprocess
import tempfile

import time

# Demonstration of how to add your own custom activation function.
# This sinc function will be available if my_sinc_function is included in the
# config file activation_options option under the DefaultGenome section.
# Note that sinc is not necessarily useful for this example, it was chosen
# arbitrarily just to demonstrate adding a custom activation function.
def sinc(x):
    return 1.0 if x == 0 else math.sin(x) / x


# number of inputs in the network
num_inputbytes = 4
# num_tests is the number of random examples each network is tested against.
num_tests = 16

def get_outputbytes(net, i):
    random.seed(i)
    inputs = [-1.0, 0.0, 1.0]+[random.random() for x in xrange(num_inputbytes)]
    output = Bits().join(map(lambda x: BitArray(uint=max(0,min(255,int(x*256))), length=8), net.activate(inputs)))
    return output.bytes

def get_commandline(argv, filename):
    if "@@" in argv:
        return " ".join([filename if x=="@@" else x for x in argv[1:]])
    else:
        return " ".join(argv[1:]) + " < " + filename

class CustomReporter(neat.reporting.BaseReporter):
    # def end_generation(self, config, population, species_set):
    #     for k,v in species_set.species.iteritems():
    #         genome = max(v.members.itervalues(), key=lambda x: x.fitness)
    #         net = neat.nn.RecurrentNetwork.create(genome, config)

    #         print("Best genome for species:", k, map(lambda i: f(net, i), xrange(num_tests)))
    #         #print("Genome for species:", k, str(genome))

    def post_evaluate(self, config, population, species, best_genome):
        #print(best_genome)
        print("STATUS %f,%f" % (time.time(), best_genome.fitness))

def stupid_eval_genome(genome, config):
    net = neat.nn.RecurrentNetwork.create(genome, config)

    best = 0
    
    for i in range(num_tests):
        output = get_outputbytes(net, i)
        if output[0] == "F":
            best += 1
            if output[1] == "U":
                best += num_tests
                if output[2] == "Z":
                    best += num_tests*num_tests
                    if output[3] == "Z":
                        best += num_tests*num_tests*num_tests

    return best

def eval_genome(genome, config):
    net = neat.nn.RecurrentNetwork.create(genome, config)

    coverage = {}

    try:

        for i in range(num_tests):
            output = get_outputbytes(net, i)

            # Write outupt to file
            with tempfile.NamedTemporaryFile(prefix="input") as f, tempfile.NamedTemporaryFile(prefix="cov") as covf:
                #print(output)
                f.write(output)
                f.flush()

                #print("afl-showmap -o %s -t 10000 -m 2000 -Q -q -- /usr/bin/identify %s" % (covf.name, f.name))
                cl = get_commandline(config.argv, f.name)
                try:
                    subprocess.call("AFL_INST_LIBS=1 /usr/local/bin/afl-showmap -o %s -t 10000 -m 2000 -Q -q -- %s" % (covf.name, cl), shell=True)

                    with open(covf.name, "r") as covr:
                        for edge in covr.read().splitlines():
                            coverage[edge] = coverage.get(edge, 0) + 1
                        #print(len(coverage))
                except Exception as e:
                    print ("Ignoring exception from afl-showmap", e)

    except Exception as e:
        print ("Exception captured, returning zero", e)
        return 0

    # We want to weight the first instance of an edge very heavily.  So the first weight is given one.
    # The second is given 1/2, and third 1/3, and so on
    return sum([sum([1/x for x in xrange(1,x+1)]) for x in coverage.values()])

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run(argv):
    # Determine path to configuration file.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    config.argv = argv

    # Demonstration of saving a configuration back to a text file.
    config.save('test_save_config.txt')

    # Demonstration of how to add your own custom activation function.
    # This sinc function will be available if my_sinc_function is included in the
    # config file activation_options option under the DefaultGenome section.
    config.genome_config.add_activation('my_sinc_function', sinc)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(CustomReporter())

    if 1:
        pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), stupid_eval_genome)
        winner = pop.run(pe.evaluate, 1000)
    else:
        winner = pop.run(eval_genomes, 1000)


    # Log statistics.
    stats.save()

    # # Show output of the most fit genome against a random input.
    print('\nBest genome:\n{!s}'.format(winner))
    print('\nOutput:')
    winner_net = neat.nn.RecurrentNetwork.create(winner, config)
    # num_correct = 0
    # for n in range(num_tests):
    #     print('\nRun {0} output:'.format(n))
    #     seq = [random.choice((0.0, 1.0)) for _ in range(N)]
    #     winner_net.reset()
    #     for s in seq:
    #         inputs = [s, 0.0]
    #         winner_net.activate(inputs)
    #         print('\tseq {0}'.format(inputs))

    #     correct = True
    #     for s in seq:
    #         output = winner_net.activate([0, 1])
    #         print("\texpected {0:1.5f} got {1:1.5f}".format(s, output[0]))
    #         correct = correct and round(output[0]) == s
    #     print("OK" if correct else "FAIL")
    #     num_correct += 1 if correct else 0

    # print("{0} of {1} correct {2:.2f}%".format(num_correct, num_tests, num_correct/num_tests))

    # node_names = {-1: 'input', -2: 'gate', 0: 'output'}
    # visualize.draw_net(config, winner, True, node_names=node_names)
    # visualize.plot_stats(stats, ylog=False, view=True)


if __name__ == '__main__':
    run(sys.argv)
