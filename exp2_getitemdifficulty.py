import ConfigParser
import random
import math
import os
import commands
import numpy as np

def run_datasets(python_command, datasets, methods, filetype):
    for dataset in datasets:
        print "########" + dataset + "########"

        for method in methods:

            if not os.path.isdir(r'./methods/' + method):
                continue

            # dataset & method

            # truthfile = r"'./datasets/" + dataset + r"/truth.csv'"
            datafile = r"'./datasets/" + dataset + r"/answer.csv'"

            if filetype in ['decision_making', 'single_label']:
                ftype = '"categorical"'
            elif filetype == 'continuous':
                ftype = '"continuous"'

            output = commands.getoutput(python_command + r'./methods/' + method + r'/method.py '
                                        + datafile + ' ' + ftype).split('\n')[-2]
            beta = eval(output)

            assert type(beta) == type({})
            # beta = dict((key, np.exp(value)) for key, value in beta.items())
            # print(beta)

            # dataset & method finished

            folder = './itemdifficulty/' + filetype
            if not os.path.isdir(folder):
                os.mkdir(folder)

            folder = folder + '/' + dataset
            if not os.path.isdir(folder):
                os.mkdir(folder)

            f = open(folder + '/' + method, 'w')
            f.write(str(beta))
            f.close()


if __name__ == '__main__':
    if not os.path.isdir('./itemdifficulty'):
        os.mkdir('./itemdifficulty')

    cf = ConfigParser.ConfigParser()
    cf.read('./config.ini')

    # decision_making
    datasets_decisionmaking = eval(cf.get("exp-1", "datasets_decisionmaking"))
    quality_decisionmaking = eval(cf.get("exp-2", "quality_decisionmaking"))
    python_command = eval(cf.get("exp-1", "python_command"))
    run_datasets(python_command, datasets_decisionmaking, quality_decisionmaking, 'decision_making')



