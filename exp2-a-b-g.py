import os
import commands
import math
import csv
import random
import ConfigParser
import time
import numpy as np


def get_label_set(datafile):
    label_set = []

    f = open(datafile, 'r')
    reader = csv.reader(f)
    next(reader)

    for line in reader:
        _, _, label = line

        if label not in label_set:
            label_set.append(label)

    return label_set


def getaccuracy(datafile, truthfile, e2lpd):
    label_set = get_label_set(datafile)
    # in case that e2lpd does not have data in the truthfile, then we randomly sample a label from label_set
    assert label_set == ['0', '1'] or label_set == ['1', '0']

    e2truth = {}
    f = open(truthfile, 'r')
    reader = csv.reader(f)
    next(reader)

    for line in reader:
        example, truth = line
        e2truth[example] = truth

    tcount = 0

    for e in e2truth:

        if e not in e2lpd:
            # randomly select a label from label_set
            truth = random.choice(label_set)
            if int(truth) == int(e2truth[e]):
                tcount += 1

            continue

        if type(e2lpd[e]) == type({}):
            temp = 0
            for label in e2lpd[e]:
                if temp < e2lpd[e][label]:
                    temp = e2lpd[e][label]

            candidate = []

            for label in e2lpd[e]:
                if temp == e2lpd[e][label]:
                    candidate.append(label)

            truth = random.choice(candidate)

        else:
            truth = e2lpd[e]

        if int(truth) == int(e2truth[e]):
            tcount += 1

    return tcount * 1.0 / len(e2truth)


def getfscore(datafile, truthfile, e2lpd):
    label_set = get_label_set(datafile)
    # in case that e2lpd does not have data in the truthfile, then we randomly sample a label from label_set
    assert label_set == ['0', '1'] or label_set == ['1', '0']

    e2truth = {}
    f = open(truthfile, 'r')
    reader = csv.reader(f)
    next(reader)

    for line in reader:
        example, truth = line
        e2truth[example] = truth

    fz = 0
    fm_pre = 0
    fm_rec = 0

    for e in e2truth:

        if int(e2truth[e]) == 1:
            fm_rec += 1

        if e not in e2lpd:
            # randomly select a label from label_set
            truth = random.choice(label_set)
            if int(truth) == 1:
                fm_pre += 1
                if int(e2truth[e]) == 1:
                    fz += 1

            continue

        if type(e2lpd[e]) == type({}):
            temp = 0
            for label in e2lpd[e]:
                if temp < e2lpd[e][label]:
                    temp = e2lpd[e][label]

            candidate = []

            for label in e2lpd[e]:
                if temp == e2lpd[e][label]:
                    candidate.append(label)

            truth = random.choice(candidate)

        else:
            truth = e2lpd[e]

        if int(truth) == 1:
            fm_pre += 1
            if int(e2truth[e]) == 1:
                fz += 1

    if fz == 0 or fm_pre == 0:
        return 0.0

    precision = fz * 1.0 / fm_pre
    recall = fz * 1.0 / fm_rec

    return 2.0 * precision * recall / (precision + recall)



def select_kfold(datafile):
    f = open(datafile, 'r')
    reader = csv.reader(f)
    next(reader)

    count = 0
    examples = {}
    for line in reader:
        example, worker, label = line
        examples[example] = 0
        count += 1

    return int(math.ceil(count * 1.0 / len(examples)))




def run_datasets(python_command, datasets, methods, iterations):
    for method in methods:
        print "########" + method + "########"

        for dataset in datasets:

            if not os.path.isdir(r'./methods/' + method):
                continue

            # dataset & method

            truthfile = r"'./datasets/" + dataset + r"/truth.csv'"

            datafile = r"'./datasets/" + dataset + '/' + r"/answer.csv'"
            output = commands.getoutput(python_command + r'./methods/' + method + r'/method.py '
                                        + datafile + ' ' + '"categorical"').split('\n')[-1]


            #TODO: change method (and qualification method) output in Run() for each different value
            output_value = commands.getoutput(python_command + r'./methods/' + method + r'/method.py '
                                             + datafile + ' ' + '"categorical"').split('\n')[-2]



            originalacc = getaccuracy(eval(datafile), eval(truthfile), eval(output))
            originalfscore = getfscore(eval(datafile), eval(truthfile), eval(output))
            originalvalue = eval(output_value)
            # assert type(originalalpha) == type({})

            accuracies = []
            fscores = []
            values = []

            for iteration in range(iterations):
                tempfile = r"'./qualification_data_kfolder/" + dataset + '/' + str(iteration) + ".csv'"

                # print datafile, tempfile

                output = commands.getoutput(python_command + r'./qualification_methods/' + method + r'/method.py '
                                            + datafile + ' ' + tempfile + ' ' + '"categorical"').split('\n')[-1]

                output_value = commands.getoutput(python_command + r'./qualification_methods/' + method + r'/method.py '
                                            + datafile + ' ' + tempfile + ' ' + '"categorical"').split('\n')[-2]

                accuracy = getaccuracy(eval(datafile), eval(truthfile), eval(output))
                fscore = getfscore(eval(datafile), eval(truthfile), eval(output))

                value = eval(output_value)

                # assert type(value) == type({})

                accuracies.append(accuracy)
                fscores.append(fscore)


                # Savg = {k: [value[j][k] for j in range(len(value))] for k in value[0].keys()}
                # value= {k: (reduce(np.add, v) / len(v)) for k, v in Savg.iteritems()}

                values.append(value)
                # print("values", values)

                print dataset + str(iteration)
                # print("values", values)
                # print("mean values", Tavg)

            accuracies.insert(0, originalacc)
            fscores.insert(0, originalfscore)
            values.insert(0, originalvalue)


            # print ("accuracies",accuracies)
            # print ("fscores", fscores)
            # print ("values", values)

            mean_acc = np.mean(np.array(accuracies).astype(np.float))
            mean_f1s_core = np.mean(np.array(fscores).astype(np.float))

            # mean (alpha or beta) value for each key of dictionary
            S = {k: [values[j][k] for j in range(len(values))] for k in values[0].keys()}
            T = {k: (reduce(np.add, v) / len(v)) for k, v in S.iteritems()}

            print("mean value:", T)
            print("mean accuracy:", mean_acc)
            print("mean f1 score:", mean_f1s_core)


            # dataset & method finished
            folder = r'./output/exp-2-alpha-beta/decision_making'
            if not os.path.isdir(folder):
                os.mkdir(folder)

            folder = folder + '/' + dataset
            if not os.path.isdir(folder):
                os.mkdir(folder)

            # accuracy
            f = open(folder + '/' + 'accuracy_' + method, 'w')
            f.write(str(accuracies))
            f.close()

            # fscore
            f = open(folder + '/' + 'fscore_' + method, 'w')
            f.write(str(fscores))
            f.close()

            #TODO: change filename for each value
            # alpha-beta-gamma
            f = open(folder + '/' + 'alpha_' + method, 'w')
            f.write(str(values))
            f.close()

if __name__ == '__main__':
    cf = ConfigParser.ConfigParser()
    cf.read('./config.ini')

    # split the data in the "./qualification_data_kfolder" folder
    import generate_qualification_kfolderdata

    iterations = eval(cf.get("exp-2", "iterations"))
    generate_qualification_kfolderdata.generate_qualification_kfolderdata(r'./qualification_data_kfolder', iterations)

    # get the results of each dataset and each method in "./output/exp-2" folder
    datasets_decisionmaking = eval(cf.get("exp-2", "datasets_decisionmaking"))
    qualification_decisionmaking = eval(cf.get("exp-2", "qualification_decisionmaking"))
    python_command = eval(cf.get("exp-2", "python_command"))
    iterations = eval(cf.get("exp-2", "iterations"))
    run_datasets(python_command, datasets_decisionmaking, qualification_decisionmaking, iterations)




