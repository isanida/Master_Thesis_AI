# Modelling task and worker correlation for crowdsourcing label aggregation.


This is the fully reproducible code of the project.


## Datasets

The ```datasets``` folder contains all the datasets that were used in order to perform the run experiments. Each dataset folder consists of an ```answer.csv``` and a ```truth.csv``` file. 


The answer file is a ```.csv``` file with columns ```question, worker, answer ```. *question* is the item_id, *worker* is the worker_id and *answer* is the given crowdsourced label.


The truth file is a ```.csv``` file with columns ```question, truth ```. *question* is the item_id, and *truth* is the inferred truth.


### ```glad_pseudo_data-gamma.ipynb``` is the notebook that generates the synthetic data.



## Methods

There are 4 implementations of inferring the true label from noisy crowdsourcing annotations. The first 3 methods consist of the baselines and the last one is the proposed method.

1. **[c_MV]** Majority Voting 
2. [Dawid and SKene](https://rss.onlinelibrary.wiley.com/doi/abs/10.2307/2346806) **[c_EM]**
3. **[c_GLAD]**. [GLAD method] (https://papers.nips.cc/paper/3644-whose-vote-should-count-more-optimal-integration-of-labels-from-labelers-of-unknown-expertise.pdf) 
4. **[c_GAMMA]** model; an extension of GLAD baseline, which integrates the abundant information associated withs task and annotators.

All methods can be found in the ```methods``` folder.

- Majority Voting method is in ```methods/c_MV/method.py```.
- D&S method is in ```methods/c_EM/method.py```.
- GLAD method is in ```methods/c_GLAD/method.py```.
- Gamma method is in ```methods/c_GAMMA/method.py```.

In order to run each method we only use the **answer_file_path** parameter which is the directory of the answer file. 


For example, in order to run the EM method using the demo asnwer file, we need to be in the ```methods/c_EM/``` directory and then run: 


```
python method.py ../../demo_answer_file.csv 
``` 


This will produce the posterior probabilities of the inferred true label for each item, and the output values of worker quality.

## Usage

In order to retrieve the inferred truth after we run each method on a dataset, where we have the given label of the worker for each item, we simply run the script ```run.py```which takes four parameters: 

1. method_file_path (directory of the method used)

2. answer_file_path (directory of the answer file)

3. result_file_path (directory of the output file)

4. task_type (it can only take the value of ```decision-making```, since we are modelling only decision-making tasks).



A complete run example is:

```python run.py methods/c_EM/method.py ./demo_answer_file.csv ./demo_result_file.csv decision-making ```



Where we use *EM* method and the *demo_answer_file* as input. The inferred true labels of the given items are in the *demo_result_file*.




## Reproducing Results


There are 2 types of experiments used, in order to evaluate the performance of the methods:


```bash
exp1_decisionmaking.py
```

For data redundancy, we define it as the number of answers collected for each task (number of collected answers/number of tasks). We observe the quality of each method, in each dataset with varying data redundancy.



```bash
exp2_decisionmaking.py
```

We see how each method can be affected by qualification test. Thus, (1) we first simulate each worker’s answers for qualification test (```generate_qualification_kfolderdata.py```); (2) then use each worker’s answering performance for them to initialise the worker’s quality (output of each method); (3) finally we run each method with the initialised worker’s quality.

```bash
exp2-a-b-g.py
```

We measure how correctly each method can learn its parameters of interest (ie: worker expertise alpha, item difficulty beta and worker-item correlation gamma). We implement the second experiment which makes use of a qualification test and this time we estimate these parameters, instead of the given label that we estimate in ```exp2_decisionmaking.py```.



```config.ini``` is the configuration file for the experiments.



## Metrics


For the first experiment, we use **Accuracy** as the metric, which is defined as the fraction of tasks whose truth are inferred correctly.


For the second experiment, we also use **F1-Score**, which is defined as the harmonic mean of Precision and Recall. 


## Data Visualization

The results of the first experiment can be visualised running ```plot_exp1_decision_making.py```. 


