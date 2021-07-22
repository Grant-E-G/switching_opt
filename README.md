>📋  A template README.md for code accompanying a Machine Learning paper

# Learning to Switch Optimizers for Quadratic Programming

This repository is the official implementation of [Learning to Switch Optimizers for Quadratic Programming](https://google.com). 

Learning to optimize (L2O) is an emerging subtopic in machine learning, where one learns an optimizer by learning an update step for an iterative optimization algorithm.

Mathematically we express optimizers as 

f(x_i, data) = x_i+1

Then the task of learning an optimizer is finding a good f. Instead of the standard practice in L2O of learning f from scratch, we train a stochastic policy that switches between standard update rules in optimization. Thus our optimizer can be greater than the sum of its parts by only using the best update rule given our current knowledge of the state of our problem. 

In practice, we use off-the-shelf reinforcement learning techniques to train our stochastic policy, and we use Quadratic Programming as our testbed for our optimizer.


## Requirements

Here are the installation instructions assuming that you want to use Conda.


```setup
conda create -n RLSO python=3.7
conda activate RLSO
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, we need to use a YAML configuration file to specify the details of our run. We have provided some example configuration files in the 'configs' subdirectory.
```train
python main.py --config='<configs/your_YAML_config_file>'  --savedata_type=2
```

This will train our model and then produce a collection of plots on test cases automatically. Thus evaluation happens right after training. Also, note that we have set the 'savedata_type' option to two, which saves everything. Options zero saves nothing but the model, and one keeps the problems generated and their best solution from testing. 'main.py' also provides command-line options for restarting training and plotting preexisting models.

```help
python main.py --help
```
The above command will provide some basic descriptions of the provided command-line options.

## Plotting Matlab experiments

The Matlab experiments use both the Matlab optimization toolbox and the python interface for Matlab. These components are not free and have to be installed separately. Please see the [Matlab Documentation] (https://www.mathworks.com/products/matlab.html) for instructions.

```matlab_cl
python matlab_cl.py --path='<Path to saved data >' 
```

The above command will run the Matlab experiments on only the selected trial data with the default options. Warning: the data has to be saved with the --savedata_type=2 for the above to work. It will also save the output figure with a file name of the following pattern 'random integer .png.' Again for more advanced options, please use the following command.

```matlab_cl
python matlab_cl.py --help 
```

## Pre-trained Models

Our pre-trained models are include in our agent subdirectory. 



## Results



## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
