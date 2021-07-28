# Learning to Switch Optimizers for Quadratic Programming

This repository is the official implementation of [Learning to Switch Optimizers for Quadratic Programming](https://google.com). 

Learning to optimize (L2O) is an emerging subtopic in machine learning, where one learns an optimizer by learning an update step for an iterative optimization algorithm.

Mathematically we express optimizers as 

f(x_i, data) = x_i+1

Then the task of learning an optimizer is finding a good f. Instead of the standard practice in L2O of learning f from scratch, we train a stochastic policy that switches between standard update rules in optimization. Thus our optimizer can be greater than the sum of its parts by only using the best update rule given our current knowledge of the state of our problem. 

In practice, we use off-the-shelf reinforcement learning techniques to train our stochastic policy, and we use convex and nonconvex Quadratic Programming as our testbed for our optimizer.


## Requirements

Here are the installation instructions assuming that you want to use Conda.


```bash
conda create -n RLSO python=3.7
conda activate RLSO
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, we need to use a YAML configuration file to specify the details of our run. We have provided some example configuration files in the 'configs' subdirectory.
```bash
python main.py --config='<configs/your_YAML_config_file>'  --savedata_type=2
```

This will train our model and then produce a collection of plots on test cases automatically. Thus evaluation happens right after training. Also, note that we have set the 'savedata_type' option to two, which saves everything. Options zero saves nothing but the model, and one keeps the problems generated and their best solution from testing. 'main.py' also provides command-line options for restarting training and plotting preexisting models.

```bash
python main.py --help
```
The above command will provide some basic descriptions of the provided command-line options.

## Plotting Matlab experiments

The Matlab experiments use both the Matlab optimization toolbox and the python interface for Matlab. These components are not free and have to be installed separately. Please see the [Matlab Documentation](https://www.mathworks.com/products/matlab.html) for instructions.

```bash
python matlab_cl.py --path='<Path to saved data >' 
```

The above command will run the Matlab experiments on only the selected trial data with the default options. Warning: the data has to be saved with the --savedata_type=2 for the above to work. It will also save the output figure with a file name of the following pattern 'random integer .png.' Again for more advanced options, please use the following command.

```python
python matlab_cl.py --help 
```

## Pre-trained Models

Our pre-trained models are include in our agent subdirectory. 



## Results

For a proper discussion of our results, please see the paper. Below we show the table summaries of our experiments. We are trying to minimize a quadratic RLSO score - Other solver is negative when RLSO performs better.  The central values are mean scores across 10000 unseen testing problems. Sigma or the standard deviation is also provided but is somewhat unenlightening as our distributions are not symmetric. Also, note that we did not reuse test problems across RLSO variants. From this ablation study, we can see that the full RLSO performs better than the sum of its parts.

We also provide a command-line utility for generating our plots.
```bash
python tables_cl.py --path='<Path to saved smalldata>'
```


### Full RLSO without retraining
| dimension           |   Score - Adam |   sigma |   Score - GD |    sigma |   Score - Random Search |   sigma |
|---------------------|----------------|---------|--------------|----------|-------------------------|---------|
| 5, 10, 15, mixed    |        -211.67 |  278.16 |      -390.74 |   835.64 |                -1124.73 |  828.92 |
| 5                   |         -59.76 |   76.6  |      -118.05 |   244.84 |                 -249.11 |  114.98 |
| 7                   |        -109.93 |  123.09 |      -198.24 |   389.66 |                 -513.63 |  177.74 |
| 10                  |        -203.7  |  204.42 |      -362.75 |   676.09 |                -1030.89 |  271.53 |
| 12                  |        -273.63 |  273.02 |      -487.49 |   881.58 |                -1437.72 |  345.27 |
| 15                  |        -387.85 |  367.2  |      -690.09 |  1205.36 |                -2138.52 |  440.8  |
| 18                  |        -490.57 |  473.99 |      -926.75 |  1563.3  |                -2908.15 |  554.38 |
| 20                  |        -571.96 |  549.95 |     -1087.58 |  1805.58 |                -3467.52 |  633.57 |
| 25                  |        -739.99 |  779.21 |     -1579.81 |  2525.38 |                -4971.17 |  858.13 |
| 30                  |        -876.39 | 1030.1  |     -2093.5  |  3295.68 |                -6607.67 | 1111.48 |
| 40                  |       -1036.85 | 1621.53 |     -3184.2  |  4865.87 |               -10231.7  | 1731.03 |
| 50                  |       -1042.84 | 2286.78 |     -4640.76 |  6756.92 |               -14186    | 2407.18 |
| 60                  |        -712.25 | 3040.49 |     -6020.77 |  8691.89 |               -18340    | 3169.99 |
| 70                  |        -182.36 | 3904.73 |     -7620.58 | 10878.4  |               -22686.6  | 4062.02 |
| 80                  |         704.14 | 4897.9  |     -9077.15 | 12842.8  |               -27069.4  | 5035.23 |
| 90                  |        1906.22 | 5857.77 |    -10731.5  | 15027.9  |               -31566.2  | 6036.55 |
| 100                 |        3341.96 | 6933.57 |    -12368.9  | 17453.7  |               -36111.6  | 7074.06 |

### Full RLSO with retraining
| dimension           |   Score - Adam |   sigma |   Score - GD |    sigma |   Score - Random Search |   sigma |
|---------------------|----------------|---------|--------------|----------|-------------------------|---------|
| 5, 10, 15, 25 mixed |        -352.04 |  513.53 |      -689.1  |  1531.57 |                -2094.48 | 1874.89 |
| 5                   |         -60    |   77.81 |      -112.85 |   236.03 |                 -250.96 |  115.89 |
| 7                   |        -108.25 |  121.2  |      -203.47 |   400.28 |                 -513.67 |  179.33 |
| 10                  |        -204.54 |  207.73 |      -363.62 |   668.03 |                -1031.7  |  277.8  |
| 12                  |        -278.43 |  267.13 |      -496.46 |   867.33 |                -1442.15 |  339.49 |
| 15                  |        -390.65 |  369.16 |      -715.58 |  1222.02 |                -2133.8  |  438.94 |
| 18                  |        -509.01 |  471.85 |      -925.36 |  1543.87 |                -2924.54 |  545.41 |
| 20                  |        -584.88 |  544.76 |     -1127.59 |  1839.29 |                -3481.33 |  618.5  |
| 25                  |        -772.94 |  759.98 |     -1610.89 |  2550.82 |                -4994.5  |  827    |
| 30                  |        -962.49 |  976.35 |     -2121.95 |  3309.67 |                -6698.99 | 1043.53 |
| 40                  |       -1217.75 | 1551.69 |     -3480.17 |  5011.33 |               -10392.4  | 1633.4  |
| 50                  |       -1339.72 | 2229.7  |     -4997.26 |  6817.12 |               -14495.6  | 2346.82 |
| 60                  |       -1298.46 | 2988.18 |     -6521.53 |  8661.62 |               -18889.8  | 3136.89 |
| 70                  |        -984.16 | 3889.82 |     -8353.74 | 10736.7  |               -23485.8  | 4055.27 |
| 80                  |        -427.65 | 4845.61 |    -10113.6  | 12758.8  |               -28198.6  | 5070.45 |
| 90                  |         496.99 | 5905.6  |    -12341.1  | 15140    |               -33005.7  | 6057.65 |
| 100                 |        1650.91 | 7032.15 |    -13989.4  | 17142.8  |               -37789.7  | 7209.86 |

### RLSO: Adam and GD without retraining
| dimension           |   Score - Adam |   sigma |   Score - GD |    sigma |   Score - Random Search |    sigma |
|---------------------|----------------|---------|--------------|----------|-------------------------|----------|
| 5, 10, 15, mixed    |         -80.94 |  240.39 |      -249.81 |   743.49 |                -1014.03 |   794.32 |
| 5                   |         -21.58 |   61.88 |       -77.33 |   225.26 |                 -212.06 |   129.14 |
| 7                   |         -41.5  |  103.41 |      -130.98 |   362.04 |                 -447.73 |   205.49 |
| 10                  |         -78.69 |  176.14 |      -237.47 |   608.92 |                 -904.97 |   323.9  |
| 12                  |        -107.71 |  242.25 |      -313.93 |   792.16 |                -1278.03 |   415.97 |
| 15                  |        -136.63 |  363.51 |      -456.17 |  1096.69 |                -1888.18 |   568.39 |
| 18                  |        -181.31 |  505.14 |      -599.85 |  1373.37 |                -2593.43 |   739.81 |
| 20                  |        -196.35 |  617.96 |      -764.26 |  1644.05 |                -3089.13 |   870.37 |
| 25                  |        -244.86 |  921.67 |     -1058.03 |  2198.12 |                -4477.93 |  1210.11 |
| 30                  |        -241.25 | 1301.51 |     -1426.89 |  2850.1  |                -5974.7  |  1627.38 |
| 40                  |        -139.93 | 2227.9  |     -2390.39 |  4257.54 |                -9326.56 |  2628.67 |
| 50                  |         167.01 | 3270.36 |     -3548.1  |  5748.13 |               -13019.1  |  3821.02 |
| 60                  |         594.15 | 4469.35 |     -4789.46 |  7336.46 |               -17009.8  |  5101.71 |
| 70                  |        1070.27 | 5617.09 |     -6285.17 |  9016.27 |               -21484.9  |  6395.52 |
| 80                  |        1878.26 | 6943.3  |     -7922.66 | 10624.7  |               -25955.3  |  7756.2  |
| 90                  |        2766.47 | 8363.57 |     -9535.89 | 12355    |               -30764.6  |  9273.84 |
| 100                 |        4120.1  | 9745.78 |    -11603.6  | 14290.9  |               -35341.5  | 10793.4  |

### RLSO: Adam and GD with retraining
| dimension           |   Score - Adam |   sigma |   Score - GD |    sigma |   Score - Random Search |   sigma |
|---------------------|----------------|---------|--------------|----------|-------------------------|---------|
| 5, 10, 15, 25 mixed |        -145.42 |  446.22 |      -480.75 |  1356.72 |                -1899.46 | 1761.39 |
| 5                   |         -19.18 |   60.71 |       -73.69 |   227.51 |                 -210.19 |  129.25 |
| 7                   |         -36.36 |  101.75 |      -130.28 |   368.19 |                 -440.93 |  210.6  |
| 10                  |         -77.21 |  178.64 |      -232.75 |   618.61 |                 -901.39 |  326.16 |
| 12                  |        -110.31 |  232.34 |      -324.82 |   801.61 |                -1279.52 |  410.55 |
| 15                  |        -169.39 |  326.45 |      -476.37 |  1111.24 |                -1920.29 |  535.8  |
| 18                  |        -220.7  |  440.83 |      -647.74 |  1411.7  |                -2634.45 |  681.03 |
| 20                  |        -251.54 |  532.88 |      -761.29 |  1611.92 |                -3150.8  |  784.43 |
| 25                  |        -321.64 |  784.82 |     -1163.38 |  2270.41 |                -4541.38 | 1088.18 |
| 30                  |        -385    | 1086.42 |     -1583.15 |  2904.94 |                -6114.47 | 1422.46 |
| 40                  |        -395.16 | 1832.61 |     -2585.81 |  4318.27 |                -9598.73 | 2296.22 |
| 50                  |        -246.61 | 2759.43 |     -3797.87 |  5817.81 |               -13424.6  | 3287.45 |
| 60                  |          57.38 | 3847.75 |     -5371.03 |  7490.82 |               -17541.6  | 4468.25 |
| 70                  |         399.14 | 4974.45 |     -6870.58 |  8990.63 |               -22076.3  | 5702.67 |
| 80                  |        1159.67 | 6220.38 |     -8872.08 | 10713    |               -26636.2  | 7095.49 |
| 90                  |        1901.38 | 7377.76 |    -10686.9  | 12455.7  |               -31543.1  | 8332.7  |
| 100                 |        2900.73 | 8808.37 |    -12832    | 14403.1  |               -36527.6  | 9916.24 |

### RLSO: Adam and Random without retraining
| dimension           |   Score - Adam |   sigma |   Score - GD |    sigma |   Score - Random Search |   sigma |
|---------------------|----------------|---------|--------------|----------|-------------------------|---------|
| 5, 10, 15, mixed    |         -38.71 |  248.37 |      -226.16 |   867.14 |                 -974.2  |  711.87 |
| 5                   |         -31.59 |   75.78 |       -86.4  |   241.62 |                 -222.13 |  114.17 |
| 7                   |         -44    |  126.27 |      -135.63 |   409.54 |                 -450.13 |  178.08 |
| 10                  |         -48.93 |  211.75 |      -218.38 |   713.73 |                 -877.34 |  273.96 |
| 12                  |         -44.6  |  265.57 |      -272.4  |   934.42 |                -1212.08 |  324.5  |
| 15                  |         -35.37 |  359.41 |      -344.35 |  1276.84 |                -1786.88 |  423.87 |
| 18                  |         -21.21 |  453.13 |      -478.73 |  1668.01 |                -2428.77 |  513.1  |
| 20                  |         -11.43 |  529.01 |      -568.69 |  1964.12 |                -2905.47 |  589.44 |
| 25                  |          33.01 |  681.62 |      -806.68 |  2683.55 |                -4194.37 |  740.16 |
| 30                  |          76.28 |  829.23 |     -1201.5  |  3536.44 |                -5660.07 |  894.56 |
| 40                  |         172.13 | 1174.71 |     -1998.34 |  5267.68 |                -9026.46 | 1219.95 |
| 50                  |         288.3  | 1492.86 |     -3324.83 |  7187.29 |               -12865.2  | 1537.16 |
| 60                  |         487.95 | 1839.38 |     -4580.36 |  9080.46 |               -17147.9  | 1871.18 |
| 70                  |         636.96 | 2183.87 |     -6740.66 | 11231    |               -21863    | 2207.21 |
| 80                  |         876.59 | 2526.72 |     -8856.14 | 13338    |               -26914.4  | 2552.56 |
| 90                  |        1076.22 | 2862.65 |    -11290    | 15459.2  |               -32394.1  | 2881.49 |
| 100                 |        1381.49 | 3317.35 |    -14405    | 17910.8  |               -38087.5  | 3254.93 |

### RLSO: Adam and Random with retraining
| dimension           |   Score - Adam |   sigma |   Score - GD |    sigma |   Score - Random Search |   sigma |
|---------------------|----------------|---------|--------------|----------|-------------------------|---------|
| 5, 10, 15, 25 mixed |         -32.15 |  398.22 |      -383.29 |  1590.65 |                -1762.79 | 1576.39 |
| 5                   |         -31.41 |   75.18 |       -86.85 |   245.34 |                 -224.43 |  115.38 |
| 7                   |         -41.56 |  125.51 |      -129.49 |   407.05 |                 -445.82 |  177.78 |
| 10                  |         -46.72 |  208.59 |      -207.73 |   705.78 |                 -876.47 |  268.69 |
| 12                  |         -47.34 |  271.82 |      -264.06 |   924.31 |                -1219.39 |  333.84 |
| 15                  |         -40.1  |  362.45 |      -331.74 |  1257.03 |                -1790.73 |  418.25 |
| 18                  |         -28.67 |  448.44 |      -463.05 |  1670.95 |                -2448.66 |  521.35 |
| 20                  |         -14.43 |  510.16 |      -549.11 |  1941.95 |                -2917.15 |  583.28 |
| 25                  |          19.14 |  669.91 |      -830.31 |  2714.89 |                -4203.11 |  731.11 |
| 30                  |          57.5  |  829.12 |     -1134.13 |  3468.62 |                -5665.52 |  895.07 |
| 40                  |         151.48 | 1171.86 |     -1970.63 |  5193.29 |                -9026.96 | 1220    |
| 50                  |         268.82 | 1453.98 |     -3282.46 |  6999.78 |               -12894.1  | 1516.93 |
| 60                  |         408.77 | 1806.53 |     -4888.4  |  9124.39 |               -17217.3  | 1846.58 |
| 70                  |         593.39 | 2164.3  |     -6663.37 | 11146.4  |               -21956.7  | 2172.3  |
| 80                  |         747.89 | 2433.59 |     -9067.86 | 13376.8  |               -27058.8  | 2494.53 |
| 90                  |         936.19 | 2770.54 |    -11630    | 15703.3  |               -32502.3  | 2841.04 |
| 100                 |        1141.58 | 3115.1  |    -14563.4  | 17775.6  |               -38272.3  | 3147.91 |

### RLSO: GD and Random without retraining
| dimension           |   Score - Adam |   sigma |   Score - GD |    sigma |   Score - Random Search |   sigma |
|---------------------|----------------|---------|--------------|----------|-------------------------|---------|
| 5, 10, 15, mixed    |        -226.1  |  283.78 |      -411.55 |   865.16 |                -1165.16 |  839.44 |
| 5                   |         -60.38 |   74.39 |      -111.65 |   234.54 |                 -252.36 |  116.18 |
| 7                   |        -113.78 |  126.03 |      -198.16 |   391.29 |                 -519.94 |  178.55 |
| 10                  |        -210.73 |  210.13 |      -372.76 |   679.32 |                -1040.55 |  272.1  |
| 12                  |        -281.88 |  276.89 |      -504.86 |   876.25 |                -1447.6  |  343.08 |
| 15                  |        -395.87 |  367.77 |      -716.78 |  1215.81 |                -2145.58 |  444.03 |
| 18                  |        -521.18 |  471.1  |      -957.12 |  1588.44 |                -2933.52 |  541.92 |
| 20                  |        -597.98 |  559.75 |     -1158.71 |  1864.14 |                -3483.22 |  624.32 |
| 25                  |        -766.96 |  785.52 |     -1620.01 |  2549.24 |                -4995.57 |  851.31 |
| 30                  |        -907.07 | 1033.99 |     -2111.88 |  3345.89 |                -6633.03 | 1099.83 |
| 40                  |       -1020.52 | 1625.07 |     -3254.24 |  5001.7  |               -10210.4  | 1715.38 |
| 50                  |        -882.48 | 2348.01 |     -4505.57 |  6826.59 |               -14024.6  | 2411.77 |
| 60                  |        -403.56 | 3195.91 |     -5595.49 |  8834.14 |               -18050    | 3285.79 |
| 70                  |         450.68 | 4114.26 |     -6810.32 | 10891.9  |               -22039.5  | 4216.62 |
| 80                  |        1824.44 | 5147.93 |     -8223.07 | 12930.7  |               -25936.9  | 5266.32 |
| 90                  |        3300.25 | 6150.99 |     -9319.46 | 15316.5  |               -30167.7  | 6277.7  |
| 100                 |        5406.74 | 7128.05 |    -10544.6  | 17457.6  |               -34056.2  | 7227.16 |

### RLSO: GD and Random with retraining
| dimension           |   Score - Adam |   sigma |   Score - GD |    sigma |   Score - Random Search |   sigma |
|---------------------|----------------|---------|--------------|----------|-------------------------|---------|
| 5, 10, 15, 25 mixed |        -372.68 |  513.86 |      -722.92 |  1584.12 |                -2121.98 | 1879.46 |
| 5                   |         -59.28 |   74.55 |      -117.52 |   246.22 |                 -251.07 |  113.97 |
| 7                   |        -114.76 |  125.53 |      -207.53 |   401.56 |                 -519.72 |  180.59 |
| 10                  |        -207.79 |  206.83 |      -365.44 |   664.59 |                -1034.56 |  269.8  |
| 12                  |        -281.46 |  269.07 |      -509.45 |   885.7  |                -1448.58 |  336.55 |
| 15                  |        -402.2  |  366.73 |      -712.18 |  1217.57 |                -2149.52 |  436.76 |
| 18                  |        -523.15 |  472.27 |      -958.36 |  1566.09 |                -2933.61 |  539.75 |
| 20                  |        -608.57 |  542.7  |     -1137.22 |  1853.89 |                -3495.36 |  605.23 |
| 25                  |        -795.56 |  755.43 |     -1615.45 |  2524.61 |                -5020    |  811.08 |
| 30                  |        -973.88 |  990.18 |     -2311.18 |  3394.68 |                -6688.25 | 1056.68 |
| 40                  |       -1237.38 | 1567.67 |     -3430.15 |  4992.78 |               -10449.1  | 1641.02 |
| 50                  |       -1314.71 | 2259.39 |     -4908.63 |  6734.34 |               -14483.1  | 2368.15 |
| 60                  |       -1153.02 | 3116.67 |     -6390.82 |  8732.64 |               -18794.3  | 3203.42 |
| 70                  |        -744.7  | 4123.61 |     -8022.7  | 10767.2  |               -23248.7  | 4238.39 |
| 80                  |          18.68 | 5172.69 |     -9680.52 | 12920.9  |               -27793.5  | 5307.09 |
| 90                  |        1083.3  | 6262.26 |    -11466.4  | 15054.6  |               -32435.1  | 6423.54 |
| 100                 |        2599.92 | 7358.91 |    -13065.8  | 17162.5  |               -36902.4  | 7567.48 |




## Contributing


