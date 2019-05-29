# Graphene nanobubbles

This code is developed to calculate equilibrium graphene nanobubble with given mass of trapped substance. Basically, it minimises total energy of the bubble:

![energy equation](http://latex.codecogs.com/gif.latex?E_%7Btotal%7D%20%3D%20E_%7Belastic%7D%20&plus;%20E_%7BvdW%7D%20&plus;%20E_%7Bsubstance%7D)

You can find detailed information about the algorithm in the [article](https://iopscience.iop.org/article/10.1088/1361-6528/ab061f/meta).

## Usage

The usage is as simple as

```
python bubble.py <mass>
```

This calculates equilibrium bubble with given trapped mass of substance. It stores information of searching algorithm in &lt;mass>.csv file, and the information of equilibrium bubble in data.csv.

## Requirements

* python 3
* numpy == 1.15.4
* scipy == 1.1.0
