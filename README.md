# RL_TP1

This code's aim is to answer Alessandro Lazaric and Emilie Kaufmann's assignments for the Reinforcement course of the MVA. The instructions for the assignments can be found on Emilie Kaufmann's page: http://chercheurs.lille.inria.fr/ekaufman/teaching.html.

Files are organized according to the questions. 
 * q1.py contains the implementation of the TreeCut problem: simulating trajectories and implementation of the Markov Decision Problem.
 * q2.py contains the implementation of a Policy, and of estimations of the corresponding value function. Two methods are implemented: 
    * Monte Carlo estimator
    * Fixed point for the Bellman operator estimator
 * q3.py implements two algorithms for optimal policy finding:
    * Value Iteration
    * Policy Iteration
 * main.py contains scripts for running the other files.
 
 It uses Elie Michel's version of Emilie Kaufmann's code, transfered from MATLAB to Python.
 
 The code is distributed under the MIT license:
 
 Copyright (c) 2016 Geoffrey Négiar

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
