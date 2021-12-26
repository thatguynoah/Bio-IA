# this file is intended to be an optimized/completed version of the origininal t-test sim done in jupyter notebooks.

# This is being done for a few reasons:
# 1. code is messy in the notebook - by compiling here I can be more clean
# 2. unoptimized - This is the larger reason, for 1000 entries of actual biological data with 50,000 points, it will take hours to compute
# considering I want to run multiple sims, this is not good practice

# to optimize the code, I will attempt a few ideas
# 1. define variables outside of multiple functions - this will stop functions from redefining redundant variables, I'm not sure how applicable this is
# 2. cythonize - by defining c types, hopefully it reduces the strain of pythons slow dynamic interpreter
# 3. multiprocessing - by splitting up the task between multiple cores, hopefully the task will run faster

#imports
import math
import statistics
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy import stats
import pandas as pd
from numba import jit
import time

#checker function for graphing
@jit()
def GraphDist(stdev, mean):
    # 100 linearly spaced numbers
    x = np.linspace(-5, 5, 100)

    # build equation
    y = (1 / (stdev * (math.sqrt(2 * math.pi)))) * math.pi ** (-.5 * ((x - mean) / stdev) ** 2)

    # setting the axes at the centre
    fig = plt.figure()

    # plot the function
    plt.plot(x, y, 'r')

    # show the plot
    plt.show()

@jit()
def MonteCarlo(stdev, mean, points, rangeVal, bin1):
    # generate x vals:
    xVals = []
    for x in range(points):
        a = random.randint(round(rangeVal[0], 0), round(rangeVal[1] - 1, 0)) + random.random()
        if abs(round(rangeVal[0], 0)) - abs(rangeVal[0]) != 0:
            divider = abs(abs(round(rangeVal[0], 0)) - abs(rangeVal[0]))
            add = random.random() * divider
            a = a + add
        xVals.append(a)

    # define bins
    valsRange = abs(rangeVal[0]) + abs(rangeVal[1])
    bins = np.arange(rangeVal[0], rangeVal[1] + valsRange / bin1, valsRange / bin1)

    # fit each x val into bin counter
    tally = [0] * (len(bins) - 1)
    for x in xVals:
        index = 0
        for y in range(len(bins) - 1):
            if x <= bins[y + 1] and x >= bins[y]:
                tally[index] += 1
            index += 1

    # get y val for each bin
    yVals = []
    for x in bins:
        y = (1.0 / (stdev * (math.sqrt(2.0 * math.pi)))) * math.pi ** (-.5 * ((x - mean) / stdev) ** 2.0)
        yVals.append(y)

    # weighted sum of each bin
    weightedSum = []
    for x in range(len(tally)):
        weightedSum.append(tally[x] * yVals[x])

    return bins, weightedSum

@jit()
def NormalizeDistro(yVals):
    totalVal = sum(yVals)
    factor = 100 / totalVal
    normalizedVals = []

    for y in yVals:
        app = y * factor
        normalizedVals.append(round(app, 0))

    return normalizedVals

# convert sampling normal dist into a list of normal distribution - recommended to use un-normalized distribution
@jit()
def MakeDist(xVals, yVals):
    # test space - building t-test distribution
    # take the popularity and create list where each xVal is added that many times, relatively creating the normal dist

    intDist = []
    #create list of lists
    for x in range(len(yVals)):
        app = [xVals[x]]*int(round(yVals[x], 0))
        intDist.append(app)

    #flatten the list (make it one dimensional)
    submitDist = []
    for x in intDist:
        for y in x:
            submitDist.append(y)

    #round the numbers to 2 decimal points for faster computation
    for x in range(len(submitDist)):
        submitDist[x] = round(submitDist[x], 2)

    return submitDist

#perform calculation
@jit()
def tTest(dist1, dist2):
    t_value,p_value = stats.ttest_ind(dist1,dist2)
    return t_value, p_value


# putting it all together
@jit()
def RandomTtest(stdev, mean, points, rangeVal, bins, stdev2, mean2, points2, rangeVal2, bins2):
    # generate distributions along curve
    # generate distribution 1
    xVals, yVals = MonteCarlo(stdev, mean, points, rangeVal, bins)

    # generate distribution 2
    xVals2, yVals2 = MonteCarlo(stdev2, mean2, points2, rangeVal2, bins2)

    # convert into usable histogramic distributions
    # convert distribution 1
    dist1 = MakeDist(xVals, yVals)

    # convert distribution 2
    dist2 = MakeDist(xVals2, yVals2)

    # perform the student's t test
    t_value, p_value = tTest(dist1, dist2)

    return t_value, p_value

if __name__=="__main__":
    t0 = time.time()
    t, p = RandomTtest(1, 0, 1_000_000, [-5, 5], 100, 1, 0, 1_000_000, [-5, 5], 100)
    t1 = time.time()
    print(p, t1-t0)

#final solution was to use numba instead of cython
#using the time module, the numba time was roughly 1.83 seconds compared to the normal 48.85 under these parameters t, p = RandomTtest(1, 0, 1_000_000, [-5, 5], 100, 1, 0, 1_000_000, [-5, 5], 100)
#that's an increase of over 26x!!