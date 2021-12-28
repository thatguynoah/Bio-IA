#The goal of this file is to multithread the distributions before being fed into the t-test function
#imports
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
import concurrent.futures

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
        #xVals = np.append(xVals, a)

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
#@jit()
def tTest(dist1, dist2):
    t_value,p_value = stats.ttest_ind(dist1,dist2)
    return t_value, p_value


# putting it all together
#@jit()
def RandomTtest(stdev, mean, points, rangeVal, bins, stdev2, mean2, points2, rangeVal2, bins2):
    # generate distributions along curve
    # generate distribution 1
    xVals, yVals = MonteCarlo(stdev, mean, points, rangeVal, bins)

    # generate distribution 2
    xVals2, yVals2 = MonteCarlo(stdev2, mean2, points2, rangeVal2, bins2)

    # multi-threading to flatten dist
    # convert into usable histogramic distributions
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    future = executor.submit(MakeDist, xVals, yVals)
    future1 = executor.submit(MakeDist, xVals2, yVals2)
    dist1 = future.result()
    dist2 = future1.result()

    # perform the student's t test
    t_value, p_value = tTest(dist1, dist2)

    return t_value, p_value

if __name__=="__main__":
    #t0 = time.time()
    #for x in range(100):
        #t, p = RandomTtest(1, 0, 100_000, [-5, 5], 100, 1, 0, 100_000, [-5, 5], 100)
    #t1 = time.time()
    #print(p, t1-t0)

    # load in the expression values
    dfQual = pd.read_csv(r"C:\Users\noahb\OneDrive\Documents\Tensorflow_Tutorial\mini_project\NSCLC_Data",
                         index_col="Unnamed: 0")
    print(dfQual.head())

    # seperate the healthy and unhealthy samples into independent dataframes pt 1 - get list of
    # patient ids

    # import the file
    df = pd.read_csv(r"C:\Users\noahb\OneDrive\Documents\Tensorflow_Tutorial\mini_project\descriptors.txt")

    # create new df
    descriptors_df = pd.DataFrame(data=None, columns=['class', 'name'])  # create df with two columns
    for x in range(0, 310):  # iterate through 155 critical rows (somereason does each twice)
        row = []  # set appending row to zero
        temp = [str(df.iloc[x]), str(df.iloc[x + 1])]  # get a holder for the two values

        if "low" in temp[0]:  # get class for first entry
            row.append("low")
        elif "high" in temp[0]:
            row.append("high")
        else:
            continue

        if "GSM" in temp[1]:  # get GSM number from each entry, it's at a strange spot
            row.append(temp[1][temp[1].find('GSM'):temp[1].find('GSM') + 10])

        row_dict = {'class': row[0], 'name': row[1]}  # create appending dictionary
        descriptors_df = descriptors_df.append(row_dict, ignore_index=True)  # append dictionary to df

    # slight modification to index
    descriptors_df = descriptors_df.set_index('name')  # set name of patient as index to their status
    print(descriptors_df.head())

    # seperate the healthy and unhealthy samples into independent dataframes pt 2
    # create list of low risk samples
    lowRisk = []
    for x in range(len(descriptors_df.iloc[:, 0])):
        if descriptors_df.iloc[x, 0] == 'low':
            lowRisk.append(descriptors_df.index[x])

    # create list of high risk samples
    highRisk = []
    for x in range(len(descriptors_df.iloc[:, 0])):
        if descriptors_df.iloc[x, 0] == 'high':
            highRisk.append(descriptors_df.index[x])

    # create low risk data frame
    dfLow = dfQual.loc[:, lowRisk]  # 1000 ids

    # create high risk data frame
    dfHigh = dfQual.loc[:, highRisk]  # 1000 ids

    # reduce the number of rows (0-1000)
    dfHigh = dfHigh.iloc[0:10_000, :]
    dfLow = dfLow.iloc[0:10_000, :]

    # qualities element - [id, mean, standard deviation]
    # track qualities of dfLow
    qualitiesLow = []
    for x in range(len(dfLow.index)):
        z = dfLow.iloc[x, :]
        identity = dfLow.index[x]
        stdev = statistics.stdev(z)
        mean = statistics.mean(z)
        qualitiesLow.append([identity, stdev, mean])
    print(qualitiesLow[0:5])

    # qualities element - [id, mean, standard deviation]
    # track qualities of dfHigh
    qualitiesHigh = []
    for x in range(len(dfHigh.index)):
        z = dfHigh.iloc[x, :]
        identity = dfHigh.index[x]
        stdev = statistics.stdev(z)
        mean = statistics.mean(z)
        qualitiesHigh.append([identity, stdev, mean])
    print(qualitiesHigh[0:5])

    # call function in order: stdev1, mean1, num_points1, range_of_values1, number_of_bins1,
    # stdev2, mean2, num_points2, range_of_values2, number_of_bins2
    t0 = time.time()
    p_val = []
    for x in range(len(qualitiesLow)):  # range(len(qualitiesLow)):
        t, p = RandomTtest(qualitiesLow[x][2], qualitiesLow[x][1], 100_000,
                           [-37 + qualitiesLow[x][1], 37 + qualitiesLow[x][1]], 100, qualitiesHigh[x][2],
                           qualitiesHigh[x][1], 100_000, [-37 + qualitiesHigh[x][1], 37 + qualitiesHigh[x][1]],
                           100)
        p_val.append(p)
    t1 = time.time()
    print(len(p_val), t1-t0)

#final solution was to use numba instead of cython
#using the time module, the numba time was roughly 1.83 seconds compared to the normal 48.85 under these parameters t, p = RandomTtest(1, 0, 1_000_000, [-5, 5], 100, 1, 0, 1_000_000, [-5, 5], 100)
#that's an increase of over 26x!!
#multithreading the flattening algorithm had an impact of reducing the time with the same parameters to roughly 1.7 seconds
#unfortunately, multitasking the distribution making had no impact or even made the time worse for some trials, this is unexpected but interesting
#by reducing the sim to 100,000 points each dist, the time is reduced to roughly 1.5s, I think this is the best selection of hyperparameter
#also experimented with removing @jit on certain functions, got down to ~1.15s