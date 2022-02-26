#####################
# CS 181, Spring 2022
# Homework 1, Problem 4
# Start Code
##################

import csv
import numpy as np
import matplotlib.pyplot as plt

csv_filename = 'data/year-sunspots-republicans.csv'
years  = []
republican_counts = []
sunspot_counts = []

with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        years.append(float(row[0]))
        sunspot_counts.append(float(row[1]))
        republican_counts.append(float(row[2]))

# Turn the data into numpy arrays.
years  = np.array(years)
republican_counts = np.array(republican_counts)
sunspot_counts = np.array(sunspot_counts)
last_year = 1985

# Plot the data.
plt.figure(1)
plt.plot(years, republican_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.figure(2)
plt.plot(years, sunspot_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Sunspots")
plt.figure(3)
plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.show()

# Create the simplest basis, with just the time and an offset.
X = np.vstack((np.ones(years.shape), years)).T

# TODO: basis functions
# Based on the letter input for part ('a','b','c','d'), output numpy arrays for the bases.
# The shape of arrays you return should be: (a) 24x6, (b) 24x12, (c) 24x6, (d) 24x26
# xx is the input of years (or any variable you want to turn into the appropriate basis).
# is_years is a Boolean variable which indicates whether or not the input variable is
# years; if so, is_years should be True, and if the input varible is sunspots, is_years
# should be false
def make_basis(xx,part='a',is_years=True):
#DO NOT CHANGE LINES 65-69
    if part == 'a' and is_years:
        xx = (xx - np.array([1960]*len(xx)))/40
        
    if part == "a" and not is_years:
        xx = xx/20

    # Turn xx into column vector
    xx = np.atleast_2d(xx).T
    
    # Include bias term of 1
    phi = np.atleast_2d(np.ones(xx.shape[0])).T
    
    if part == 'a':
        # Let all the a_j = 1
        for j in range(1, 6):
            phi_j = xx ** j
            phi = np.hstack((phi, phi_j))

    elif part == 'b':
        for mu_j in range(1960, 2015, 5):
            phi_j = np.exp(-np.power(xx - mu_j, 2) / 25)
            phi = np.hstack((phi, phi_j))
        
    elif part == 'c':
        for j in range(1, 6):
            phi_j = np.cos(xx / j)
            phi = np.hstack((phi, phi_j))

    elif part == 'd':
        for j in range(1, 26):
            phi_j = np.cos(xx / j)
            phi = np.hstack((phi, phi_j))
            
    else:
        raise ValueError("Invalid part")
    
    return phi

# Nothing fancy for outputs.
Y = republican_counts

# Find the regression weights using the Moore-Penrose pseudoinverse.
def find_weights(X,Y):
    w = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))
    return w

# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!
grid_years = np.linspace(1960, 2005, 200)

# Year v. Number of Republicans in the Senate
for part in ['a', 'b', 'c', 'd']:
    # Find weights based on training data and print error
    X = make_basis(years, part, is_years=True)
    w = find_weights(X, Y)
    Yhat = np.dot(X, w)
    rss_error = np.sum(np.power(Y - Yhat, 2))
    print("The residual sum of squares error for part " + part +
          " is " + str(rss_error))

    # Calculate regression line
    grid_X = make_basis(grid_years, part, is_years=True)
    grid_Yhat = np.dot(grid_X, w)

    # Plot the data and the regression line.
    plt.title("Part " + part)
    plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat, '-')
    plt.xlabel("Year")
    plt.ylabel("Number of Republicans in Congress")
    plt.savefig(part + ".png", facecolor="white")
    plt.show()
    
# Number of Sunspots v. Number of Republicans in the Senate
for part in ['a', 'c', 'd']:
    # Only use data from before 1985
    republican_counts_pre_1985, sunspot_counts_pre_1985 = zip(*[(r, s) for y, r, s in zip(years, republican_counts, sunspot_counts) if y < 1985])
    republican_counts_pre_1985 = np.array(republican_counts_pre_1985)
    sunspot_counts_pre_1985 = np.array(sunspot_counts_pre_1985)
    
    # Find weights based on training data and print error
    X = make_basis(sunspot_counts_pre_1985, part, is_years=False)
    Y = republican_counts_pre_1985
    w = find_weights(X, Y)
    Yhat = np.dot(X, w)
    rss_error = np.sum(np.power(Y - Yhat, 2))
    print("The residual sum of squares error for part " + part +
          " is " + str(rss_error))

    # Calculate regression line
    grid_sunspots = np.linspace(0, 160, 200)
    grid_X = make_basis(grid_sunspots, part, is_years=False)
    grid_Yhat = np.dot(grid_X, w)

    # Plot the data and the regression line.
    plt.title("Part " + part)
    plt.plot(sunspot_counts_pre_1985, republican_counts_pre_1985, 'o', grid_sunspots, grid_Yhat, '-')
    plt.xlabel("Number of Sunspots")
    plt.ylabel("Number of Republicans in Congress")
    plt.savefig(part + "_sunspots.png", facecolor="white")
    plt.show()