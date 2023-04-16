""" Just to test things """
# import numpy as np
# import projects.xsabr_fit.sabrgenerator as sabr


alpha = [1, 2, 3, 4]
beta = [0.1, 0.2, 0.3, 0.4]

gamma = []

for (a, b) in zip(alpha, beta):
    gamma.append(a + b)

print(gamma)
