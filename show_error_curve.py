from matplotlib import pyplot as plt
import seaborn as sns
import sys
import pickle

filename = sys.argv[1]
errG, errD = pickle.load(open(filename, 'rb'), encoding='latin1')

print(errD)
