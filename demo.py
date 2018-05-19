#!coding=utf8

import numpy as np
from OS_ELM import OS_ELM

if __name__ == '__main__':
	raw = np.loadtxt(open("data/test.csv", "r"), delimiter=",", skiprows=1)
	elm = OS_ELM(hidden_neuron=200, input_neuron=19)
	network = elm.fit_init(data=raw)