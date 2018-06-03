#!coding=utf8

import numpy as np
from OS_ELM import OS_ELM

if __name__ == '__main__':
	raw = np.loadtxt(open("data/test.csv", "r"), delimiter=",", skiprows=1)
	# 按照4:1的比例分配训练集和测试集
	train = raw[:650]
	test = raw[650:]
	# 原始数据已经归一化过，不使用模型本身的归一化
	elm = OS_ELM(hidden_neuron=100, input_neuron=19, id="test_model", norm="no")
	network = elm.fit_init(data=train)
	# 重新训练模型直至准确率符合要求
	while not network:
		np.random.shuffle(train)
		network = elm.fit_init(data=train)

	res = network.predict(data=test[:,1:])
	print(res)