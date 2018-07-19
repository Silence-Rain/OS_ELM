#!coding=utf8

import numpy as np
from OS_ELM import OS_ELM

if __name__ == '__main__':
	# 读取数据
	raw = np.loadtxt(open("data/test.csv", "r"), delimiter=",", skiprows=1)
	# 按照4:1的比例分配训练集和测试集
	train = raw[:650]
	test = raw[650:]
	# 设置模型参数：
	# 隐层节点：100
	# 输入节点：19（根据输入数据的维度确定）
	# 模型id：test_model
	# 不进行归一化处理（原始数据已经归一化）
	elm = OS_ELM(hidden_neuron=100, input_neuron=19, id="test_model", norm="no")
	# 训练模型
	network = elm.fit_init(data=train)
	# 输出模型预测结果
	res = network.predict(data=test[:,1:])
	print(res)