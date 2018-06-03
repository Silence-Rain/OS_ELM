#!coding:UTF-8

import numpy as np

# 在线顺序极限学习机
# 构造参数：隐层节点数，输入节点数，模型id（用于在输出中标识不同的模型，默认为空），归一化方法（默认0-1归一化）
class OS_ELM(object):
	def __init__(self, hidden_neuron, input_neuron, id="", norm="0-1"):
		self.num_hidden_neurons = hidden_neuron
		self.num_input_neurons = input_neuron
		self.id = id
		self.norm = norm
		self.clear()

	def clear(self):
		self.Iw = np.mat(np.random.rand(self.num_hidden_neurons, self.num_input_neurons) * 2 - 1)
		self.bias = np.mat(np.random.rand(1, self.num_hidden_neurons))
		self.M = None
		self.beta = None

	# 0-1归一化
	def zeroone(self, x, min, max):
		return (x - min) / (max - min)

	# 使用0-1归一化，处理除label以外所有列的数据
	# 参数：初始训练数据（np.array），label列的下标（默认为-1，无下标）
	def normalize(self, data, label_index=-1):
		for ind in range(1,len(data[0])):
			if ind != label_index:
				maxi = np.max(data[:,ind:ind+1])
				mini = np.min(data[:,ind:ind+1])
				for i in range(len(data)):
					data[i][ind] = self.zeroone(data[i][ind], mini, maxi)

		return data

	# 激活函数
	# 参数：tData: 样本矩阵
	#	Iw: 输入层权重
	#	bias: 隐层单元偏置
	# 返回值：隐层输出矩阵
	def sig(self, tData, Iw, bias):
		#样本数*隐含神经元个数
		v = tData * Iw.T
		bias_1 = np.ones((len(tData), 1)) * bias
		v = v + bias_1
		H = 1./(1 + np.exp(-v, dtype="float64"))
		return H

	# 获取数据中label列的取值范围
	# 参数：label列
	# 返回值：（label取值区间长度，label最小值）
	def get_label_range(self, label):
		bot = min(label)
		top = max(label)
		return top - bot + 1, bot

	# 使用初始数据训练网络
	# 参数：初始训练数据（np.array），label列的下标（默认为0）
	# 返回值：训练后的网络
	def fit_init(self, data, label_index=0):
		label = []
		matrix = []
		# 归一化数据
		if self.norm == "0-1":
			data = self.normalize(data, label_index)
		data_size = len(data)
		# 划分训练集和测试集
		train_set = data[:int(0.8 * data_size)]
		test_set = data[int(0.8 * data_size):]
		print(len(train_set), len(test_set))
		for row in train_set:
			# 记录样本label
			temp = []
			label.append(int(row[label_index]))
			# 获取特征数据
			for index, item in enumerate(row):
				if index != label_index:
					temp.append(item)
			matrix.append(temp)
			
		# 获得训练数据中label取值范围，记为网络的一个参数
		self.ran = self.get_label_range(label)
		p0 = np.mat(matrix)
		T0 = np.zeros((len(matrix), self.ran[0]))
		# 处理样本标签
		for index, item in enumerate(label):
			T0[index][item - self.ran[1]] = 1
		T0 = T0 * 2 - 1
		# 计算隐层输出矩阵
		H0 = self.sig(p0, self.Iw, self.bias)
		self.M = (H0.T * H0).I
		# 计算输出权重
		self.beta = self.M * H0.T * T0

		# 训练准确率不足80%或测试准确率不足70%，则重新训练
		train_err = self.error_calc(train_set, text="%s训练" % self.id)
		test_err = self.error_calc(test_set, text="%s测试" % self.id)
		if train_err < 0.80 or test_err < 0.70:
			self.clear()
			return False
		else:
			return self
		# return self

	# 使用在线数据更新网络
	# 参数：在线训练数据（np.array），label列的下标（默认为0）
	# 返回值：更新后的网络
	def fit_train(self, data, label_index=0):
		# 归一化数据
		if self.norm == "0-1":
			data = self.normalize(data, label_index)
		# 逐条使用数据，对网络进行更新
		for row in data:
			Tn = np.zeros((1, self.ran[0]))
			# 处理样本标签
			b = int(row[0])
			Tn[0][b - self.ran[1]] = 1
			Tn = Tn * 2 - 1
			# 获取特征数据
			matrix = []
			for index, item in enumerate(row):
				if index != label_index:
					matrix.append(item)
			pn = np.mat(matrix)
			# 更新隐层输出矩阵
			H = self.sig(pn, self.Iw, self.bias)
			self.M = self.M - self.M * H.T * (np.eye(1,1) + H * self.M * H.T).I * H * self.M
			# 更新输出权重
			self.beta = self.beta + self.M * H.T * (Tn - H * self.beta)

		return self

	# 使用现有模型对数据分类
	# 参数：需要分类的数据（np.array）
	# 返回值：预测的label行
	def predict(self, data):
		res = []
		# 归一化数据
		if self.norm == "0-1":
			data = self.normalize(data)
		for row in data:
			# 处理特征
			matrix = []
			for item in row:
				matrix.append(item)
			p = np.mat(matrix)
			HTrain = self.sig(p, self.Iw, self.bias)
			Y = HTrain * self.beta
			# 返回预测label
			res.append(np.argmax(Y) + self.ran[1])

		return res

	# 计算训练的误差值
	# 参数：训练数据，label列的下标（默认为0），输出内容的附加文本（默认为模型id）
	def error_calc(self, data, label_index=0, text=""):
		# 附加文本默认为模型id
		if len(text) == 0:
			test = self.id
		correct = 0
		sum = 0
		for row in data:
			# 处理特征
			matrix = []
			for index, item in enumerate(row):
				if index != label_index:
					matrix.append(item)
			p = np.mat(matrix)
			HTrain = self.sig(p, self.Iw, self.bias)
			Y = HTrain * self.beta
			# 若预测结果和实际结果相同则计数
			if np.argmax(Y) + self.ran[1] == int(row[label_index]):
				correct += 1
			sum += 1
		print("%s准确性为：%f" % (text, correct / sum))
		return correct / sum


if __name__ == '__main__':
	raw=np.loadtxt(open("data/trainData1111.csv","r"),delimiter=",",skiprows=0)
	elm = OS_ELM(hidden_neuron=60, input_neuron=4, id="local_detection", norm="no")
	network = elm.fit_init(data=raw)
	while not network:
		np.random.shuffle(raw)
		network = elm.fit_init(data=raw)

	res = network.predict(data=test)
	print(res)
