



class GMM:
	#高斯权重初始化
	def __init__(self.Data,k,weights=Noe,means=None,covars=None):
		"""
        这是GMM（高斯混合模型）类的构造函数
        :param Data: 训练数据
        :param K: 高斯分布的个数
        :param weigths: 每个高斯分布的初始概率（权重）
        :param means: 高斯分布的均值向量
        :param covars: 高斯分布的协方差矩阵集合
        """
		
		self.Data = Data
        self.K = K
		#初始化参数的维度和样本特征点维数有关
		col = np.shape(self.Data)[1]
        if weights is not None:
            self.weights = weights
        else:
            self.weights  = np.random.rand(self.K)
            self.weights /= np.sum(self.weights)        # 归一化
			
		if means is not None:
            self.means = means
        else:
            self.means = []
            for i in range(self.K):
                mean = np.random.rand(col)
                #mean = mean / np.sum(mean)        # 归一化
                self.means.append(mean)
				
		if covars is not None:
            self.covars = covars
        else:
            self.covars  = []
            for i in range(self.K):
                cov = np.random.rand(col,col)
                #cov = cov / np.sum(cov)                    # 归一化
                self.covars.append(cov)                     # cov是np.array,但是self.covars是list
				
				
	def Gaussian(self,x,mean,cov):
        """
        这是自定义的高斯分布概率密度函数
        :param x: 输入数据
        :param mean: 均值数组
        :param cov: 协方差矩阵
        :return: x的概率
        """
        dim = np.shape(cov)[0]
        # cov的行列式为零时的措施
		#np.linalg.det数组a的行列式
        covdet = np.linalg.det(cov + np.eye(dim) * 0.001)
		#np.linalg.inv()：矩阵求逆
        covinv = np.linalg.inv(cov + np.eye(dim) * 0.001)
		#求的是x-u
        xdiff = (x - mean).reshape((1,dim))
        #根据公式求概率密度
		#需要知道协方差、均值、维度数
        prob = 1.0/(np.power(np.power(2*np.pi,dim)*np.abs(covdet),0.5))*\
               np.exp(-0.5*xdiff.dot(covinv).dot(xdiff.T))[0][0]
		#为什么取[0][0]？
		#什么时候用np.dot()
        return prob
		
	def GMM_EM(self):
        """
        这是利用EM算法进行优化GMM参数的函数
        :return: 返回各组数据的属于每个分类的概率
        """
        loglikelyhood = 0
        oldloglikelyhood = 1
        len,dim = np.shape(self.Data)
        # gamma表示第n个样本属于第k个混合高斯的概率
        gammas = [np.zeros(self.K) for i in range(len)]
        while np.abs(loglikelyhood-oldloglikelyhood) > 0.00000001:
            oldloglikelyhood = loglikelyhood
            # E-step
            for n in range(len):
                # respons是GMM的EM算法中权重为w的后验概率
                respons = [self.weights[k] * self.Gaussian(self.Data[n], self.means[k], self.covars[k])
                                                    for k in range(self.K)]
                respons = np.array(respons)
                sum_respons = np.sum(respons)
                gammas[n] = respons/sum_respons
            # M-step
            for k in range(self.K):
                #nk表示N个样本中有多少属于第k个高斯
                nk = np.sum([gammas[n][k] for n in range(len)])
                # 更新每个高斯分布的概率
                self.weights[k] = 1.0 * nk / len
                # 更新高斯分布的均值
                self.means[k] = (1.0/nk) * np.sum([gammas[n][k] * self.Data[n] for n in range(len)], axis=0)
                # 更新高斯分布的协方差矩阵
				xdiffs = self.Data - self.means[k]
                self.covars[k] = (1.0/nk)*np.sum([gammas[n][k]*xdiffs[n].reshape((dim,1)).dot(xdiffs[n].reshape((1,dim))) for n in range(len)],axis=0)
            loglikelyhood = []
            for n in range(len):
                tmp = [np.sum(self.weights[k]*self.Gaussian(self.Data[n],self.means[k],self.covars[k])) for k in range(self.K)]
                tmp = np.log(np.array(tmp))
                loglikelyhood.append(list(tmp))
            loglikelyhood = np.sum(loglikelyhood)
        for i in range(len):
            gammas[i] = gammas[i]/np.sum(gammas[i])
        self.posibility = gammas
        self.prediction = [np.argmax(gammas[i]) for i in range(len)]
		
		
if __name__=="__main__":
# GMM模型
    K = 3
    gmm = GMM(data,K)
    gmm.GMM_EM()
    y_pre = gmm.prediction

	
	