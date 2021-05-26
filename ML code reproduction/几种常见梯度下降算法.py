#-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
#写一个算法需要理清楚
#1、算法流程
#2、算法目标、算法初始态、算法转换态、算法迭代方式
#3、算法的输入、中间数据、输出数据（物理意义）如何转成数据格式
#    （是数字，数组还是向量等）
#4、用什么数据结构去描述算法中的数据
#5、算法的数学推导简化出来
#6、对于图像机器学习深度学习中要搞清楚数据维度
#7、要知道推导出来的公式在哪个地方带入，推导出来的公式是不是要放在迭代循环中
#8、遇到有下标的公式，一定要清楚下标含义，是不是一个通项式中的某一个用了j表示
#    在计算的时候，是不是每一项都得求，是单独一项一项的求，还是矩阵直接求了所有项

#注意：写算法最需要花时间的是理解清楚迭代和数据怎么表达
#尽可能抛出所有有歧义（理解不清楚的问题）去思考

#-------------------------------------------
#以逻辑回归为例，写几个梯度下降算法
#梯度算法目标：求取最优解，即使得目标函数最小化的参数模型


#-------------------------------------------
#假设给出样本特征X,样本标签Y
#主程序
if __name__=="__main__":
	#假设已经给出的
	train_data_X,train_label_Y
	#获取样本个数
	m=train_X.shape[0]
	#axis=1表示对应行的数组进行拼接
	#思考：此处为什么加np.ones((m,1))
	#求函数模型截距 即参数b
	#函数应该是hθ(x)=b + θ0*x0+θ1*x1+.....+
	X=np.concatenate((np.ones((m,1)),train_data_X),axis=1)
	#需要传入学习率、最小误差、最大迭代次数三个参数，此处学习率是固定的，其实也可以设置的
	thetas,costs,iterationCount=gradient_sgd(X, train_label_Y, alpha=0.01, epsilon=0.00001, maxloop=1000)

	
#需要搞清楚，需要的是什么样式的数据，单个样本，还是多个样本，还是单个值
#这里也是多个样本一起计算的
def sigmoid(x):
	return 1.0/(1+np.exp(-x))
	
	
#这个函数是所有样本一起计算，python强大的数组能力可以让我们一起计算
#不排除也是可以单独计算再累起来
def J(theta,X,Y,lamda=0):#lambda是正则化因子
	m,n=X.shape
	#先计算模型输出y_,即根据hθ(x)=θ0*x0+θ1*x1+.....+计算
	#这里是矩阵乘法
	y_=np.dot(X,theta)
	#在通过激活函数给出最终判决概率
	h=sigmoid(y_)
	#可以写损失函数了,使用的是交叉熵损失函数
	J=(-1.0/m)*(np.dot(np.log(h).T,Y)+np.dot(np.log(1-h).T,1-Y)) /
		+(Lambda/(2.0*m))*np.sum(np.square(theta[1:]))
		#其实这个推导公式就是对所有的样本求损失，最终只有一个值
		
	#后面一部分是加入的正则项，用来约束θ值过大
	#这里是从theta1开始的取的，说明只用来约束theta1 2 3... 
	#实际中从0开始也是没多大影响的
	
	#其实J里面只有一个数值，需要取出该数值，可以比划一下维度
	return J.flatten()[0] 
		
	
#--------------------------随机梯度下降求解算法---------------------------------------------
#算法中有两个循环，迭代次数和所有样本便利完	
def  gradient_sgd(X,Y,train_label_Y, alpha=0.01, epsilon=0.00001, maxloop=1000):
	m,n=X.shape
	
	#初始化参数θ，不一定得初始化为0,也是可以其他值的，一般都是0开始
	#参数的维度和样本维度有关，因为hθ(x)=b+θ0*x0+θ1*x1+.....
	# 初始化模型参数，n个特征对应n个参数
	theta=np.zeros((n,1))
	
	#给了参数，就有了决策函数，可以计算根据损失函数计算损失
	cost=J(theta,X,Y)# 当前所有样本的误差
	costs=[cost,]# 存放迭代中的每一轮的误差
	thetas=[theta]
	
	#迭代次数
	cout=0;
	#如果精度已经满足了就直接停止
	flag=False
	
	#开始迭代
	while count<maxloop:
		if flag:
			break
			
		#每次只使用单个训练样本来更新θ参数，依次遍历训练集，而不是一次更新中考虑所有的样本？？
		for i in range(m)：
			h=sigmoid(np.dot(X[i].reshape((1,n)),theta))
			#此处是直接带入推导公式
			#沿着梯度方向更新模型参数是最快收敛到最小值的方法
			#theta是n*1的矩阵？
			theta=theta - alpha * ((1.0 / m) * X[i].reshape((n, 1)) * (h - Y[i]) /
					+ (Lambda / m) * np.r_[[[0]], theta[1:]])
	
			#theta值被更新了
			thetas.append(theta)
			#用新的thea求损失
			cost=J(theta,X,Y,Lambda)
			costs.append(cost)
			if abs(costs[-1]-costs[-2])<epsilon:
				flag=True
				break
				
		cout+=1
		#没100次打印一次数据
		if count % 100==0:
			print("cost: ",cost)
	
	return thetas,costs,count


		
#--------------------------批量梯度下降求解算法---------------------------------------------
	
def  gradient_bgd(X,Y,train_label_Y, alpha=0.01, epsilon=0.00001, maxloop=1000):
	
	#跟以上的区别在于，不是单个样本更新一次，而是随机选择多个样本进行梯度更新
		
	


#--------------------------自适应梯度（AdaGrad）下降求解算法--------------------------------------------	
#实际中不同参数的重要性不同，所以对不同参数进行动态调整更新，采取不同的学习率
#将每一个参数的每一次迭代的梯度取平方，然后累加并开方得到 r，最后用全局学习率除以 r，作为学习率的动态更新。
#令 α 表示全局学习率，r 为梯度累积变量，初始值为 0

def gradient_adagrad(X,Y,alpha=0.01,sigmoid=1e-7,epsilon=0.00001, maxloop=1000, Lambda=0.0):

	m,n=X.shape
	
	theta=np.zeros((n,1))
	
	#r是梯度累积变量，初始值为0
	#这是list中嵌套list结构
	r=[[0.0] for _ in range(n)]
	
	cost=J(theta,X,Y)
	costs=[cost]
	thetas=[theta]
	
	count=0
	flag=False
	#开始迭代
	while count<maxloop:
		if flag:
			break
		#还是以每个样本进行更新
		for i in range(m):
			h=sigmoid(np.dot(X[i].reshape((1,n)),theta)
			#求梯度
			grad=(1.0/m)*X[i].reshape((n,1))*(h-Y[i])
			#将每一个参数的每一次迭代的梯度取平方，然后累加并开方得到 r，最后用全局学习率除以 r，作为学
			for j in range(n):
				r[j].append(grad[j]**2+r[j][-1])
				theta[j]=theta[j]-alpha*grad[j]/(sigma + math.sqrt(r[j][-1]))

			thetas.append(theta)
			cost=J(theta,X,Y,lambda)
			costs.append(cost)
			
			if abs(costs[-1]-costs[-2])<epsilon:
				flag=True
				break
					
		cout+=1
		
		#没100次打印一次数据
		if count % 100==0:
			print("cost: ",cost)	

#--------------------------牛顿法求解算法---------------------------------------------------
def  gradient_sgd(X,Y,train_label_Y, alpha=0.01, epsilon=0.00001, maxloop=1000):
	m,n=X.shape
	

	theta=np.zeros((n,1))
	

	cost=J(theta,X,Y)# 当前所有样本的误差
	costs=[cost,]# 存放迭代中的每一轮的误差
	thetas=[theta]
	
	#迭代次数
	cout=0;
	#如果精度已经满足了就直接停止
	flag=False
	
	#开始迭代
	while count<maxloop:
	
		#这里使用牛顿法更新参数
		#初始化
		delta_J=0.0
		H=0.0
		#这里便利所有样本是公式需求，这里不是每个样本更新			
		for i in range(m)：
		
			h=sigmoid(np.dot(X[i].reshape((1,n)),theta))
			#根据公式推导写出的
			delta_J+=X[i]*(h-Y[i])
			#根据公式推导写出的
			H+=h.T*(1-h)*X[i]*X[i].T
		
		#根据公式推导写的
		delta_J/=m
		H/=m
		
		print(H,delta_J)
		
		#此处才到了牛顿法的迭代公式
		theta = theta - 1.0 / H * delta_J #这里应该还有一个lamda
		thetas.append(theta)
		cost=J(theta,X,Y,Lambda)
		costs.append(cost)
		
		if abs(costs[-1]-costs[-2])<epsilon:
			flag=True
			break
				
		cout+=1
		#没100次打印一次数据
		if count % 100==0:
			print("cost: ",cost)
	
	return thetas,costs,count
	

#----------------------------------------------------------------------------
#上述的方案学习率都是确定的，只研究了梯度下降幅度权重
#下面研究调整学习率，优化算法收敛速度
#----------------------------------------------------------------------------


#--------------------------AdaDelta算法--------------------------------------
#AdaDelta算法是对Adagrad算法的扩展，
def gradient_adadelta(X,Y,rho=0.01,alpha=0.01,sigma=1e-7,epsilon=0.00001, maxloop=1000, Lambda=0.0):
	m,n=X.shape
	
	theta=np.zeros((n,1))
	
	r = [[0.0] for _ in range(n)]
	deltax = [[0.0] for _ in range(n)]
	deltax_ = [[1.0] for _ in range(n)]
	
	cost = J(theta, X, Y)
	costs = [cost]
	thetas = [theta]
	
	count = 0
	flag = False
	while count < maxloop:
		if flag:
			break
			
		for i in range(m):
			h = sigmoid(np.dot(X[i].reshape((1, n)), theta))
			grad = (1.0 / m) * X[i].reshape((n, 1)) * (h - Y[i])
			
			for j in range(n):
				#动态平均值
				r[j].append((1-rho) * grad[j]**2 + rho * r[j][-1])
				
				deltax[j].append(- (math.sqrt(deltax_[j][-1] / sigma + r[j][-1]))*alpha)
				#更新参数
				theta[j] = theta[j] + deltax[j][-1]
				
				deltax_[j].append((1-rho)*deltax[j][-1]**2+rho*deltax_[j][-1])
				
			thetas.append(theta)
			cost=J(theta,X,Y,Lambda)
			cost.append(cost)
			if abs(costs[-1]-costs[-2])<epsilon:
				flag=True
				break
		cout+=1
		
		if count % 100 == 0:
		print("cost:", cost)

    return thetas, costs, count



#--------------------------RMSProp算法--------------------------------------
#因为Adagrad算法会出现提前停止的现象，RMSProp它采用指数加权平均的思想，
#只将最近的梯度进行累加计算平方	

def gradient_RMSProp(X, Y, rho=0.01, alpha=0.01, sigma=1e-7, epsilon=0.00001, maxloop=1000, theLambda=0.0):

    m, n = X.shape

    theta = np.zeros((n, 1))

    r = [[0.0] for _ in range(n)]


    cost = J(theta, X, Y)
    costs = [cost]
    thetas = [theta]

    count = 0
    flag = False
    while count < maxloop:
        if flag:
            break

        for i in range(m):
            h = sigmoid(np.dot(X[i].reshape((1, n)), theta))

            grad = (1.0 / m) * X[i].reshape((n, 1)) * (h - Y[i])

            for j in range(n):
                r[j].append((1 - rho)*grad[j]**2+rho*r[j][-1])
                theta[j] = theta[j] - alpha * grad[j] / (sigma + math.sqrt(r[j][-1]))

            thetas.append(theta)
            cost = J(theta, X, Y, theLambda)
            costs.append(cost)
            if abs(costs[-1] - costs[-2]) < epsilon:
                print(costs)
                flag = True
                break
        count += 1

        if count % 100 == 0:
            print("cost:", cost)
    return thetas, costs, count

	
#--------------------------Adam算法--------------------------------------
#Adam算法和传统的随机梯度下降不同。
#随机梯度下降保持单一的学习率（即alpha）更新所有的权重学习率在训练过程中并不会改变。
#Adam通过随机梯度的一阶矩估计和二阶矩估计而为不同的参数设计独立的自适应性学习率。

def gradient_adam(X, Y, rho1=0.9, rho2=0.999, alpha=0.01, sigma=1e-7, epsilon=0.00001, maxloop=1000, theLambda=0.0):

    m, n = X.shape

    theta = np.zeros((n, 1))
	#这是list中嵌套list结构
    s = [[0.0] for _ in range(n)]
    r = [[0.0] for _ in range(n)]

    cost = J(theta, X, Y)
    costs = [cost]
    thetas = [theta]

    count = 0
    flag = False
    while count < maxloop:
        if flag:
            break

        for i in range(m):
            h = sigmoid(np.dot(X[i].reshape((1, n)), theta))

            grad = (1.0 / m) * X[i].reshape((n, 1)) * (h - Y[i])

            for j in range(n):
                r[j].append((1 - rho2)*grad[j]**2+rho2*r[j][-1])
                s[j].append((1 - rho1)*grad[j]+rho1*r[j][-1])

                theta[j] = theta[j] - alpha * (s[j][-1]/(1-rho1**2))/(math.sqrt(r[j][-1]/(1-rho2**2))+sigma)
			
            thetas.append(theta)
            cost = J(theta, X, Y, theLambda)
            costs.append(cost)
            if abs(costs[-1] - costs[-2]) < epsilon:
                print(costs)
                flag = True
                break
        count += 1

        if count % 100 == 0:
            print("cost:", cost)
    return thetas, costs, count	
	




