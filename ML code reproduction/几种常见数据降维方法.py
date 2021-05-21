
#写公式一定要搞清楚下标怎么解决

#降维是为了解决维数灾难，将特征空间从一个高维空间变换到另一个低维空间




#--------------------------------------------------------------------
#---------MDS:MultiDimensional Scaling 多维尺度变换------------------


def calculate_distanc(x,y):
	d=np.sqrt(np.sum((x-y)**2)
	return d
	
# 计算矩阵各行之间的欧式距离；x矩阵的第i行与y矩阵的第0-j行继续欧式距离计算，构成新矩阵第i行[i0、i1...ij]	
def calculate_distanc_matrix(x,y):
	d=metrics.pairwise_distance(x,y)
	return d

def  cla_B(D):
	(n1,n2)=D.shape
	DD=np.square(D)
	Di=np.sum(DD,axis=1)/n1
	Dj=np.sum(DD,axis=0)/n1
	Dij=np.sum(DD)/(n1**2)
	B=np.zeros((n1,n1))
	for i in range(n1):
		for j in range(n2):
			B[i,j]=(-1/2)*(Dij+DD[i,j]-Di[i]-Dj[j])
	
	return B
	
def MDS(data,n=2):
	D=calculate_distanc_matrix(data,data)
	B=cal_B(D)
	#奇异值分解
	Be,Bv=np.linalg.eigh(B)
	Be_sort=np.argsort(-Be)
	#特征值从大到小排序
	Be=Be[Be_sort]
	Bv=Bv[:,Be_sort]
	# 前n个特征值对角矩阵
	Bez=np.diag(Be[0:n])
	 # 前n个归一化特征向量
	Bvz=Bv=[:,0:n]
	Z=np.dot(np.sqrt(Bez),Bvz.T).T
	return Z
	
	
#-----------------------------------------------------------
#----------------------------ISOMAP-------------------------
#与MDS区别：ISOMAP用图中两点的最短路径距离代替MDS中欧氏空间距离
#输入：样本集D={x1,x2,...,xm},近邻参数k,低维空间维数
#过程：
#	for i =1,2,3,...,m 
#		do 确定xi的k近邻
#		xi与k近邻点之间的距离设置为欧氏距离，与其他点距离设置为无强大
#    end  for
# 调用最短路径算法计算任意两个样本点之间的距离dist(xi,xj)
# 将dist(xi,xj)作为MDS的算法的输入
# return MDS算法的输出
#输出： 样本集D在低维空间的投影Z={z1,z2,..,}

#算法描述的时候是任意两个点，在代码和应用中，是样本中特征维度与特征维度

def my_mds(dist,n_dims):
	
	n=dist.shape[0]
	
	dist=dist**2
	T1=np.ones((n,n))*np.sum(dist)/n**2
	T2=np.sum(dist,axis=1)/n
	T3=np.sum(dist,axis=0)/n
	
	B=-(T1-T2-T3+dist)/2
	
	eig_val，eig_vector=np.linalg.eig(B)
	#从大到小排
	index_=np.argsort(-eig_val)[:n_dims]
	picked_eig_val=eig_val[index_].real
	picked_eig_vector=eig_vector[:,index_]
	
	return picked_eig_val*picked_eig_val**(0.5)
	



#返回两个点在图中最短距离的算法
def floyd(D,n_neighbors):
		
	n1,n2=D.shape
	k=n_neighbors
	#初始化，设置为无穷大
	Max=np.max(D)*1000
	D1=np.one((n1,n1))*Max
	
	#argsort()函数是将向量中的元素从小到大排列，axis=1按行排列
	#提取其对应的index(索引)，然后输出到y，也就是说D_rag是索引
	#画出图，可以理解出来，实际就是把每个点到所有的点的距离进行排列
	D_rag=np.argsort(D,axis=1)
	#遍历所有点，找到距离这个点的最近k个
	for i in range(n1):
		#选取最近邻k
		D1[i,D_rag[i,0:k+1]]=D[i,D_arg[i,0:k+1]]
	#图论中求最短路径算法
	for m in range(n1):
		for i in range(n1):
			for j in range(n1):
				if D1[i,m]+D1[m,j]<D1[i,j]:
					D1[i,j]=D1[i,m]+D1[m,j]
					
	return D1
	

def cal_pairwise_dist(x):
    '''计算pairwise 距离, x是matrix
    (a-b)^2 = a^2 + b^2 - 2*a*b
    '''
	#np.square计算各元素的平方 
    sum_x = np.sum(np.square(x), 1)
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    #返回任意两个点之间距离的平方
    return dist

def my_isomap(data,n=2,n_neighbors=30):
	D=cal_pariwise_dist(data)
	#做一个处理，也可以没有
	D[D<0]=0
	D=D**0.5
	#D就是算法流程中所说的图
	#在图中计算k近邻
	D_floyd=floyd(D,n_neighbors)
	#此时就可以调用my_mds
	data_n=my_mds(D_floyd,n_dims=n)
	return data_n

#------------------------------------------------------------
#-----------------------PCA----------------------------------

def pca(data, n):

    data = np.array(data)
    # 均值
    mean_vector = np.mean(data, axis=0)
    # 协方差
    cov_mat = np.cov(data - mean_vector, rowvar=0)
    # 特征值 特征向量
    fvalue, fvector = np.linalg.eig(cov_mat)
    # 排序
    fvaluesort = np.argsort(-fvalue)
    # 取前几大的序号
    fValueTopN = fvaluesort[:n]
    # 保留前几大的数值
    newdata = fvector[:, fValueTopN]
    new = np.dot(data, newdata)
    return new
	
	
#-------------------------------------------------------------
#-------------LDA：Linear Discriminant Analysis---------------
#与PCA一样，是一种线性降维算法。
#不同于PCA只会选择数据变化最大的方向，
#由于LDA是有监督的（分类标签），
#所以LDA会主要以类别为思考因素，使得投影后的样本尽可能可分。


def lda_num2(data1,  data2,  n=2):
    mu0 = data2.mean(0)
    mu1 = data1.mean(0)
    print(mu0)
    print(mu1)

    sum0 = np.zeros((mu0.shape[0], mu0.shape[0]))
    for i in range(len(data2)):
        sum0 += np.dot((data2[i] - mu0).T, (data2[i] - mu0))
    sum1 = np.zeros(mu1.shape[0])
    for i in range(len(data1)):
        sum1 += np.dot((data1[i] - mu1).T, (data1[i] - mu1))

    s_w = sum0 + sum1
    print(s_w)
    w = np.linalg.pinv(s_w) * (mu0 - mu1)

    new_w = w[:n].T

    new_data1 = np.dot(data1, new_w)
    new_data2 = np.dot(data2, new_w)

    return new_data1, new_data2
	
	
#-------------------------------------------------------------
#-------------t-SNE Stochastic Neighbor Embedding--------------
#前面是保证距离不变，tSNE保证的是概率分布不变
#t-SNE是SNE的改进版，使用t分布替代高斯分布表达两点之间的相似度
#SNE是先将欧几里得距离转换为条件概率来表达点与点之间的相似度
#计算概率Pij，是人为构建的


#计算每一个样本的信息熵
def cal_entropy(D,beta):
	#信息熵如何计算？？好像和公式不一样啊
	#beta {float} -- 即1/(2sigma^2)
    
	P=np.exp(-D*beta)
    sumP=sum(P)
    sumP=np.maximum(sumP,1e-200)
	#H计算公式好像不一样啊？
    H=np.log(sumP) + beta * np.sum(D * P) / sumP
    return H


def cal_p(D,entropy,K=50):
	#每一行的计算都需要找到一个合适beta
	#delta是依据最大熵原理来决定，最大熵不能超过entropy
	#给定一个初始值，但不是最优值
	beta=1.0
	#计算每一个样本的信息熵
	H=cal_entropy(D,beta)
	error=H-entropy
	k=0
	betmin=-np.inf
	betmax=np.inf
	#每一行的计算都需要找到一个合适beta，使得这一行的分布熵小于等于log(neighbors)
	#二分搜索寻找beta
	while error>=1e-4 and k<=K:
		#说明还可以取最优值
		if error>0:
			betmin=copy.deepcopy(beta)
			if betmax=np.inf:
				beta=beta*2
			else:
				beta=(beta+betmax)/2
		
		else:
		#说明取值取过了
            betamax=copy.deepcopy(beta)
            if betamin==-numpy.inf:
                beta=beta/2
            else:
                beta=(beta+betamin)/2
				
		#重新计算
		H=cal_entropy(D,beta)
		error=H-entropy
        k+=1
	#根据公式求
	P=numpy.exp(-D*beta)
    P=P/numpy.sum(P)
    return P

#计算高维空间分布
def cal_matrix_P(x,neigbors):
	#最大信息熵不能超多此值
	entropy=np.log(neigbors)
	n1,n2=x.shape
	D=np.square(metrics.pairwise_distance(x))
	#这里偷了个懒，没有找邻域点，而是对数据进行了排序，选取排序前面k个作为邻域点
	D_sort=np.argsort(D,axis=1)
	#p(i,j)表示的是第i个样本在第j个样本周围的概率
	#所以P的大小是n1
	P=np.zeros((n1,n1))
	for i in xrange(n1):
		Di=D[i,D_sort[i,1:]]
		P[i,D_sort[i,1:]]=cal_p(Di,entropy=entropy)
	
	#根据公式求的	p=(pij+pji)/2*n
	P=(P+np.transpose(P))/(2*n1)
	P=np.maximum(P,1e-100)
	return P
	
	
	
#计算低维空间分布Q
def cal_matrix_Q(Y):
	n1,n2=Y.shape
    D=numpy.square(metrics.pairwise_distances(Y))
	#根据公式求
	Q=(1/(1+D))/(np.sum(1/(1+D))-n1)
	#后面几步感觉不写也行
	Q=Q/(np.sum(Q)-np.sum(Q[range(n1),range(n1)]))
	Q[range(n1),range(n1)]=0
	Q=np.maximum(Q,1e-100)
	return Q


#计算损失函数KL散度
def cal_loss(P,Q):
	C=np.sum(p*np.log(p/Q))
	return C

	
#计算梯度	
#和计算公式不一样
def cal_gradients(P,Q,Y):
    n1,n2=Y.shape
    DC=numpy.zeros((n1,n2))
    for i in xrange(n1):
	
        E=(1+np.sum((Y[i,:]-Y)**2,axis=1))**(-1)
        F=Y[i,:]-Y
        G=(P[i,:]-Q[i,:])
		
        E=E.reshape((-1,1))
        G=G.reshape((-1,1))
        G=np.tile(G,(1,n2))
        E=np.tile(E,(1,n2))
        DC[i,:]=np.sum(4*G*E*F,axis=0)
    return DC
		

		

def tsne(X,n=2,neighbors=30,max_iter=200):

    n1,n2=X.shape
    P=cal_matrix_P(X,neighbors)
    Y=numpy.random.randn(n1,n)*1e-4
	
    Q = cal_matrix_Q(Y)
	#梯度
    DY = cal_gradients(P, Q, Y)
    A=200.0
    B=0.1
	#开始迭代
    for i  in xrange(max_iter):
        data.append(Y)
		#第一次特殊
        if i==0:
            Y=Y-A*DY
            Y1=Y
            error1=cal_loss(P,Q)
		#第二次特殊
        elif i==1:
            Y=Y-A*DY
            Y2=Y
            error2=cal_loss(P,Q)
        else:
			#公式
            YY=Y-A*DY+B*(Y2-Y1)
            QQ = cal_matrix_Q(YY)
			#散度衡量误差的
            error=cal_loss(P,QQ)
            if error>error2:
                A=A*0.7
                continue
            elif (error-error2)>(error2-error1):
                A=A*1.2
            Y=YY
            error1=error2
            error2=error
            Q = QQ
			#更细梯度
            DY = cal_gradients(P, Q, Y)
            Y1=Y2
            Y2=Y
        if cal_loss(P,Q)<1e-3:
            return Y
        if numpy.fmod(i+1,10)==0:
            print '%s iterations the error is %s, A is %s'%(str(i+1),str(round(cal_loss(P,Q),2)),str(round(A,3)))
    tsne_dat['data']=data
    tsne_dat.close()
    return Y
	



