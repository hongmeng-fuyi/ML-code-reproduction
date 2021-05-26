
#-------------------------------------------------------------------
#k均值聚类算法是一种迭代求解的聚类分析算法




#----------------------------K-means--------------------------------
#无监督分类算法
#算法流程
#1、从N个样本随机选取K个作为质心
#2、每个样本测量其到质心的距离，并把他归于最近质心的类
#3、新的聚集出来之后，计算每个聚集的新中心点，就是求平均
#4、迭代（2）-（3），直到迭代终止（选择中心点不在变化为止）

#读完算法流程值得思考的问题？
#1、k的选择是已知的还是未知的，是因为知道有多少类别而选择的K吗
#2、损失怎么约束，作用在那一块，迭代退出条件吗？
#		kmeans是无标签的分类算法，没有利用标签不存在损失
#       迭代条件应该是可以给出次数，也可以判别中心点是否发生变化
#		值得距离中心是否发生变化可能得以两个点的误差给一个值来判别，
#		因为存在精度问题不可能直接比较点是否相同
#3、步骤（2）中计算每个样本点到质心的距离是所有还是其余


def dist(a,b,ax=1):
	#这里是用范数计算距离
	return np.linalg.norm(a-b,axis=ax)
	
# 设定分区数
k = 3
# 随机获得中心点的X轴坐标
C_x = np.random.randint(0, np.max(X)-20, size=k)
# 随机获得中心点的Y轴坐标
C_y = np.random.randint(0, np.max(X)-20, size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
#其实也可以直接选三个样本点作为中心点，这种设置中心点的方式也行


# 用于保存中心点更新前的坐标
C_old = np.zeros(C.shape)
print(C)
# 用于保存数据所属中心点
clusters = np.zeros(len(X))
# 迭代标识位，通过计算新旧中心点的距离
iteration_flag = dist(C, C_old, 1)

tmp = 1
# 若中心点不再变化或循环次数不超过20次(此限制可取消)，则退出循环
while iteration_flag.any() != 0 and tmp < 20:
    # 循环计算出每个点对应的最近中心点
    for i in range(len(X)):
        # 计算出每个点与中心点的距离
		#这里循环是不是错了？
		#应该是每个点和三个中心点计算三个距离选择最近的一个
		#好像是函数放在一起计算了
        distances = dist(X[i], C, 1)
        # print(distances)
        # 记录0 - k-1个点中距离近的点
        cluster = np.argmin(distances) 
        # 记录每个样例点与哪个中心点距离最近
        clusters[i] = cluster
        
    # 采用深拷贝将当前的中心点保存下来
    # print("the distinct of clusters: ", set(clusters))
    C_old = deepcopy(C)
    # 从属于中心点放到一个数组中，然后按照列的方向取平均值
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        # print(points)
        # print(np.mean(points, axis=0))
        C[i] = np.mean(points, axis=0)
        # print(C[i])
    # print(C)
    
    # 计算新旧节点的距离
    print ('循环第%d次' % tmp)
    tmp = tmp + 1
    iteration_flag = dist(C, C_old, 1)
    print("新中心点与旧点的距离：", iteration_flag)