
#随机森林是用多棵树对样本进行训练并预测的一种分类器

#随机在于随机选择特征的数目，随机选择训练数据
#本质就是样本和特征都进行了采样

#随机森林实际是一种特殊的bagging方法，将决策树用做bagging中
#用bootstrap方法生成m个训练集，对每个训练集构造一颗决策树，节点找特征时
#不是选择单个信息熵最大的特征，而是随机抽取一部分特征


#--------------------------------------------------------------------
#-------------------随机森林----------------------------------------

#算法
#用N来表示训练用例（样本）的个数，M表示特征数目。
#输入特征数目m，用于确定决策树上一个节点的决策结果；其中m应远小于M。
#从N个训练用例（样本）中以有放回抽样的方式，取样N次，形成一个训练集（即bootstrap取样），并用未抽到的用例（样本）作预测，评估其误差。
#对于每一个节点，随机选择m个特征，决策树上每个节点的决定都是基于这些特征确定的。根据这m个特征，计算其最佳的分裂方式。
#每棵树都会完整成长而不会剪枝，这有可能在建完一棵正常树状分类器后会被采用）。



#基尼系数
#数据集D的纯度可以用基尼值来度量
#Gini（D）反映了从数据集D中随机选取两个样本，其类别标记不一致的概率，
#因此Gini（D）越小，则数据集D的纯度越高。


#计算基尼系数
#i是表示选几个特征吗？？？？
def gini(data, i):

    num = len(data)
    label_counts = [0, 0, 0, 0]

    p_count = [0, 0, 0, 0]

    gini_count = [0, 0, 0, 0]

    for d in data:
        label_counts[d[i]] += 1

    for l in range(len(label_counts)):
        for d in data:
            if label_counts[l] != 0 and d[0] == 1 and d[i] == l:
                p_count[l] += 1


    for l in range(len(label_counts)):
        if label_counts[l] != 0:
            gini_count[l] = 2*(p_count[l]/label_counts[l])*(1 - p_count[l]/label_counts[l])

    gini_p = 0
    for l in range(len(gini_count)):
        gini_p += (label_counts[l]/num)*gini_count[l]



    return gini_p


#候选属性集合A中，选择那个使得划分后基尼指数最小的属性作为最优划分属性

def get_best_feature(data, category):
    if len(category) == 2:
        return 1, category[1]

    feature_num = len(category) - 1
    data_num = len(data)

    feature_gini = []

    for i in range(1, feature_num+1):
        feature_gini.append(gini(data, i))

    min = 0

    for i in range(len(feature_gini)):
        if feature_gini[i] < feature_gini[min]:
            min = i

    print(feature_gini)
    print(category)
    print(min+1)
    print(category[min+1])

    return min+1, category[min + 1]
	
	

class Node(object):
	def __init__(self,item):
		self.name=item
		self.lchild=None
		self.rchild=None
		
def creat_tree(data, labels, feature_labels=[]):
# 三种结束情况
    # 取分类标签(survivor or death)
    class_list = [exampel[0] for exampel in data]

    if class_list == []:
        return Node(0)
    # 如果类别完全相同则停止分类
    if class_list.count(class_list[0]) == len(class_list):
        return Node(class_list[0])
    # 遍历完所有特征时返回出现次数最多的类标签
    if len(data[0]) == 1:
        return Node(majority_cnt(class_list))

    # 最优特征的标签
    best_feature_num, best_feature_label = get_best_feature(data, labels)

    feature_labels.append(best_feature_label)

    node = Node(best_feature_label)

    ldata = []
    rdata = []

    for d in data:
        if d[best_feature_num] == 1:
            del(d[best_feature_num])
            ldata.append(d)
        else:
            del(d[best_feature_num])
            rdata.append(d)

    labels2 = copy.deepcopy(labels)
    del(labels2[best_feature_num])

    tree = node
    tree.lchild = creat_tree(ldata, labels2, feature_labels)
    tree.rchild = creat_tree(rdata, labels2, feature_labels)

    return tree

#统计分类
def majority_cnt(class_list):
    class_count = {}
    # 统计class_list中每个元素出现的次数
    for vote in class_list:
        if vote not in class_count:
            class_count[vote] = 0
        class_count[vote] += 1
        # 根据字典的值降序排列
        sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]

#预测代码
def pre(t_test, labels, tree):
    result = []
    r = []

    for i in range(len(t_test)):
            label = []
            label = copy.deepcopy(labels[i])
            print(label)
            breadth_travel(tree[i])
            r.append(prediction(tree[i], t_test[i], label))
    rr = []
    for i in range(len(r[0])):
        rr.append([])

    for i in range(len(rr)):
        for j in range(len(r)):
            rr[i].append(r[j][i])

    print(rr)

    for i in range(len(rr)):
        result.append(majority_cnt(rr[i]))
    return result
	

#------------main--------------

#生成10颗树
	tree_num=10
	bootsrapping=[]
	b_category=[]

	#创建10颗树
	for i in range(tree_num):
        b_category.append(copy.deepcopy(category))
		
        bootstrapping.append([])
		#数据集中随机选取样本
		#为什么这里感觉是把全部样本都选择了，只是随机了一下顺序而已
		#由于是有放回的随机抽样，也就是说，尽管抽样了和抽样数目和数据集数目一样
		#但抽样的数据集内容不等于原有数据集
		#当然这里抽样数目多少应该是可以自己定的
        for j in range(len(data_set)):
            bootstrapping[i].append(copy.deepcopy(data_set[int(np.floor(np.random.random() * len(data_set)))]))
			
	
	#对每个样本集随机选取特征,原有特征数是5，随机选取特征数为2个
	for i in range(tree_num):
		#这里得到的是下标
		n_num_category[i].append(random.sample(range(1, 5), 2))
	
	#需要将特征值转换过来重新组成特征矩阵
	#网上代码没看懂啥意思 ，感觉应该是这个意思？？？
	b_category<------n_num_category

	
	#为每个集构建树
	for i in range(tree_num):
		my_tree.append(creat_tree(bootstrapping[i], b_category[i]))