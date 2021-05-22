
#不能只看数学公式，一定要搞清楚数学公式转换成实际的物理含义
#比如要知道条件熵的条件是如何处理的写成代码的
#你要知道对于离散型 都是如何转换成概率的

#决策树学习本质上是从训练数据集中归纳出一组分类规则。
#决策树学习的算法通常是一个递归地选择最优特征
#包含特征选择、决策树的生成和决策树的剪枝过程。


#-------------------------------------------------------
#----------------ID3------------------------------------
#ID3算法以信息增益来度量属性的选择,选择分裂后信息增益最大的属性进行分裂

#这里并不是真正构建一个树结构，而是一个树的逻辑判决结构，先判断啥再判断啥
#{{{{}}}}括号套括号的存放
#找出出现次数最多的分类名称
def majorityCnt(classList):  
    classCount = {}  
    for vote in classList:  
        if vote not in classCount.keys(): classCount[vote] = 0  
        classCount[vote] += 1  
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0] 


#除去最优特征后得到新的子集
def splitDataSet(data_set, axis, value):
    ret_dataset = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis + 1:])
            ret_dataset.append(reduced_feat_vec)
    return ret_dataset

#计算熵
def calcShannonEnt(dataSet):
	#统计数据数量
	numEntries=len(dataset)
	#存储每个label出现的次数
	label_counts={}
	#统计label出现的次数
	for featVec in dataSet:
		current_label=featVec[-1]:
		if current_label not in label_counts:
			label_counts[current_label]=0
		label_counts[current_label]+=1
	
	shannon_ent=0
	#计算经验熵
	for key in label_counts:
	#根据公式计算
		prob=float(label_counts[key])/numEntries
		shannon_ent-=prob*log(prob,2)
	
	return shannon_ent
	
#选择最好的特征进行分支
def chooseBestFeatureToSplit(dataSet):
	#特征数量
	num_features=len(dataSet[0])-1
	#计算熵
	base_entropy=calcShannonEnt(dataSet)
    # 信息增益
    best_info_gain = 0.0
	#最优特征索引值
	best_feature=-1
	#遍历所有特征，以当前特征计算条件熵
	#仔细读论文发现计算当前特征条件熵，就是以当前特征取出去除后的子集进行计算得到的熵为条件熵
	for i in range(num_features)
		#获取dataset第i个特征
		feat_list=[exampe[i] for exampe in dataSet]
		#创建set集合，元素不可重合
		unique_val=set(feat_list)
		#根据公式计算信息特征的信息增益
		for value in unique_val:
			sub_dataset=splitDataSet(dataSet,i,value)
			#这一个计算是不是有问题
			#计算子集出现的概率
			prob=len(sub_dataset)/float(len(dataset))
			#计算经验条件熵---公式
			#仔细读论文发现计算当前特征条件熵，就是以当前特征取出去除后的子集进行计算得到的熵为条件熵
			new_entropy+=prob*calcShannonEnt(sub_dataset)
		#信息增益--公式
		info_gain=base_entropy-new_entropy
		
		# 打印每个特征的信息增益
        print("第%d个特征的信息增益为%.3f" % (i, info_gain))
		
		if info_gain > best_info_gain:
            # 更新信息增益
            best_info_gain = info_gain
            # 记录信息增益最大的特征的索引值
            best_feature = i
		
	print("最优索引值：" + str(best_feature))
    print()
    return best_feature

#获取决策树深度	
def get_tree_depth(my_tree):
    max_depth = 0       # 初始化决策树深度
    firsr_str = next(iter(my_tree))     # python3中myTree.keys()返回的是dict_keys,不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    second_dict = my_tree[firsr_str]    # 获取下一个字典
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':     # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth      # 更新层数
    return max_depth

#分类就是遍历树模型到叶子节点
#next() 返回迭代器的下一个项目
#next() 函数要和生成迭代器的 iter() 函数一起使用。
	
def classify(input_tree, feat_labels, test_vec):
    # 获取决策树节点
    first_str = next(iter(input_tree))
    # 下一个字典
    second_dict = input_tree[first_str]
    feat_index = feat_labels.index(first_str)

    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                class_label = second_dict[key]
    return class_label
	
	
def create_tree(dataSet,Labels,featLabels):
	# 数据集
    dataSet = [[0, 0, 0, 0, 'no'],
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
              # [1, 0, 0, 0, 'yes'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    # 分类属性
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
	
	
	
	#取分类标签（是否放贷：yes or no）
	class_list=[exampel[-1] for exampel in dataSet]	
	
	# 只剩下一列类别时就不用分类了
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 遍历完所有特征时返回出现次数最多的类标签
	#当dataSet[0]只有一个长度的时候，表示已经是最后一个特征了
	#这里是不是直接求dataset长度就行 不用定位到[0]??
    if len(dataSet[0]) == 1:
        return majority_cnt(class_list)
	
	
	#选择最优特征
	best_feature = chooseBestFeatureToSplit(dataSet)
	# 最优特征的标签
    best_feature_label = labels[best_feature] 
	featLabels.append(best_feature_label)
	
	#根据最优特征标签生成树
	my_tree={best_feature_label:{}}#假设{年龄：{年龄（1）：结果1，{...}，年龄（2）：结果2，{.....}}}
	#得先删除已经使用标签
	del(labels[best_feature])
	
	# 得到训练集中所有最优特征的属性值(某一列的值)
    feat_value = [exampel[best_feature] for exampel in dataSet]
    # 去掉重复属性值
    unique_vls = set(feat_value)
    for value in unique_vls:
        my_tree[best_feature_label][value] = 
				creat_tree(splitDataSet(dataSet, best_feature, value), labels, featLabels)
	
	return my_tree
	
if __name___=="__main__":
	
	#
	featLabels=[]
	
	#创建树
	myTree=create_tree(dataSet,labels,featLabels)

	
