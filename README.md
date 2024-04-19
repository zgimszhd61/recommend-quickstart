# recommend-quickstart

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 示例数据：用户-物品评分矩阵
# 行代表用户，列代表物品，值代表评分
ratings = np.array([
    [5, 3, 0, 0],
    [4, 0, 4, 1],
    [1, 1, 0, 5],
    [1, 0, 4, 4],
    [0, 1, 5, 4],
])

# 计算物品之间的相似度
item_similarity = cosine_similarity(ratings.T)

# 打印物品相似度矩阵
print("物品相似度矩阵:")
print(item_similarity)

# 推荐函数
def recommend(user_index, ratings, item_similarity, top_n=3):
    # 对于指定用户，获取其未评分的物品的索引
    zero_indices = np.where(ratings[user_index, :] == 0)[0]
    # 计算未评分物品的预测评分
    scores = item_similarity[:, zero_indices].T.dot(ratings[user_index, :])
    # 获取预测评分最高的top_n个物品的索引
    recommended_item_indices = np.argsort(scores)[::-1][:top_n]
    # 返回推荐的物品索引
    return zero_indices[recommended_item_indices]

# 对第一个用户进行推荐
recommended_items = recommend(0, ratings, item_similarity)
print("推荐给用户0的物品索引:", recommended_items)

-----

在上面提供的个性化推荐引擎示例中，推荐给用户0的物品索引的逻辑可以分为以下几个步骤：

### 1. 数据准备
首先，我们有一个用户-物品评分矩阵，其中行代表不同的用户，列代表不同的物品，矩阵中的值代表用户对物品的评分。例如，如果用户1对物品2的评分是5，则表示用户1非常喜欢物品2。

### 2. 计算物品之间的相似度
使用余弦相似度来计算物品之间的相似度。余弦相似度是一种衡量两个向量在方向上的相似程度的方法，它的计算公式是两个向量的点积与它们各自范数（长度）的乘积的比值。在这个例子中，每个物品的向量就是所有用户对这个物品的评分。

### 3. 推荐逻辑
对于特定的用户（例如用户0），我们首先找出该用户未评分的物品。这是因为我们只需要对用户尚未互动过的物品进行推荐。

#### 推荐步骤详解：
- **找出未评分的物品**：对于用户0，我们检查其评分向量，找出所有评分为0的位置，这些位置对应的物品即为用户0未评分的物品。
- **计算预测评分**：对于每个未评分的物品，我们利用已有的物品相似度矩阵和用户0对其他物品的评分来预测用户0对未评分物品的评分。具体来说，对于每个未评分的物品，我们将该物品与所有其他物品的相似度与用户0对这些其他物品的评分相乘，然后求和，得到一个预测评分。
- **选择最高评分的物品**：根据计算出的预测评分，选择评分最高的几个物品作为推荐。这里的`top_n`参数可以定义我们想要推荐的物品数量。

### 4. 输出推荐结果
最后，函数返回预测评分最高的物品索引，这些索引对应的就是推荐给用户0的物品。

这个过程通过数学上的向量和矩阵运算实现，确保了推荐的准确性和效率。通过这种方式，我们可以为每个用户提供个性化的推荐，帮助他们发现可能感兴趣的新物品。

Citations:
[1] https://cloud.tencent.com/developer/article/1487831
[2] https://www.cnblogs.com/pinard/p/6349233.html
[3] https://www.niaogebiji.com/article-25842-1.html
[4] https://www.easemob.com/news/10998
[5] https://www.cnblogs.com/yoyo1216/p/12618472.html
[6] https://blog.csdn.net/duyibo123/article/details/110915485
[7] https://www.fenghui100.com/2534.html
[8] https://www.cnblogs.com/Matrix_Yao/p/15037865.html
[9] https://developer.aliyun.com/article/54541
[10] https://blog.csdn.net/mudan97/article/details/111466918
[11] https://cloud.tencent.com/developer/article/1005510
[12] https://www.jiqizhixin.com/graph/technologies/3781437f-1f14-4acd-b256-ca7129c6d0c4
[13] https://openmlsys.github.io/chapter_recommender_system/system_architecture.html
[14] https://www.jiqizhixin.com/graph/technologies/6ca1ea2d-6bca-45b7-9c93-725d288739c3
[15] https://www.volcengine.com/theme/3134114-T-7-1
[16] https://machinelearninggao.readthedocs.io/zh/latest/16.%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F/
[17] https://www.cnblogs.com/eilearn/p/9972243.html
[18] https://www.nvidia.cn/glossary/data-science/recommendation-system/
[19] https://developer.aliyun.com/article/763945
