# recommend-quickstart

```
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
```

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


-------

基于内容的过滤（Content-Based Filtering, CBF）推荐系统的核心思想是利用物品的特征信息来推荐用户可能喜欢的物品。以下是一个简单的基于内容的推荐系统的Python例子，它可以在Google Colab上运行。

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# 示例数据：物品的描述
items = {
    'Item1': 'Horror movie scary ghost',
    'Item2': 'Thriller movie suspense',
    'Item3': 'Adventure movie outdoor action',
    'Item4': 'Comedy movie funny laugh',
    'Item5': 'Drama movie serious performance',
}

# 用户喜好的物品描述
user_likes = 'Horror ghost thriller suspense'

# 使用TF-IDF向量化器将文本转换为向量
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(items.values())

# 将用户喜好也转换为向量
user_likes_vector = tfidf.transform([user_likes])

# 计算物品之间的余弦相似度
cosine_similarities = linear_kernel(user_likes_vector, tfidf_matrix).flatten()

# 获取相似度最高的物品的索引
top_item_index = cosine_similarities.argsort()[-2::-1]

# 输出推荐的物品
print("根据用户喜好推荐的物品:")
for index in top_item_index:
    print(list(items.keys())[index], '-', list(items.values())[index])
```

### 代码解释：
1. **数据准备**：定义一个包含物品描述的字典`items`，以及一个表示用户喜好的字符串`user_likes`。
2. **TF-IDF向量化**：使用`TfidfVectorizer`将物品描述转换为TF-IDF特征向量。这个过程可以提取文本中的关键信息，并将其转换为数值型特征，以便进行计算。
3. **计算相似度**：使用`linear_kernel`计算用户喜好向量与每个物品向量之间的余弦相似度。余弦相似度是一种衡量两个向量在方向上的相似程度的方法，常用于文本相似度计算。
4. **推荐物品**：根据相似度排序，选择相似度最高的物品作为推荐。这里我们排除了最后一个元素，因为它是用户喜好本身。

这个简单的CBF推荐系统可以直接在Google Colab中运行，不需要额外安装库，适合作为基于内容推荐的入门示例。

Citations:
[1] https://github.com/7hang/cyber-security-interview
[2] https://juejin.cn/s/python%20%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%20%E5%8D%8F%E5%90%8C%E8%BF%87%E6%BB%A4
[3] https://juejin.cn/post/6844903912705622030
[4] https://github.com/ictar/python-doc/blob/master/Science%20and%20Data%20Analysis/Python%E4%B8%AD%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%9A%84%E5%9F%BA%E4%BA%8E%E5%86%85%E5%AE%B9%E7%9A%84%E6%8E%A8%E8%8D%90%E5%BC%95%E6%93%8E.md
[5] https://github.com/fcitx/fcitx-table-data/blob/master/erbi.txt
[6] https://blog.csdn.net/m0_62554628/article/details/134574807
[7] https://blog.csdn.net/qq_29762941/article/details/93001289
