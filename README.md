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
