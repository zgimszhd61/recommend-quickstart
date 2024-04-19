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

-------

为了使上述基于内容的推荐系统支持中文，并且适应协同推荐的场景，我们需要对代码进行一些调整。这包括使用适合中文的分词工具来处理文本数据，以及调整数据结构以适应协同推荐的需求。以下是修改后的代码，它可以在Google Colab上运行：

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import jieba  # 导入jieba中文分词库

# 示例数据：物品的描述（中文）
items = {
    'Item1': '恐怖电影 鬼魂',
    'Item2': '惊悚电影 悬疑',
    'Item3': '冒险电影 户外 动作',
    'Item4': '喜剧电影 搞笑',
    'Item5': '剧情电影 严肃 表演',
}

# 用户喜好的物品描述（中文）
user_likes = '恐怖 鬼魂 惊悚 悬疑'

# 使用jieba进行中文分词
def chinese_tokenizer(text):
    return jieba.lcut(text)

# 使用TF-IDF向量化器将文本转换为向量，指定分词函数为jieba分词
tfidf = TfidfVectorizer(tokenizer=chinese_tokenizer, stop_words=['的', '是', '和', '在'])
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
1. **中文分词**：使用`jieba`库进行中文分词。由于中文文本不像英文那样有明显的单词分隔符，因此需要使用分词工具来将句子切分成词语。
2. **TF-IDF向量化**：在`TfidfVectorizer`中指定`tokenizer`参数为`chinese_tokenizer`，这是一个使用`jieba`分词的函数。同时，指定一些常见的停用词以提高分析的准确性。
3. **余弦相似度计算**：与之前相同，计算用户喜好向量与各个物品向量之间的余弦相似度。
4. **推荐物品**：根据相似度排序，选择相似度最高的物品作为推荐。

这个修改后的代码支持中文文本处理，并且可以在Google Colab中直接运行，适合用于中文内容的基于内容的推荐系统示例。


-------

为了适应主要由100字以内短文组成的新闻个性化推荐系统，我们需要对之前的基于内容的推荐系统代码进行一些调整。这些调整主要是为了更好地处理短文本的特性，并提高推荐的准确性和相关性。以下是修改后的代码示例，可以在Google Colab上运行：

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import jieba  # 导入jieba中文分词库

# 示例数据：新闻的标题和短内容（中文）
news = {
    'News1': '美国总统访问中国，外交关系紧张。',
    'News2': '人工智能技术突破，新一代AI模型发布。',
    'News3': 'NBA季后赛火热进行，球星表现抢眼。',
    'News4': '知名歌手发行新专辑，粉丝热情高涨。',
    'News5': '股市大跌，投资者情绪不稳。',
}

# 用户喜好的新闻描述（中文）
user_likes = '科技突破，AI发展'

# 使用jieba进行中文分词
def chinese_tokenizer(text):
    return jieba.lcut(text)

# 使用TF-IDF向量化器将文本转换为向量，指定分词函数为jieba分词
tfidf = TfidfVectorizer(tokenizer=chinese_tokenizer, stop_words=['的', '是', '和', '在'])
tfidf_matrix = tfidf.fit_transform(news.values())

# 将用户喜好也转换为向量
user_likes_vector = tfidf.transform([user_likes])

# 计算新闻之间的余弦相似度
cosine_similarities = linear_kernel(user_likes_vector, tfidf_matrix).flatten()

# 获取相似度最高的新闻的索引
top_news_index = cosine_similarities.argsort()[-2::-1]

# 输出推荐的新闻
print("根据用户喜好推荐的新闻:")
for index in top_news_index:
    print(list(news.keys())[index], '-', list(news.values())[index])
```

### 代码调整说明：
1. **数据准备**：新闻数据现在包括标题和100字以内的简短内容。这要求我们在处理时更加注重文本的关键信息提取。
2. **中文分词**：使用`jieba`库进行中文分词，这对于中文文本处理至关重要，尤其是在短文本中，每个词的权重更加明显。
3. **TF-IDF向量化**：在`TfidfVectorizer`中指定`tokenizer`参数为`chinese_tokenizer`，并设置适当的停用词，以提高分析的准确性。对于短文本，正确的停用词设置尤其重要，因为无关词汇可能会对结果产生较大影响。
4. **余弦相似度计算**：计算用户喜好向量与各个新闻向量之间的余弦相似度，以评估内容的相关性。
5. **推荐新闻**：根据相似度排序，选择相似度最高的新闻作为推荐。

这个修改后的代码适用于处理包含标题和简短内容的中文新闻推荐，可以直接在Google Colab中运行，适合用于短文本新闻的个性化推荐系统示例。

Citations:
[1] https://arxiv.org/pdf/2106.08934.pdf
[2] https://www.mdpi.com/2078-2489/14/2/60
[3] https://link.springer.com/article/10.1007/s41019-023-00228-5
[4] https://dl.acm.org/doi/10.1145/3584946
[5] https://www.mdpi.com/2673-6470/4/1/3
[6] https://paperswithcode.com/task/news-recommendation
[7] https://mobidev.biz/blog/how-to-build-text-based-recommender-system-with-nlp
[8] https://aclanthology.org/P19-1033
[9] https://arxiv.org/pdf/2104.07413.pdf
[10] http://www.sjisr.org/download/sjisr-4-4-473-484.pdf
[11] https://milvus.io/blog/building-an-intelligent-news-recommendation-system-inside-sohu-news-app.md
[12] https://dl.acm.org/doi/abs/10.1145/3404835.3463069
[13] https://ieeexplore.ieee.org/document/9404101
[14] https://dergipark.org.tr/tr/download/article-file/2728602


-----------

为了融合用户评分反馈因素，并对未评分的新闻进行预测，我们需要对之前的代码进行进一步的改进。这次我们将扩充新闻样本到10条，并且只对用户未评分的新闻进行预测，推荐那些预测用户最可能喜欢的新闻。以下是修改后的代码示例，可以在Google Colab上运行：

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import jieba  # 导入jieba中文分词库

# 示例数据：新闻的标题和短内容（中文），扩充到10条
news = {
    'News1': '美国总统访问中国，外交关系紧张。',
    'News2': '人工智能技术突破，新一代AI模型发布。',
    'News3': 'NBA季后赛火热进行，球星表现抢眼。',
    'News4': '知名歌手发行新专辑，粉丝热情高涨。',
    'News5': '股市大跌，投资者情绪不稳。',
    'News6': '国际油价上涨，经济形势复杂。',
    'News7': '本地教育改革，学生家长意见大。',
    'News8': '新款电动汽车发布，续航里程刷新纪录。',
    'News9': '国际旅游业开始复苏，旅客数量激增。',
    'News10': '全球气候变暖，极端天气频发。'
}

# 用户评分反馈：喜欢(1)、未评分(0)、不喜欢(-1)
user_feedback = {
    'News1': -1,
    'News2': 1,
    'News3': 0,
    'News4': 1,
    'News5': -1,
    'News6': 0,
    'News7': 0,
    'News8': 1,
    'News9': 0,
    'News10': -1
}

# 使用jieba进行中文分词
def chinese_tokenizer(text):
    return jieba.lcut(text)

# 使用TF-IDF向量化器将文本转换为向量，指定分词函数为jieba分词
tfidf = TfidfVectorizer(tokenizer=chinese_tokenizer, stop_words=['的', '是', '和', '在'])
tfidf_matrix = tfidf.fit_transform(news.values())

# 构建用户喜好向量，考虑用户反馈
user_profile = np.zeros(tfidf_matrix.shape[1])
for news_id, feedback in user_feedback.items():
    if feedback != 0:  # 只考虑用户已评分的新闻
        user_profile += feedback * tfidf_matrix[list(news.keys()).index(news_id)].toarray().flatten()

# 计算新闻之间的余弦相似度
cosine_similarities = linear_kernel(user_profile.reshape(1, -1), tfidf_matrix).flatten()

# 获取未评分新闻的索引
unrated_news_indices = [index for index, news_id in enumerate(news) if user_feedback[news_id] == 0]

# 获取未评分新闻的相似度
unrated_news_similarities = cosine_similarities[unrated_news_indices]

# 获取相似度最高的未评分新闻的索引
top_unrated_news_index = np.argsort(unrated_news_similarities)[-3:][::-1]  # 推荐相似度最高的三条未评分新闻

# 输出推荐的未评分新闻
print("根据用户喜好推荐的未评分新闻:")
for index in top_unrated_news_index:
    news_index = unrated_news_indices[index]
    print(list(news.keys())[news_index], '-', list(news.values())[news_index])
```

### 代码调整说明：
1. **扩充新闻样本**：新闻数据现在包括10条新闻的标题和简短内容。
2. **用户评分反馈**：用户对每条新闻的评分反馈被更新，包括喜欢、未评分和不喜欢。
3. **构建用户喜好向量**：根据用户的评分反馈和新闻的TF-IDF向量，构建一个代表用户喜好的向量。
4. **预测未评分新闻**：计算用户喜好向量与所有新闻向量之间的余弦相似度，然后只针对用户未评分的新闻进行预测。
5. **推荐逻辑**：从未评分的新闻中选择预测相似度最高的几条新闻作为推荐。

这个修改后的代码能够有效地融合用户的评分反馈，并对未评分的新闻进行预测，推荐那些用户最可能喜欢的新闻。


-----------

评估协同推荐系统的效果通常涉及多个标准，主要包括以下几种：

1. **准确率（Accuracy）**：通常通过计算预测评分和实际评分之间的误差来衡量，常用的指标包括均方误差（MSE）和均方根误差（RMSE）。

2. **精确度（Precision）和召回率（Recall）**：
   - **精确度**：推荐系统推荐的项中，用户实际喜欢的比例。
   - **召回率**：用户实际喜欢的项中，被推荐系统推荐出来的比例。

3. **F1分数**：精确度和召回率的调和平均，是一个综合考虑精确度和召回率的指标。

4. **覆盖率（Coverage）**：推荐系统能够推荐出多少比例的物品。

5. **多样性（Diversity）**：推荐列表中物品之间的不相似度。

为了在上述代码中加入评估模块，我们可以假设有一组用户的实际喜好数据用于测试。以下是更新后的代码，包括一个简单的评估模块来计算精确度和召回率：

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import jieba

# 示例数据和用户反馈
news = {
    'News1': '美国总统访问中国，外交关系紧张。',
    'News2': '人工智能技术突破，新一代AI模型发布。',
    'News3': 'NBA季后赛火热进行，球星表现抢眼。',
    'News4': '知名歌手发行新专辑，粉丝热情高涨。',
    'News5': '股市大跌，投资者情绪不稳。',
    'News6': '国际油价上涨，经济形势复杂。',
    'News7': '本地教育改革，学生家长意见大。',
    'News8': '新款电动汽车发布，续航里程刷新纪录。',
    'News9': '国际旅游业开始复苏，旅客数量激增。',
    'News10': '全球气候变暖，极端天气频发。'
}

user_feedback = {
    'News1': -1,
    'News2': 1,
    'News3': 0,
    'News4': 1,
    'News5': -1,
    'News6': 0,
    'News7': 0,
    'News8': 1,
    'News9': 0,
    'News10': -1
}

# 假设的用户实际喜好（用于评估）
actual_likes = ['News2', 'News4', 'News8', 'News9']

# 分词和TF-IDF向量化
def chinese_tokenizer(text):
    return jieba.lcut(text)

tfidf = TfidfVectorizer(tokenizer=chinese_tokenizer, stop_words=['的', '是', '和', '在'])
tfidf_matrix = tfidf.fit_transform(news.values())

# 构建用户喜好向量
user_profile = np.zeros(tfidf_matrix.shape[1])
for news_id, feedback in user_feedback.items():
    if feedback != 0:
        user_profile += feedback * tfidf_matrix[list(news.keys()).index(news_id)].toarray().flatten()

# 计算相似度并推荐
cosine_similarities = linear_kernel(user_profile.reshape(1, -1), tfidf_matrix).flatten()
recommended_indices = cosine_similarities.argsort()[-3:][::-1]
recommended_news = [list(news.keys())[i] for i in recommended_indices]

# 评估模块
def evaluate(recommended, actual):
    recommended_set = set(recommended)
    actual_set = set(actual)
    precision = len(recommended_set & actual_set) / len(recommended_set)
    recall = len(recommended_set & actual_set) / len(actual_set)
    return precision, recall

precision, recall = evaluate(recommended_news, actual_likes)
print("推荐的新闻:", recommended_news)
print("精确度:", precision)
print("召回率:", recall)
```

这段代码首先定义了一个假设的用户实际喜好列表`actual_likes`，用于评估推荐系统的精确度和召回率。然后，它计算了推荐新闻与实际喜好之间的精确度和召回率，这可以帮助我们了解推荐系统的效果。
