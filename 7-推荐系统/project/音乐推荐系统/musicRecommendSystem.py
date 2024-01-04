import pandas as pd
import numpy as np
import time
import sqlite3
import matplotlib.pyplot as plt

# 1. 数据集清洗
# 1.1 读取数据
data_home = 'E:\\machineLearning\\kaggle\\dataset\\'
save_home = 'E:\\machineLearning\\kaggle\\projectResult\\musicRecommend\\'
triplet_dataset = pd.read_csv(filepath_or_buffer=data_home + 'train_triplets.txt',
                              sep='\t', header=None,
                              names=['user', 'song', 'play_count']) #有一部分是数据库文件
triplet_dataset.shape

# 数据情况
triplet_dataset.info()
triplet_dataset.head(n=10)

# 1.2 统计分析
# 1.2.1 统计每一个用户的播放总量
output_dict = {}
with open(data_home+'train_triplets.txt') as f:
    for line_number, line in enumerate(f):
        user = line.split('\t')[0] #用户
        play_count = int(line.split('\t')[2]) # 该用户的播放总量数据
        if user in output_dict:
            play_count += output_dict[user]
            output_dict.update({user: play_count})
        output_dict.update({user: play_count}) #判断结果字典中是否已有该用户
output_list = [{'user': k, 'play_count': v} for k, v in output_dict.items()] #将结果字典转换为列表形式
play_count_df = pd.DataFrame(output_list) #转换为DF格式
play_count_df = play_count_df.sort_values(by='play_count', ascending=False) #排序
play_count_df.to_csv(path_or_buf=save_home + 'user_playcount_df.csv', index=False) #保存结果

# 1.2.2 统计每一首歌的播放总量，如上
output_dict = {}
with open(data_home+'train_triplets.txt') as f:
    for line_number, line in enumerate(f):
        song = line.split('\t')[1]
        play_count = int(line.split('\t')[2])
        if song in output_dict:
            play_count += output_dict[song]
            output_dict.update({song: play_count})
        output_dict.update({song: play_count})
output_list = [{'song': k, 'play_count': v} for k, v in output_dict.items()]
song_count_df = pd.DataFrame(output_list)
song_count_df = song_count_df.sort_values(by='play_count', ascending=False)
song_count_df.to_csv(path_or_buf=save_home + 'song_playcount_df.csv', index=False)

# 1.2.3 展示统计结果
play_count_df = pd.read_csv(filepath_or_buffer=save_home + 'user_playcount_df.csv')
play_count_df.head(n=10)

song_count_df = pd.read_csv(filepath_or_buffer=save_home + 'song_playcount_df.csv')
song_count_df.head(10)

# 1.3 选取目标集合
total_play_count = sum(song_count_df.play_count)
play_count_subset = play_count_df.head(n=100000)
song_count_subset = song_count_df.head(n=30000)
# 前10万名用户的播放量占总体的比例
print((float(play_count_df.head(n=100000).play_count.sum()) / total_play_count)*100)
# 前3万首歌曲的播放量占总体的比例
print((float(song_count_df.head(n=30000).play_count.sum()) / total_play_count)*100)

# 1.4 原始数据集清洗
user_subset = list(play_count_subset.user)
song_subset = list(song_count_subset.song)
# 读取原始数据集
triplet_dataset = pd.read_csv(filepath_or_buffer=data_home + 'train_triplets.txt',
                              sep='\t', header=None, names=['user', 'song', 'play_count'])
# 只保留前10万用户
triplet_dataset_sub = triplet_dataset[triplet_dataset.user.isin(user_subset)]
del(triplet_dataset)
# 只保留前3万歌曲
triplet_dataset_sub_song = triplet_dataset_sub[triplet_dataset_sub.song.isin(song_subset)]
del(triplet_dataset_sub)
# 保存过滤后的数据集
triplet_dataset_sub_song.to_csv(path_or_buf=save_home + 'triplet_dataset_sub_song.csv', index=False)
triplet_dataset_sub_song.shape

# 1.5 数据集整合
# 读取每首歌曲的其他详细信息
conn = sqlite3.connect(data_home + 'track_metadata.db')
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
cur.fetchall()
track_metadata_df = pd.read_sql(con=conn, sql='select * from songs')
track_metadata_df_sub = track_metadata_df[track_metadata_df.song_id.isin(song_subset)]
track_metadata_df_sub.to_csv(path_or_buf=save_home + 'track_metadata_df_sub.csv', index=False) # 保存读取后的数据
triplet_dataset_sub_song = pd.read_csv(filepath_or_buffer=save_home+'triplet_dataset_sub_song.csv', encoding="ISO-8859-1")
track_metadata_df_sub = pd.read_csv(filepath_or_buffer=save_home+'track_metadata_df_sub.csv', encoding="ISO-8859-1")
track_metadata_df_sub.head() # 包含详细信息的前3万首歌曲

# 1.6 选择指定特征
del(track_metadata_df_sub['track_id'])
del(track_metadata_df_sub['artist_mbid']) # 去掉无用
track_metadata_df_sub = track_metadata_df_sub.drop_duplicates(['song_id']) # 去掉重复
triplet_dataset_sub_song_merged = pd.merge(triplet_dataset_sub_song, track_metadata_df_sub,
                                           how='left', left_on='song', right_on='song_id') # 整合该音乐信息数据和之前的播放数据
triplet_dataset_sub_song_merged.rename(columns={'play_count': 'listen_count'}, inplace=True) # 改变列名
del(triplet_dataset_sub_song_merged['song_id'])
del(triplet_dataset_sub_song_merged['artist_id'])
del(triplet_dataset_sub_song_merged['duration'])
del(triplet_dataset_sub_song_merged['artist_familiarity'])
del(triplet_dataset_sub_song_merged['artist_hotttnesss'])
del(triplet_dataset_sub_song_merged['track_7digitalid'])
del(triplet_dataset_sub_song_merged['shs_perf'])
del(triplet_dataset_sub_song_merged['shs_work']) # 去掉不需要指标
triplet_dataset_sub_song_merged.head(n=10)

# 1.7 统计
# 1.7.1 按歌曲名字来统计其播放量的总数
popular_songs = triplet_dataset_sub_song_merged[['title', 'listen_count']].groupby('title').sum().reset_index()
popular_songs_top_20 = popular_songs.sort_values('listen_count', ascending=False).head(n=20) # 对结果进行排序
# 绘图
objects = (list(popular_songs_top_20['title'])) # 转换成list格式方便画图
y_pos = np.arange(len(objects)) # 设置位置
performance = list(popular_songs_top_20['listen_count']) # 对应结果值
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects, rotation='vertical')
plt.ylabel('Item count')
plt.title('Most popular songs')
plt.show()

# 1.7.2 按专辑名字来统计播放总量
popular_release = triplet_dataset_sub_song_merged[['release', 'listen_count']].groupby('release').sum().reset_index()
popular_release_top_20 = popular_release.sort_values('listen_count', ascending=False).head(n=20)
objects = (list(popular_release_top_20['release']))
y_pos = np.arange(len(objects))
performance = list(popular_release_top_20['listen_count'])
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects, rotation='vertical')
plt.ylabel('Item count')
plt.title('Most popular Release')
plt.show()

# 1.7.3 用户播放的分布情况
user_song_count_distribution = (triplet_dataset_sub_song_merged[['user', 'title']].
                                groupby('user').count().reset_index().sort_values(by='title', ascending=False))
user_song_count_distribution.title.describe()

# 绘图展示
x = user_song_count_distribution.title
n, bins, patches = plt.hist(x, 50, facecolor='green', alpha=0.75)
plt.xlabel('Play Counts')
plt.ylabel('Num of Users')
plt.title(r'$\mathrm{Histogram\ of\ User\ Play\ Count\ Distribution}\ $')
plt.grid(True)
plt.show()


# 2. 基于相似度的推荐系统
# 2.1 排行榜推荐（用户冷启动）
def create_popularity_recommendation(train_data, user_id, item_id):
    # 根据指定的特征来统计其播放情况，可以选择歌曲名，专辑名，歌手名
    train_data_grouped = train_data.groupby([item_id]).agg({user_id: 'count'}).reset_index()
    # 为了直观展示，我们用得分来表示其结果
    train_data_grouped.rename(columns={user_id: 'score'}, inplace=True)
    # 排行榜单需要排序
    train_data_sort = train_data_grouped.sort_values(['score', item_id], ascending=[0, 1])
    # 加入一项排行等级，表示其推荐的优先级
    train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')
    # 返回指定个数的推荐结果
    popularity_recommendations = train_data_sort.head(20)
    return popularity_recommendations

recommendations = create_popularity_recommendation(triplet_dataset_sub_song_merged,'user','title')
recommendations.head()

# 2.2 基于歌曲相似度推荐（计算歌曲之间相似度）


# 3. 基于矩阵分解的推荐
# 3.1 使用SVD奇异矩阵分解，定义用户对歌曲打分值为 该用户播放该歌曲量/该用户总播放量
triplet_dataset_sub_song_merged_sum_df = triplet_dataset_sub_song_merged[['user', 'listen_count']].groupby('user').sum().reset_index()
triplet_dataset_sub_song_merged_sum_df.rename(columns={'listen_count': 'total_listen_count'}, inplace=True)
triplet_dataset_sub_song_merged = pd.merge(triplet_dataset_sub_song_merged,triplet_dataset_sub_song_merged_sum_df)
triplet_dataset_sub_song_merged.head()
# 计算比例
triplet_dataset_sub_song_merged['fractional_play_count'] = (triplet_dataset_sub_song_merged['listen_count'] /
                                                            triplet_dataset_sub_song_merged['total_listen_count'])

# 3.2 制作简单的id索引，构建矩阵
from scipy.sparse import coo_matrix
small_set = triplet_dataset_sub_song_merged
user_codes = small_set.user.drop_duplicates().reset_index()
song_codes = small_set.song.drop_duplicates().reset_index()
user_codes.rename(columns={'index': 'user_index'}, inplace=True)
song_codes.rename(columns={'index': 'song_index'}, inplace=True)
song_codes['so_index_value'] = list(song_codes.index)
user_codes['us_index_value'] = list(user_codes.index)
small_set = pd.merge(small_set, song_codes, how='left')
small_set = pd.merge(small_set, user_codes, how='left')
mat_candidate = small_set[['us_index_value', 'so_index_value', 'fractional_play_count']]
data_array = mat_candidate.fractional_play_count.values
row_array = mat_candidate.us_index_value.values
col_array = mat_candidate.so_index_value.values
data_sparse = coo_matrix((data_array, (row_array, col_array)),dtype=float)

# 3.3 SVD矩阵分解
import math as mt
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix
# 3.3.1 执行SVD，指定K值
def compute_svd(urm, K):
    U, s, Vt = svds(urm, K)
    dim = (len(s), len(s))
    S = np.zeros(dim, dtype=np.float32)
    for i in range(0, len(s)):
        S[i, i] = mt.sqrt(s[i])
    U = csc_matrix(U, dtype=np.float32)
    S = csc_matrix(S, dtype=np.float32)
    Vt = csc_matrix(Vt, dtype=np.float32)
    return U, S, Vt
# 参数选择
K = 50
urm = data_sparse
MAX_PID = urm.shape[1]
MAX_UID = urm.shape[0]
U, S, Vt = compute_svd(urm, K)

# 3.3.2 将小矩阵还原
def compute_estimated_matrix(urm, U, S, Vt, uTest, K, test):
    rightTerm = S * Vt
    max_recommendation = 250
    estimatedRatings = np.zeros(shape=(MAX_UID, MAX_PID), dtype=np.float16)
    recomendRatings = np.zeros(shape=(MAX_UID, max_recommendation), dtype=np.float16)
    for userTest in uTest:
        prod = U[userTest, :] * rightTerm
        estimatedRatings[userTest, :] = (
            prod.todense())
        recomendRatings[userTest, :] = (-estimatedRatings[userTest, :]).argsort()[:max_recommendation]
    return recomendRatings

# 3.4 选取测试用户
uTest = [4,5,6,7,8,873,23]
# 对用户进行推荐
uTest_recommended_items = compute_estimated_matrix(urm, U, S, Vt, uTest, K, True)
# 将结果按照得分值排序
for user in uTest:
    print("Recommendation for user with user id {}". format(user))
    rank_value = 1
    for i in uTest_recommended_items[user, 0:10]:
        song_details = small_set[small_set.so_index_value == i].drop_duplicates('so_index_value')[['title','artist_name']]
        print("The number {} recommended song is {} BY {}".format(rank_value, list(song_details['title'])[0],list(song_details['artist_name'])[0]))
        rank_value += 1


