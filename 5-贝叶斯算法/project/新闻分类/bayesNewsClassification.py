import pandas as pd
import jieba as jb
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
import jieba.analyse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#读取数据
df_news = pd.read_table('E:\\machineLearning\\kaggle\\dataset\\news_data.txt',
                        names=['category', 'theme', 'URL', 'content'], encoding='utf-8')
df_news = df_news.dropna()
df_news.head()

#数据清洗
content = df_news.content.values.tolist()
print(content[100])

#对文章进行分词
content_S = []
for line in content:
    current_segment = jb.lcut(line)
    if len(current_segment) > 1 and current_segment != '\r\n':
        content_S.append(current_segment)
#展示分词结果
df_content = pd.DataFrame({'content_S':content_S})
df_content.head()

#加载停用词
stopwords=pd.read_csv("E:\\machineLearning\\kaggle\\dataset\\stopwords.txt",
                      index_col=False,sep="\t",quoting=3,names=['stopword'], encoding='utf-8')
stopwords.head(10)

#构造停用词过滤函数
def drop_stopwords(contents, stopwords):
    contents_clean = []
    all_words = []
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
            all_words.append(str(word))
        contents_clean.append(line_clean)
    return contents_clean, all_words
contents = df_content.content_S.values.tolist()
stopwords = stopwords.stopword.values.tolist()
contents_clean, all_words = drop_stopwords(contents, stopwords)
df_content=pd.DataFrame({'contents_clean': contents_clean}) #分词后的清洗结果放列表里
df_all_words=pd.DataFrame({'all_words': all_words}) #分词后的清洗结果单个数据保存
df_content.tail()

#绘制词云图
#%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)
woordcloud_data = df_all_words.all_words.value_counts()[:100] #截取要呈现的目标数据
wordcloud = WordCloud(font_path="E:\\machineLearning\\kaggle\\projectResult\\simhei.ttf",
                    background_color="white", max_font_size=80) #初始化
wordcloud = wordcloud.fit_words(woordcloud_data)
plt.imshow(wordcloud)



#TF-IDF关键词提取
index = 2400
content_S_str = "".join(content_S[index]) #分词结果组合在一起，形成一个句子
print (content_S_str)
print ("  ".join(jieba.analyse.extract_tags(content_S_str, topK=5, withWeight=False))) #选出来5个核心词



#词袋模型
#数据预处理
#标签
df_train = pd.DataFrame({'contents_clean': contents_clean, 'label': df_news['category']})
df_train.tail()

#df_train.label.unique() 查看总共有多少个唯一的标签，然后在进行映射
label_mapping = {"汽车": 1, "财经": 2, "科技": 3, "健康": 4, "体育": 5, "教育": 6, "文化": 7, "军事": 8, "娱乐": 9, "时尚": 0}
df_train['label'] = df_train['label'].map(label_mapping) #构建一个映射方法
df_train.tail()

#数据集切分
x_train, x_test, y_train, y_test = train_test_split(df_train['contents_clean'].values,
                                                    df_train['label'].values, random_state=1)
words = []
#训练集数据格式转换
for line_index in range(len(x_train)):
    try:
        words.append(' '.join(x_train[line_index]))
    except:
        print()
test_words = []
#测试集数据格式转换
for line_index in range(len(x_test)):
    try:
        test_words.append(' '.join(x_test[line_index]))
    except:
         print()

#建模
vec = CountVectorizer(analyzer='word', max_features=4000, lowercase=False)
feature = vec.fit_transform(words)
print(feature.shape)

#贝叶斯模型
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(feature, y_train)
classifier.score(vec.transform(test_words), y_test)

#TF-IDF特征
vectorizer = TfidfVectorizer(analyzer='word', max_features=4000, lowercase=False)
vectorizer.fit(words)
classifier = MultinomialNB()
classifier.fit(vectorizer.transform(words), y_train)
classifier.score(vectorizer.transform(test_words), y_test)