#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#参数解读https://blog.csdn.net/TeFuirnever/article/details/99818078


# ### 导入数据

# In[1]:


import numpy as np
import pandas as pd


# In[41]:


data = pd.read_excel('data_test_train.xlsx')
print(data.head())


# ### 朴素贝叶斯

# #### 数据预处理

# In[42]:


#根据需要做处理
#去重、去除停用词


# #### jieba分词

# In[43]:


import jieba

def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))

data['cut_comment'] = data.comment.apply(chinese_word_cut)


# In[44]:


data.head()


# #### 提取特征

# In[45]:


from sklearn.feature_extraction.text import CountVectorizer

def get_custom_stopwords(stop_words_file):
    with open(stop_words_file) as f:
        stopwords = f.read()
    stopwords_list = stopwords.split('\n')
    custom_stopwords_list = [i for i in stopwords_list]
    return custom_stopwords_list

stop_words_file = '哈工大停用词表.txt'
stopwords = get_custom_stopwords(stop_words_file)

vect = CountVectorizer(max_df = 0.8,
                       min_df = 3,
                       token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b',
                       stop_words=list(stopwords))


# #### 划分数据集

# In[46]:


#划分数据集
X = data['cut_comment']
y = data.sentiment

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)


# In[47]:
# print(vect.get_feature_names_out())

#特征展示
test = pd.DataFrame(vect.fit_transform(X_train).toarray(), columns=vect.get_feature_names_out())
print(test.head())



from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3) #,weights='distance'

X_train_vect = vect.fit_transform(X_train)
knn.fit(X_train_vect, y_train)
train_score = knn.score(X_train_vect, y_train)
print(train_score)  # train_score 模型在训练数据上的效果


# #### 测试模型

# In[33]:


X_test_vect = vect.transform(X_test)
print(knn.score(X_test_vect, y_test)) # 模型在测试数据上的效果


# #### 分析数据 

# In[11]:


data = pd.read_excel("data.xlsx")
data.head()


# In[12]:


data = pd.read_excel("data.xlsx")
data['cut_comment'] = data.comment.apply(chinese_word_cut)
X=data['cut_comment']
X_vec = vect.transform(X)
knn_result = knn.predict(X_vec)
#predict_proba(X)[source] 返回概率
data['knn_sentiment'] = knn_result


# In[13]:


data.to_excel("data_result.xlsx",index=False)


# In[ ]:




