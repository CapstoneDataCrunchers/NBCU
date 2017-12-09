import collections
import math 
import os
import random
import spacy
import zipfile
import numpy as np
import sys
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def read_data():
    
    #import stoplist
    nlp = spacy.load('en')
    stoplist = []
    with open('stoplist.txt') as f:
        for word in f.readlines():
            stoplist.append(word[:-1])
            
    #import scripts data, output plain text
    path     =  '/Users/zhanzhuoyang/Desktop/Columbia University/MSAA Curriculum/2017 Fall/Captsone/scripts/'
    namelist =  os.listdir(path)
    docs     =  []
    for name in namelist:
        if name[0] != '.':
            doc = []
            with open(path + name) as f:
                a = f.readlines()
                for line in a[1:]:
                    if (line[0].isdigit() == False) and (line != '\n'):
                        if (line[0] == '-'):
                            doc.append(line[1:-1])
                        elif (line[1] == '-'):
                            doc.append(line[2:-1])
                        else:
                            doc.append(line[:-1])
                docs.append(doc)
    return docs, stoplist


def bag_of_words(docs, stoplist):    #cut every episode into 4 same length chapter
    texts = []
    for doc in docs:
        text    =   []
        for line in doc:
            b = line.split()
            for word in b:
                if word[-1] in {'.',',','!','?'}:
                    word = word[:-1]
                if word.lower() not in stoplist:
                    text.append(word.lower())
            texts.append(text)
            
    return texts


def word2vec():
    if sys.version_info[0] >= 3:
        from urllib.request import urlretrieve
    else:
        from urllib import urlretrieve

    #Download data for word2vec model
    url = 'http://mattmahoney.net/dc/'

    def maybe_download(filename, excepted_bytes):
        if not os.path.exists(filename):
            filename, _ = urlretrieve(url + filename, filename)
        statinfo = os.stat(filename)
        if statinfo.st_size == excepted_bytes:
            print("Found and verified", filename)

        else:
            print(statinfo.st_size)
            raise Exception(
                "Failed to verfy" + filename + "Can you get to it with browser?")
        return filename
    filename = maybe_download('text8.zip', 31344016)

    #定义读取数据的函数，并把数据转成列表
    def read_data(filename):
        with zipfile.ZipFile(filename) as f:
            data = tf.compat.as_str(f.read(f.namelist()[0])).split()
        return data

    words = read_data(filename)
    print('Data size', len(words))

    #创建词汇表，选取前50000频数的单词，其余单词认定为Unknown,编号为0
    vocabulary_size = 50000

    def build_dataset(words):
        count = [['UNK', -1]]
        count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0
                unk_count += 1
            data.append(index)
        count[0][1] = unk_count
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return data, count, dictionary, reverse_dictionary
    data, count, dictionary, reverse_dictionary = build_dataset(words)

    #为了节约内存删除原始单词列表，打印出最高频出现的词汇及其数量
    del words
    print ('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

    #生成训练样本，assert断言：申明其布尔值必须为真的判定，如果发生异常，就表示为假
    data_index = 0

    def generate_batch(batch_size, num_skips, skip_window):
        global data_index
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape = (batch_size), dtype = np.int32)
        labels = np.ndarray(shape = (batch_size, 1), dtype = np.int32)
        span = 2 * skip_window + 1
        buffer = collections.deque(maxlen = span)
        for _ in range(span):
            buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)
        for i in range(batch_size // num_skips):
            target = skip_window
            targets_to_avoid = [skip_window]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[target]
            buffer.append(data[data_index])
            data_index = (data_index + 1)%len(data) 
        return batch, labels

    #调用generate_batch函数简单测试一下功能
    batch, labels = generate_batch(batch_size = 8, num_skips = 2, skip_window = 1)
    for i in range(8):
        print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

    #定义训练参数
    batch_size = 128
    embedding_size = 128
    skip_window = 1
    num_skips = 2

    valid_size = 16
    valid_window = 100
    valid_examples = np.random.choice(valid_window, valid_size, replace = False)
    num_sampled = 64

    #定义Skip-Gram Word2Vec模型的网络结构
    graph = tf.Graph()
    with graph.as_default():

        train_inputs = tf.placeholder(tf.int32, shape = [batch_size])
        train_labels = tf.placeholder(tf.int32, shape = [batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype = tf.int32)

        with tf.device('/cpu:0'):
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size], stddev = 1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weights,
                                            biases = nce_biases,
                                            labels = train_labels,
                                            inputs = embed,
                                            num_sampled = num_sampled,
                                            num_classes = vocabulary_size))

        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims = True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b = True)

        init = tf.global_variables_initializer()


    #定义最大迭代次数，创建并设置默认的session
    num_steps = 100001

    with tf.Session(graph = graph, config=tf.ConfigProto(log_device_placement=True)) as session:
        init.run()
        print("Initialized")

        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            _, loss_val = session.run([optimizer, loss], feed_dict = feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                print("Average loss at step ", step, ":", average_loss)
                average_loss = 0

            if step % 10000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8
                    nearest = (-sim[i, :]).argsort()[1: top_k+1]
                    log_str = "Nearest to %s:" % valid_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = "%s  %s," %(log_str, close_word)
                    print(log_str)
        final_embeddings = normalized_embeddings.eval()

def counter():
    a = collections.Counter(text)
    l = []
    for i in a.keys():
        freq = a[i]
        l.append((i,freq))

    sorted_i = sorted(l, key=lambda x:-x[1])
    sorted_text = [i[0] for i in sorted_i[:220]]

    id_list = []
    vectors = []
    for word in sorted_text:
        try:
            id_ = dictionary[word]
            vector = final_embeddings[id_]
            id_list.append(id_)
            vectors.append(vector)
        except:
            print(word)

    id_list = []
    vectors = []
    for word in sorted_text:
        try:
            id_ = dictionary[word]
            vector = final_embeddings[id_]
            id_list.append(id_)
            vectors.append(vector)
        except:
            print(word)
            
    return id_list
        
def plot_with_labels_(low_dim_embs, labels, filename = 'GFG_EPS301_s.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize= (18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy = (x, y), xytext= (5, 2), textcoords = 'offset points', ha = 'right', va = 'bottom')
    plt.savefig(filename) 

 


if __name__ == '__main__':
    docs, stoplist = read_data()
    texts = bag_of_words(docs, stoplist)
    #Only generate for episode one
    text = texts[0]
    word2vec()
    id_list = counter()
    tsne = TSNE(perplexity = 30, n_components = 2, init = 'pca', n_iter = 5000)
    low_dim_embs = tsne.fit_transform(vectors)
    labels = [reverse_dictionary[i] for i in id_list]
    plot_with_labels_(low_dim_embs, labels) 