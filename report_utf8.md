Report
=============================

一个用Julia实现的Averaged Perceptron序列标注工具

(测试集上的结果在tst.result中)


Usage
--------------
julia> require("pumpkin_tagger.jl")
julia> t = Tagger();
julia> train_tagger(t, "datasets/all.crfsuite.txt", max_iter = 7);
julia> result = decode(t, "datasets/tst.crfsuite.txt")


File Format
-------------
这个工具采用了和crfsuite工具一样的输入文件格式以提高实用性，具体来说: 
    1. 每行一个词
    2. 一行用tab分割为多个域
    3. 一行的第一个域是正确的标签(即使是标签未知的测试集，也需要写一个任意的标签)
    4. 接下来每一个域表示一个特征，举例来说，如果出现了funny_feature_name这个域，则这个词funny_feature_name这个特征的取值为1
    5. 一个特征后面可以接一个":"字符然后跟一个实数值f，表示这个特征的取值不是1而是f。比如说某个域为continous_feature:0.005，则表示特征continous_feature的取值为0.005
    6. 特征"__BOS__"以及"__EOS__"分别用来标记一个句子的开始和结束


Algorithm
------------
采用的模型是二阶averaged perceptron，decode的算法是简单的Viterbi算法

训练算法伪代码如下：
    for t in 1:max_iter
        for i in sequence
            y* = argmax sum( λ(k, y*) * f(k, i), for active feature k )
            if y* != y_i
                for active feature k of word i:
                    λ(k, y*)  -= f(k, i)
                    λ(k, y_i) += f(k, i)
                end
            end
        end
        λ_avg += λ
    end


Feature Engineering
-----------
对于POS Tagging这个任务，我尝试了如下种类的特征：

    1. 周围的词，w[-2], w[-1], w[0], w[1], w[2], 分别表示往前两个词和往后两个词
    2. 周围词的bigram比如trigram, 比如w[-1]|w[0]，w[0]|w[1]，w[-1]|w[0]|w[1]
    3. 周围词的前缀、后缀，比如如果词是“北京“，那么把后缀”京“当作特征
    4. 把词在布朗聚类得到的层次树上的路径的所有前缀作为特征，Percy Liang 2005
    5. 把词的CCA embedding当作特征，Stratos 2015
    6. 把词的word embedding当作特征，Mikolov 2013

然后，仅仅使用上1,2,3的所有特征，模型的参数个数就达到了130万个左右，超过了训练集的样本个数60万，从而遇到严重的overfitting问题，所以最后只使用了周围5个词的unigram，周围两个词的bigram，前缀和后缀也限制到长度不超过2

另外，5中考虑使用的CCA embedding算法跑得太慢，所以没有用这个特征，word embedding则是因为每个词都加上100维的稠密的实数特征后也会使得训练过程变慢很多倍，所以也没有使用 :-(

对于布朗聚类的想法，因为布朗聚类的复杂度为O(N C^2)，所以C（类别个数）只取到100，所以似乎并没有发挥太大作用。。

所以最后在dev set得到的准确率只有惨淡的92%。。。.


Conclusion
----------
1. perceptron非常好实现，但是有收敛性的问题(a decreasing learning rate is necessary!) 以及overfitting(we need structured SVM!)
2. crfsuite的实现也有问题，在IO上非常的慢搞得完全跑不起来。。
3. 最终也没有达到孙薇薇老师说的一来就有94%准确率...
