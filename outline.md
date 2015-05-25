outline
==================
1. 将输入格式转变为crfsuite格式
2. training:
       1. 找到所有label，所有特征f : Dict{String, Int}
          weights: Array{2, Float64} w[ label, feature ] -> weight
       2. for iter in 1:T
              for w in sequence
                  (predict)
                  features += last_tag
                  for active features f:
                      for each label l:
                          score[l] += w[ l, f ] * scale[f];
                  z = argmax score[l]
                  if z != y:
                      for active features f:
                          w[ y, f ] += scale[f]
                          w[ z, f ] -= scale[f]

  decoding:
       1. for w in setence:
            for features f:
                for each tag t:
                    score[t] += w[ f, t ];
            for each tag t:
                score[t] += max w[ t', t ] + f[w-1, t'] for all tag t'

feature extraction
----------------------
lexical features:
    1. words nearby : w[0]  w[1]  w[2]   w[-1]  w[-2]   w[-1]|w[0]   w[0]|w[1]
    2. suffixes of nearby word: w[0][-1]  w[0][-2:-1] ... 
