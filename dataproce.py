import pandas as pd
import re
import os
import numpy as np
import csv
import math
import nltk
import sys
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import gensim
from gensim import corpora
from sklearn.decomposition import TruncatedSVD
from heapq import nlargest
import datetime
import pickle
EMPTY_VERTEX_NAME = ""
TITLE_VERTEX_NAME = "_TITLE_"
MAX_CONCEPTS_NUM = 50
fm=open("..\idfnew", "rb")
idf = pickle.load(fm)
glove = open('..\..\glove.6B.300d.txt', encoding='ISO-8859-1')
wordDict = {}
while (True):
    line = glove.readline()
    if line == '':
        break
    line = re.sub('[\n\r]','',line)
    line = line.split(' ')
    listtmp = list(map(float, line[1:]))
    wordDict[line[0]] = np.array(listtmp)
def compute_pc(X,npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_

def remove_pc(X, npc=1):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    pc = compute_pc(X, npc)
    if npc==1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX
#计算余弦相似度
def Comp(a,b):
    c = np.multiply(np.array(a),np.array(b))
    mo = np.linalg.norm(np.array(a))*np.linalg.norm(np.array(b))
    c = c.tolist()
    if mo<=0.00001:return 0.0
    return (1.0+sum(c)/mo)/2.0
def pre(fname):
    data = []
    f = open(fname,encoding='ISO-8859-1',newline = '')
    reader = csv.reader(f)
    stp = open('..\..\stopwords.txt', encoding='utf-8')
    glove = open('..\..\glove.6B.300d.txt',encoding='ISO-8859-1')
    glove = re.split('\n', glove.read())
    glove = [re.split(' ', i)[0] for i in glove]
    line = ''
    id = 1
    while(1):
        try:
            line = next(reader)
        except StopIteration:
            break
        if(line[0]=='index'):continue
        line[2] = re.sub(r'[(].[)]', ' ', line[2])
        for i in range(len(line[2])):
            if(i==0):continue
            if(i==len(line[2])-1):
                line[2]+='_'
                break
            if((line[2][i]=='.' or line[2][i]==';') and (line[2][i+1]!='.' and line[2][i-1]!='.')):
                line[2] = line[2][:i]+' _ '+line[2][i+1:]
        id+=1
        line[2] = re.sub(r'[^\w\s]', ' ', line[2])
        line[2] = re.sub('\n', ' ', line[2])
        line[2] = re.sub('\d', ' ', line[2])
        line[2] = [str.lower(i) for i in line[2].split() if str.lower(i) not in stp]
        line[1] = re.sub(r'[(].[)]', ' ', line[1])
        line[1] = re.sub(r'[^\w\s]', ' ', line[1])
        line[1] = re.sub('\n', ' ', line[1])
        line[1] = re.sub('\d', ' ', line[1])
        line[1] = [str.lower(i) for i in line[1].split() if str.lower(i) not in stp]
        tmplist = []
        head = 0
        cnt = 0
        for i in line[2]:
            if (i not in glove) and i!='_':
                line[2][cnt] = '*'
            if(line[2][cnt]=='_'or cnt == len(line[2])-1):
                tmplist.append([k for k in line[2][head:cnt] if k!='*'])
                head = cnt+1
            cnt += 1
        cnt = 0
        for i in line[1]:
            if i not in glove:
                line[1][cnt] = '*'
            cnt += 1

        data.append([line[0],[i for i in line[1] if i != '*'],tmplist])
    np.save(namehash[fname],np.array(data))
    return data

def hasPre():

    return os.path.isfile("..\dataSGBnewforju.npy")

def assign_sentences_to_concepts(sentences, concepts):
    """
    Assign a list of sentences to different concept communities.
    :param sentences: a list of sentences
    :param concepts: a list of concepts.
    :return: a dictionary of (concept_name, sentence index list)
    """
    concept_sentidxs = {}
    concept_sentidxs[EMPTY_VERTEX_NAME] = []
    assigned_sentidxs = []
    for concept in concepts:
        concept_sentidxs[concept] = []
        for i in range(len(sentences)-1):
            words = sentences[i]
            if concept in words:
                concept_sentidxs[concept].append(i)
                assigned_sentidxs.append(i)

    concept_sentidxs[EMPTY_VERTEX_NAME] = [x for x in range(len(sentences)-1)
                                           if x not in assigned_sentidxs]
    concept_sentidxs["TITLE"]=[len(sentences)-1]
    return concept_sentidxs

def keywords(text):
    tokenstf = {}
    toknum=0
    for i in ('doc1', 'doc2'):
        for token in text[i][0]:
            toknum += 1
            if (token not in idf.keys()): continue
            tokenstf[token] = 1 if token not in tokenstf.keys() else tokenstf[token] + 1
        for j in range(len(text[i][1])):
            for token in text[i][1][j]:
                toknum+=1
                if (token not in idf.keys()): continue
                tokenstf[token] = 1 if token not in tokenstf.keys() else tokenstf[token] + 1
    tmpcpt = []
    cnt=0
    for key, value in tokenstf.items():
        tmp=value

        tmp=float(tmp)/float(toknum)*idf[key]
        #print(key, tmp)
        if (tmp > 0.2):
            tmpcpt.append(key)
            cnt+=1
        if(cnt>MAX_CONCEPTS_NUM):break
    return tmpcpt


namehash={'..\..\SG bankruptcy act1.csv':'dataSGBnewforju','..\..\(done)Bankruptcy Ordinance HK.csv':'dataHKBnewforju', \
          '..\..\(done)Companies Ordinance HK.csv':'dataHKCnewforju','..\..\(done)Insolvency Act(UK)1986.csv':'dataUKBnewforju', \
          '..\..\(done)Companies Act(UK).csv':'dataUKCnewforju','..\..\(done)Companies Actpdfzhuan SG1.csv':'dataSGCnewforju'}

def embd(data):

    tot = 0
    a = 0.01

    Vec = []
    namelist = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            namelist.append({0:i,1:data[i][j][0],2:j})
            Vec.append([data[i][j][1],data[i][j][2]])
    return  namelist,Vec

def calc_text_pair_features(text1, text2):
    features = []
    #features.append(cal_bm25_sim(BM25MODEL, text1, text2))
    features.append(tfidf_cos_sim(text1, text2, idf))
    features.append(tf_cos_sim(text1, text2))
    # features.append(lcs(text1, text2))
    # features.append(num_common_words(text1, text2))
    features.append(jaccard_common_words(text1, text2))
    features.append(ochiai_common_words(text1, text2))
    # features.append(Levenshtein_distance(text1, text2))
    # features.append(Levenshtein_ratio(text1, text2))
    # features.append(Levenshtein_jaro(text1, text2))
    # features.append(Levenshtein_jaro_winkler(text1, text2))
    # NOTICE: add other features here
    return features
def jaccard_common_words(text1, text2):
    str1 = set(text1)
    str2 = set(text2)
    if len(str1) == 0 or len(str2) == 0:
        return 0.0
    return float(len(str1 & str2)) / len(str1 | str2)


def ochiai_common_words(text1, text2):
    str1 = set(text1)
    str2 = set(text2)
    if len(str1) == 0 or len(str2) == 0:
        return 0.0
    return float(len(str1 & str2)) / math.sqrt(len(str1) * len(str2))
def tfidf_cos_sim(text1,text2,idf):
    v1=[0.0 for x in range(300)]
    v2=[0.0 for x in range(300)]
    for token in text1:
        v1+=idf[token]*wordDict[token]
    for token in text2:
        v2+=idf[token]*wordDict[token]
    return Comp(v1,v2)

def tf_cos_sim(text1,text2):
    v1=[0.0 for x in range(300)]
    v2=[0.0 for x in range(300)]
    for token in text1:
        v1+=wordDict[token]
    for token in text2:
        v2+=wordDict[token]
    return Comp(v1,v2)

def CompNoTitleNode(v1,v2):
    text={'doc1':v1,'doc2':v2}
    csent = {}
    cpt = keywords(text)
    print(cpt)
    concept_vertexidxs_map = {}
    for tt in range(len(cpt)):
        concept_vertexidxs_map[cpt[tt]]=tt
    sentences = {'doc1':v1[1]+[v1[0]],'doc2':v2[1]+[v2[0]]}
    sents=v1[1]+v2[1]
    for j in ('doc1', 'doc2'):
        csent[j] = assign_sentences_to_concepts(sentences[j], cpt)

    cpt.append("")
    e = []  # (i,j)表明ij之间有一条边
    edgeV = [0.0 for x in range(3000)]  # 边的权值
    enear = [[] for x in range(len(cpt))]  # 边的邻接表
    edgeA = [[0 for x in range(MAX_CONCEPTS_NUM+2)] for xx in range(MAX_CONCEPTS_NUM+2)]  # 邻接矩阵
    nodeV = [[] for x in range(len(cpt))]
    vfeatures = [[0.0 for x in range(4)] for xx in range(len(cpt))]
    cnt=0
    for cp in cpt:
        text1=[]
        text2=[]
        for x in csent['doc1'][cp]:
            text1 += sentences['doc1'][x]
        for x in csent['doc2'][cp]:
            text2+= sentences['doc2'][x]
        vfeatures[cnt]=calc_text_pair_features(text1,text2)
        cnt+=1

    edge_idx_map={}#边和idx的映射，通过（i，j）的形式获得边的idx（在e中的idx）
    for sent_idx in range(len(sents)):
        sent = sents[sent_idx]
        words = sent
        intersect = set(words).intersection(set(cpt))
        if len(intersect) == 0:
            continue
        related_vertexidxs = []
        for c in intersect:
            related_vertexidxs.append(concept_vertexidxs_map[c])
        related_vertexidxs = list(set(related_vertexidxs))
        num_related_v = len(related_vertexidxs)
        if num_related_v < 2:
            continue
        for j in range(num_related_v):
            v1_idx = related_vertexidxs[j]
            tmp=[]
            for k in range(j, num_related_v):
                if j == k:
                    continue
                v2_idx = related_vertexidxs[k]
                source_idx = min(v1_idx, v2_idx)
                target_idx = max(v1_idx, v2_idx)
                if v2_idx not in tmp :
                    tmp.append(v2_idx)
                if (source_idx, target_idx) not in e:
                    e.append((source_idx, target_idx))
                    e.append((target_idx,source_idx))
                    edge_idx_map[(source_idx,target_idx)]=len(e)-1
            enear[v1_idx]=tmp
    for i in range(len(cpt)):
        for j in range(len(cpt)):
            if i==j:
                edgeA[i][j]=1.0
                continue
            if (i,j)not in edge_idx_map:continue
            text1=[]
            text2=[]
            for t1 in range(len(csent['doc1'][cpt[i]])):
                text1 += sentences['doc1'][t1]
            for t1 in range(len(csent['doc2'][cpt[i]])):
                text1 += sentences['doc2'][t1]
            for t2 in range(len(csent['doc1'][cpt[j]])):
                text2 += sentences['doc1'][t2]
            for t2 in range(len(csent['doc2'][cpt[j]])):
                text2 += sentences['doc2'][t2]
            edgeV[edge_idx_map[(i,j)]]=tfidf_cos_sim(text1,text2,idf)#利用concept之间的文本相似度计算边的权重
            edgeA[i][j]=edgeV[edge_idx_map[(i,j)]]
    return edgeA,vfeatures

def CompNoTitle(v1,v2):
    text={'doc1':v1,'doc2':v2}
    csent = {}
    cpt = keywords(text)
    print(cpt)
    concept_vertexidxs_map = {}
    for tt in range(len(cpt)):
        concept_vertexidxs_map[cpt[tt]]=tt
    sentences = {'doc1':v1[1],'doc2':v2[1]}
    sents=v1[1]+v2[1]
    for j in ('doc1', 'doc2'):
        csent[j] = assign_sentences_to_concepts(sentences[j], cpt)

    cpt.append("")
    e = []  # (i,j)表明ij之间有一条边
    edgeV = [0.0 for x in range(3000)]  # 边的权值
    enear = [[] for x in range(len(cpt))]  # 边的邻接表
    edgeA = [[0 for x in range(MAX_CONCEPTS_NUM+2)] for xx in range(MAX_CONCEPTS_NUM+2)]  # 邻接矩阵
    nodeV = [[] for x in range(len(cpt))]
    vfeatures = [[0.0 for x in range(4)] for xx in range(len(cpt))]
    cnt=0
    for cp in cpt:
        text1=[]
        text2=[]
        for x in csent['doc1'][cp]:
            text1 += sentences['doc1'][x]
        for x in csent['doc2'][cp]:
            text2+= sentences['doc2'][x]
        vfeatures[cnt]=calc_text_pair_features(text1,text2)
        cnt+=1

    edge_idx_map={}#边和idx的映射，通过（i，j）的形式获得边的idx（在e中的idx）
    for sent_idx in range(len(sents)):
        sent = sents[sent_idx]
        words = sent
        intersect = set(words).intersection(set(cpt))
        if len(intersect) == 0:
            continue
        related_vertexidxs = []
        for c in intersect:
            related_vertexidxs.append(concept_vertexidxs_map[c])
        related_vertexidxs = list(set(related_vertexidxs))
        num_related_v = len(related_vertexidxs)
        if num_related_v < 2:
            continue
        for j in range(num_related_v):
            v1_idx = related_vertexidxs[j]
            tmp=[]
            for k in range(j, num_related_v):
                if j == k:
                    continue
                v2_idx = related_vertexidxs[k]
                source_idx = min(v1_idx, v2_idx)
                target_idx = max(v1_idx, v2_idx)
                if v2_idx not in tmp :
                    tmp.append(v2_idx)
                if (source_idx, target_idx) not in e:
                    e.append((source_idx, target_idx))
                    e.append((target_idx,source_idx))
                    edge_idx_map[(source_idx,target_idx)]=len(e)-1
            enear[v1_idx]=tmp
    for i in range(len(cpt)):
        for j in range(len(cpt)):
            if i==j:
                edgeA[i][j]=1.0
                continue
            if (i,j)not in edge_idx_map:continue
            text1=[]
            text2=[]
            for t1 in range(len(csent['doc1'][cpt[i]])):
                text1 += sentences['doc1'][t1]
            for t1 in range(len(csent['doc2'][cpt[i]])):
                text1 += sentences['doc2'][t1]
            for t2 in range(len(csent['doc1'][cpt[j]])):
                text2 += sentences['doc1'][t2]
            for t2 in range(len(csent['doc2'][cpt[j]])):
                text2 += sentences['doc2'][t2]
            edgeV[edge_idx_map[(i,j)]]=tfidf_cos_sim(text1,text2,idf)#利用concept之间的文本相似度计算边的权重
            edgeA[i][j]=edgeV[edge_idx_map[(i,j)]]
    return edgeA,vfeatures

def Comp1(v1,v2):
    text={'doc1':v1,'doc2':v2}
    csent = {}
    cpt = keywords(text)
    print(cpt)
    concept_vertexidxs_map = {}
    for tt in range(len(cpt)):
        concept_vertexidxs_map[cpt[tt]]=tt
    sentences = {'doc1':v1[1]+[v1[0]],'doc2':v2[1]+[v2[0]]}
    sents=v1[1]+v2[1]
    for j in ('doc1', 'doc2'):
        csent[j] = assign_sentences_to_concepts(sentences[j], cpt)

    cpt.append("")
    cpt.append("TITLE")
    e = []  # (i,j)表明ij之间有一条边
    edgeV = [0.0 for x in range(3000)]  # 边的权值
    enear = [[] for x in range(len(cpt))]  # 边的邻接表
    edgeA = [[0 for x in range(MAX_CONCEPTS_NUM+3)] for xx in range(MAX_CONCEPTS_NUM+3)]  # 邻接矩阵
    nodeV = [[] for x in range(len(cpt))]
    vfeatures = [[0.0 for x in range(4)] for xx in range(len(cpt))]
    cnt=0
    texts=[]
    for cp in cpt:
        text1=[]
        text2=[]
        for x in csent['doc1'][cp]:
            text1 += sentences['doc1'][x]
        for x in csent['doc2'][cp]:
            text2+= sentences['doc2'][x]
        texts.append([text1,text2])
        vfeatures[cnt]=calc_text_pair_features(text1,text2)
        cnt+=1

    edge_idx_map={}#边和idx的映射，通过（i，j）的形式获得边的idx（在e中的idx）
    for sent_idx in range(len(sents)):
        sent = sents[sent_idx]
        words = sent
        intersect = set(words).intersection(set(cpt))
        if len(intersect) == 0:
            continue
        related_vertexidxs = []
        for c in intersect:
            related_vertexidxs.append(concept_vertexidxs_map[c])
        related_vertexidxs = list(set(related_vertexidxs))
        num_related_v = len(related_vertexidxs)
        if num_related_v < 2:
            continue
        for j in range(num_related_v):
            v1_idx = related_vertexidxs[j]
            tmp=[]
            for k in range(j, num_related_v):
                if j == k:
                    continue
                v2_idx = related_vertexidxs[k]
                source_idx = min(v1_idx, v2_idx)
                target_idx = max(v1_idx, v2_idx)
                if v2_idx not in tmp :
                    tmp.append(v2_idx)
                if (source_idx, target_idx) not in e:
                    e.append((source_idx, target_idx))
                    e.append((target_idx,source_idx))
                    edge_idx_map[(source_idx,target_idx)]=len(e)-1
            enear[v1_idx]=tmp
    for i in range(len(cpt)):
        for j in range(len(cpt)):
            if i==j:
                edgeA[i][j]=1.0
                continue
            if (i,j)not in edge_idx_map:continue
            text1=[]
            text2=[]
            for t1 in range(len(csent['doc1'][cpt[i]])):
                text1 += sentences['doc1'][t1]
            for t1 in range(len(csent['doc2'][cpt[i]])):
                text1 += sentences['doc2'][t1]
            for t2 in range(len(csent['doc1'][cpt[j]])):
                text2 += sentences['doc1'][t2]
            for t2 in range(len(csent['doc2'][cpt[j]])):
                text2 += sentences['doc2'][t2]
            edgeV[edge_idx_map[(i,j)]]=tfidf_cos_sim(text1,text2,idf)#利用concept之间的文本相似度计算边的权重
            if i==len(cpt)-1 or j==len(cpt)-1:
                edgeA[i][j]=0
            else:
                edgeA[i][j]=edgeV[edge_idx_map[(i,j)]]

    return edgeA,vfeatures,texts


def loadinfile():
    infile=open("label.txt",'r')
    infile=infile.readlines()
    inp=[{0:0,1:0,2:-1} for x in range(len(infile))]
    cnt=0
    for line in infile:
        tmp = line.split('|')
        tmp[2] = tmp[2][0]
        inp[cnt][0]=tmp[0]
        inp[cnt][1] = tmp[1]
        inp[cnt][2] = int(tmp[2])
        cnt+=1
    return inp

def getidx(namelist):
    return '0'+str(namelist[0])+namelist[1]

def getlawid(namelist):
    return str(namelist[0])

def getid(namelist):
    return str(namelist[1])
def cal(namelist,lenlist,InputV,dataV):
    sim = []
    #inp = loadinfile()
    #of=open('prcdata+titlewithtexts','wb')
    of = open('sgb2ukb','wb')
    dataout=[]
    for i in range(len(InputV)):
        tmplist = []
        idx1 = getlawid(namelist[i])
        for j in range(len(dataV)):
            idx2=getlawid(namelist[len(InputV)+j])
            #for x in inp:
            if(idx1=='0'and idx2=='3'):
                tmpdict={}
                tmpdict[0]=getid(namelist[i])
                tmpdict[1]=getid(namelist[len(InputV)+j])
                print(tmpdict[0],tmpdict[1])
                adj,v_features,texts=Comp1(InputV[i], dataV[j])
                #print(adj)
                #print(v_features)
                tmpdict[3]=adj
                tmpdict[4]=v_features
                tmpdict[5]=0
                tmpdict[6]=texts
                dataout.append(tmpdict)
    pickle.dump(dataout,of)

def main():
    data = [[],[],[],[],[],[]]
    if(not hasPre()):
        dataSGB = pre('..\..\SG bankruptcy act1.csv')
        dataHKB = pre('..\..\(done)Bankruptcy Ordinance HK.csv')
        dataHKC = pre('..\..\(done)Companies Ordinance HK.csv')
        dataUKB = pre('..\..\(done)Insolvency Act(UK)1986.csv')
        dataUKC = pre('..\..\(done)Companies Act(UK).csv')
        dataSGC = pre('..\..\(done)Companies Actpdfzhuan SG1.csv')
    else:
        data[0] =  dataSGB = np.load('..\dataSGBnewforju.npy',allow_pickle=True).tolist()
        data[1] = dataHKB = np.load('..\dataHKBnewforju.npy',allow_pickle=True).tolist()
        data[2] = dataHKC = np.load('..\dataHKCnewforju.npy',allow_pickle=True).tolist()
        data[3] = dataUKB = np.load('..\dataUKBnewforju.npy',allow_pickle=True).tolist()
        data[4] = dataUKC = np.load('..\dataUKCnewforju.npy',allow_pickle=True).tolist()
        data[5] = dataSGC = np.load('..\dataSGCnewforju.npy',allow_pickle=True).tolist()
    indexname,vec = embd(data)
    lenlist = [len(data[1]),len(data[2]),len(data[3]),len(data[4]),len(data[5])]
    sim = [[],[],[],[],[]]
    cal(indexname,lenlist,vec[0:len(data[0])],vec[len(data[0]):])



main()
