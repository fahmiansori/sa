import math
import pandas as pd

class InfoGain():
    def entropy(self,data,s,totaldata=0,totalalldata=0,col=''):
        attrs = data
        if col:
            attrs = data[col]
        if totaldata == 0:
            for c in attrs:
                totaldata+=s[c]
        i = 0
        for attr in attrs:
            try:
                i+=(-s[attr]/totaldata)*math.log2(s[attr]/totaldata)
            except:
                i+=0
        # print(i)
        try:
            prob = totaldata/totalalldata
        except:
            prob = 0
        # print(i,totaldata,'/',totalalldata,':',prob)
        en = {'i':i,'prob':prob}
        return en

    def gain(self,S,clas,totaldata,s):
        entropyClass = self.entropy(clas,s)

        for x in S:
            ig = 0
            for xx in S[x]:
                en = self.entropy(clas,S[x][xx],totalalldata=totaldata)
                # print('entropy',x,xx,':',en)
                ig += (en['prob']*en['i'])
            ig = entropyClass['i'] - ig
            print('information gain',x,':',ig)

    def run(self,data,exceptional_feature,colclas='clas'):
        cols = []
        dataframe = data['vsm']
        columns = data['column']
        totalclass = dataframe[colclas].value_counts()
        clas = getattr(dataframe,colclas)
        clas = clas.unique()
        totaldata = len(dataframe)

        nulval = dataframe[colclas].isnull().sum()

        if totalclass.empty or nulval > 0:
            print("There are an empty class in training data! Please check your data.")
        else:
            for i in columns:
                if i not in exceptional_feature:
                    cols.append(i)

            S = {}
            for i in cols:
                category = {}

                c_count = {}
                for c in clas:
                    c_count[c] = 0
                    for j in dataframe.loc[dataframe[colclas]==c,i]:
                        if j > 0:
                            c_count[c] += 1
                category['1'] = c_count

                c_count = {}
                for c in clas:
                    c_count[c] = 0
                    for j in dataframe.loc[dataframe[colclas]==c,i]:
                        if j < 1:
                            c_count[c] += 1
                category['0'] = c_count

                S[i] = category
                # break
            # print(S)
            s = {'no':3,'yes':3}
            ss = {}
            for c in clas:
                ss[c] = dataframe['clas'].where(dataframe['clas']==c).count()
            s = ss
            self.gain(S,clas,totaldata,ss)

        return True



# clas = ['no','yes']
# totaldata = 6
# # totaldata = 14
# s = {'no':3,'yes':3}
# # s = {'no':5,'yes':9}
#
# def info(data,s,totaldata=0,totalalldata=0,col=''):
#     attrs = data
#     if col:
#         attrs = data[col]
#     if totaldata == 0:
#         for c in attrs:
#             totaldata+=s[c]
#     i = 0
#     for attr in attrs:
#         try:
#             i+=(-s[attr]/totaldata)*math.log2(s[attr]/totaldata)
#         except:
#             i+=0
#     # print(i)
#     try:
#         prob = totaldata/totalalldata
#     except:
#         prob = 0
#     # print(i,totaldata,'/',totalalldata,':',prob)
#     en = {'i':i,'prob':prob}
#     return en
#
# entropyClass = info(clas,s)
# print(entropyClass['i'])
#
# S={}
# # S['outlook'] = {'overcast':{'no':0,'yes':4},'rainy':{'no':2,'yes':3},'sunny':{'no':3,'yes':2}}
# S['kecewa'] = {'1':{'no':1,'yes':0},'0':{'no':2,'yes':3}}
# S['perintah'] = {'1':{'no':2,'yes':1},'0':{'no':1,'yes':2}}
# S['sekarang'] = {'1':{'no':1,'yes':0},'0':{'no':2,'yes':3}}
# S['gagal'] = {'1':{'no':1,'yes':0},'0':{'no':2,'yes':3}}
# S['wujud'] = {'1':{'no':1,'yes':0},'0':{'no':2,'yes':3}}
# S['sejahtera'] = {'1':{'no':1,'yes':0},'0':{'no':2,'yes':3}}
# S['jokowi'] = {'1':{'no':1,'yes':3},'0':{'no':2,'yes':0}}
# S['kurang'] = {'1':{'no':1,'yes':0},'0':{'no':2,'yes':3}}
# S['tegas'] = {'1':{'no':1,'yes':0},'0':{'no':2,'yes':3}}
# S['pimpin'] = {'1':{'no':1,'yes':0},'0':{'no':2,'yes':3}}
# S['puas'] = {'1':{'no':0,'yes':1},'0':{'no':3,'yes':2}}
# S['terima'] = {'1':{'no':0,'yes':1},'0':{'no':3,'yes':2}}
# S['kasih'] = {'1':{'no':0,'yes':1},'0':{'no':3,'yes':2}}
# S['bangun'] = {'1':{'no':0,'yes':1},'0':{'no':3,'yes':2}}
# S['papua'] = {'1':{'no':0,'yes':1},'0':{'no':3,'yes':2}}
# S['banding'] = {'1':{'no':0,'yes':1},'0':{'no':3,'yes':2}}
# S['belum'] = {'1':{'no':0,'yes':1},'0':{'no':3,'yes':2}}
# S['baik'] = {'1':{'no':0,'yes':1},'0':{'no':3,'yes':2}}
#
# for x in S:
#     ig = 0
#     for xx in S[x]:
#         en = info(clas,S[x][xx],totalalldata=totaldata)
#         print('entropy',x,xx,':',en)
#         ig += (en['prob']*en['i'])
#     ig = entropyClass['i'] - ig
#     print('information gain',x,':',ig)
#
# #contoh hitung atribut jk, dgn 2 value
# # info_a_jk1 = -1* (())
