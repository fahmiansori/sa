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

        # try:
        #     prob = totaldata/totalalldata
        # except:
        #     prob = 0
        if totalalldata != 0:
            prob = totaldata/totalalldata
        else:
            prob = 0
        # print(i,totaldata,'/',totalalldata,':',prob)
        en = {'i':i,'prob':prob}
        return en

    def gain(self,S,clas,totaldata,s):
        entropyClass = self.entropy(clas,s)

        igSubset = {}
        for subset in S:
            ig = 0
            for categoryVal in S[subset]:
                en = self.entropy(clas,S[subset][categoryVal],totalalldata=totaldata)
                # print('entropy',subset,category,':',en)
                ig += (en['prob']*en['i'])
            ig = entropyClass['i'] - ig
            # print('information gain',subset,':',ig)
            igSubset[subset] = ig

        # so = sorted(igSubset,key=igSubset.__getitem__,reverse=True)
        # for i in so:
        #     print(i,":",igSubset[i])
        return igSubset

    def run(self,data,threshold=-0.1,take_feature=0,exceptional_feature=[],colclas='clas'):
        cols = []
        dataframe = data['vsm']
        columns = data['column']
        totalclass = dataframe[colclas].value_counts()
        clas = getattr(dataframe,colclas)
        clas = clas.unique()
        totaldata = len(dataframe)

        nulval = dataframe[colclas].isnull().sum()

        if totalclass.empty or nulval > 0:
            print("There are an empty class or more in training data! Please check your data.")
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
                category['1'] = c_count # >> Karena hanya ada 2 kategori yg dipakai untuk kasus ini, yaitu muncul atau tidak, 1 untuk muncul

                c_count = {}
                for c in clas:
                    c_count[c] = 0
                    for j in dataframe.loc[dataframe[colclas]==c,i]:
                        if j < 1:
                            c_count[c] += 1
                category['0'] = c_count # >> Karena hanya ada 2 kategori yg dipakai untuk kasus ini, yaitu muncul atau tidak, 0 untuk ketidakmunculan
                S[i] = category
                # break
            # print(S)
            ss = {}
            for c in clas:
                ss[c] = dataframe[colclas].where(dataframe[colclas]==c).count()
            s = ss
            igSubset = self.gain(S,clas,totaldata,ss)

            so = sorted(igSubset,key=igSubset.__getitem__,reverse=True)
            num = 1
            feature_to_delete = []
            column = []
            for i in so:
                if igSubset[i] > threshold and num <= take_feature or take_feature == 0:
                    column.append(i)
                    # print(num,".",i,":",igSubset[i])
                else:
                    feature_to_delete.append(i)
                num+=1
            columnlen = len(column)
            if len(feature_to_delete) > 0:
                for i in feature_to_delete:
                    dataframe.drop(i,axis=1,inplace=True) # axis = 1->kolom,0->rows,inplace=True->no asignment

            vs_model = {'vsm':dataframe,'column':column,'columnlen':columnlen}
            return vs_model

        return False
