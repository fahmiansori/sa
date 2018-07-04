import math
import pandas as pd

class InfoGain():
    def entropy(self,data,s,totaldata=0,totalalldata=0,col='',qc=None):
        attrs = data
        if col:
            attrs = data[col]
        if totaldata == 0:
            for c in attrs:
                totaldata+=s[c]
                if qc:
                    qc.processEvents()
        i = 0
        for attr in attrs:
            try:
                i+=(-s[attr]/totaldata)*math.log2(s[attr]/totaldata)
            except:
                i+=0
            if qc:
                qc.processEvents()
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

    def gain(self,S,clas,totaldata,s,qc=None):
        entropyClass = self.entropy(clas,s,qc=qc)

        igSubset = {}
        for subset in S:
            ig = 0
            for categoryVal in S[subset]:
                en = self.entropy(clas,S[subset][categoryVal],totalalldata=totaldata,qc=qc)
                # print('entropy',subset,category,':',en)
                ig += (en['prob']*en['i'])
                if qc:
                    qc.processEvents()
            ig = entropyClass['i'] - ig
            # print('information gain',subset,':',ig)
            igSubset[subset] = ig
            if qc:
                qc.processEvents()

        # so = sorted(igSubset,key=igSubset.__getitem__,reverse=True)
        # for i in so:
        #     print(i,":",igSubset[i])
        return igSubset

    def run(self,data,take_feature=0,threshold=0,exceptional_feature=[],colclas='clas',qc=None):
        cols = []
        take_feature=int(take_feature)
        threshold=float(threshold)
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
                if qc:
                    qc.processEvents()

            S = {}
            for i in cols:
                category = {}
                c_count = {}
                for c in clas:
                    c_count[c] = 0
                    for j in dataframe.loc[dataframe[colclas]==c,i]:
                        if j > 0:
                            c_count[c] += 1
                        if qc:
                            qc.processEvents()
                    if qc:
                        qc.processEvents()

                category['1'] = c_count # >> Karena hanya ada 2 kategori yg dipakai untuk kasus ini, yaitu muncul atau tidak, 1 untuk muncul

                c_count = {}
                for c in clas:
                    c_count[c] = 0
                    for j in dataframe.loc[dataframe[colclas]==c,i]:
                        if j < 1:
                            c_count[c] += 1
                        if qc:
                            qc.processEvents()
                    if qc:
                        qc.processEvents()
                category['0'] = c_count # >> Karena hanya ada 2 kategori yg dipakai untuk kasus ini, yaitu muncul atau tidak, 0 untuk ketidakmunculan
                S[i] = category
                # break
            # print(S)
            ss = {}
            for c in clas:
                ss[c] = dataframe[colclas].where(dataframe[colclas]==c).count()
                if qc:
                    qc.processEvents()
            s = ss
            igSubset = self.gain(S,clas,totaldata,ss,qc=qc)

            so = sorted(igSubset,key=igSubset.__getitem__,reverse=True)
            num = 1
            feature_to_delete = []
            column = []
            featureDf = pd.DataFrame(columns=['feature','ig'])
            text_t = {}
            for i in so:
                if igSubset[i] > threshold and num <= take_feature or take_feature == 0 and threshold == 0 or take_feature == 0 and igSubset[i] > threshold or num <= take_feature and threshold == 0:
                    column.append(i)
                    text_t['feature'] = i
                    text_t['ig'] = igSubset[i]
                    featureDf = featureDf.append(text_t,ignore_index=True)
                    # print(num,".",i,":",igSubset[i])
                else:
                    feature_to_delete.append(i)
                num+=1
                if qc:
                    qc.processEvents()
            columnlen = len(column)
            if len(feature_to_delete) > 0:
                for i in feature_to_delete:
                    dataframe.drop(i,axis=1,inplace=True) # axis = 1->kolom,0->rows,inplace=True->no asignment
                    if qc:
                        qc.processEvents()

            vs_model = {'vsm':dataframe,'column':column,'columnlen':columnlen,'feature':featureDf}
            return vs_model

        return False
