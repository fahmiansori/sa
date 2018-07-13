import re

class ECSP():
    def stem(self,sentence):
        tokenized = sentence.lower()
        tokenized = tokenized.split(" ")
        no = 1
        for word in tokenized:
            print(no,self.doStemming(word))
            no+=1

    def isSixChar(self,term):
        if re.match(r'^[a-z]{1,5}$',term):  # cek apakah kurang dr 6 karakter
            return True
        return False

    def doStemming(self,term):
        # 1. Cek apakah karakter kurang dari 6
        if self.isSixChar(term):
            return term
        else:
            # 2. Cek apakah illegal affix
            if re.match(r'^(ke)([A-Za-z\-]+)(i|kan)$',term):
                return term
            elif re.match(r'^(se)([A-Za-z\-]+)(i|kan)$',term):
                return term
            elif re.match(r'^(peng)([A-Za-z\-]+)(i|kan)$',term):
                return term
            elif re.match(r'^(tar)([A-Za-z\-]+)(an)$',term):
                return term
            else:
                temp_term = ""

                # 3.1. hapus kata ganti kepunyaan yang berada di depan
                if re.search(r'^(ku|kau)\S{1,}', term):
                    if re.search(r'^(ku)[a]\S{4,}', term):
                        pass
                    else:
                        term = re.sub(r'^(ku|kau)','', term)
                    # if self.isSixChar(term):
                    #     return term
                # 3.2. awalan asing
                elif re.search(r'^(adi|antar|anti|dwi|eka|infra|maha|manca|multi|nara|pasca|pari|pramu|pra|sapta|semi|swa|tri|catur|ultra)\S{1,}', term):
                    term = re.sub(r'^(adi|antar|anti|dwi|eka|infra|maha|manca|multi|nara|pasca|pari|pramu|pra|sapta|semi|swa|tri|catur|ultra)','', term)
                    if self.isSixChar(term):
                        return term

                # 4.1. hapus inflectional particle
                if re.search(r'(lah|kah|tah|pun)$', term):
                    term = re.sub(r'(lah|kah|tah|pun)$','', term)
                    if self.isSixChar(term):
                        return term
                    # term = re.sub(r'kah$','', term)
                    # term = re.sub(r'tah$','', term)
                    # term = re.sub(r'pun$','', term)
                # 4.2. hapus possessive prounon
                elif re.search(r'(ku|mu|nya)$', term):
                    term = re.sub(r'(ku|mu|nya)$','', term)
                    if self.isSixChar(term):
                        return term
                    # term = re.sub(r'mu$','', term)
                    # term = re.sub(r'nya$','', term)

                # 5.1. hapus awalan yang tidak bermofologi
                if re.search(r'^(di|ke|se)\S{1,}', term):
                    if re.search(r'^((se)[r]|(kera))\S{1,}', term):
                        pass
                    elif re.search(r'^(seks|semangat|sempit|senonoh|diskusi)\S{1,}', term):
                        pass
                    elif re.search(r'^(ke)[c]\S{1,}', term):
                        pass
                    else:
                        term = re.sub(r'^(di|ke|se)','', term) #??? distribusi
                    if self.isSixChar(term):
                        return term
                    # term = re.sub(r'^ke','', term)#???
                    # term = re.sub(r'^se','', term)#???
                # 5.2. hapus awalan bermofologi
                if re.search(r'^(be|te|pe|me)\S{1,}', term): # modifikasi [^km]
                    if re.search(r'^(me)(?!ng)\S{1,}', term) and re.search(r'^(me)[^nm]\S{1,}', term):
                        term = re.sub(r'^(me)','', term)
                    if re.search(r'^(be|te|pe)[^kmrbl]\S{1,}', term):
                        if re.search(r'^(pe)[n]\S{1,}', term):
                            pass
                        else:
                            term = re.sub(r'^(be|te|pe)','', term) #???
                    if self.isSixChar(term):
                        return term

                if re.search(r'^(de)(?!ng)\S{6,}', term): # tambahan sendiri
                    term = re.sub(r'^(de)','', term)

                # implementasi pemenggalan kata, recording tabel
                if re.search(r'^(be|te|me|pe)\S{1,}', term):
                    if re.search(r'^(be)\S{1,}', term):
                        if re.search(r'^(ber)[aiueo]\S{1,}', term): #1
                            if not re.search(r'^(berita)\S{0,}', term):
                                term = re.sub(r'^(ber)','', term)
                        elif re.search(r'^(ber)[^aiueor]([A-Za-z\-]+)(?!er)\S{1,}', term): #2
                            term = re.sub(r'^(ber)','', term)
                        elif re.search(r'^(ber)[^aiueor]([A-Za-z\-]+)er[aiueo]\S{1,}', term): #3
                            term = re.sub(r'^(ber)','', term)
                        elif re.search(r'^belajar\S{0,}', term): #4
                            term = re.sub(r'^(bel)','', term)
                        elif re.search(r'^(be)[^aiueolr]er[^aiueo]\S{1,}', term): #5
                            term = re.sub(r'^(be)','', term)
                    elif re.search(r"^(te)\S{1,}", term):
                        if re.search(r'^(ter)[aiueo]\S{1,}', term): #6
                            temp_term = re.sub(r'^(ter)','', term)
                            # check to db, if not match check to the seconds
                            temp_term = re.sub(r'^(ter)','r', term)
                        elif re.search(r'^(ter)[^aiueor]er[aiueo]\S{1,}', term): #7
                            term = re.sub(r'^(ter)','', term)
                        elif re.search(r'^(ter)[^aiueor](?!er)\S{1,}', term): #8
                            term = re.sub(r'^(ter)','', term)
                        elif re.search(r'^(te)[^aiueor]er[^aiueo]\S{1,}', term): #9
                            term = re.sub(r'^(te)','', term)
                        elif re.search(r'^(ter)[^aiueor]er[^aiueo]\S{1,}', term): #34
                            term = re.sub(r'^(ter)','', term)
                    elif re.search(r"^(me)\S{1,}",term):
                        if re.search(r'^(me)[lrwyv][aiueo]', term): #10
                            term = re.sub(r'^(me)','', term)
                        elif re.search(r'^(mem)[bfvp]\S{1,}', term): #11
                            term = re.sub(r'^(mem)','', term)
                        elif re.search(r'^(mempe)\S{1,}', term): #12
                            term = re.sub(r'^(mem)','', term)
                        elif re.search(r'^(mem)((r[aiueo])|[aiueo])\S{1,}', term): #13
                            temp_term = re.sub(r'^(me)','', term)
                            # check to db, if not match check to the second
                            temp_term = re.sub(r'^(mem)','p', term)
                        elif re.search(r'^(men)[cdjszt]\S{1,}', term): #14
                            term = re.sub(r'^(men)','', term)
                        elif re.search(r'^(men)[aiueo]\S{1,}', term): #15
                            temp_term = re.sub(r'^(me)','', term)
                            # check to db, if not match check to the second
                            temp_term = re.sub(r'^(men)','t', term)
                        elif re.search(r'^(meng)[ghqk]\S{1,}', term): #16
                            term = re.sub(r'^(meng)','', term)
                        elif re.search(r'^(meng)[aiueo]\S{1,}', term): #17
                            # if re.search(r'^(menge)\S{1,}', term):
                            #     temp_term = re.sub(r'^(menge)','', term)
                            # else:
                            temp_term = re.sub(r'^(meng)','', term)
                            # check to db, if not match check to the second
                            temp_term = re.sub(r'^(meng)','k', term)
                        elif re.search(r'^(meny)[aiueo]\S{1,}', term): #18
                            term = re.sub(r'^(meny)','s', term)
                        elif re.search(r'^(memp)[^e]\S{1,}', term): #19
                            term = re.sub(r'^(mem)','', term)
                        elif re.search(r'^(mem)\S{1,}', term): #38
                            term = re.sub(r'^(mem)','', term)
                    elif re.search(r"^(pe)\S{1,}", term):
                        if re.search(r'^(pe)[wy][aiueo]\S{1,}', term): #20
                            term = re.sub(r'^(pe)','', term)
                        elif re.search(r'^(per)[aiueo]\S{1,}', term): #21
                            if re.search(r'^(per)[^i]\S{1,}', term):
                                temp_term = re.sub(r'^(per)','', term)
                                # check to db, if not match check to the second
                                temp_term = re.sub(r'^(pe)','', term)
                        elif re.search(r'^(per)[^aiueor]([A-Za-z\-]+)(?!er)\S{1,}', term): #22
                            term = re.sub(r'^(per)','', term)
                        elif re.search(r'^(per)[^aiueor]([A-Za-z\-]+)(er)[aiueo]\S{1,}', term): #23
                            term = re.sub(r'^(per)','', term)
                        elif re.search(r'^(pem)[bfv]\S{1,}', term): #24
                            term = re.sub(r'^(pem)','', term)
                        elif re.search(r'^(pem)(r[aiueo]|[aiueo])\S{1,}', term): #25
                            temp_term = re.sub(r'^(pe)','', term)
                            # check to db, if not match check to the second
                            temp_term = re.sub(r'^(pem)','p', term)
                        elif re.search(r'^(pen)[cdjz]\S{1,}', term): #26
                            term = re.sub(r'^(pen)','', term)
                        elif re.search(r'^(pen)[aiueo]\S{1,}', term): #27
                            temp_term = re.sub(r'^(pe)','', term)
                            # check to db, if not match check to the second
                            temp_term = re.sub(r'^(pen)','t', term)
                        elif re.search(r'^(peng)[^aiueo]\S{1,}', term): #28
                            term = re.sub(r'^(peng)','', term)
                        elif re.search(r'^(peng)[aiueo]\S{1,}', term): #29
                            if re.search(r'^(penge)\S{1,}', term):
                                temp_term = re.sub(r'^(penge)','', term)
                            else:
                                temp_term = re.sub(r'^(peng)','', term)
                                # check to db, if not match check to the second
                                temp_term = re.sub(r'^(peng)','k', term)
                        elif re.search(r'^(peny)[aiueo]\S{1,}', term): #30
                            temp_term = re.sub(r'^(peny)','s', term)
                            # check to db, if not match check to the second
                            temp_term = re.sub(r'^(pe)','', term)
                        elif re.search(r'^(pel)[aiueo]\S{1,}', term): #31
                            if not re.search(r'^(pelajar)\S{0,}', term):
                                term = re.sub(r'^(pe)','', term)
                            else:
                                term = re.sub(r'^(pel)','', term)
                        elif re.search(r'^(pe)[^rwylmn]er[aiueo]\S{1,}', term): #32
                            term = re.sub(r'^(pe)','', term)
                        elif re.search(r'^(pe)[^rwylmn](?!er)\S{1,}', term): #33
                            term = re.sub(r'^(pe)','', term)
                        elif re.search(r'^(pe)[^aiueorwylmn]er[^aiueo]\S{1,}', term): #35
                            term = re.sub(r'^(pe)','', term)
                        #36 entah kemana, ditabel aturan tidak ada!!!:3
                        elif re.search(r'^(pem)[aiueo]\S{1,}', term): #39
                            term = re.sub(r'^(pem)','p', term)
                        elif re.search(r'^(pe)[ctsz]\S{1,}', term): #40
                            term = re.sub(r'^(pe)','', term)
                    if temp_term:
                        term = temp_term
                    if self.isSixChar(term):
                        return term

                if re.search(r'^([^aiueo])e\1[^aiueo][aiueo]\S{1,}', term): #37 > Aturan belum fix, masih kurang jelas :v
                    # C1PC2V > C1P-C2V, dimana C1=C2 dan P=’e’
                    # if C1 == C2:
                    term = re.sub(r'^([^aiueo])e','', term)
                    if self.isSixChar(term):
                        return term

                # 7. hapus akhiran serapan asing
                if re.search(r'(wati|wan|isme|is|iah|isasi|er|wi|in|logi)$', term):
                    if re.search(r'^(notaris)$', term):
                        pass
                    else:
                        term = re.sub(r'(wati|wan|isme|is|iah|isasi|er|wi|in|logi)$','', term)
                    if self.isSixChar(term):
                        return term

                # 6. hapus akhiran
                if re.search(r'(i|kan|an)$', term):
                    if re.search(r'^(pribadi|handai|korban)$', term):
                        pass
                    else:
                        term = re.sub(r'(i|kan|an)$','', term)
                    if self.isSixChar(term):
                        return term

                # 8. hapus infiks atau sisipan
                # >>>>>>>>>>>>>>>>>>>>>>>>>>> CHECK POINT
                if re.match(r'([gjlt])el([aiueo])',term):
                    term = re.sub(r'el','', term)
                elif re.match(r'([cjkgt])em([aiueo])',term):
                    term = re.sub(r'em','', term)
                elif re.match(r'([sgkr])er([aiueo])',term):
                    term = re.sub(r'er','', term)
                elif re.match(r'([kst])in([aiueo])',term):
                    term = re.sub(r'in','', term)

        return term
