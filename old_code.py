import codecs


def delblankline(infile1,infile2,trainfile,validfile,testfile):
#### 2 是test，3 是valid的我写错了

    info1 = codecs.open(infile1,'r', 'UTF-8')
    info2 = codecs.open(infile2,'r', 'UTF-8')
    train= codecs.open(trainfile,'w', 'UTF-8')
    valid= codecs.open(validfile,'w', 'UTF-8')
    test=codecs.open(testfile,'w', 'UTF-8')
    lines1 = info1.readlines()
    lines2 = info2.readlines()
    for i in range(1,len(lines1)):
        t1=lines1[i].replace("-LRB-","(")
        t2=t1.replace("-RRB-",")")
        ###把括号部分还原
        k=lines2[i].strip().split(",")
        t=t2.strip().split('\t')
        if k[1]=='1':
            train.writelines(t[1])
            train.writelines("\n")
        elif(k[1]=='2'):
            test.writelines(t[1])
            test.writelines("\n")
        elif(k[1]=='3'):
            valid.writelines(t[1])
            valid.writelines("\n")
    print("end")
    info1.close()
    info2.close()
    train.close()
    valid.close()
    test.close()

def tag(infile1,infile2,outputfile3):
    info1 = codecs.open(infile1,'r', 'UTF-8')
    info2 = codecs.open(infile2,'r', 'UTF-8')
    info3=codecs.open(outputfile3,'w', 'UTF-8')
    lines1 = info1.readlines()
    lines2 = info2.readlines()
    text={}
    for i in range(0,len(lines1)):
        s=lines1[i].strip().split("|")
        text[s[1]]=s[0]
    for j in range(1,len(lines2)):
        k=lines2[j].strip().split("|")
        if text.get(k[0]) is not None:
            info3.writelines(text[k[0]])
            info3.writelines("\n")
            info3.writelines(k[1])
            info3.writelines("\n")
        else:
            print("{} not exist in dictionary".format(k[0]))


    print("end2d1")
    info1.close()
    info2.close()
    info3.close()


def tag1(infile0,infile1,infile2,infile3,infile4,infile5,infile6):
    info0 = codecs.open(infile0,'r', 'UTF-8')
    info1 = codecs.open(infile1,'r', 'UTF-8')
    info2 = codecs.open(infile2,'r', 'UTF-8')
    info3 = codecs.open(infile3,'r', 'UTF-8')
    info4 = codecs.open(infile4,'w', 'UTF-8')
    info5 = codecs.open(infile5,'w', 'UTF-8')
    info6 = codecs.open(infile6,'w', 'UTF-8')
    lines0 = info0.readlines()
    lines1 = info1.readlines()
    lines2 = info2.readlines()
    lines3 = info3.readlines()
    for i in range(0,len(lines0),2):
        if lines0[i].strip() in lines1:
            info4.writelines(lines0[i])
            info4.writelines(lines0[i+1])
        if lines0[i].strip() in lines2:
            info5.writelines(lines0[i])
            info5.writelines(lines0[i+1])
        if lines0[i].strip() in lines3:
            info6.writelines(lines0[i])
            info6.writelines(lines0[i+1])

    print("end3d1")
    info0.close()
    info1.close()
    info2.close()
    info3.close()
    info4.close()
    info5.close()
    info6.close()



if __name__ == "__main__":
    # delblankline("/Users/feili/dataset/sst/stanfordSentimentTreebank/datasetSentences.txt",
    #              "/Users/feili/dataset/sst/stanfordSentimentTreebank/datasetSplit.txt",
    #              "sst/train.txt",
    #              "sst/valid.txt",
    #              "sst/test.txt")

    # tag("/Users/feili/dataset/sst/stanfordSentimentTreebank/dictionary.txt",
    #     "/Users/feili/dataset/sst/stanfordSentimentTreebank/sentiment_labels.txt",
    #     "sst/allsentimet.txt")

    tag1("sst/allsentimet.txt", "sst/train.txt", "sst/valid.txt", "sst/test.txt",
         "sst/train1.txt", "sst/valid1.txt", "sst/test1.txt")