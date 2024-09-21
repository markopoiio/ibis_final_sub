from random import randint, sample
from numpy import array, zeros, float64

rev_comp_dict = {
    "A":"T",
    "C":"G",
    "G":"C",
    "T":"A",
    "N":"N",
    "W":"S",
    "S":"W",
    "M":"K",
    "K":"M",
    "R":"Y",
    "Y":"R",
    "B":"V",
    "V":"B",
    "D":"H",
    "H":"D",
    "X":"X",
}

def rev_comp(seq):
    out = ""
    for s in seq:
        out = rev_comp_dict[s] + out
    return out

def onehot(seq):
    out = zeros(shape=(len(seq.strip()), 4), dtype=float64)
    for i in range(len(seq.strip())):
        s = seq[i].upper()
        if s == "A":
            out[i][0] = 1.0
        elif s == "C":
            out[i][1] = 1.0
        elif s == "G":
            out[i][2] = 1.0
        elif s == "T":
            out[i][3] = 1.0
        elif s == "N":
            out[i][:] += .25
        elif s == "W":
            out[i, 0] = 1/2
            out[i, 3] = 1/2
        elif s == "S":
            out[i, 1] = 1/2
            out[i, 2] = 1/2
        elif s == "M":
            out[i, 0] = 1/2
            out[i, 1] = 1/2
        elif s == "K":
            out[i, 2] = 1/2
            out[i, 3] = 1/2
        elif s == "R":
            out[i, 0] = 1/2
            out[i, 2] = 1/2
        elif s == "Y":
            out[i, 1] = 1/2
            out[i, 3] = 1/2
        elif s == "B":
            out[i, 1] = 1/3
            out[i, 2] = 1/3
            out[i, 3] = 1/3
        elif s == "D":
            out[i, 0] = 1/3
            out[i, 2] = 1/3
            out[i, 3] = 1/3
        elif s == "H":
            out[i, 0] = 1/3
            out[i, 1] = 1/3
            out[i, 3] = 1/3
        elif s == "V":
            out[i, 0] = 1/3
            out[i, 2] = 1/3
        elif s == "X":
            pass
        else:
            print(s)
            raise BaseException("unknown base:" + s)
    return out

TF_list = ['LEF1', 'NACC2', 'NFKB1', 'RORB', 'TIGD3']
def tf_generator(b_size, TF, ratio=1):
    X = []
    Y = []

    F_names = ["G2A_train_data/negatives_train.txt.shuf"] ## оставляем рандомные негативы из генома
    F_names += [f"A2G_train_data/{tf}_train.txt.shuf" for tf in TF_list if tf != TF]
    F_neg = [open(f) for f in F_names]

    F_pos = open(f"A2G_train_data/{TF}_train.txt.shuf")
       
    pos = 0
    F_neg_id = 0
    while True:
       
        if pos == 0:
            for i in range(ratio):
                
                s = F_neg[F_neg_id].readline().strip().upper()
                if not s:
                    F_neg[F_neg_id].close()
                    F_neg[F_neg_id] = open(F_names[F_neg_id])
                    s = F_neg[F_neg_id].readline().strip().upper()
                # if "P" in s:
                #     # print(s)
                #     raise BaseException("found P in " + s)
                X.append(s)
                Y.append(pos)        
                F_neg_id = (F_neg_id + 1) % len(F_names)
        else:
            s = F_pos.readline().strip().upper()
            if not s:
                F_pos.close()
                F_pos = open(f"A2G_train_data/{TF}_train.txt.shuf")
                s = F_pos.readline().strip().upper()
            # if "P" in s:
            #     # print(s)
            #     raise BaseException("found P in " + s)
            X.append(s)
            Y.append(pos)        
        pos = 1 - pos

        if len(X) >= b_size:
            X_sent, Y_sent = X[:b_size], Y[:b_size]
            X, Y = X[b_size:], Y[b_size:]

            batch_l = 0
            for x, y in zip(X_sent,Y_sent):
                if y == 1:
                    batch_l = max(batch_l, len(x))
            X_arr = []
            Y_arr = []
            for x,y in zip(X_sent,Y_sent):
                if randint(0,1):
                    x = rev_comp(x)
                if len(x) > batch_l:
                    dislp = randint(0, len(x)-batch_l)
                    x = x[dislp:dislp+batch_l]
                else:
                    x = "X"*(batch_l-len(x)) + x
                X_arr.append(onehot(x))
                Y_arr.append(y)
            yield array(X_arr), array(Y_arr, dtype=float64)


TF_final = "CREB3L3 GCM1 MSANTD1 SP140L ZBTB47 ZNF286B ZNF721 ZNF831".split()
TF_final += "FIZ1 MKX MYPOP TPRX1 ZFTA ZNF500 ZNF780B".split()

def tf_generator_final(b_size, TF, ratio=1):
    X = []
    Y = []

    F_names = ["train_final/negatives_train.txt.shuf"] ## оставляем рандомные негативы из генома
    F_names += [f"train_final/a2g_train_data/{tf}_train.txt.shuf" for tf in TF_final if tf != TF]
    F_neg = [open(f) for f in F_names]

    F_pos = open(f"train_final/a2g_train_data/{TF}_train.txt.shuf")
       
    pos = 0
    F_neg_id = 0
    while True:
       
        if pos == 0:
            for i in range(ratio):
                
                s = F_neg[F_neg_id].readline().strip().upper()
                if not s:
                    F_neg[F_neg_id].close()
                    F_neg[F_neg_id] = open(F_names[F_neg_id])
                    s = F_neg[F_neg_id].readline().strip().upper()
                X.append(s)
                Y.append(pos)        
                F_neg_id = (F_neg_id + 1) % len(F_names)
        else:
            s = F_pos.readline().strip().upper()
            if not s:
                F_pos.close()
                F_pos = open(f"train_final/a2g_train_data/{TF}_train.txt.shuf")
                s = F_pos.readline().strip().upper()
            X.append(s)
            Y.append(pos)        
        pos = 1 - pos

        if len(X) >= b_size:
            X_sent, Y_sent = X[:b_size], Y[:b_size]
            X, Y = X[b_size:], Y[b_size:]

            batch_l = 0
            for x, y in zip(X_sent,Y_sent):
                if y == 1:
                    batch_l = max(batch_l, len(x))
            X_arr = []
            Y_arr = []
            for x,y in zip(X_sent,Y_sent):
                if randint(0,1):
                    x = rev_comp(x)
                if len(x) > batch_l:
                    dislp = randint(0, len(x)-batch_l)
                    x = x[dislp:dislp+batch_l]
                else:
                    x = "X"*(batch_l-len(x)) + x
                X_arr.append(onehot(x))
                Y_arr.append(y)
            yield array(X_arr), array(Y_arr, dtype=float64)
