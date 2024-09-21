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
            raise BaseException("unknown base:" + s)
    return out

TF_list = ["GABPA", "PRDM5", "SP140", "ZNF362", "ZNF407"]

batch_l = 1000
def generator(b_size=32, use_softmax=True):
    if use_softmax:
        y_len = 6
    else:
        y_len = 5
    
    while True:
        files = [0,1,2,3,4,5]
        # f_lens = [113_441, 112_946, 112_182, 116_988, 115_199, 202_965]
        f_lens = [100_000, 100_000, 100_000, 100_000, 100_000, 200_000]
        F = [open("G2A_train_data/" + TF + "_train.txt") for TF in TF_list]
        f_neg = open("G2A_train_data/negatives_train.txt")
        
        X = []
        Y = []
        for f_num in sample(files, counts=f_lens, k=sum(f_lens)):
            if f_num < 5: # positive
                f = F[f_num]
                X.append(f.readline().strip().upper())
                Y.append(f_num)
            else:
                X.append(f_neg.readline().strip().upper())
                Y.append(5)
            if len(X) == b_size:
                # b_length = max(len(x) for x in X)
                b_length = batch_l
                X_arr, Y_arr  = [], []
                for x,y in zip(X,Y):
                    if randint(0,1):
                        x = rev_comp(x)
                    # X_arr.append(onehot("N"*(b_length-len(x)) + x))
                    X_arr.append(onehot("X"*(b_length-len(x[:b_length])) + x[:b_length]))
                    y_arr = [0]*y_len
                    if y < y_len:
                        y_arr[y] += 1
                    Y_arr.append(y_arr)
                yield array(X_arr), array(Y_arr)
                X, Y = [], []
        
# batch_l = 1000


def tf_generator(b_size, TF, ratio=1):
    X = []
    Y = []

    F_names = ["G2A_train_data/negatives_train.txt.shuf"]
    F_names += [f"G2A_train_data/{tf}_train.txt.shuf" for tf in TF_list if tf != TF]
    F_neg = [open(f) for f in F_names]

    F_pos = open(f"G2A_train_data/{TF}_train.txt.shuf")
       
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
                F_pos = open(f"G2A_train_data/{TF}_train.txt.shuf")
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
                x = x[:batch_l]
                x = "X"*(batch_l-len(x)) + x
                X_arr.append(onehot(x))
                Y_arr.append(y)
            yield array(X_arr), array(Y_arr, dtype=float64)


            
TF_final = "CAMTA1  MYF6    SALL3  ZBED2  ZNF20   ZNF367  ZNF493   ZNF648".split()
TF_final += "LEUTX   PRDM13  USF3   ZBED5  ZNF251  ZNF395  ZNF518B".split()
def tf_generator_final(b_size, TF, ratio=1):
    X = []
    Y = []

    F_names = ["train_final/negatives_train.txt.shuf"]
    F_names += [f"train_final/g2a_train_data/{tf}_train.txt.shuf" for tf in TF_final if tf != TF]
    F_neg = [open(f) for f in F_names]

    F_pos = open(f"train_final/g2a_train_data/{TF}_train.txt.shuf")
       
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
                F_pos = open(f"train_final/g2a_train_data/{TF}_train.txt.shuf")
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
                x = x[:batch_l]
                x = "X"*(batch_l-len(x)) + x
                X_arr.append(onehot(x))
                Y_arr.append(y)
            yield array(X_arr), array(Y_arr, dtype=float64)

