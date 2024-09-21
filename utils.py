import numpy as np
def one_hot(seq):
    """Encode sequence to one hot, where A:1000, C:0100, G:0010, T:0001
    X corresponds to 0000
    """
    seq = seq.strip()
    encoded = np.zeros([len(seq), 4])
    for i, s in enumerate(seq):
        if s == "A":
            encoded[i, 0] = 1
        elif s == "C":
            encoded[i, 1] = 1
        elif s == "G":
            encoded[i, 2] = 1
        elif s == "T":
            encoded[i, 3] = 1
        elif s == "N":
            encoded[i,:] += .25
        elif s == "W":
            encoded[i, 0] = 1/2
            encoded[i, 3] = 1/2
        elif s == "S":
            encoded[i, 1] = 1/2
            encoded[i, 2] = 1/2
        elif s == "M":
            encoded[i, 0] = 1/2
            encoded[i, 1] = 1/2
        elif s == "K":
            encoded[i, 2] = 1/2
            encoded[i, 3] = 1/2
        elif s == "R":
            encoded[i, 0] = 1/2
            encoded[i, 2] = 1/2
        elif s == "Y":
            encoded[i, 1] = 1/2
            encoded[i, 3] = 1/2
        elif s == "B":
            encoded[i, 1] = 1/3
            encoded[i, 2] = 1/3
            encoded[i, 3] = 1/3
        elif s == "D":
            encoded[i, 0] = 1/3
            encoded[i, 2] = 1/3
            encoded[i, 3] = 1/3
        elif s == "H":
            encoded[i, 0] = 1/3
            encoded[i, 1] = 1/3
            encoded[i, 3] = 1/3
        elif s == "V":
            encoded[i, 0] = 1/3
            encoded[i, 1] = 1/3
            encoded[i, 2] = 1/3
        elif s == "X":
            continue
        else:
            print(f"Char '{s}' found in seq")
    return encoded

# from g2a_dataset import onehot

def prepare_batch(seqs):
    L = max(len(seq) for seq in seqs)
    # L = 1000
    X = []
    for seq in seqs:
        X.append( one_hot("X"*(L-len(seq[:L]))+seq[:L]) )
    return np.array(X)

def make_predictions(fout_name,
                     experiment,
                     tags,
                     columns,
                     model,
                     total_lines=1_000_000,
                     b_size=2048,
                    ):
    fout = open(fout_name, "w")
    print("tag" + ( "\t{}"*len(tags) ).format(*tags), file=fout)
    
    all_values = np.zeros( (total_lines, len(tags)) )
    all_keys = []

    written = 0

    with open(f"testdata/{experiment}_participants.fasta") as f:
        labels, seqs = [], []
        for meta, fasta in zip(f,f):
            label = meta.split(" ")[0][1:].strip()
            labels.append(label)
            seqs.append(fasta.strip())

            if len(labels) == b_size:
                batch = prepare_batch(seqs)
                # return batch
                predictions = model.predict(batch, batch_size=b_size, verbose=0)

                for label, pred in zip(labels, predictions):
                    all_values[written] = pred[columns]
                    # print(f"{label}\t{pred[0]:.5f}", file=fout)
                    written += 1
                print(written, end="\r")
                all_keys += labels
                labels, seqs = [], []

        if len(labels):
            batch = prepare_batch(seqs)
            predictions = model.predict(batch, batch_size=b_size, verbose=0)

            for label, pred in zip(labels, predictions):
                all_values[written] = pred[columns]
                written += 1
            print(written, end="\r")
            all_keys += labels

    all_values = all_values - all_values.min(axis=0)
    all_values = all_values / all_values.max(axis=0)
    for k, v in zip(all_keys, all_values):
        s = k + ( "\t{:.5f}"*len(v) ).format(*v)
        print(s, file=fout)

    fout.flush()
    fout.close()
    

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

    
def make_predictions_2strands(fout_name,
                              experiment,
                              tags,
                              columns,
                              model,
                              total_lines=1_000_000,
                              b_size=2048,
                              test_folder="testdata",
                             ):
    fout = open(fout_name, "w")
    print("tag" + ( "\t{}"*len(tags) ).format(*tags), file=fout)
    
    all_values = np.zeros( (total_lines, len(tags)) )
    all_keys = []

    written = 0

    with open(f"{test_folder}/{experiment}_participants.fasta") as f:
        labels, seqs = [], []
        for meta, fasta in zip(f,f):
            label = meta.split(" ")[0][1:].strip()
            labels.append(label)
            seqs.append(fasta.strip())

            if len(labels) == b_size:
                batch = prepare_batch(seqs)
                predictions_f = model.predict(batch, batch_size=b_size, verbose=0)
                
                seqs_r = [rev_comp(seq) for seq in seqs]
                batch_r = prepare_batch(seqs_r)
                predictions_r = model.predict(batch_r, batch_size=b_size, verbose=0)
            
                predictions = np.concatenate([predictions_f, predictions_r], axis=1)
                coef = np.abs(predictions - 0.5)
                weighted_predictions = (predictions * coef).sum(axis=1) / coef.sum(axis=1)
                
                for pred in weighted_predictions:
                    all_values[written] = pred
                    written += 1
                print(written, end="\r")
                all_keys += labels
                labels, seqs = [], []

        if len(labels):
            batch = prepare_batch(seqs)
            predictions_f = model.predict(batch, batch_size=b_size, verbose=0)

            seqs_r = [rev_comp(seq) for seq in seqs]
            batch_r = prepare_batch(seqs_r)
            predictions_r = model.predict(batch_r, batch_size=b_size, verbose=0)
            
            predictions = np.concatenate([predictions_f, predictions_r], axis=1)
            coef = np.abs(predictions - 0.5)
            weighted_predictions = (predictions * coef).sum(axis=1) / coef.sum(axis=1)
            
            for pred in weighted_predictions:
                all_values[written] = pred
                written += 1
            print(written, end="\r")
            all_keys += labels

    all_values = all_values - all_values.min(axis=0)
    all_values = all_values / all_values.max(axis=0)
    for k, v in zip(all_keys, all_values):
        s = k + ( "\t{:.5f}"*len(v) ).format(*v)
        print(s, file=fout)

    fout.flush()
    fout.close()
