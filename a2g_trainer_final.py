from random import randint
from a2g_dataset import tf_generator_final
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow import argmax, convert_to_tensor, float32
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import tcn
from utils_imported import get_wbfc
from models import build_model_self_att, build_model

# ---

b_size = 4096

TF_final = "CREB3L3 GCM1 MSANTD1 SP140L ZBTB47 ZNF286B ZNF721 ZNF831".split()
TF_final += "FIZ1 MKX MYPOP TPRX1 ZFTA ZNF500 ZNF780B".split()


for TF in TF_final:
    print(TF)
    model = build_model(
        input_shape=(None,4),
        inner_dimension=64,
        mlp_units=[16,8],
        num_att_blocks=1,
        att_dropout=.1,
        mlp_dropout=.1,
        noatt=True,
        local_dilations=[1,3,7],
        local_k=5,
        do_global=False
    )
    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["AUC", "Precision", "Recall"])
    
    steps = sum(1 for line in open(f"train_final/a2g_train_data/{TF}_train.txt.shuf")) // b_size // 4
    
    model.fit(x=tf_generator_final(b_size, TF, 1),
              batch_size=b_size,
              steps_per_epoch=steps,
              epochs=8,
              verbose=2)
    print(f"=== TF {TF} done ===")
    model.save(f"final_models/{TF}.keras")
