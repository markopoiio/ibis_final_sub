from random import randint
from g2a_dataset import tf_generator_final
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow import argmax, convert_to_tensor, float32
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import tcn
from utils_imported import get_wbfc

from models import build_model

# ---

b_size = 64
steps_balanced = 100_000 // b_size

TF_final = "CAMTA1  MYF6    SALL3  ZBED2  ZNF20   ZNF367  ZNF493   ZNF648".split()
TF_final += "LEUTX   PRDM13  USF3   ZBED5  ZNF251  ZNF395  ZNF518B".split()

for TF in TF_final:
    print(TF)
    model = build_model(
        input_shape=(None,4),
        inner_dimension=64,
        mlp_units=[16,8],
        num_att_blocks=1,
        att_dropout=.1,
        mlp_dropout=.1,
        do_global=False,
        local_dilations=[1,2,4,8],
        local_k=3,
        noatt=True,
    )
    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["AUC", "Precision", "Recall"])
    
    model.fit(x=tf_generator_final(b_size, TF, 1),
              batch_size=b_size,
              steps_per_epoch=steps_balanced,
              epochs=8,
              verbose=2)

    print(f"=== TF {TF} done ===")
    model.save(f"final_models/{TF}.keras")
