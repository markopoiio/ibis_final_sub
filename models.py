import tensorflow as tf
from tensorflow import argmax, convert_to_tensor, float32
from tensorflow.keras import layers
import tcn


def att_encoder(inputs, D, dropout, dilations=[1,2,4,8], k_size=3, noatt=False):
    activations = tcn.TCN(nb_filters=D, 
                          kernel_size=k_size,
                          dilations=dilations,
                          padding="same",
                          return_sequences=True,
                          dropout_rate=dropout,
                         ) (inputs)
    if noatt:
        return layers.LayerNormalization(epsilon=1e-6) (activations)
    
    attention = layers.Activation("tanh") (activations)
    attention = layers.Dense(1) (attention)
    attention = layers.Flatten() (attention)
    attention = layers.RepeatVector(D) (attention)
    attention = layers.Permute([2,1]) (attention)
    
    x = layers.Multiply() ([activations, attention])
    # x = x + inputs
    # x = tcn.TCN(nb_filters=D, 
    #             kernel_size=3, 
    #             dilations=[1,2], 
    #             padding="same",
    #             return_sequences=True,
    #             dropout_rate=dropout,
    #            ) (x)
    # x = layers.Conv1D(filters=D, kernel_size=1, activation="relu") (x)
    out = layers.LayerNormalization(epsilon=1e-6) (x + activations)
    return out

def build_model(input_shape,
                inner_dimension,
                num_att_blocks,
                mlp_units,
                # num_classes,
                att_dropout=0.3,
                mlp_dropout=0.3,

                local_dilations=[1,2,4,8],
                local_k=3,
                global_dilations=[128,256],
                global_k=3,
                # use_softmax=True,
                noatt=False,
                do_global=True,
         ):
    
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(inner_dimension) (inputs)
    for i in range(num_att_blocks):
        # x = att_encoder(x, inner_dimension, att_dropout, noatt=noatt, k_size=5, dilations=[1,3,9]) # local tcn -- это для PRDM5
        x = att_encoder(x, inner_dimension, att_dropout, noatt=noatt, k_size=local_k, dilations=local_dilations) # local tcn -- это для A2G
        if do_global:
            x = att_encoder(x, inner_dimension, att_dropout, dilations=global_dilations, noatt=noatt, k_size=global_k) # global tcn
    
    out = x
    out = layers.GlobalMaxPool1D(name="max_pool") (out)
    # out = layers.GlobalAveragePooling1D(name="avg_pool") (out)
    for d in mlp_units:
        out = layers.Dense(d) (out)
        out = layers.Dropout(mlp_dropout) (out)

    # if use_softmax:
        # out = layers.Dense(num_classes+1, activation="softmax") (out)
    # else:
    out = layers.Dense(1, activation="sigmoid", ) (out)
        
    model = tf.keras.models.Model(inputs=inputs, outputs=out)
    return model


def positional_encoder(ff_dim, inputs_shape):
    L, ff_dim = inputs_shape[1:]
    
    pe = tf.zeros((L, ff_dim))
    position = tf.range(0, L, dtype=tf.float32)
    position = tf.expand_dims(position, 1)
    div_term = tf.exp(
        tf.range(ff_dim, dtype=tf.float32) * (-tf.math.log(10000.0) / tf.cast(ff_dim, tf.float32))
                     )
    pe += tf.sin(position * div_term)
    pe += tf.cos(position * div_term)
    return pe


def self_attention_encoder(inputs, 
                           head_size, 
                           num_heads,
                           ff_dim, 
                           dropout
                     ):
    """Applies self attention to inputs 
    """
 
    x = layers.MultiHeadAttention(key_dim=head_size, 
                                  num_heads=num_heads,
                                  kernel_regularizer=tf.keras.regularizers.L2(1e-6)
                                 ) (inputs, inputs)
    x = layers.LayerNormalization(epsilon=1e-6) (x)
    res = x + inputs

    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu") (x)
    x = layers.Dropout(dropout) (x)
    out = layers.LayerNormalization(epsilon=1e-6) (x+res)
    
    return out

def build_model_self_att(
    input_shape,
    ff_dim,
    mlp_units,
    num_att_blocks,
    dropout=0,
    mlp_dropout=0,
):
    inputs = layers.Input(shape=input_shape)

    x = layers.Dense(ff_dim) (inputs)
    # x += positional_encoder(ff_dim, tf.shape(x))
    for _ in range(num_att_blocks):
        x += positional_encoder(ff_dim, tf.shape(x))
        x = self_attention_encoder(x, head_size=256, num_heads=4, ff_dim=ff_dim, dropout=dropout)        
    
    x = layers.GlobalMaxPool1D(name="maxpool") (x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu") (x)
        x = layers.Dropout(mlp_dropout) (x)
        
    outputs = layers.Dense(1, activation="sigmoid") (x)
    # outputs = layers.Flatten() (outputs)
    return tf.keras.models.Model(inputs, outputs)
