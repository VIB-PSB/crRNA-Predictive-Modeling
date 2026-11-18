#!/usr/bin/env python

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

class PearsonCorr(tf.keras.metrics.Metric):
    # Create a custom metric for the correlation coefficient #
    def __init__(self, **kwargs):
        super(PearsonCorr, self).__init__(**kwargs)
        self.correlation = self.add_weight(name="correlation", initializer="zeros")
        self.total_samples = self.add_weight(name="total_samples", initializer="zeros")
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        x = y_true
        y = y_pred
        
        xm = tf.math.reduce_mean(x)
        ym = tf.math.reduce_mean(y)
        
        xs = x - xm
        ys = y - ym        
        num = tf.math.reduce_sum(xs*ys)
        
        xs_sqrt = tf.math.sqrt(tf.math.reduce_sum(xs*xs))
        ys_sqrt = tf.math.sqrt(tf.math.reduce_sum(ys*ys))
        denom = xs_sqrt * ys_sqrt
        
        self.correlation.assign_add(num/(denom + 1e-5))
        self.total_samples.assign_add(1.0)         
    
    def result(self):
        return self.correlation/self.total_samples
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config}


class OneHotSeries(tf.keras.layers.Layer):
    def __init__(self, depth, **kwargs):
        super(OneHotSeries, self).__init__(**kwargs)
        self.depth = depth
        
    def call(self, inputs):
        oh_encoded_inputs = tf.one_hot(tf.cast(inputs, tf.int32), self.depth, dtype=tf.float32)
        return oh_encoded_inputs

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "depth": self.depth}


class OneHot(tf.keras.layers.Layer):
    def __init__(self, depth, **kwargs):
        super(OneHot, self).__init__(**kwargs)
        self.depth = depth
        
    def call(self, inputs):
        oh_encoded_inputs = tf.one_hot(tf.cast(inputs, tf.int32), self.depth, dtype=tf.float32)
        return tf.reshape(oh_encoded_inputs, [tf.shape(inputs)[0], tf.shape(inputs)[1]*self.depth])

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "depth": self.depth}


class Stratify(tf.keras.layers.Layer):
    def __init__(self, wt_threshold, hz_threshold, wt_level=0.0, he_level=1.0, hz_level=2.0, na_level=-1.0, **kwargs):
        super(Stratify, self).__init__(**kwargs)
        self.wt_threshold = wt_threshold
        self.hz_threshold = hz_threshold
        self.wt_level = wt_level
        self.he_level = he_level
        self.hz_level = hz_level
        self.na_level = na_level        
    
    def call(self, inputs):        
        x = inputs
        x = tf.where(((inputs >= 0.0) & (inputs < self.wt_threshold)), self.wt_level, x)
        x = tf.where(((inputs >= self.wt_threshold) & (inputs < self.hz_threshold)), self.he_level, x)
        x = tf.where((inputs >= self.hz_threshold), self.hz_level, x)
        x = tf.where((inputs < 0.0), self.na_level, x)
        return x
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "wt_threshold": self.wt_threshold, "hz_threshold": self.hz_threshold, "wt_level": self.wt_level, "he_level": self.he_level, "hz_level": self.hz_level, "na_level": self.na_level}


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, mask_zero=False, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.token_embeddings = layers.Embedding(input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mask_zero = mask_zero
        
    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_positions + embedded_tokens
    
    def compute_mask(self, inputs, mask=None):
        if self.mask_zero:
            return tf.math.not_equal(inputs, 0)
        else:
            return None
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "output_dim": self.output_dim, "sequence_length": self.sequence_length, "input_dim": self.input_dim, "mask_zero": self.mask_zero}


class Transformer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, neurons, num_heads, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.neurons = neurons
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense = keras.Sequential([layers.Dense(neurons, activation="relu"), layers.Dense(embed_dim)])
        self.layer_norm1 = layers.LayerNormalization()
        self.layer_norm2 = layers.LayerNormalization()
    
    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layer_norm1(inputs+attention_output)
        proj_output = self.dense(proj_input)
        return self.layer_norm2(proj_input + proj_output)
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "embed_dim": self.embed_dim, "neurons": self.neurons, "num_heads": self.num_heads}


def preprocess_RNA(inputs):
    f = tf.strings.bytes_split(inputs).to_tensor()
    x = tf.fill(dims=tf.shape(f), value=-1)
    x = tf.where(f=="T", 0, x)
    x = tf.where(f=="U", 0, x)
    x = tf.where(f=="A", 1, x)
    x = tf.where(f=="C", 2, x)
    x = tf.where(f=="G", 3, x)
    return x


def preprocess_RNA_embedding(inputs, seq_length, strides):
    base = inputs
    num = len(base[0])

    positions = tf.tile(tf.range(0, num-seq_length+1, strides)[tf.newaxis], [len(base), 1])
    length = tf.tile(tf.constant([seq_length])[tf.newaxis], tf.shape(positions))
    inputs_base = tf.tile(base[...,np.newaxis], [1, tf.shape(positions)[1]])
    
    return tf.strings.substr(input=inputs_base, pos=positions, len=length)
    

class RNA_tokenizer():
    def __init__(self, max_tokens=None):
        self.vocabulary = {}
        self.max_tokens=max_tokens
        self.counts = {}  
    
    def fit(self, inputs):
        uniques, counts = np.unique(inputs.numpy().reshape((-1,)), return_counts=True)
        sorted_index = np.argsort(counts)[::-1]
        uniques = uniques[sorted_index]
        counts = counts[sorted_index]
        self.counts = {k:v for k, v in zip(uniques, counts)}
        if self.max_tokens != None:
            uniques = uniques[:min(self.max_tokens, len(uniques))]
        self.vocabulary = {k:v for k, v in zip(uniques, np.arange(2,len(uniques)+2))}
        self.vocabulary[b'****'] = 1
    
    def transform(self, inputs):
        inputs_c = inputs.numpy()
        inputs_c = np.where(np.isin(inputs_c,list(self.vocabulary.keys())),inputs_c, b'****')
        output = np.vectorize(self.vocabulary.get)(inputs_c)
        return output
    
    def fit_transform(self, inputs):
        self.fit(inputs)
        return self.transform(inputs)


def size(model): # Compute number of params in a model (the actual number of floats)
    return sum([np.prod(K.get_value(w).shape) for w in model.trainable_weights])


def cross_validate(model_function, model_param, X_train, y_train, batch_size=32, folds=5, max_epochs=100, callbacks=[]):
    # Perform k-fold crossvalidation to find out what the optimal number of epochs is #
    length = len(X_train)//folds
    result = dict()
    model_size = 0

    for i in range(folds):
        model = model_function(**model_param)
        X_val_k = X_train[i*length:(i+1)*length,:]
        y_val_k = y_train[i*length:(i+1)*length]
        X_train_k = np.concatenate([X_train[:i*length,:], X_train[(i+1)*length:,:]], axis=0)
        y_train_k = np.concatenate([y_train[:i*length], y_train[(i+1)*length:]], axis=0)
        history = model.fit(X_train_k, y_train_k, validation_data=(X_val_k, y_val_k), batch_size=batch_size, epochs=max_epochs, verbose=0, callbacks=callbacks)
        result[i] = history.history
        if i==0:
            model_size = size(model)
    
    return result, model_size


def extract_best_performance(history_dict, par_to_extract):
    
    keys = list(history_dict.keys())
    result_mat = np.full(shape=[len(par_to_extract) + 1, len(keys) + 1], fill_value=np.nan)
    
    for key_num in range(len(keys)):
        epoch = np.argmin(history_dict[keys[key_num]]["val_loss"])
        result_mat[-1][key_num] = epoch + 1
        for par_num in range(len(par_to_extract)):
            result_mat[par_num][key_num] = history_dict[keys[key_num]][par_to_extract[par_num]][epoch]
    
    result_mat[:,-1] = np.mean(result_mat[:,:-1], axis=1)
    result = pd.DataFrame(result_mat, index=par_to_extract+["epoch"], columns=keys+["mean"])
    
    return result


def gridsearch_DNN(model_function, hyperparameters_dict, X_train, y_train, batch_size=32, folds=5, max_epochs=100, callbacks=[], par_to_extract=["val_loss"], output_name_base=""):
    import itertools
    #https://stackoverflow.com/questions/12935194/permutations-between-two-lists-of-unequal-length
    keys = list(hyperparameters_dict.keys())
    items = list(itertools.product(*list(hyperparameters_dict.values())))
    
    all_parameters = []
    all_results = []
    all_val_losses = []
    all_epochs = []
    all_sizes = []
    
    for item in items:
        run_param = dict()
        for i in range(len(keys)):
            run_param[keys[i]] = item[i]
        result, model_size = cross_validate(model_function=model_function, model_param=run_param, 
                                X_train=X_train, y_train=y_train, batch_size=batch_size, 
                                folds=folds, max_epochs=max_epochs, callbacks=callbacks)
        result = extract_best_performance(result, par_to_extract)
        
        
        all_parameters.append(run_param)
        all_results.append(result)
        all_val_losses.append(result.loc["val_loss","mean"])
        all_epochs.append(np.round(result.loc["epoch","mean"]))
        all_sizes.append(model_size)
    
    #Get the best parameters:
    best_index = np.argmin(all_val_losses)
    best_val_loss = all_val_losses[best_index]
    best_result = all_results[best_index]
    best_epoch = all_epochs[best_index]
    best_parameter = all_parameters[best_index]
    best_model_size = all_sizes[best_index]
    
    #Write out the best parameters
    best_df = pd.DataFrame({"parameter": ["run parameters", "index", "val_loss", "epoch", "size"], 
                            "value": [str(best_parameter), best_index, best_val_loss, best_epoch, best_model_size]})
    str_all_parameters = [str(i) for i in all_parameters]
    all_df = pd.DataFrame({"parameter": str_all_parameters, "val_loss": all_val_losses, "epoch": all_epochs, "size": all_sizes})
    
    best_df.to_csv(output_name_base+"_properties.csv", sep="\t", index=False)
    best_result.to_csv(output_name_base+"_cvresults.csv", sep="\t", index=True)
    all_df.to_csv(output_name_base+"_allresult.csv", sep="\t", index=False)
    
    return [best_parameter, best_epoch]


def train_full_model(model_function, best_param, best_epoch, X_train, y_train, X_test, y_test, batch_size=32, repeats=10, loss_fns=[], output_name_base=""):
    final_model = np.nan
    y_true = y_test
    
    result_mat = np.full(shape=[repeats, len(loss_fns)], fill_value=np.nan)
    
    for i in range(repeats + 1):
        model = model_function(**best_param)
        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=best_epoch, verbose=0)
        if i == repeats:
            final_model = model
        else:
            y_pred = model.predict(X_test)
            y_pred = y_pred.reshape((-1,))
            for num, loss_fn in enumerate(loss_fns):
                result_mat[i, num] = loss_fn(y_true, y_pred)
    
    fn_names = [str(i).split(" ")[1] for i in loss_fns]
    result = pd.DataFrame(result_mat, columns=fn_names)
    result.to_csv(output_name_base+"_full_model_results.csv", sep="\t", index=True)
    return final_model



def gridsearch_DNN_val(model_function, hyperparameters_dict, X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32, max_epochs=100, callbacks=[], output_name_base=""):
    import itertools
    #https://stackoverflow.com/questions/12935194/permutations-between-two-lists-of-unequal-length
    keys = list(hyperparameters_dict.keys())
    items = list(itertools.product(*list(hyperparameters_dict.values())))
    
    all_parameters = []
    all_results_mse = []
    all_results_pearson = []
    all_results_spearman = []    
    all_sizes = []
    
    best_model = ""    
    best_index = -1
    best_results_mse = -1
    best_results_pearson = -1
    best_results_spearman = -1
    best_parameter = -1
    best_model_size = -1
    
    for index, item in enumerate(items):
        run_param = dict()
        for i in range(len(keys)):
            run_param[keys[i]] = item[i]
        model = model_function(**run_param)
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=max_epochs, verbose=0, callbacks=callbacks)
        model_size = size(model)
        
        y_pred = model.predict(X_val, verbose=0)
        y_pred = y_pred.reshape((-1,))
        
        result_mse = MSE(y_val, y_pred)
        result_pearson = pearson_correlation(y_val, y_pred)
        result_spearman = spearman_correlation(y_val, y_pred)        
        
        all_parameters.append(run_param)        
        all_results_mse.append(result_mse)
        all_results_pearson.append(result_pearson)
        all_results_spearman.append(result_spearman)        
        all_sizes.append(model_size)
        
        #Get the best parameters:
        if (index == 0) | (result_mse < best_results_mse):
            
            best_model = model
            
            best_index = index            
            best_results_mse = result_mse
            best_results_pearson = result_pearson
            best_results_spearman = result_spearman   
            best_parameter = run_param
            best_model_size = model_size
    
    #Write out the best parameters    
    y_pred = best_model.predict(X_test, verbose=0)
    y_pred = y_pred.reshape((-1,))
    
    test_mse = MSE(y_test, y_pred)
    test_pearson = pearson_correlation(y_test, y_pred)
    test_spearman = spearman_correlation(y_test, y_pred) 
    
    best_df = pd.DataFrame({"parameter": ["run parameters", "index", "size", "validation mse", "validation pearson", "validation spearman", "test mse", "test pearson", "test spearman"], 
                            "value": [str(best_parameter), best_index, best_model_size, best_results_mse, best_results_pearson, best_results_spearman, test_mse, test_pearson, test_spearman]})
    str_all_parameters = [str(i) for i in all_parameters]
    all_df = pd.DataFrame({"parameter": str_all_parameters, "mse": all_results_mse, "pearson": all_results_pearson, "spearman": all_results_spearman, "size": all_sizes})
    
    best_df.to_csv(output_name_base+"_properties.csv", sep="\t", index=False)
    all_df.to_csv(output_name_base+"_allresult.csv", sep="\t", index=False)
      
    return best_model


def pearson_correlation(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0][1]


def spearman_correlation(y_true, y_pred):
    return spearmanr(y_true, y_pred)[0]


def MSE(y_true, y_pred):
    mse = tf.keras.metrics.MeanSquaredError()
    return mse(y_true, y_pred).numpy()


def MAE(y_true, y_pred):
    mae = tf.keras.metrics.MeanAbsoluteError()
    return mae(y_true, y_pred).numpy() 


def positional_encoding(length, depth):
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :]/depth

    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1) 

    return tf.cast(pos_encoding, dtype=tf.float32)
  
 
class PositionalEmbedding_X(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = layers.Embedding(vocab_size, d_model, mask_zero=True) 
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)        
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "d_model": self.d_model, "vocab_size": self.embedding.input_dim}


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = layers.MultiHeadAttention(**kwargs)
        self.layernorm = layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True)

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.dff = dff
        self.dropout_rate = dropout_rate
        
        self.seq = tf.keras.Sequential([
          layers.Dense(dff, activation='relu'),
          layers.Dense(d_model),
          layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x) 
        return x
       
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "d_model": self.d_model, "dff": self.dff, "dropout_rate": self.dropout_rate}


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "d_model": self.d_model, "num_heads": self.num_heads, "dff": self.dff, "dropout_rate": self.dropout_rate}


class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
        super().__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate

        self.pos_embedding = PositionalEmbedding_X(vocab_size=vocab_size, d_model=d_model)

        self.enc_layers = [EncoderLayer(d_model=d_model,
                                        num_heads=num_heads,
                                        dff=dff,
                                        dropout_rate=dropout_rate)
                                        for _ in range(num_layers)]
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x):
        x = self.pos_embedding(x)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "num_layers":self.num_layers, "d_model": self.d_model, "num_heads": self.num_heads, "dff": self.dff, "vocab_size": self.vocab_size, "dropout_rate": self.dropout_rate}
 

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.self_attention(x=x)
        x = self.cross_attention(x=x, context=context)
        x = self.ffn(x)
        return x
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "d_model": self.d_model, "num_heads": self.num_heads, "dff": self.dff, "dropout_rate": self.dropout_rate}


class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate

        self.pos_embedding = PositionalEmbedding_X(vocab_size=vocab_size, d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate) for _ in range(num_layers)]

    def call(self, x, context):
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
          x  = self.dec_layers[i](x, context)
        return x
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "num_layers": self.num_layers, "d_model": self.d_model, "num_heads": self.num_heads, "dff": self.dff, "vocab_size": self.vocab_size, "dropout_rate": self.dropout_rate}
        
