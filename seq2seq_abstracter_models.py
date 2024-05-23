import tensorflow as tf
from tensorflow.keras import layers, Model,Sequential
import time
from transformer_modules import Encoder, Decoder, AddPositionalEmbedding,FeedForward
from abstracters import SymbolicAbstracter, RelationalAbstracter, AblationAbstractor
from tensorflow.keras.regularizers import l2
from multi_head_relation import MultiHeadRelation

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
tf.random.set_seed(2023)



from tensorflow.keras import layers

class BipolarDenseLayer(layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(BipolarDenseLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer="glorot_uniform",
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer="zeros",
                                 trainable=True)

    def call(self, inputs, training=False):
        weights = tf.math.sign(self.w)
        output = tf.matmul(inputs, weights) + self.b
        if self.activation is not None:
            output = self.activation(output)
        return output
                    


class HDSymoblicAttention(Layer):
    def __init__(self, d_model, dim, n_seq, **kwargs):
        super(HDSymoblicAttention, self).__init__(**kwargs)
        self.d_model = d_model  # Dimensionality of the model
        self.dim = dim
        self.n_seq = n_seq

    def cosine_similarity(self, a, b):
        # Compute the cosine similarity as dot product divided by magnitudes
        dot_product = tf.reduce_sum(tf.math.sign(a) * tf.math.sign(b), axis=-1)/self.d_model
        return dot_product   
    
    
    def create_cosine_similarity_matrix(self,X):
        X_expanded = tf.expand_dims(X, 2)  # Shape: (batch_size, N, 1, D)
        X_repeated = tf.repeat(X_expanded, repeats=tf.shape(X)[1], axis=2)  # Shape: (batch_size, N, N, D)
    
        X_i_expanded = tf.expand_dims(X, 1)  # Shape: (batch_size, 1, N, D)
        X_i_repeated = tf.repeat(X_i_expanded, repeats=tf.shape(X)[1], axis=1)  # Shape: (batch_size, N, N, D)
    
        #X_i_plus_X_j = X_i_repeated + X_repeated  # Broadcasting adds the matrices element-wise
        X_i_plus_X_j = X_repeated
        
        S = self.cosine_similarity(X_i_repeated, X_i_plus_X_j)  # Shape: (batch_size, N, N)
    
        return tf.nn.softmax(S)
            
    def build(self, input_shape):
                                                  
        normal_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.Symbols = tf.Variable(
                normal_initializer(shape=(self.n_seq, input_shape[-1])),
                name='symbols', trainable=True)          
        self.dense_layers = [BipolarDenseLayer(self.d_model, activation='tanh') for _ in range(self.dim)]
        self.bn1 = layers.BatchNormalization(synchronized=True)
        super(HDSymoblicAttention, self).build(input_shape)
    def call(self, values):
    
        # Unpack the inputs (queries, keys, values)
        self.S3 = tf.zeros_like(values)
        self.S3 = self.S3 + self.Symbols 
        h_vectors = [self.dense_layers[i](values[:, i, :]) for i in range(len(self.dense_layers))]
        h_vectors2 = [self.dense_layers[i](self.S3[:, i, :]) for i in range(len(self.dense_layers))]
        values_projected = tf.stack(h_vectors, axis=1)
        symbol_projected = tf.stack(h_vectors2, axis=1)
        scores  = self.create_cosine_similarity_matrix(values_projected)
        attention_output = tf.einsum('bij,bkk->bkj', values_projected, scores) 
        return  tf.nn.swish(attention_output * symbol_projected)    
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.d_model)
        
        
                

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, num_heads, dff,
            input_vocab, target_vocab, embedding_dim, output_dim,
            dropout_rate=0.1, name='transformer'):
        """A transformer model.

        Args:
            num_layers (int): # of layers in encoder and decoder
            num_heads (int): # of attention heads in attention operations
            dff (int): dimension of feedforward laeyrs
            input_vocab (int or str): if input is tokens, the size of vocabulary as an int. 
                if input is vectors, the string 'vector'. used to create embedder.
            target_vocab (int): if target is tokens, the size of the vocabulary as an int. 
                if input is vectors, the string 'vector'. used to create embedder.
            embedding_dim (int): embedding dimension to use. this is the model dimension.
            output_dim (int): dimension of final output. e.g.: # of classes.
            dropout_rate (float, optional): dropout rate. Defaults to 0.1.
            name (str, optional): name of model. Defaults to 'transformer'.
        """

        super().__init__(name=name)

        if isinstance(input_vocab, int):
            self.source_embedder = layers.Embedding(input_vocab, embedding_dim, name='source_embedder')
        elif input_vocab == 'vector':
            self.source_embedder = layers.TimeDistributed(layers.Dense(embedding_dim), name='source_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        if isinstance(target_vocab, int):
            self.target_embedder = layers.Embedding(target_vocab, embedding_dim, name='target_embedder')
        elif target_vocab == 'vector':
            self.target_embedder = layers.TimeDistributed(layers.Dense(embedding_dim), name='target_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        self.encoder = Encoder(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate, name='encoder')

        self.pos_embedding_adder_input = AddPositionalEmbedding(name='add_pos_embedding_input')
        self.pos_embedding_adder_target = AddPositionalEmbedding(name='add_pos_embedding_target')

        #self.decoder = Decoder()
        self.final_layer = layers.Dense(output_dim, name='final_layer')
        self.decoder = Decoder(num_layers=num_layers, num_heads=num_heads, dff=dff,
          dropout_rate=dropout_rate, name='decoder')
   


    def call(self, inputs):
        source, target  = inputs
        x = self.source_embedder(source)
        x = self.pos_embedding_adder_input(x)
        encoder_context = self.encoder(x)
        target_embedding = self.target_embedder(target)
        target_embedding = self.pos_embedding_adder_target(target_embedding)
        x = self.decoder(x=target_embedding, context=encoder_context)
        logits = self.final_layer(x)
        try:
          # Drop the keras mask, so it doesn't scale the losses/metrics.
          # b/250038731
          del logits._keras_mask
        except AttributeError:
          pass
        return logits


class Seq2SeqRelationalAbstracter(tf.keras.Model):
    """
    Sequence-to-Sequence Relational Abstracter.
    Uses the architecture X -> Encoder -> RelationalAbstracter -> Decoder -> y.

    Note: 'autoregressive_abstractor.py' implements a more general seq2seq
    abstractor architecture.
    """
    def __init__(self, num_layers, num_heads, dff, rel_attention_activation,
            input_vocab, target_vocab, embedding_dim, output_dim,
            dropout_rate=0.1, name='seq2seq_relational_abstracter'):
        """
        Args:
            num_layers (int): # of layers in encoder and decoder
            num_heads (int): # of attention heads in attention operations
            dff (int): dimension of feedforward layers
            rel_attention_activation (str): the activation function to use in relational attention.
            input_vocab (int or str): if input is tokens, the size of vocabulary as an int. 
                if input is vectors, the string 'vector'. used to create embedder.
            target_vocab (int): if target is tokens, the size of the vocabulary as an int. 
                if input is vectors, the string 'vector'. used to create embedder.
            embedding_dim (int): embedding dimension to use. this is the model dimension.
            output_dim (int): dimension of final output. e.g.: # of classes.
            dropout_rate (float, optional): dropout rate. Defaults to 0.1.
            name (str, optional): name of model. Defaults to 'seq2seq_relational_abstracter'.
        """

        super().__init__(name=name)

        if isinstance(input_vocab, int):
            self.source_embedder = layers.Embedding(input_vocab, embedding_dim, name='source_embedder')
        elif input_vocab == 'vector':
            self.source_embedder = layers.TimeDistributed(layers.Dense(embedding_dim), name='source_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        if isinstance(target_vocab, int):
            self.target_embedder = layers.Embedding(target_vocab, embedding_dim, name='target_embedder')
        elif target_vocab == 'vector':
            self.target_embedder = layers.TimeDistributed(layers.Dense(embedding_dim), name='target_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        self.pos_embedding_adder_input = AddPositionalEmbedding(name='add_pos_embedding_input')
        self.pos_embedding_adder_target = AddPositionalEmbedding(name='add_pos_embedding_target')

        self.encoder = Encoder(num_layers=2, num_heads=2, dff=dff, dropout_rate=dropout_rate, name='encoder')
        self.abstracter = RelationalAbstracter(num_layers=2, num_heads=2, dff=dff,
            mha_activation_type=rel_attention_activation, dropout_rate=dropout_rate, name='abstracter')
        self.decoder = Decoder(num_layers=num_layers, num_heads=num_heads, dff=dff,
          dropout_rate=dropout_rate, name='decoder')
        #self.decoder = Decoder()
        self.final_layer = layers.Dense(output_dim, name='final_layer')
   
    def call(self, inputs):
        source, target  = inputs
        x = self.source_embedder(source)
        x = self.pos_embedding_adder_input(x)
        x = self.encoder(x)
        abstracted_context = self.abstracter(x)
        target_embedding = self.target_embedder(target)
        target_embedding = self.pos_embedding_adder_target(target_embedding)
        x = self.decoder(x=target_embedding, context=abstracted_context)
        logits = self.final_layer(x)
        try:
          # Drop the keras mask, so it doesn't scale the losses/metrics.
          # b/250038731
          del logits._keras_mask
        except AttributeError:
          pass
        return logits
              


class Seq2SeqSymbolicAbstracter(tf.keras.Model):
    def __init__(self, num_layers, num_heads, dff, rel_attention_activation,
            input_vocab, target_vocab, embedding_dim, output_dim,
            dropout_rate=0.1, name='seq2seq_symbolic_abstracter'):
        """
        Sequence-to-Sequence Symbolic Abstracter.

        Args:
            num_layers (int): # of layers in encoder and decoder
            num_heads (int): # of attention heads in attention operations
            dff (int): dimension of feedforward layers
            rel_attention_activation (str): the activation function to use in relational attention.
            input_vocab (int or str): if input is tokens, the size of vocabulary as an int. 
                if input is vectors, the string 'vector'. used to create embedder.
            target_vocab (int): if target is tokens, the size of the vocabulary as an int. 
                if input is vectors, the string 'vector'. used to create embedder.
            embedding_dim (int): embedding dimension to use. this is the model dimension.
            output_dim (int): dimension of final output. e.g.: # of classes.
            dropout_rate (float, optional): dropout rate. Defaults to 0.1.
            name (str, optional): name of model. Defaults to 'seq2seq_symbolic_abstracter'.
        """

        super().__init__(name=name)

        if isinstance(input_vocab, int):
            self.source_embedder = layers.Embedding(input_vocab, embedding_dim, name='source_embedder')
        elif input_vocab == 'vector':
            self.source_embedder = layers.TimeDistributed(layers.Dense(embedding_dim), name='source_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        if isinstance(target_vocab, int):
            self.target_embedder = layers.Embedding(target_vocab, embedding_dim, name='target_embedder')
        elif target_vocab == 'vector':
            self.target_embedder = layers.TimeDistributed(layers.Dense(embedding_dim), name='target_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        self.pos_embedding_adder_input = AddPositionalEmbedding(name='add_pos_embedding_input')
        self.pos_embedding_adder_target = AddPositionalEmbedding(name='add_pos_embedding_target')

        self.encoder = Encoder(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate, name='encoder')
        self.abstracter = SymbolicAbstracter(num_layers=num_layers, num_heads=num_heads, dff=dff,
            mha_activation_type=rel_attention_activation, dropout_rate=dropout_rate, name='abstracter')
        self.decoder = Decoder(num_layers=num_layers, num_heads=num_heads, dff=dff,
          dropout_rate=dropout_rate, name='decoder')
        self.final_layer = layers.Dense(output_dim, name='final_layer')


    def call(self, inputs):
        source, target  = inputs

        x = self.source_embedder(source)
        x = self.pos_embedding_adder_input(x)

        encoder_context = self.encoder(x)

        abstracted_context = self.abstracter(encoder_context)

        target_embedding = self.target_embedder(target)
        target_embedding = self.pos_embedding_adder_target(target_embedding)

        x = self.decoder(x=target_embedding, context=abstracted_context)

        logits = self.final_layer(x)

        try:
          # Drop the keras mask, so it doesn't scale the losses/metrics.
          # b/250038731
          del logits._keras_mask
        except AttributeError:
          pass

        return logits


class AutoregressiveAblationAbstractor(tf.keras.Model):
    def __init__(self, num_layers, num_heads, dff, mha_activation_type,
            input_vocab, target_vocab, embedding_dim, output_dim,
            use_encoder, use_self_attn,
            dropout_rate=0.1, name='seq2seq_ablation_abstractor'):
        """
        Sequence-to-Sequence Ablation Abstracter.

        A Seq2Seq Abstractor model where the abstractor's cross-attention
        scheme is standard cross-attention rather than relation cross-attention.
        Used to isolate for the effect of relational cross-attention in abstractor models.

        Args:
            num_layers (int): # of layers in encoder and decoder
            num_heads (int): # of attention heads in attention operations
            dff (int): dimension of feedforward layers
            mha_activation_type (str): the activation function to use in AblationAbstractor's cross-attention.
            input_vocab (int or str): if input is tokens, the size of vocabulary as an int. 
                if input is vectors, the string 'vector'. used to create embedder.
            target_vocab (int): if target is tokens, the size of the vocabulary as an int. 
                if input is vectors, the string 'vector'. used to create embedder.
            embedding_dim (int): embedding dimension to use. this is the model dimension.
            output_dim (int): dimension of final output. e.g.: # of classes.
            use_encoder (bool): whether to use a (Transformer) Encoder as first step of processing.
            use_self_attn (bool): whether to use self-attention in AblationAbstractor.
            dropout_rate (float, optional): dropout rate. Defaults to 0.1.
            name (str, optional): name of model. Defaults to 'seq2seq_relational_abstracter'.
        """

        super().__init__(name=name)

        if isinstance(input_vocab, int):
            self.source_embedder = layers.Embedding(input_vocab, embedding_dim, name='source_embedder')
        elif input_vocab == 'vector':
            self.source_embedder = layers.TimeDistributed(layers.Dense(embedding_dim), name='source_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        if isinstance(target_vocab, int):
            self.target_embedder = layers.Embedding(target_vocab, embedding_dim, name='target_embedder')
        elif target_vocab == 'vector':
            self.target_embedder = layers.TimeDistributed(layers.Dense(embedding_dim), name='target_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        self.pos_embedding_adder_input = AddPositionalEmbedding(name='add_pos_embedding_input')
        self.pos_embedding_adder_target = AddPositionalEmbedding(name='add_pos_embedding_target')

        self.use_encoder = use_encoder
        self.use_self_attn = use_self_attn
        if self.use_encoder:
            self.encoder = Encoder(num_layers=num_layers, num_heads=num_heads, dff=dff,
            dropout_rate=dropout_rate, name='encoder')
        self.abstractor = AblationAbstractor(num_layers=num_layers, num_heads=num_heads, dff=dff,
            mha_activation_type=mha_activation_type, use_self_attn=use_self_attn, dropout_rate=dropout_rate,
            name='ablation_abstractor')
        self.decoder = Decoder(num_layers=num_layers, num_heads=num_heads, dff=dff,
          dropout_rate=dropout_rate, name='decoder')
        self.final_layer = layers.Dense(output_dim, name='final_layer')


    def call(self, inputs):
        source, target  = inputs

        x = self.source_embedder(source)
        x = self.pos_embedding_adder_input(x)

        if self.use_encoder:
            encoder_context = self.encoder(x)
        else:
            encoder_context = x

        abstracted_context = self.abstractor(encoder_context)
        target_embedding = self.target_embedder(target)
        target_embedding = self.pos_embedding_adder_target(target_embedding)

        x = self.decoder(x=target_embedding, context=abstracted_context)

        logits = self.final_layer(x)

        try:
          # Drop the keras mask, so it doesn't scale the losses/metrics.
          # b/250038731
          del logits._keras_mask
        except AttributeError:
          pass

        return logits
        





class Seq2SeqCorelNet(tf.keras.Model):
    def __init__(self, num_layers, num_heads, dff, rel_attention_activation,
            input_vocab, target_vocab, embedding_dim, output_dim,
            dropout_rate=0.1, name='seq2seq_relational_abstracter'):
        """
        Sequence-to-Sequence Ablation Abstracter.

        A Seq2Seq Abstractor model where the abstractor's cross-attention
        scheme is standard cross-attention rather than relation cross-attention.
        Used to isolate for the effect of relational cross-attention in abstractor models.

        Args:
            num_layers (int): # of layers in encoder and decoder
            num_heads (int): # of attention heads in attention operations
            dff (int): dimension of feedforward layers
            mha_activation_type (str): the activation function to use in AblationAbstractor's cross-attention.
            input_vocab (int or str): if input is tokens, the size of vocabulary as an int. 
                if input is vectors, the string 'vector'. used to create embedder.
            target_vocab (int): if target is tokens, the size of the vocabulary as an int. 
                if input is vectors, the string 'vector'. used to create embedder.
            embedding_dim (int): embedding dimension to use. this is the model dimension.
            output_dim (int): dimension of final output. e.g.: # of classes.
            use_encoder (bool): whether to use a (Transformer) Encoder as first step of processing.
            use_self_attn (bool): whether to use self-attention in AblationAbstractor.
            dropout_rate (float, optional): dropout rate. Defaults to 0.1.
            name (str, optional): name of model. Defaults to 'seq2seq_relational_abstracter'.
        """

        super().__init__(name=name)

        if isinstance(input_vocab, int):
            self.source_embedder = layers.Embedding(input_vocab, embedding_dim, name='source_embedder')
        elif input_vocab == 'vector':
            self.source_embedder = layers.TimeDistributed(layers.Dense(embedding_dim), name='source_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        if isinstance(target_vocab, int):
            self.target_embedder = layers.Embedding(target_vocab, embedding_dim, name='target_embedder')
        elif target_vocab == 'vector':
            self.target_embedder = layers.TimeDistributed(layers.Dense(embedding_dim), name='target_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        self.pos_embedding_adder_input = AddPositionalEmbedding(name='add_pos_embedding_input')
        self.pos_embedding_adder_target = AddPositionalEmbedding(name='add_pos_embedding_target')


        self.decoder = Decoder(num_layers=num_layers, num_heads=num_heads, dff=dff,
          dropout_rate=dropout_rate, name='decoder')
        self.final_layer = layers.Dense(output_dim, name='final_layer')
        self.mhr = MultiHeadRelation(rel_dim=64, proj_dim=None, symmetric=True, dense_kwargs=dict(use_bias=False))


    def call(self, inputs):
        source, target  = inputs

        x = self.source_embedder(source)
        x = self.pos_embedding_adder_input(x)
        abstracted_context = self.mhr(x)
        target_embedding = self.target_embedder(target)
        target_embedding = self.pos_embedding_adder_target(target_embedding)
        x = self.decoder(x=target_embedding, context=abstracted_context)

        logits = self.final_layer(x)

        try:
          # Drop the keras mask, so it doesn't scale the losses/metrics.
          # b/250038731
          del logits._keras_mask
        except AttributeError:
          pass

        return logits
                


class Seq2SeqLARS_VSA(tf.keras.Model):
    def __init__(self, num_layers, num_heads, num_heads_H, dff, rel_attention_activation,
            input_vocab, target_vocab, embedding_dim, output_dim, VSA_dim, seq_N,
            dropout_rate=0.1, name='seq2seq_relational_abstracter'):
        """
        Args:
            num_layers (int): # of layers in encoder and decoder
            num_heads (int): # of attention heads in attention operations
            dff (int): dimension of feedforward layers
            rel_attention_activation (str): the activation function to use in relational attention.
            input_vocab (int or str): if input is tokens, the size of vocabulary as an int. 
                if input is vectors, the string 'vector'. used to create embedder.
            target_vocab (int): if target is tokens, the size of the vocabulary as an int. 
                if input is vectors, the string 'vector'. used to create embedder.
            embedding_dim (int): embedding dimension to use. this is the model dimension.
            output_dim (int): dimension of final output. e.g.: # of classes.
            dropout_rate (float, optional): dropout rate. Defaults to 0.1.
            name (str, optional): name of model. Defaults to 'seq2seq_relational_abstracter'.
        """

        super().__init__(name=name)

        if isinstance(input_vocab, int):
            self.source_embedder = layers.Embedding(input_vocab, embedding_dim, name='source_embedder')
        elif input_vocab == 'vector':
            self.source_embedder = layers.TimeDistributed(layers.Dense(embedding_dim), name='source_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        if isinstance(target_vocab, int):
            self.target_embedder = layers.Embedding(target_vocab, embedding_dim, name='target_embedder')
        elif target_vocab == 'vector':
            self.target_embedder = layers.TimeDistributed(layers.Dense(embedding_dim), name='target_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        self.pos_embedding_adder_input = AddPositionalEmbedding(name='add_pos_embedding_input')
        self.pos_embedding_adder_target = AddPositionalEmbedding(name='add_pos_embedding_target')
        self.VSA_dim = VSA_dim
        self.H = num_heads_H
        self.mha = [HDSymoblicAttention(self.VSA_dim,output_dim,output_dim) for _ in range(num_heads_H)]
        self.dropout = layers.Dropout(dropout_rate)
        self.decoder = Decoder(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate, name='decoder')
        self.final_layer = layers.Dense(output_dim, name='final_layer')
        self.bn_e = layers.BatchNormalization(synchronized=True)
        self.bns = [layers.BatchNormalization(synchronized=True) for _ in range(self.H)]
        self.bn_f = layers.BatchNormalization(synchronized=True)
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
    def call(self, inputs):
        source, target  = inputs
        x = self.source_embedder(source)
        x = self.pos_embedding_adder_input(x)
        target_embedding = self.target_embedder(target)
        target_embedding = self.pos_embedding_adder_target(target_embedding)
        x = self.bn_e(x)
        x = self.dropout(x)
        h_glob = tf.reshape(self.mha[0](x),(-1,self.output_dim,self.VSA_dim//self.embedding_dim,self.embedding_dim))
        for i in range(1,self.H,1):
            h_glob += self.bns[i](tf.reshape(self.mha[i](x),(-1,self.output_dim,self.VSA_dim//self.embedding_dim,self.embedding_dim)))
        abstracted_context = tf.reduce_mean(h_glob,axis=2)
        abstracted_context = self.dropout(abstracted_context)  
        abstracted_context = self.bn_f(abstracted_context) 
        x = self.decoder(x=target_embedding, context=abstracted_context)
        logits = self.final_layer(x)
        try:
          # Drop the keras mask, so it doesn't scale the losses/metrics.
          # b/250038731
          del logits._keras_mask
        except AttributeError:
          pass
        return logits
        
        
        
        
        
                
        


