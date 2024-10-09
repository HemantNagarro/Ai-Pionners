import tensorflow as tf
import os
import json
import pandas as pd
import re
import numpy as np
import collections
import random
import requests
import pickle
import json
from math import sqrt
from PIL import Image
from tqdm.auto import tqdm
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.utils import register_keras_serializable
import os.path
import keras
from huggingface_hub import HfApi
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download


model_path = hf_hub_download(repo_id="aks07hat/apparel-keywords-generator", filename="apparelKeywordGeneratorModel.keras")

weights_path = hf_hub_download(repo_id="aks07hat/apparel-keywords-generator", filename="trainedModel.weights.h5")

tokenizer_path = hf_hub_download(repo_id="aks07hat/apparel-keywords-generator", filename="tokenizer.pkl")

with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

@register_keras_serializable(package="Custom", name="CNN_Encoder")
class CNN_Encoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inception_v3 = tf.keras.applications.InceptionV3(
            include_top=False,  
            weights='imagenet'  
        )  
        output = self.inception_v3.output
        output = tf.keras.layers.Reshape(
            (-1, output.shape[-1]))(output)
        self.cnn_model = tf.keras.models.Model(
            inputs=self.inception_v3.input,
            outputs=output
        )
    def call(self, inputs):
        return self.cnn_model(inputs)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        instance = cls(**config)
        return instance
    

@register_keras_serializable(package="Custom", name="TransformerEncoderLayer")
class TransformerEncoderLayer(tf.keras.layers.Layer):
    
    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.layer_norm_1 = tf.keras.layers.LayerNormalization()
        self.layer_norm_2 = tf.keras.layers.LayerNormalization()
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense = tf.keras.layers.Dense(embed_dim, activation="relu")
    
    def call(self, x, training):
        x = self.layer_norm_1(x)
        x = self.dense(x)

        attn_output = self.attention(
            query=x,
            value=x,
            key=x,
            attention_mask=None,
            training=training
        )

        x = self.layer_norm_2(x + attn_output)
        return x

    def get_config(self):
        # Return a dictionary of the configuration to serialize
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "layer_norm_1": tf.keras.layers.serialize(self.layer_norm_1),
            "layer_norm_2": tf.keras.layers.serialize(self.layer_norm_2),
            "attention": tf.keras.layers.serialize(self.attention),
            "dense": tf.keras.layers.serialize(self.dense),
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Deserialize the configuration to recreate the object
        layer_norm_1 = tf.keras.layers.deserialize(config.pop('layer_norm_1'))
        layer_norm_2 = tf.keras.layers.deserialize(config.pop('layer_norm_2'))
        attention = tf.keras.layers.deserialize(config.pop('attention'))
        dense = tf.keras.layers.deserialize(config.pop('dense'))

        # Create an instance with deserialized layers
        instance = cls(**config)
        instance.layer_norm_1 = layer_norm_1
        instance.layer_norm_2 = layer_norm_2
        instance.attention = attention
        instance.dense = dense
        return instance
@register_keras_serializable(package="Custom", name="Embeddings")
class Embeddings(tf.keras.layers.Layer):

    def __init__(self, vocab_size, embed_dim, max_len, **kwargs):
        super(Embeddings, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_len = max_len
        
        # Initialize the embedding layers
        self.token_embeddings = tf.keras.layers.Embedding(
            vocab_size, embed_dim)
        self.position_embeddings = tf.keras.layers.Embedding(
            max_len, embed_dim, input_shape=(None, max_len))
    
    def call(self, input_ids):
        # Determine the length of the input
        length = tf.shape(input_ids)[-1]
        # Create position ids based on the length of the input sequence
        position_ids = tf.range(start=0, limit=length, delta=1)
        position_ids = tf.expand_dims(position_ids, axis=0)

        # Generate token and position embeddings
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # Combine the token and position embeddings
        return token_embeddings + position_embeddings

    def get_config(self):
        # Serializing the necessary parameters (vocab_size, embed_dim, max_len)
        config = super(Embeddings, self).get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'max_len': self.max_len
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Reconstructing the layer from its serialized config
        instance = cls(**config)
        vocab_size = (config.pop('vocab_size'))
        embed_dim = (config.pop('embed_dim'))
        max_len = (config.pop('max_len'))
        instance.vocab_size = vocab_size
        instance.embed_dim = embed_dim
        instance.max_len = max_len
        return instance
    
@register_keras_serializable(package="Custom", name="TransformerDecoderLayer")
class TransformerDecoderLayer(tf.keras.layers.Layer):
    
    def __init__(self, embed_dim, units, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.units = units
        self.num_heads = num_heads

        self.embedding = Embeddings(tokenizer.vocabulary_size(), embed_dim, 40)

        self.attention_1 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )
        self.attention_2 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )

        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()
        self.layernorm_3 = tf.keras.layers.LayerNormalization()

        self.ffn_layer_1 = tf.keras.layers.Dense(units, activation="relu")
        self.ffn_layer_2 = tf.keras.layers.Dense(embed_dim)

        self.out = tf.keras.layers.Dense(tokenizer.vocabulary_size(), activation="softmax")

        self.dropout_1 = tf.keras.layers.Dropout(0.3)
        self.dropout_2 = tf.keras.layers.Dropout(0.5)

    def call(self, input_ids, encoder_output, training, mask=None):
        embeddings = self.embedding(input_ids)

        combined_mask = None
        padding_mask = None
        
        if mask is not None:
            causal_mask = self.get_causal_attention_mask(embeddings)
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)

        attn_output_1 = self.attention_1(
            query=embeddings,
            value=embeddings,
            key=embeddings,
            attention_mask=combined_mask,
            training=training
        )

        out_1 = self.layernorm_1(embeddings + attn_output_1)

        attn_output_2 = self.attention_2(
            query=out_1,
            value=encoder_output,
            key=encoder_output,
            attention_mask=padding_mask,
            training=training
        )

        out_2 = self.layernorm_2(out_1 + attn_output_2)

        ffn_out = self.ffn_layer_1(out_2)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.ffn_layer_2(ffn_out)

        ffn_out = self.layernorm_3(ffn_out + out_2)
        ffn_out = self.dropout_2(ffn_out, training=training)
        preds = self.out(ffn_out)
        return preds

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0
        )
        return tf.tile(mask, mult)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "units": self.units,
            "num_heads": self.num_heads,
            "embedding": tf.keras.layers.serialize(self.embedding),
            "attention_1": tf.keras.layers.serialize(self.attention_1),
            "attention_2": tf.keras.layers.serialize(self.attention_2),
            "layernorm_1": tf.keras.layers.serialize(self.layernorm_1),
            "layernorm_2": tf.keras.layers.serialize(self.layernorm_2),
            "layernorm_3": tf.keras.layers.serialize(self.layernorm_3),
            "ffn_layer_1": tf.keras.layers.serialize(self.ffn_layer_1),
            "ffn_layer_2": tf.keras.layers.serialize(self.ffn_layer_2),
            "out": tf.keras.layers.serialize(self.out),
            "dropout_1": tf.keras.layers.serialize(self.dropout_1),
            "dropout_2": tf.keras.layers.serialize(self.dropout_2),
        })
        return config

    @classmethod
    def from_config(cls, config):
        embedding = tf.keras.layers.deserialize(config.pop("embedding"))
        attention_1 = tf.keras.layers.deserialize(config.pop("attention_1"))
        attention_2 = tf.keras.layers.deserialize(config.pop("attention_2"))
        layernorm_1 = tf.keras.layers.deserialize(config.pop("layernorm_1"))
        layernorm_2 = tf.keras.layers.deserialize(config.pop("layernorm_2"))
        layernorm_3 = tf.keras.layers.deserialize(config.pop("layernorm_3"))
        ffn_layer_1 = tf.keras.layers.deserialize(config.pop("ffn_layer_1"))
        ffn_layer_2 = tf.keras.layers.deserialize(config.pop("ffn_layer_2"))
        out = tf.keras.layers.deserialize(config.pop("out"))
        dropout_1 = tf.keras.layers.deserialize(config.pop("dropout_1"))
        dropout_2 = tf.keras.layers.deserialize(config.pop("dropout_2"))

        instance = cls(**config)
        instance.embedding = embedding
        instance.attention_1 = attention_1
        instance.attention_2 = attention_2
        instance.layernorm_1 = layernorm_1
        instance.layernorm_2 = layernorm_2
        instance.layernorm_3 = layernorm_3
        instance.ffn_layer_1 = ffn_layer_1
        instance.ffn_layer_2 = ffn_layer_2
        instance.out = out
        instance.dropout_1 = dropout_1
        instance.dropout_2 = dropout_2

        return instance

word2idx = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=tokenizer.get_vocabulary())

idx2word = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=tokenizer.get_vocabulary(),
    invert=True)

@register_keras_serializable(package="Custom", name="ImageCaptioningModel")
class ImageCaptioningModel(tf.keras.Model):

    def __init__(self, cnn_model, encoder, decoder, image_aug=None):
        super().__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.image_aug = image_aug
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.acc_tracker = tf.keras.metrics.Mean(name="accuracy")


    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)


    def calculate_accuracy(self, y_true, y_pred, mask):
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)
    

    def compute_loss_and_acc(self, img_embed, captions, training=True):
        encoder_output = self.encoder(img_embed, training=True)
        y_input = captions[:, :-1]
        y_true = captions[:, 1:]
        mask = (y_true != 0)
        y_pred = self.decoder(
            y_input, encoder_output, training=True, mask=mask
        )
        loss = self.calculate_loss(y_true, y_pred, mask)
        acc = self.calculate_accuracy(y_true, y_pred, mask)
        return loss, acc

    
    def train_step(self, batch):
        imgs, captions = batch

        if self.image_aug:
            imgs = self.image_aug(imgs)
        
        img_embed = self.cnn_model(imgs)

        with tf.GradientTape() as tape:
            loss, acc = self.compute_loss_and_acc(
                img_embed, captions
            )
    
        train_vars = (
            self.encoder.trainable_variables + self.decoder.trainable_variables
        )
        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)

        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}
    

    def test_step(self, batch):
        imgs, captions = batch

        img_embed = self.cnn_model(imgs)

        loss, acc = self.compute_loss_and_acc(
            img_embed, captions, training=False
        )

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)

        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]
    
    
    def get_config(self):
        config = super(ImageCaptioningModel, self).get_config()
        config.update({
            "cnn_model": self.cnn_model,
            "encoder": self.encoder,
            "decoder": self.decoder,
            "image_aug": self.image_aug,
        })
        print("While saving in config", config)
        return config
    

    @classmethod
    def from_config(cls, config):
        cnn_model = tf.keras.layers.deserialize(config["cnn_model"])
        encoder = tf.keras.layers.deserialize(config["encoder"])
        decoder = tf.keras.layers.deserialize(config["decoder"])
        image_aug = config.pop("image_aug", None)
        return cls(cnn_model, encoder, decoder, image_aug=image_aug)
    
    def call(self, img_path, training=False):
        if ('http' in img_path):
            im = Image.open(requests.get(img_path, stream=True).raw)
            im = im.convert('RGB')
            im.save('tmp.jpg')
            img_path = 'tmp.jpg'
            
        img = tf.io.read_file(img_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.cast(img, tf.float32) 
        img = tf.keras.layers.Resizing(299, 299)(img)
        img = tf.keras.applications.inception_v3.preprocess_input(img)

        img = tf.expand_dims(img, axis=0)
        img_embed = self.cnn_model(img)
        img_embed = tf.reshape(img_embed, (img_embed.shape[0], -1, img_embed.shape[-1]))
        img_encoded = self.encoder(img_embed, training=training)

        y_inp = ''
        for i in range(40-1):
            tokenized = tokenizer([y_inp])[:, :-1]
            mask = tf.cast(tokenized != 0, tf.int32)
            pred = self.decoder(
                tokenized, img_encoded, training=training, mask=mask)

            pred_idx = np.argmax(pred[0, i, :])
            pred_idx = tf.convert_to_tensor(pred_idx)
            pred_word = idx2word(pred_idx).numpy().decode('utf-8')
            if pred_word == '[end]':
                break

            y_inp += ' ' + pred_word

        y_inp = y_inp.replace('[start] ', '')
        return y_inp
#         caption_set = set(y_inp.split(','))
#         return list(caption_set)


def loadModel():
    caption_model = keras.saving.load_model(model_path)
    caption_model.load_weights(weights_path)
    return caption_model