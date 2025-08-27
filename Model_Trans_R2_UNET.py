import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Model, Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, BatchNormalization, Activation
from tensorflow.keras.layers import Add, Dropout
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy


# Define Transformer Block
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = models.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim)]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# Define Recurrent Residual Block
def recurrent_block(x, filters, recurrent_steps):
    for _ in range(recurrent_steps):
        x1 = Conv2D(filters, (3, 3), padding="same")(x)
        x1 = BatchNormalization()(x1)
        x1 = Activation("relu")(x1)
        x1 = Conv2D(filters, (3, 3), padding="same")(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation("relu")(x1)
        if x.shape[-1] != filters:
            x = Conv2D(filters, (1, 1), padding="same")(x)
        x = Add()([x, x1])
    return x


# Define Residual U-Net++ Block
def unet_plus_plus_block(x, filters, depth, recurrent_steps):
    skips = []
    for d in range(depth):
        x = recurrent_block(x, filters, recurrent_steps)
        skips.append(x)
        x = MaxPooling2D((2, 2))(x)
        filters *= 2

    x = recurrent_block(x, filters, recurrent_steps)

    for d in reversed(range(depth)):
        filters //= 2
        x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same")(x)
        x = concatenate([x, skips[d]])
        x = recurrent_block(x, filters, recurrent_steps)

    return x


# Define Model
def Trans_R2Unet_plusplus(input_shape, num_classes, depth=3, initial_filters=32, recurrent_steps=2, num_heads=4,
                                ff_dim=128):
    inputs = Input(input_shape)
    x = Conv2D(initial_filters, (3, 3), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = unet_plus_plus_block(x, initial_filters, depth, recurrent_steps)

    x_shape = x.shape
    transformer_input = tf.reshape(x, [-1, x_shape[1] * x_shape[2], x_shape[3]])  # Flatten spatial dimensions
    x = TransformerBlock(embed_dim=x.shape[-1], num_heads=num_heads, ff_dim=ff_dim)(transformer_input)
    x = tf.reshape(x, [-1, x_shape[1], x_shape[2], x.shape[-1]])  # Reshape back to spatial dimensions

    x = Conv2D(num_classes, (1, 1), activation="softmax")(x)

    return Model(inputs, x)


def Model_Trans_R2Unet_plusplus(Data, Target):
    input_shape = (32, 32, 3)
    num_classes = 3

    Original_Images = np.zeros((Data.shape[0], input_shape[0], input_shape[1], input_shape[2]))
    for i in range(Data.shape[0]):
        Original_Images[i, :] = np.resize(Data[i], (input_shape[0], input_shape[1], input_shape[2]))
    X = Original_Images.reshape(Original_Images.shape[0], input_shape[0], input_shape[1], input_shape[2])

    Ground_truth = np.zeros((Target.shape[0], input_shape[0], input_shape[1], num_classes))
    for i in range(Target.shape[0]):
        Ground_truth[i, :] = np.resize(Target[i], (input_shape[0], input_shape[1], num_classes))
    Y = Ground_truth.reshape(Ground_truth.shape[0], input_shape[0], input_shape[1], num_classes)

    model = Trans_R2Unet_plusplus(input_shape, num_classes)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(X, Y, batch_size=4, epochs=5, steps_per_epoch=5)
    pred = model.predict(X)
    return pred




