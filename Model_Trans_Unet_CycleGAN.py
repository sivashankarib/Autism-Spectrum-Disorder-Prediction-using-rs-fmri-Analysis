import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np


def conv_block(x, filters, kernel_size=(3, 3), padding='same', activation='relu'):
    x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x


def encoder_block(x, filters, kernel_size=(3, 3), padding='same', activation='relu'):
    x = conv_block(x, filters, kernel_size, padding, activation)
    p = layers.MaxPooling2D(pool_size=(2, 2))(x)
    return x, p


def decoder_block(x, skip_features, filters, kernel_size=(3, 3), padding='same', activation='relu'):
    x = layers.Conv2DTranspose(filters, kernel_size, strides=(2, 2), padding=padding)(x)
    x = layers.Concatenate()([x, skip_features])
    x = conv_block(x, filters, kernel_size, padding, activation)
    return x


def transformer_block(x, num_heads, ff_dim, dropout=0.1):
    x1 = layers.LayerNormalization(epsilon=1e-6)(x)
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(x1, x1)
    attn_output = layers.Dropout(dropout)(attn_output)
    x2 = layers.Add()([x, attn_output])
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    ff_output = layers.Dense(ff_dim, activation='relu')(x3)
    ff_output = layers.Dropout(dropout)(ff_output)
    ff_output = layers.Dense(x.shape[-1])(ff_output)
    return layers.Add()([x2, ff_output])


def TransUNet(input_shape, num_classes, num_transformer_blocks=4, num_heads=4, ff_dim=256):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # Bottleneck with Transformer blocks
    b = conv_block(p4, 1024)
    b = layers.Reshape((b.shape[1] * b.shape[2], b.shape[3]))(b)
    for _ in range(num_transformer_blocks):
        b = transformer_block(b, num_heads, ff_dim)
    b = layers.Reshape((p4.shape[1], p4.shape[2], 1024))(b)

    # Decoder
    d1 = decoder_block(b, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = layers.Conv2D(num_classes, (1, 1), padding='same', activation='sigmoid')(d4)

    model = models.Model(inputs, outputs)
    return model


def CycleGAN(generator_G, generator_F, discriminator_X, discriminator_Y, lambda_cycle=10.0, lambda_identity=0.5):
    class CycleGAN(tf.keras.Model):
        def __init__(self):
            super(CycleGAN, self).__init__()
            self.gen_G = generator_G
            self.gen_F = generator_F
            self.disc_X = discriminator_X
            self.disc_Y = discriminator_Y
            self.lambda_cycle = lambda_cycle
            self.lambda_identity = lambda_identity

        def compile(self, gen_G_optimizer, gen_F_optimizer, disc_X_optimizer, disc_Y_optimizer, gen_loss_fn,
                    disc_loss_fn, cycle_loss_fn, identity_loss_fn):
            super(CycleGAN, self).compile()
            self.gen_G_optimizer = gen_G_optimizer
            self.gen_F_optimizer = gen_F_optimizer
            self.disc_X_optimizer = disc_X_optimizer
            self.disc_Y_optimizer = disc_Y_optimizer
            self.gen_loss_fn = gen_loss_fn
            self.disc_loss_fn = disc_loss_fn
            self.cycle_loss_fn = cycle_loss_fn
            self.identity_loss_fn = identity_loss_fn

        def train_step(self, batch_data):
            real_x, real_y = batch_data

            with tf.GradientTape(persistent=True) as tape:
                fake_y = self.gen_G(real_x, training=True)
                cycled_x = self.gen_F(fake_y, training=True)

                fake_x = self.gen_F(real_y, training=True)
                cycled_y = self.gen_G(fake_x, training=True)

                same_y = self.gen_G(real_y, training=True)
                same_x = self.gen_F(real_x, training=True)

                disc_real_x = self.disc_X(real_x, training=True)
                disc_real_y = self.disc_Y(real_y, training=True)
                disc_fake_x = self.disc_X(fake_x, training=True)
                disc_fake_y = self.disc_Y(fake_y, training=True)

                gen_G_loss = self.gen_loss_fn(disc_fake_y)
                gen_F_loss = self.gen_loss_fn(disc_fake_x)

                cycle_loss_G = self.cycle_loss_fn(real_x, cycled_x, self.lambda_cycle)
                cycle_loss_F = self.cycle_loss_fn(real_y, cycled_y, self.lambda_cycle)

                id_loss_G = self.identity_loss_fn(real_y, same_y, self.lambda_cycle, self.lambda_identity)
                id_loss_F = self.identity_loss_fn(real_x, same_x, self.lambda_cycle, self.lambda_identity)

                total_loss_G = gen_G_loss + cycle_loss_G + id_loss_G
                total_loss_F = gen_F_loss + cycle_loss_F + id_loss_F

                disc_X_loss = self.disc_loss_fn(disc_real_x, disc_fake_x)
                disc_Y_loss = self.disc_loss_fn(disc_real_y, disc_fake_y)

            grads_G = tape.gradient(total_loss_G, self.gen_G.trainable_variables)
            grads_F = tape.gradient(total_loss_F, self.gen_F.trainable_variables)

            disc_X_grads = tape.gradient(disc_X_loss, self.disc_X.trainable_variables)
            disc_Y_grads = tape.gradient(disc_Y_loss, self.disc_Y.trainable_variables)

            self.gen_G_optimizer.apply_gradients(zip(grads_G, self.gen_G.trainable_variables))
            self.gen_F_optimizer.apply_gradients(zip(grads_F, self.gen_F.trainable_variables))

            self.disc_X_optimizer.apply_gradients(zip(disc_X_grads, self.disc_X.trainable_variables))
            self.disc_Y_optimizer.apply_gradients(zip(disc_Y_grads, self.disc_Y.trainable_variables))

            return {
                "G_loss": total_loss_G,
                "F_loss": total_loss_F,
                "D_X_loss": disc_X_loss,
                "D_Y_loss": disc_Y_loss
            }

    return CycleGAN()


def unet_generator(input_shape=(256, 256, 1), output_channels=1):
    inputs = tf.keras.layers.Input(shape=input_shape)

    down_stack = [
        layers.Conv2D(64, (4, 4), strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(128, (4, 4), strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(256, (4, 4), strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(512, (4, 4), strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU()
    ]

    up_stack = [
        layers.Conv2DTranspose(256, (4, 4), strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2DTranspose(128, (4, 4), strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2DTranspose(64, (4, 4), strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU()
    ]

    x = inputs

    for layer in down_stack:
        x = layer(x)

    for layer in up_stack:
        x = layer(x)

    last = layers.Conv2DTranspose(output_channels, (4, 4), strides=2, padding='same')
    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def discriminator(input_shape=(256, 256, 1)):
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = layers.Input(shape=input_shape, name='input_image')
    tar = layers.Input(shape=input_shape, name='target_image')

    x = layers.concatenate([inp, tar])

    down1 = layers.Conv2D(64, 4, strides=2, padding='same', kernel_initializer=initializer)(x)
    down1 = layers.LeakyReLU()(down1)

    down2 = layers.Conv2D(128, 4, strides=2, padding='same', kernel_initializer=initializer)(down1)
    down2 = layers.BatchNormalization()(down2)
    down2 = layers.LeakyReLU()(down2)

    down3 = layers.Conv2D(256, 4, strides=2, padding='same', kernel_initializer=initializer)(down2)
    down3 = layers.BatchNormalization()(down3)
    down3 = layers.LeakyReLU()(down3)

    zero_pad1 = layers.ZeroPadding2D()(down3)

    conv = layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer)(zero_pad1)
    conv = layers.BatchNormalization()(conv)
    conv = layers.LeakyReLU()(conv)

    zero_pad2 = layers.ZeroPadding2D()(conv)

    last = layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


def Model_Trans_Unet_CycleGAN(Data, Target):

    input_shape = (256, 256, 1)
    num_classes = 1
    epoch = 100

    transunet = TransUNet(input_shape, num_classes)
    transunet.summary()

    generator_G = unet_generator(input_shape, num_classes)
    generator_F = unet_generator(input_shape, num_classes)
    discriminator_X = discriminator(input_shape)
    discriminator_Y = discriminator(input_shape)

    cycle_gan = CycleGAN(generator_G, generator_F, discriminator_X, discriminator_Y)

    # Compile the CycleGAN model
    cycle_gan.compile(
        gen_G_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
        gen_F_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
        disc_X_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
        disc_Y_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
        gen_loss_fn=tf.keras.losses.MeanSquaredError(),
        disc_loss_fn=tf.keras.losses.MeanSquaredError(),
        cycle_loss_fn=lambda x, y, lambda_cycle: lambda_cycle * tf.reduce_mean(tf.abs(x - y)),
        identity_loss_fn=lambda x, y, lambda_cycle, lambda_identity: lambda_identity * lambda_cycle * tf.reduce_mean(
            tf.abs(x - y))
    )

    Train_Temp = np.zeros((Data.shape[0], input_shape[0], input_shape[1], input_shape[2]))
    for i in range(Data.shape[0]):
        Train_Temp[i, :] = np.resize(Data[i], (input_shape[0], input_shape[1], input_shape[2]))
    X = Train_Temp.reshape(Train_Temp.shape[0], input_shape[0], input_shape[1], input_shape[2])

    Test_Temp = np.zeros((Target.shape[0], input_shape[0], input_shape[1], input_shape[2]))
    for i in range(Target.shape[0]):
        Test_Temp[i, :] = np.resize(Target[i], (input_shape[0], input_shape[1], input_shape[2]))
    Y = Test_Temp.reshape(Test_Temp.shape[0], input_shape[0], input_shape[1], input_shape[2])

    Pred = []
    for i in range(epoch):
        result = cycle_gan.train_step((X, Y))
        print(f"Training step {i + 1}: {result}")
        Pred.append(result)
    Pred = Pred[-1]
    return Pred
