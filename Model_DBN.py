import numpy as np
from Evaluation import evaluation
from tensorflow.keras.layers import Input, Dense
from keras.src.models import Model
from tensorflow.keras.optimizers import SGD
from keras.src.utils import to_categorical


class RBM(object):
    def __init__(self, visible_units, hidden_units, learning_rate=0.01, batch_size=10, epochs=10):
        self.visible_units = visible_units
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        self.W = np.random.randn(visible_units, hidden_units) * 0.1
        self.hb = np.zeros(hidden_units)
        self.vb = np.zeros(visible_units)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def train(self, data):
        for epoch in range(self.epochs):
            for i in range(0, data.shape[0], self.batch_size):
                batch = data[i:i + self.batch_size]

                # Positive phase
                pos_hidden_activations = np.dot(batch, self.W) + self.hb
                pos_hidden_probs = self.sigmoid(pos_hidden_activations)
                pos_hidden_states = pos_hidden_probs > np.random.rand(self.batch_size, self.hidden_units)
                pos_associations = np.dot(batch.T, pos_hidden_probs)

                # Negative phase
                neg_visible_activations = np.dot(pos_hidden_states, self.W.T) + self.vb
                neg_visible_probs = self.sigmoid(neg_visible_activations)
                neg_hidden_activations = np.dot(neg_visible_probs, self.W) + self.hb
                neg_hidden_probs = self.sigmoid(neg_hidden_activations)

                neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

                # Update weights and biases
                self.W += self.learning_rate * (pos_associations - neg_associations) / self.batch_size
                self.vb += self.learning_rate * np.mean(batch - neg_visible_probs, axis=0)
                self.hb += self.learning_rate * np.mean(pos_hidden_probs - neg_hidden_probs, axis=0)

    def transform(self, data):
        hidden_activations = np.dot(data, self.W) + self.hb
        hidden_probs = self.sigmoid(hidden_activations)
        return hidden_probs


class DBN(object):
    def __init__(self, layer_sizes, learning_rate=0.01, batch_size=10, rbm_epochs=10, finetune_epochs=10):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.rbm_epochs = rbm_epochs
        self.finetune_epochs = finetune_epochs
        self.rbms = []

        for i in range(len(layer_sizes) - 1):
            rbm = RBM(layer_sizes[i], layer_sizes[i + 1], learning_rate, batch_size, rbm_epochs)
            self.rbms.append(rbm)

    def pretrain(self, data):
        input_data = data
        for rbm in self.rbms:
            rbm.train(input_data)
            input_data = rbm.transform(input_data)

    def finetune(self, train_data, train_labels, test_data, test_labels):
        input_data = Input(shape=(self.layer_sizes[0],))
        current_input = input_data

        for rbm in self.rbms:
            current_input = Dense(rbm.hidden_units, activation='sigmoid', trainable=False)(current_input)

        final_output = Dense(train_labels.shape[1], activation='softmax')(current_input)

        self.model = Model(inputs=input_data, outputs=final_output)
        self.model.compile(optimizer=SGD(learning_rate=self.learning_rate), loss='categorical_crossentropy',
                           metrics=['accuracy'])

        # Set RBM weights
        layer_num = 1
        for rbm in self.rbms:
            self.model.layers[layer_num].set_weights([rbm.W, rbm.hb])
            layer_num += 1

        self.model.fit(train_data, train_labels, epochs=self.finetune_epochs, batch_size=self.batch_size,
                       validation_data=(test_data, test_labels))

    def predict(self, data):
        return self.model.predict(data)


def Model_DBN(train_data, train_target, test_data, test_target, BS=None, sol=None):
    if sol is None:
        sol = [5, 10, 5]
    if BS is None:
        BS = 10

    IMG_SIZE = [28, 28]
    Train_Temp = np.zeros((train_data.shape[0], IMG_SIZE[0], IMG_SIZE[1]))
    for i in range(train_data.shape[0]):
        Train_Temp[i, :] = np.resize(train_data[i], (IMG_SIZE[0], IMG_SIZE[1]))
    train_X = Train_Temp.reshape(Train_Temp.shape[0], IMG_SIZE[0], IMG_SIZE[1])

    Test_Temp = np.zeros((test_data.shape[0], IMG_SIZE[0], IMG_SIZE[1]))
    for i in range(test_data.shape[0]):
        Test_Temp[i, :] = np.resize(test_data[i], (IMG_SIZE[0], IMG_SIZE[1]))
    test_X = Test_Temp.reshape(Test_Temp.shape[0], IMG_SIZE[0], IMG_SIZE[1])

    # Normalize and reshape the data
    Train_X = train_X.astype('float32') / 255.0
    Test_X = test_X.astype('float32') / 255.0

    Train_X = Train_X.reshape((Train_X.shape[0], -1)).astype('int')  # Flatten the images
    Test_X = Test_X.reshape((Test_X.shape[0], -1)).astype('int')  # Flatten the images

    # Convert labels to one-hot encoding
    # train_target = to_categorical(train_target, train_target.shape[-1])
    # test_target = to_categorical(test_target, train_target.shape[-1])

    # Define DBN
    dbn = DBN(layer_sizes=[784, 256, 128], learning_rate=0.01, batch_size=int(train_data.shape[0] / 5), rbm_epochs=int(sol[1]),
              finetune_epochs=10)

    # Pretrain DBN
    dbn.pretrain(Train_X)
    # Finetune DBN
    dbn.finetune(Train_X, train_target, Test_X, test_target)
    # Predict
    pred = dbn.predict(Test_X)

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    Eval = evaluation(test_target, pred)
    return Eval, pred


