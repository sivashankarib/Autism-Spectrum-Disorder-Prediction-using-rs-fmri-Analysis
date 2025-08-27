from stellargraph.layer import GraphConvolution
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
import keras
import numpy as np
from Evaluation import evaluation
from sklearn.model_selection import train_test_split
from keras import backend as K


def model_GCN(n_nodes, n_features, n_classes):
    kernel_initializer = "glorot_uniform"
    bias_initializer = "zeros"

    x_features = Input(shape=(n_nodes, n_features))
    x_adjacency = Input(shape=(n_nodes, n_nodes))

    x = Dropout(0.5)(x_features)
    x = GraphConvolution(32, activation='relu', use_bias=True, kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer)([x, x_adjacency])
    x = Dropout(0.5)(x)
    x = GraphConvolution(32, activation='relu', use_bias=True, kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer)([x, x_adjacency])

    # Apply Dense layer directly without GatherIndices
    x = Dense(n_classes, activation='sigmoid')(x)
    x = keras.layers.GlobalAveragePooling1D()(x)  # Ensure the output shape is (None, classes)

    model = Model(inputs=[x_features, x_adjacency], outputs=x)
    model.summary()

    return model


def create_adjacency_and_indices(data, n_nodes):
    n_samples = data.shape[0]
    adjacency_matrices = np.ones((n_samples, n_nodes, n_nodes)) - np.eye(n_nodes)
    indices = np.tile(np.arange(n_nodes), (n_samples, 1))
    return adjacency_matrices, indices


def Model_GCN(X, Y, test_X, test_Y, BS=None):
    if BS is None:
        BS = 32
    n_nodes = 10
    n_features = 100
    n_classes = Y.shape[-1]

    Train_X = np.zeros((X.shape[0], n_nodes, n_features))
    for i in range(X.shape[0]):
        temp = np.resize(X[i], (n_nodes, n_features))
        Train_X[i] = np.reshape(temp, (n_nodes, n_features))

    Test_X = np.zeros((test_X.shape[0], n_nodes, n_features))
    for i in range(test_X.shape[0]):
        temp = np.resize(test_X[i], (n_nodes, n_features))
        Test_X[i] = np.reshape(temp, (n_nodes, n_features))

    train_adjacency, _ = create_adjacency_and_indices(X, n_nodes)
    test_adjacency, _ = create_adjacency_and_indices(test_X, n_nodes)

    model = model_GCN(n_nodes, n_features, n_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit([Train_X, train_adjacency], Y, epochs=10, batch_size=BS, steps_per_epoch=1, validation_data=([Test_X, test_adjacency], test_Y))

    pred = model.predict([Test_X, test_adjacency])
    Eval = evaluation(pred, test_Y)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    return Eval, pred


def Model_GCN_Feat(X, Y, BS):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=104, test_size=0.25, shuffle=True)

    n_nodes = 10
    n_features = 100
    n_classes = Y.shape[-1]

    Train_X = np.zeros((X_train.shape[0], n_nodes, n_features))
    for i in range(X_train.shape[0]):
        temp = np.resize(X_train[i], (n_nodes, n_features))
        Train_X[i] = np.reshape(temp, (n_nodes, n_features))

    Test_X = np.zeros((X_test.shape[0], n_nodes, n_features))
    for i in range(X_test.shape[0]):
        temp = np.resize(X_test[i], (n_nodes, n_features))
        Test_X[i] = np.reshape(temp, (n_nodes, n_features))

    train_adjacency, _ = create_adjacency_and_indices(X_train, n_nodes)
    test_adjacency, _ = create_adjacency_and_indices(X_test, n_nodes)

    model = model_GCN(n_nodes, n_features, n_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit([Train_X, train_adjacency], y_train, epochs=50, batch_size=BS, steps_per_epoch=1,
              validation_data=([Test_X, test_adjacency], y_test))

    inp = model.input  # input placeholder
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    functors = [K.function([inp], [out]) for out in outputs]  # evaluation functions
    layerNo = 1
    data = X
    Feats = []
    for i in range(data.shape[0]):
        print(i, data.shape[0])
        test = data[i, :][np.newaxis, ...]
        layer_out = np.asarray(functors[layerNo]([test])).squeeze()
        Feats.append(layer_out)
    Feats = np.asarray(Feats)
    Feature = np.resize(Feats, (data.shape[0], Feats.shape[1] * Feats.shape[2]))
    return Feature


