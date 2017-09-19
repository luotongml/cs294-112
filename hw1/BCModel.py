"""Behavioral Cloning Model"""

import pickle
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.activations import relu, linear
from keras.optimizers import RMSprop

class BCModel:
    HIDDEN_LAYERS_DEFAULT = 1
    UNITS_DEFAULT = 32
    EPOCHS_DEFAULT = 50
    LEARNING_RATE_DEFAULT = 0.001

    def __init__(self, hidden_layers, units, epochs, learning_rate):
        self.hidden_layers = hidden_layers or BCModel.HIDDEN_LAYERS_DEFAULT
        self.units = units or BCModel.UNITS_DEFAULT
        self.epochs = epochs or BCModel.EPOCHS_DEFAULT
        self.learning_rate = learning_rate or BCModel.LEARNING_RATE_DEFAULT

        self.brain = None
        self.history = None

    def train(self, observations, actions):
        self.brain = Sequential()
        for _ in range(self.hidden_layers):
            self.brain.add(Dense(self.units, activation=relu, input_shape=(observations.shape[1],)))
        self.brain.add(Dense(actions.shape[1], activation=linear))
        self.brain.compile(optimizer=RMSprop(lr=self.learning_rate), loss='mse')

        history = self.brain.fit(observations, actions, epochs=self.epochs)

        self.history = history.history
        return self.history

    def predict(self, observations):
        return self.brain.predict(observations)
    
    def save(self, save_name):
        self.brain.save('trained_models/{}.h5'.format(save_name))
        with open('trained_models/{}_params.pkl'.format(save_name), 'wb') as f:
            pickle.dump((self.hidden_layers, self.units, self.epochs, self.learning_rate), f)
        with open('trained_models/{}_history.pkl'.format(save_name), 'wb') as f:
            pickle.dump(self.history, f)

    @staticmethod
    def load(saved_model):
        with open('trained_models/{}_params.pkl'.format(saved_model), 'rb') as f:
            model = BCModel(*pickle.load(f))
        with open('trained_models/{}_history.pkl'.format(saved_model), 'rb') as f:
            model.history = pickle.load(f)
        model.brain = load_model('trained_models/{}.h5'.format(saved_model))

        return model
