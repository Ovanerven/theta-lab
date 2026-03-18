from models.ode_rnn import OdeRNN
from models.ode_rnn_analytical import AnalyticalOdeRNN
from models.neural_ode import NeuralODE

MODELS: dict = {
    "ode_rnn":            OdeRNN,
    "ode_rnn_analytical": AnalyticalOdeRNN,
    "neural_ode":         NeuralODE,
}
