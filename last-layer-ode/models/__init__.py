from models.ode_rnn import OdeRNN
from models.ode_rnn_analytical import AnalyticalOdeRNN
from models.ode_rnn_2020 import OdeRNN2020
from models.neural_ode import NeuralODE
from models.ode_transformer import OdeTransformer
from models.ode_mlp import OdeMLP

MODELS: dict = {
    "ode_rnn":            OdeRNN,
    "ode_rnn_analytical": AnalyticalOdeRNN,
    "ode_rnn_2020":       OdeRNN2020,
    "neural_ode":         NeuralODE,
    "ode_transformer":    OdeTransformer,
    "ode_mlp":            OdeMLP,
}
