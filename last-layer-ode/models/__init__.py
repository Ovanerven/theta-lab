from models.ode_rnn import OdeRNN
from models.ode_rnn_analytical import AnalyticalOdeRNN
from models.ode_rnn_2020 import OdeRNN2020
from models.neural_ode import NeuralODE
from models.ode_transformer import OdeTransformer
from models.ode_mlp import OdeMLP
from models.ode_fixed_theta import OdeFixedTheta
from models.ode_sample_theta import OdeSampleTheta
from models.neural_ode_gru import NeuralOdeGRU
from models.ode_mamba import OdeMamba
from models.lstm_rnn import LstmRNN

try:
    from models.ode_mamba_ssm import OdeMambaSSM
    _mamba_ssm_available = True
except ImportError:
    _mamba_ssm_available = False

MODELS: dict = {
    "ode_rnn":            OdeRNN,
    "lstm_rnn":           LstmRNN,
    "ode_rnn_analytical": AnalyticalOdeRNN,
    "ode_rnn_2020":       OdeRNN2020,
    "neural_ode":         NeuralODE,
    "neural_ode_mlp":     NeuralODE,       # alias — same class, explicit name for ablation table
    "ode_transformer":    OdeTransformer,
    "ode_mlp":            OdeMLP,
    "ode_fixed_theta":    OdeFixedTheta,
    "ode_sample_theta":   OdeSampleTheta,
    "neural_ode_gru":     NeuralOdeGRU,
    "ode_mamba":          OdeMamba,
    **({"ode_mamba_ssm": OdeMambaSSM} if _mamba_ssm_available else {}),
}
