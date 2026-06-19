from models.ode_rnn import OdeRNN
from models.ode_rnn_sparse_theta import OdeRNNSparseTheta
from models.neural_ode import NeuralODE
from models.ode_transformer import OdeTransformer
from models.ode_fixed_theta import OdeFixedTheta
from models.ode_fixed_theta_nn import NeuralOdeCorrection
from models.neural_ode_gru import NeuralOdeGRU
from models.ode_mamba import OdeMamba
from models.ode_lstm import OdeLSTM
from models.ode_slstm import OdesLSTM
from models.ode_mingru import OdeMinGRU
from models.ode_lmu import OdeLMU


try:
    from models.ode_mamba_ssm import OdeMambaSSM
    _mamba_ssm_available = True
except ImportError:
    _mamba_ssm_available = False

try:
    from models.ode_mambapy import OdeMambapySSM
    _mambapy_available = True
except ImportError:
    _mambapy_available = False

# Canonical model zoo used in the thesis.
#   ode_rnn ............... CMVF with a GRU encoder (the main model)
#   ode_rnn_sparse_theta .. K-anchor CMVF (piecewise-constant theta)
#   ode_{transformer,slstm,mingru,mamba,lmu} .. CMVF encoder ablations
#   lstm_rnn .............. CMVF with an LSTM encoder
#   neural_ode_mlp / neural_ode_gru / neural_ode_correction .. NODE baselines
#   ode_fixed_theta ....... static/global-theta mechanistic baseline
MODELS: dict = {
    "ode_rnn":            OdeRNN,
    "ode_rnn_sparse_theta": OdeRNNSparseTheta,
    "lstm_rnn":           OdeLSTM,
    "neural_ode":         NeuralODE,
    "neural_ode_mlp":     NeuralODE,       # alias — same class, explicit name for ablation table
    "ode_transformer":    OdeTransformer,
    "ode_fixed_theta":    OdeFixedTheta,
    "neural_ode_correction": NeuralOdeCorrection,
    "neural_ode_gru":     NeuralOdeGRU,
    "ode_slstm":          OdesLSTM,
    "ode_mingru":         OdeMinGRU,
    "ode_lmu":            OdeLMU,     # Legendre Memory Unit (long input-free retention)
    "ode_mamba":          OdeMamba,   # from-scratch pure-PyTorch selective SSM (full BPTT, jit-able)
    **({"ode_mamba_ssm": OdeMambaSSM} if _mamba_ssm_available else {}),
    **({"ode_mambapy": OdeMambapySSM} if _mambapy_available else {}),
}
