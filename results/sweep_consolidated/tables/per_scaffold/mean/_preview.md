# Per-scaffold tables — NRMSE mean

### arch_sweep — Enzyme/2  (NRMSE mean)

| Model | n=3 | n=10 | n=100 | n=1000 |
|---|---|---|---|---|
| *Full observation* | | | | |
| GRU | -- | -- | 0.046 | -- |
| LSTM | -- | -- | 0.079 | -- |
| sLSTM | **0.533** | **0.114** | 0.064 | **0.012** |
| Transformer | -- | -- | **0.042** | -- |
| Mamba | -- | -- | 0.160 | -- |
| *First--last observation* | | | | |
| GRU | -- | -- | **0.046** | -- |
| LSTM | -- | -- | -- | -- |
| sLSTM | -- | -- | -- | -- |
| Transformer | -- | -- | -- | -- |
| Mamba | -- | -- | -- | -- |

### arch_sweep — Glyc/12  (NRMSE mean)

| Model | n=3 | n=10 | n=100 | n=1000 |
|---|---|---|---|---|
| *Full observation* | | | | |
| GRU | 0.610 | 0.266 | 0.148 | **0.072** |
| LSTM | 0.371 | **0.191** | 0.395 | 0.083 |
| sLSTM | 0.338 | 0.312 | 0.143 | 0.077 |
| Transformer | **0.222** | 0.221 | **0.119** | 0.073 |
| Mamba | 0.249 | 0.257 | 0.162 | 0.125 |
| *First--last observation* | | | | |
| GRU | **0.309** | 0.376 | 0.508 | 0.478 |
| LSTM | 0.384 | **0.319** | 0.508 | **0.393** |
| sLSTM | 0.347 | 0.793 | 0.551 | 0.465 |
| Transformer | 0.396 | 0.359 | 0.416 | 0.449 |
| Mamba | 0.333 | 0.455 | **0.341** | 0.772 |

### arch_sweep — Glyc/8  (NRMSE mean)

| Model | n=3 | n=10 | n=100 | n=1000 |
|---|---|---|---|---|
| *Full observation* | | | | |
| GRU | 0.146 | 0.112 | 0.129 | **0.113** |
| LSTM | 0.269 | 0.132 | **0.122** | 8.0e6 |
| sLSTM | 0.184 | 0.111 | -- | -- |
| Transformer | **0.117** | 0.114 | 0.129 | 0.123 |
| Mamba | 0.200 | **0.108** | 0.130 | 0.168 |
| *First--last observation* | | | | |
| GRU | 0.250 | 0.485 | 0.742 | 0.887 |
| LSTM | 0.206 | **0.300** | 0.736 | 0.939 |
| sLSTM | 0.154 | 0.704 | -- | -- |
| Transformer | 0.201 | 0.537 | 0.631 | **0.821** |
| Mamba | **0.131** | 0.387 | **0.564** | 0.880 |

### arch_sweep — MOF/4  (NRMSE mean)

| Model | n=3 | n=10 | n=100 | n=1000 |
|---|---|---|---|---|
| *Full observation* | | | | |
| GRU | 0.181 | 0.155 | 0.119 | 0.478 |
| LSTM | 0.163 | 0.151 | 0.125 | 0.487 |
| sLSTM | 0.188 | 0.155 | 0.124 | **0.438** |
| Transformer | **0.147** | **0.126** | **0.100** | 0.440 |
| Mamba | 0.222 | 0.152 | 0.117 | 1.146 |
| *First--last observation* | | | | |
| GRU | 1.050 | 0.540 | 0.766 | 12.024 |
| LSTM | 0.935 | 0.681 | 0.788 | 19.292 |
| sLSTM | 0.848 | **0.414** | **0.321** | **7.404** |
| Transformer | 0.831 | 0.585 | 0.774 | 10.535 |
| Mamba | **0.798** | 0.693 | 0.569 | 7.0e12 |

### arch_sweep — MOF/6  (NRMSE mean)

| Model | n=3 | n=10 | n=100 | n=1000 |
|---|---|---|---|---|
| *Full observation* | | | | |
| GRU | -- | **0.067** | 0.057 | **0.039** |
| LSTM | -- | 0.080 | 0.058 | 0.044 |
| sLSTM | **0.122** | 0.085 | **0.056** | 0.067 |
| Transformer | -- | -- | 0.060 | 0.049 |
| Mamba | -- | -- | 0.065 | 0.163 |
| *First--last observation* | | | | |
| GRU | -- | 0.130 | 0.121 | **0.154** |
| LSTM | -- | -- | **0.116** | 0.162 |
| sLSTM | **0.351** | **0.130** | 0.176 | 0.173 |
| Transformer | -- | -- | 0.129 | 0.199 |
| Mamba | -- | -- | 0.178 | 0.339 |

### data_ablation — Enzyme/2  (NRMSE mean)

| Model | n=3 | n=10 | n=100 | n=1000 |
|---|---|---|---|---|
| *Full observation* | | | | |
| CMVF | 0.526 | 0.148 | 0.046 | 0.015 |
| CMVF-L1 | 0.526 | 0.148 | **0.044** | 0.016 |
| CMVF-L2 | -- | -- | -- | 0.016 |
| CMVF-unbounded | 0.606 | **0.080** | 0.055 | **0.012** |
| NODE-GRU | 0.776 | 0.103 | 0.083 | 0.020 |
| NODE-MLP | 0.729 | 0.116 | 0.181 | 0.319 |
| NODE-correction | **0.424** | 0.083 | 0.177 | 0.305 |
| Global ($\theta$) | 0.466 | 0.348 | 0.241 | 0.413 |
| Initial-condition ($\theta$) | 0.500 | 0.180 | 0.266 | 0.422 |
| *First--last observation* | | | | |
| CMVF | 0.526 | 0.148 | 0.046 | -- |
| CMVF-L1 | 0.526 | 0.148 | **0.044** | -- |
| CMVF-L2 | -- | -- | -- | -- |
| CMVF-unbounded | 0.606 | **0.080** | 0.055 | -- |
| NODE-GRU | 0.776 | 0.103 | 0.083 | -- |
| NODE-MLP | 0.729 | 0.116 | 0.181 | -- |
| NODE-correction | **0.424** | 0.083 | 0.177 | -- |
| Global ($\theta$) | 0.466 | 0.348 | 0.241 | **0.413** |
| Initial-condition ($\theta$) | 0.500 | 0.180 | 0.266 | 0.422 |

### data_ablation — Enzyme/4  (NRMSE mean)

| Model | n=3 | n=10 | n=100 | n=1000 |
|---|---|---|---|---|
| *Full observation* | | | | |
| CMVF | 0.031 | 0.074 | **0.005** | 0.004 |
| CMVF-L1 | 0.038 | 0.041 | 0.009 | 0.007 |
| CMVF-unbounded | 0.046 | 0.032 | 0.015 | **0.004** |
| NODE-GRU | 0.733 | 0.139 | 0.031 | 0.011 |
| NODE-MLP | 0.509 | 0.184 | 0.042 | 0.006 |
| NODE-correction | 0.039 | 0.031 | 0.035 | 1.1e6 |
| Global ($\theta$) | 0.031 | 0.030 | 0.036 | 0.042 |
| Initial-condition ($\theta$) | **0.030** | **0.028** | 0.036 | 1.1e6 |
| *First--last observation* | | | | |
| CMVF | **0.030** | 0.060 | **0.005** | **0.004** |
| CMVF-L1 | 0.038 | 0.060 | 0.009 | 0.007 |
| CMVF-unbounded | 0.037 | **0.024** | 0.006 | 1.1e6 |
| NODE-GRU | 0.520 | 0.478 | 0.305 | 0.257 |
| NODE-MLP | 0.279 | 0.193 | 0.265 | 0.291 |
| NODE-correction | 0.067 | 0.050 | 0.103 | 0.086 |
| Global ($\theta$) | 0.031 | 0.031 | 0.037 | 0.042 |
| Initial-condition ($\theta$) | 0.031 | 0.028 | 0.036 | 1.1e6 |

### data_ablation — Enzyme/6  (NRMSE mean)

| Model | n=3 | n=10 | n=100 | n=1000 |
|---|---|---|---|---|
| *Full observation* | | | | |
| CMVF | 0.013 | 0.007 | **0.003** | **0.002** |
| CMVF-L1 | **0.007** | 0.010 | 0.008 | 0.005 |
| CMVF-unbounded | 0.011 | 0.005 | 0.004 | 0.002 |
| NODE-GRU | 4.3e6 | 1.9e6 | 6.2e5 | 1.6e5 |
| NODE-MLP | 7.2e6 | 1.0e6 | 4.1e5 | 6.9e4 |
| NODE-correction | 3.0e6 | 2.4e5 | 1.0e5 | 1.3e4 |
| Global ($\theta$) | 0.019 | 0.014 | 0.014 | 0.007 |
| Initial-condition ($\theta$) | 0.009 | **0.002** | 0.003 | 0.004 |
| *First--last observation* | | | | |
| CMVF | 0.012 | 0.009 | **0.003** | **0.003** |
| CMVF-L1 | 0.010 | 0.010 | 0.004 | 0.005 |
| CMVF-unbounded | 0.012 | 0.005 | 0.005 | 0.003 |
| NODE-GRU | 5.4e7 | 6.6e7 | 1.6e7 | 2.2e7 |
| NODE-MLP | 1.4e7 | 5.0e7 | 1.9e7 | 2.7e7 |
| NODE-correction | 7.7e6 | 2.4e7 | 2.1e7 | 1.5e7 |
| Global ($\theta$) | 0.020 | 0.021 | 0.014 | 0.008 |
| Initial-condition ($\theta$) | **0.009** | **0.002** | 0.003 | 0.004 |

### data_ablation — Glyc/12  (NRMSE mean)

| Model | n=3 | n=10 | n=100 | n=1000 |
|---|---|---|---|---|
| *Full observation* | | | | |
| CMVF | 0.610 | 0.266 | **0.148** | **0.072** |
| CMVF-L1 | 0.259 | 0.308 | 0.161 | 0.077 |
| CMVF-unbounded | **0.215** | 0.266 | 0.172 | 0.088 |
| NODE-GRU | 0.282 | 0.314 | 0.196 | 0.178 |
| NODE-MLP | 0.252 | 0.312 | 0.221 | 0.143 |
| NODE-correction | 0.266 | 0.266 | 0.166 | 0.174 |
| Global ($\theta$) | 0.334 | 0.609 | 0.413 | 0.184 |
| Initial-condition ($\theta$) | 0.253 | **0.211** | 0.175 | 0.182 |
| *First--last observation* | | | | |
| CMVF | 0.309 | 0.376 | 0.508 | 0.478 |
| CMVF-L1 | 0.307 | 0.323 | 0.449 | 9.4e6 |
| CMVF-unbounded | **0.300** | 0.644 | 0.593 | **0.265** |
| NODE-GRU | 1.627 | 2.464 | 1.339 | 7.157 |
| NODE-MLP | 2.120 | 2.667 | 3.563 | 86.196 |
| NODE-correction | 2.021 | 2.678 | 1.361 | 5.618 |
| Global ($\theta$) | 0.333 | 0.553 | **0.356** | 0.387 |
| Initial-condition ($\theta$) | 0.353 | **0.272** | 0.525 | 0.674 |

### data_ablation — Glyc/22  (NRMSE mean)

| Model | n=3 | n=10 | n=100 | n=1000 |
|---|---|---|---|---|
| *Full observation* | | | | |
| CMVF | 0.375 | 0.069 | 0.021 | 1.2e13 |
| CMVF-L1 | 0.812 | 0.069 | 0.027 | 0.006 |
| CMVF-unbounded | **0.095** | **0.041** | **0.016** | **0.004** |
| NODE-GRU | 562.012 | 663.087 | 1.7e4 | 8.0e3 |
| NODE-MLP | 2.337 | 3.1e5 | 1.2e4 | 7.1e3 |
| NODE-correction | 0.285 | 0.231 | 5.1e3 | -- |
| Global ($\theta$) | 0.397 | 0.343 | 0.299 | 0.042 |
| Initial-condition ($\theta$) | 0.310 | 0.112 | 0.030 | 0.015 |
| *First--last observation* | | | | |
| CMVF | 0.638 | 0.273 | **0.185** | 0.215 |
| CMVF-L1 | 0.634 | 0.387 | 0.221 | 0.254 |
| CMVF-unbounded | 0.564 | 0.423 | 0.296 | **0.108** |
| NODE-GRU | 6.7e6 | 1.4e7 | 1.2e7 | 9.3e6 |
| NODE-MLP | 7.0e6 | 3.0e6 | 1.9e6 | 1.0e8 |
| NODE-correction | 0.436 | 1.768 | 5.2e6 | -- |
| Global ($\theta$) | **0.434** | 0.388 | 0.413 | -- |
| Initial-condition ($\theta$) | 0.809 | **0.270** | 0.294 | -- |

### data_ablation — Glyc/4  (NRMSE mean)

| Model | n=3 | n=10 | n=100 | n=1000 |
|---|---|---|---|---|
| *Full observation* | | | | |
| CMVF | **0.234** | 0.226 | **0.111** | **0.077** |
| CMVF-L1 | 0.236 | 0.253 | 0.140 | 0.079 |
| CMVF-unbounded | 0.261 | 0.545 | 0.258 | 0.088 |
| NODE-GRU | 0.239 | 0.165 | 0.145 | 0.108 |
| NODE-MLP | 0.268 | 0.141 | 0.178 | 0.146 |
| NODE-correction | 0.295 | **0.100** | 0.125 | 0.137 |
| Global ($\theta$) | 0.480 | 0.876 | 0.556 | 0.284 |
| Initial-condition ($\theta$) | 0.267 | 0.225 | 0.215 | 0.294 |
| *First--last observation* | | | | |
| CMVF | 0.493 | 1.247 | 1.891 | 4.511 |
| CMVF-L1 | 0.347 | 0.859 | 1.094 | 1.697 |
| CMVF-unbounded | 2.589 | 2.058 | 1.669 | 28.351 |
| NODE-GRU | **0.329** | **0.438** | **0.388** | **0.273** |
| NODE-MLP | 0.532 | 0.572 | 0.523 | 7.494 |
| NODE-correction | 0.378 | 1.114 | 1.102 | 0.348 |
| Global ($\theta$) | 0.565 | 1.062 | 0.909 | 0.892 |
| Initial-condition ($\theta$) | 0.523 | 0.775 | 1.022 | 0.934 |

### data_ablation — Glyc/8  (NRMSE mean)

| Model | n=3 | n=10 | n=100 | n=1000 |
|---|---|---|---|---|
| *Full observation* | | | | |
| CMVF | **0.146** | 0.112 | 0.129 | 0.113 |
| CMVF-L1 | 0.181 | 0.146 | 0.149 | 0.101 |
| CMVF-unbounded | 0.192 | **0.078** | **0.119** | **0.082** |
| NODE-GRU | 0.169 | 0.135 | 0.135 | 0.141 |
| NODE-MLP | 0.271 | 0.112 | 0.122 | 0.133 |
| NODE-correction | 0.247 | 0.169 | 0.136 | 0.140 |
| Global ($\theta$) | 0.569 | 1.015 | 0.960 | 0.339 |
| Initial-condition ($\theta$) | 0.182 | 0.442 | 0.254 | 0.273 |
| *First--last observation* | | | | |
| CMVF | 0.250 | 0.485 | 0.742 | **0.887** |
| CMVF-L1 | 0.279 | 0.772 | 0.752 | 1.275 |
| CMVF-unbounded | 0.251 | 0.593 | **0.699** | 0.953 |
| NODE-GRU | 0.377 | 0.419 | 0.914 | 0.893 |
| NODE-MLP | 0.311 | **0.378** | 2.009 | 5.701 |
| NODE-correction | 0.902 | 2.396 | 2.288 | 6.509 |
| Global ($\theta$) | 0.564 | 1.013 | 0.860 | 1.739 |
| Initial-condition ($\theta$) | **0.161** | 0.461 | 3.689 | 4.234 |

### data_ablation — MOF/12  (NRMSE mean)

| Model | n=3 | n=10 | n=100 | n=1000 |
|---|---|---|---|---|
| *Full observation* | | | | |
| CMVF | **0.086** | **0.014** | 7.430 | 2.144 |
| CMVF-L1 | 0.167 | 0.027 | 38.581 | **2.113** |
| CMVF-unbounded | 0.122 | 0.021 | 11.565 | 2.866 |
| NODE-GRU | 0.450 | 0.339 | 9.7e5 | 3.2e4 |
| NODE-MLP | 0.543 | 0.272 | 1.0e7 | 3.4e4 |
| NODE-correction | 0.200 | 0.111 | 5.3e4 | 5.9e3 |
| Global ($\theta$) | 0.193 | 0.160 | 1.0e3 | 122.809 |
| Initial-condition ($\theta$) | 0.094 | 0.025 | **5.848** | 2.269 |
| *First--last observation* | | | | |
| CMVF | **0.119** | 0.112 | 5.486 | 3.049 |
| CMVF-L1 | 0.119 | 0.113 | 23.798 | 1.174 |
| CMVF-unbounded | 0.173 | **0.080** | 21.744 | 8.517 |
| NODE-GRU | 3.777 | 3.715 | 1.6e7 | 7.8e6 |
| NODE-MLP | 1.358 | 2.121 | 1.9e7 | 4.3e6 |
| NODE-correction | 0.185 | 0.314 | 2.6e5 | 7.7e4 |
| Global ($\theta$) | 0.169 | 0.167 | 930.252 | 83.288 |
| Initial-condition ($\theta$) | 0.142 | 0.141 | **2.412** | **1.043** |

### data_ablation — MOF/4  (NRMSE mean)

| Model | n=3 | n=10 | n=100 | n=1000 |
|---|---|---|---|---|
| *Full observation* | | | | |
| CMVF | 0.181 | 0.155 | **0.119** | 0.478 |
| CMVF-L1 | 0.210 | 0.147 | 0.121 | 0.672 |
| CMVF-L2 | -- | -- | -- | 0.490 |
| CMVF-unbounded | 0.534 | 0.133 | 0.138 | **0.428** |
| NODE-GRU | 0.407 | 0.159 | 2.0e4 | 2.1e4 |
| NODE-MLP | 1.045 | 0.132 | 1.3e6 | 5.0e5 |
| NODE-correction | 0.401 | **0.131** | 0.164 | -- |
| Global ($\theta$) | 0.593 | 0.501 | 0.330 | 4.018 |
| Initial-condition ($\theta$) | **0.178** | 0.159 | 0.135 | 3.413 |
| *First--last observation* | | | | |
| CMVF | 1.050 | 0.540 | **0.766** | 12.024 |
| CMVF-L1 | 1.043 | 0.602 | 11.351 | 9.588 |
| CMVF-L2 | -- | -- | -- | -- |
| CMVF-unbounded | **0.735** | **0.424** | 23.357 | **3.835** |
| NODE-GRU | 1.989 | 3.658 | 8.8e5 | 1.6e7 |
| NODE-MLP | 0.871 | 3.890 | 2.9e7 | 1.2e7 |
| NODE-correction | 0.979 | 0.865 | 2.9e5 | 1.2e5 |
| Global ($\theta$) | 0.948 | 0.668 | 0.859 | 16.990 |
| Initial-condition ($\theta$) | 0.743 | 0.571 | 0.833 | 15.714 |

### data_ablation — MOF/6  (NRMSE mean)

| Model | n=3 | n=10 | n=100 | n=1000 |
|---|---|---|---|---|
| *Full observation* | | | | |
| CMVF | 0.156 | **0.067** | **0.057** | **0.039** |
| CMVF-L1 | 0.250 | 0.075 | 0.057 | 0.074 |
| CMVF-unbounded | 0.196 | 0.079 | 0.537 | 0.126 |
| NODE-GRU | 0.393 | 0.181 | 1.9e5 | 1.7e4 |
| NODE-MLP | 0.408 | 0.182 | 1.9e5 | 5.2e4 |
| NODE-correction | 0.317 | 0.140 | 0.076 | 1.4e4 |
| Global ($\theta$) | 0.240 | 0.255 | 0.109 | 0.236 |
| Initial-condition ($\theta$) | **0.111** | 0.154 | 0.094 | 0.246 |
| *First--last observation* | | | | |
| CMVF | 0.338 | **0.130** | **0.121** | **0.154** |
| CMVF-L1 | 0.273 | 0.144 | 0.130 | 0.459 |
| CMVF-unbounded | 0.264 | 0.132 | 8.915 | 0.223 |
| NODE-GRU | 1.540 | 1.173 | 1.6e7 | 7.4e6 |
| NODE-MLP | 5.559 | 16.193 | 3.5e7 | 5.6e6 |
| NODE-correction | 0.328 | 0.309 | 1.1e4 | 4.7e5 |
| Global ($\theta$) | **0.229** | 0.240 | 0.133 | 0.353 |
| Initial-condition ($\theta$) | 0.265 | 0.172 | 0.191 | 0.213 |

### data_ablation — MOF/8  (NRMSE mean)

| Model | n=3 | n=10 | n=100 | n=1000 |
|---|---|---|---|---|
| *Full observation* | | | | |
| CMVF | 0.215 | **0.052** | **0.021** | **0.015** |
| CMVF-L1 | 0.303 | 0.114 | 0.029 | 0.020 |
| CMVF-unbounded | **0.200** | 0.118 | 0.860 | 0.074 |
| NODE-GRU | 0.370 | 0.246 | 7.6e4 | 3.0e3 |
| NODE-MLP | 0.388 | 0.216 | 4.4e5 | 1.1e4 |
| NODE-correction | 0.346 | 0.173 | 4.5e4 | 1.0e4 |
| Global ($\theta$) | 0.300 | 0.265 | 0.160 | 0.095 |
| Initial-condition ($\theta$) | 0.249 | 0.130 | 0.094 | 0.095 |
| *First--last observation* | | | | |
| CMVF | 0.265 | **0.151** | **0.070** | **0.042** |
| CMVF-L1 | 0.289 | 0.156 | 0.170 | 0.250 |
| CMVF-unbounded | 0.268 | 0.152 | 1.496 | 0.093 |
| NODE-GRU | 3.369 | 3.087 | 1.2e8 | 2.6e7 |
| NODE-MLP | 3.183 | 10.429 | 4.3e7 | 1.9e7 |
| NODE-correction | 0.409 | 0.332 | 1.7e5 | 1.1e6 |
| Global ($\theta$) | 0.272 | 0.259 | 0.168 | 0.150 |
| Initial-condition ($\theta$) | **0.240** | 0.160 | 0.216 | 0.202 |

