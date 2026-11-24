# Biology & Mathematical Formulation

## 1. Connectome Data & Dale's Law
- **Source:** FlyWire v783 whole-brain connectome (`Completeness_783.csv`, `Connectivity_783.parquet` from `philshiu/Drosophila_brain_model`) + `Supplemental_file1_neuron_annotations.tsv` (`flyconnectome/flywire_annotations`).
- **Scale:** `138,639` neurons, `15,091,983` directed edges, `54,492,922` total synapses.
- **Six Canonical Dorsal Regions:** `Sensory`, `Antennal Lobe`, `Mushroom Body`, `Central Complex`, `Lateral Horn`, `Motor/Output`.
- **Dale's Law:** Each presynaptic neuron is assigned `+1` (excitatory, e.g. acetylcholine) or `-1` (inhibitory, e.g. GABA/glutamate). All plastic weight updates clamp excitatory synapses to $[0, w_{\max}]$ and inhibitory synapses to $[-w_{\max}, 0]$.

## 2. Batched Leaky Integrate-and-Fire (LIF) Dynamics
At $\Delta t = 1.0\text{ ms}$ over $T = 60\text{ ms}$ with $\tau_m = 20.0\text{ ms}$ ($\alpha = e^{-\Delta t / \tau_m}$), $\tau_{\text{syn}} = 5.0\text{ ms}$ ($\beta = e^{-\Delta t / \tau_{\text{syn}}}$), $V_{\text{rest}} = V_{\text{reset}} = -52.0\text{ mV}$, $V_{\text{thresh}} = -45.0\text{ mV}$, and $\tau_{\text{ref}} = 2\text{ ms}$:
1. $I_{\text{syn}}(t) = \beta I_{\text{syn}}(t-1) + W_{\text{eff}}^\top s(t-1) + I_{\text{ext}}(t)$
2. $V(t) = \alpha V(t-1) + (1 - \alpha)(V_{\text{rest}} + R_m I_{\text{syn}}(t))$ (for neurons with refractory counter $= 0$)
3. $s(t) = \mathbb{I}[V(t) \ge V_{\text{thresh}}]$, resetting spiking neurons to $V_{\text{reset}}$ and setting refractory counter $= 2$.

## 3. Three-Factor Dopamine R-STDP & REINFORCE
- **R-STDP:** Plastic synapses (Mushroom Body & Motor/Output targets) track decaying pre/post spike traces and eligibility $e_{ij}(t+1) = \lambda_e e_{ij}(t) + A_+ x_i(t) s_j(t) - A_- s_i(t) y_j(t)$ ($\lambda_e = 0.95$). At trial end, $\Delta W_{ij} = \eta (r - \bar{r}) e_{ij}(T)$ with $\bar{r} \leftarrow 0.95\bar{r} + 0.05 r$, followed by Dale's Law clamping.
- **REINFORCE:** Samples $y \sim \text{softmax}(\text{logits} / \tau)$ and updates policy weights via $\nabla_\theta \log \pi_\theta(y \mid x)(r - \bar{r})$.
