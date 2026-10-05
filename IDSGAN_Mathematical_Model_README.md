# IDSGAN — Mathematical Model and Weight Update

## Paper

**IDSGAN: Generative Adversarial Networks for Attack Generation against Intrusion Detection**

The paper proposes an improved GAN-based framework for generating adversarial malicious traffic records against a **black-box intrusion detection system (IDS)**. The generator creates modified malicious traffic records, while a discriminator learns the behavior of the black-box IDS from its query outputs and provides differentiable feedback to the generator.

---

# 1. Problem Definition

An intrusion detection system classifies a traffic record as normal or malicious.

Let

$$
\mathbf{x} \in \mathbb{R}^{m}
$$

represent a preprocessed malicious traffic record.

The objective of IDSGAN is to construct an adversarial record

$$
\mathbf{x}_{adv}
$$

such that:

1. It remains related to the original malicious traffic record.
2. Functional features of the original attack are preserved.
3. The black-box IDS is more likely to misclassify the adversarial record.

The paper uses the NSL-KDD dataset. The original dataset contains 41 features: 9 discrete and 32 continuous features. Categorical features are encoded and numerical features are normalized to $[0,1]$.

---

# 2. Main Components

IDSGAN contains three conceptual components:

$$
\boxed{
\text{Generator }G_\theta
}
$$

$$
\boxed{
\text{Black-box IDS }B
}
$$

$$
\boxed{
\text{Discriminator / Surrogate }D_\phi
}
$$

Their roles are:

| Component | Role |
|---|---|
| $G_\theta$ | Generates candidate adversarial traffic |
| $B$ | Target black-box IDS |
| $D_\phi$ | Learns/imitates the behavior of $B$ and provides differentiable feedback to $G$ |

The paper assumes that the internal architecture and parameters of $B$ are unknown, but its classification results can be obtained by querying it.

---

# 3. Notation

| Symbol | Meaning |
|---|---|
| $\mathbf{x}$ | Original malicious traffic record |
| $\mathbf{n}$ | Random noise vector |
| $m$ | Dimension of preprocessed traffic vector |
| $k$ | Noise dimension; paper uses $k=9$ |
| $\theta$ | Generator parameters |
| $\phi$ | Discriminator parameters |
| $G_\theta$ | Generator |
| $D_\phi$ | Discriminator / surrogate IDS |
| $B$ | Black-box target IDS |
| $\mathbf{M}$ | Feature modification mask |
| $\tilde{\mathbf{x}}$ | Raw generated traffic record |
| $\mathbf{x}_{adv}$ | Restricted adversarial traffic record |
| $L_G$ | Generator loss |
| $L_D$ | Discriminator loss |
| $\eta$ | Learning rate |
| $\rho$ | RMSProp decay parameter |
| $\epsilon$ | Numerical stability constant |

---

# 4. Overall Mathematical Flow

The complete IDSGAN pipeline can be represented as:

$$
\boxed{
\mathbf{x}
\xrightarrow[\mathbf{n}]{G_\theta}
\tilde{\mathbf{x}}
\xrightarrow{\text{Restricted Modification}}
\mathbf{x}_{adv}
\xrightarrow{B}
y_B
}
$$

The discriminator observes the same traffic samples and learns the behavior of the black-box IDS:

$$
\boxed{
\mathbf{x}_{adv}
\xrightarrow{D_\phi}
\hat{y}
}
$$

The discriminator provides a differentiable path for optimizing the generator:

$$
\boxed{
D_\phi
\rightarrow
L_G
\rightarrow
\nabla_\theta L_G
\rightarrow
G_\theta
}
$$

while the black-box IDS provides labels/predictions used to train the discriminator:

$$
\boxed{
B(\mathbf{x})
\rightarrow
\text{IDS prediction}
\rightarrow
L_D
\rightarrow
\nabla_\phi L_D
\rightarrow
D_\phi
}
$$

---

# 5. Input and Noise

Let:

$$
\mathbf{x}\in[0,1]^m
$$

be the normalized malicious traffic record.

The paper uses a noise vector:

$$
\mathbf{n}\sim U(0,1)^k
$$

with:

$$
\boxed{k=9}
$$

The generator input is the concatenation:

$$
\boxed{
\mathbf{z}=
[\mathbf{x};\mathbf{n}]
}
$$

Therefore:

$$
\mathbf{z}\in\mathbb{R}^{m+k}
$$

---

# 6. Generator Mathematical Model

The paper describes a generator consisting of five linear layers. ReLU is used after the first four layers, and the output has the same dimension as the traffic vector.

### Layer 1

$$
\mathbf{h}_1
=
\text{ReLU}
\left(
W_1\mathbf{z}+\mathbf{b}_1
\right)
$$

### Layer 2

$$
\mathbf{h}_2
=
\text{ReLU}
\left(
W_2\mathbf{h}_1+\mathbf{b}_2
\right)
$$

### Layer 3

$$
\mathbf{h}_3
=
\text{ReLU}
\left(
W_3\mathbf{h}_2+\mathbf{b}_3
\right)
$$

### Layer 4

$$
\mathbf{h}_4
=
\text{ReLU}
\left(
W_4\mathbf{h}_3+\mathbf{b}_4
\right)
$$

### Output Layer

$$
\boxed{
\tilde{\mathbf{x}}
=
W_5\mathbf{h}_4+\mathbf{b}_5
}
$$

where:

$$
\text{ReLU}(a)=\max(0,a)
$$

Therefore the complete generator is:

$$
\boxed{
\tilde{\mathbf{x}}
=
G_\theta(\mathbf{x},\mathbf{n})
}
$$

where:

$$
\theta=
\{W_1,\mathbf{b}_1,\ldots,W_5,\mathbf{b}_5\}
$$

---

# 7. Output Constraint

The paper restricts generated feature values to the normalized range $[0,1]$.

Therefore:

$$
\boxed{
\tilde{x}_i
=
\text{clip}(\tilde{x}_i,0,1)
}
$$

or:

$$
\boxed{
\tilde{\mathbf{x}}
=
\text{clip}
\left(
\tilde{\mathbf{x}},0,1
\right)
}
$$

For binary features, the paper uses a threshold of $0.5$:

$$
\tilde{x}_i =
\begin{cases}
1,&\tilde{x}_i\geq0.5\\
0,&\tilde{x}_i<0.5
\end{cases}
$$

---

# 8. Restricted Modification Mechanism

This is a central contribution of the paper.

Not every feature should be modified.

Define a binary mask:

$$
\mathbf{M}\in\{0,1\}^{m}
$$

where:

$$
M_i=
\begin{cases}
0,&\text{feature must remain unchanged}\\
1,&\text{feature may be modified}
\end{cases}
$$

The actual adversarial record is:

$$
\boxed{
\mathbf{x}_{adv}
=
(\mathbf{1}-\mathbf{M})\odot\mathbf{x}
+
\mathbf{M}\odot\tilde{\mathbf{x}}
}
$$

where $\odot$ denotes element-wise multiplication.

For a fixed feature:

$$
M_i=0
$$

therefore:

$$
x_{adv,i}=x_i
$$

For a modifiable feature:

$$
M_i=1
$$

therefore:

$$
x_{adv,i}=\tilde{x}_i
$$

Hence:

$$
\boxed{
\text{Mask}=\text{constraint on which features can change}
}
$$

The paper identifies functional feature groups for different attack categories and keeps those functional features unchanged.

---

# 9. Black-Box IDS

Let the target IDS be:

$$
\boxed{
B:\mathbb{R}^{m}\rightarrow\mathcal{Y}
}
$$

where $\mathcal{Y}$ represents the IDS prediction.

For example:

$$
B(\mathbf{x})=
\begin{cases}
0,&\text{normal}\\
1,&\text{malicious}
\end{cases}
$$

The important assumption is that the attacker does not know the internal structure or parameters of $B$.

Instead, it can query:

$$
\boxed{
y_B=B(\mathbf{x})
}
$$

For an adversarial sample:

$$
\boxed{
y_{B,adv}=B(\mathbf{x}_{adv})
}
$$

---

# 10. Discriminator / Surrogate Model

The discriminator is trained to learn the behavior of the black-box IDS.

Therefore:

$$
\boxed{
D_\phi(\mathbf{x})\approx B(\mathbf{x})
}
$$

The black-box IDS provides the current predictions:

$$
y_B=B(\mathbf{x})
$$

These predictions are used as training information for $D_\phi$.

Conceptually:

$$
\boxed{
\text{Black-box IDS}
\rightarrow
\text{query}
\rightarrow
\text{prediction}
\rightarrow
D_\phi
}
$$

The important point is that $D_\phi$ is differentiable, while the black-box IDS may not be differentiable.

---

# 11. Why the Discriminator Is Needed

Suppose:

$$
B=\text{Random Forest}
$$

The generator is implemented using PyTorch.

Directly calculating:

$$
\frac{\partial B}{\partial\theta}
$$

is not generally available through the normal neural-network backpropagation path.

Instead:

$$
\boxed{
G_\theta
\rightarrow
\mathbf{x}_{adv}
\rightarrow
D_\phi
\rightarrow
L_G
}
$$

allows us to calculate:

$$
\frac{\partial L_G}{\partial\theta}
$$

using ordinary backpropagation.

Therefore $D_\phi$ acts as a differentiable approximation of the black-box IDS for generator optimization.

---

# 12. Generator Loss

The paper defines the generator loss as:

$$
\boxed{
L_G
=
\mathbb{E}_{\mathbf{x}\in S_{attack},\mathbf{n}}
\left[
D_\phi
\left(
G_\theta(\mathbf{x},\mathbf{n})
\right)
\right]
}
$$

with the restricted modification mechanism applied to the generated sample.

In implementation form:

$$
\boxed{
L_G
=
\mathbb{E}
\left[
D_\phi(\mathbf{x}_{adv})
\right]
}
$$

The generator minimizes this objective:

$$
\boxed{
\theta^*
=
\arg\min_\theta L_G
}
$$

---

# 13. Generator Gradient

The generator parameters are updated through the discriminator.

Using the chain rule:

$$
\boxed{
\frac{\partial L_G}{\partial\theta}
=
\frac{\partial L_G}{\partial D}
\frac{\partial D}{\partial\mathbf{x}_{adv}}
\frac{\partial\mathbf{x}_{adv}}
{\partial G_\theta}
\frac{\partial G_\theta}{\partial\theta}
}
$$

Because:

$$
\mathbf{x}_{adv}
=
(\mathbf{1}-\mathbf{M})\odot\mathbf{x}
+
\mathbf{M}\odot G_\theta(\mathbf{x},\mathbf{n})
$$

we obtain:

$$
\boxed{
\frac{\partial\mathbf{x}_{adv}}
{\partial G_\theta}
=
\mathbf{M}
}
$$

Therefore:

$$
\boxed{
\frac{\partial L_G}{\partial\theta}
=
\frac{\partial L_G}{\partial D}
\frac{\partial D}{\partial\mathbf{x}_{adv}}
\mathbf{M}
\frac{\partial G_\theta}{\partial\theta}
}
$$

This shows mathematically how the modification mask controls the gradient flow.

If:

$$
M_i=0
$$

the corresponding feature receives no generator gradient through the modification path.

---

# 14. Discriminator Loss

The paper defines:

$$
\boxed{
L_D
=
\mathbb{E}_{s\in B_{normal}}[D_\phi(s)]
-
\mathbb{E}_{s\in B_{attack}}[D_\phi(s)]
}
$$

where the normal and adversarial traffic records are associated with predictions obtained from the black-box IDS.

The discriminator is optimized according to the WGAN-style formulation used by the paper.

### Important implementation note

The paper states the mathematical loss and says it uses RMSProp and discriminator weight clipping. When implementing with a framework optimizer that performs minimization, the equivalent sign convention is commonly written as:

$$
\boxed{
L_D^{min}
=
\mathbb{E}[D_\phi(x_{adv})]
-
\mathbb{E}[D_\phi(x_{normal})]
}
$$

The sign convention must be kept consistent with the optimizer.

---

# 15. Weight Update — Core Mathematical Part

The paper uses **RMSProp** with learning rate:

$$
\boxed{
\eta=0.0001
}
$$

for both generator and discriminator.

The following gives the mathematical weight-update mechanism.

---

## 15.1 Generator Gradient

At iteration $t$:

$$
\boxed{
g_{\theta,t}
=
\nabla_\theta L_G
}
$$

For each generator parameter $\theta_i$:

$$
g_{\theta_i,t}
=
\frac{\partial L_G}
{\partial\theta_i}
$$

---

## 15.2 RMSProp Accumulator

Maintain a moving average of squared gradients:

$$
\boxed{
v_{\theta,t}
=
\rho v_{\theta,t-1}
+
(1-\rho)
g_{\theta,t}^{\,2}
}
$$

where the square is element-wise.

---

## 15.3 Generator Weight Update

The RMSProp update is:

$$
\boxed{
\theta_{t+1}
=
\theta_t
-
\eta
\frac{g_{\theta,t}}
{\sqrt{v_{\theta,t}}+\epsilon}
}
$$

Therefore:

$$
\boxed{
\text{Generator}
\rightarrow
\text{gradient}
\rightarrow
\text{RMSProp}
\rightarrow
\text{new generator weights}
}
$$

---

# 16. Discriminator Weight Update

Calculate the discriminator gradient:

$$
\boxed{
g_{\phi,t}
=
\nabla_\phi L_D
}
$$

Maintain its squared-gradient moving average:

$$
\boxed{
v_{\phi,t}
=
\rho v_{\phi,t-1}
+
(1-\rho)
g_{\phi,t}^{\,2}
}
$$

Then:

$$
\boxed{
\phi_{t+1}
=
\phi_t
-
\eta
\frac{g_{\phi,t}}
{\sqrt{v_{\phi,t}}+\epsilon}
}
$$

---

# 17. Discriminator Weight Clipping

The paper uses a weight clipping threshold:

$$
\boxed{
c=0.01
}
$$

After updating discriminator parameters:

$$
\boxed{
\phi_{t+1}
\leftarrow
\text{clip}
(
\phi_{t+1},
-c,
c
)
}
$$

Therefore:

$$
\boxed{
\phi_{t+1}
\in[-0.01,0.01]
}
$$

This is part of the WGAN-style implementation described by the paper.

---

# 18. Complete Generator Weight-Update Derivation

The complete path is:

$$
\mathbf{x}
\rightarrow
G_\theta
\rightarrow
\mathbf{x}_{adv}
\rightarrow
D_\phi
\rightarrow
L_G
$$

Therefore:

### Forward

$$
\mathbf{x}_{adv}
=
(\mathbf{1}-\mathbf{M})\odot\mathbf{x}
+
\mathbf{M}\odot G_\theta(\mathbf{x},\mathbf{n})
$$

### Loss

$$
L_G=D_\phi(\mathbf{x}_{adv})
$$

### Gradient

$$
g_{\theta}
=
\nabla_\theta L_G
$$

### RMSProp

$$
v_{\theta}
=
\rho v_{\theta}
+
(1-\rho)g_{\theta}^2
$$

### Update

$$
\boxed{
\theta
\leftarrow
\theta
-
\eta
\frac{g_{\theta}}
{\sqrt{v_{\theta}}+\epsilon}
}
$$

---

# 19. Complete Discriminator Weight-Update Derivation

The discriminator receives normal and adversarial samples.

### Forward

$$
D_\phi(x_{normal})
$$

and:

$$
D_\phi(x_{adv})
$$

### Loss

Using the minimization form:

$$
L_D^{min}
=
E[D_\phi(x_{adv})]
-
E[D_\phi(x_{normal})]
$$

### Gradient

$$
g_\phi
=
\nabla_\phi L_D^{min}
$$

### RMSProp accumulator

$$
v_\phi
=
\rho v_\phi
+
(1-\rho)g_\phi^2
$$

### Weight update

$$
\boxed{
\phi
\leftarrow
\phi
-
\eta
\frac{g_\phi}
{\sqrt{v_\phi}+\epsilon}
}
$$

### Clip

$$
\boxed{
\phi
\leftarrow
\text{clip}(\phi,-0.01,0.01)
}
$$

---

# 20. Alternating Optimization

IDSGAN does not update both networks only once.

The paper's Algorithm 1 alternates:

1. Generator steps
2. Discriminator steps

The algorithm is:

$$
\boxed{
\begin{aligned}
&\text{Initialize }G_\theta,D_\phi\\
&\\
&\text{repeat:}\\
&\qquad \text{for }G\text{-steps:}\\
&\qquad\qquad \mathbf{x}_{adv}=G_\theta(\mathbf{x},\mathbf n)\\
&\qquad\qquad \theta\leftarrow\text{RMSProp update}\\
&\\
&\qquad \text{for }D\text{-steps:}\\
&\qquad\qquad y_B=B(\mathbf{x}_{normal},\mathbf{x}_{adv})\\
&\qquad\qquad \phi\leftarrow\text{RMSProp update}\\
&\qquad\qquad \phi\leftarrow\text{clip}(\phi,-0.01,0.01)\\
&\\
&\text{until convergence}
\end{aligned}
}
$$

This corresponds to Algorithm 1 in the paper.

---

# 21. Complete Mathematical Model

The entire model can be summarized as:

$$
\boxed{
\begin{aligned}
\mathbf n&\sim U(0,1)^9\\
\tilde{\mathbf x}
&=
G_\theta(\mathbf x,\mathbf n)\\
\mathbf x_{adv}
&=
(\mathbf 1-\mathbf M)\odot\mathbf x
+
\mathbf M\odot\tilde{\mathbf x}\\
y_B
&=
B(\mathbf x_{adv})\\
D_\phi(\mathbf x_{adv})
&\approx
B(\mathbf x_{adv})
\end{aligned}
}
$$

Generator objective:

$$
\boxed{
\theta^*
=
\arg\min_\theta
E[D_\phi(\mathbf x_{adv})]
}
$$

Discriminator objective in the paper's WGAN-style notation:

$$
\boxed{
L_D
=
E[D_\phi(x_{normal})]
-
E[D_\phi(x_{attack})]
}
$$

Weight updates:

$$
\boxed{
\theta_{t+1}
=
\theta_t
-
\eta
\frac{\nabla_\theta L_G}
{\sqrt{v_{\theta,t}}+\epsilon}
}
$$

$$
\boxed{
v_{\theta,t}
=
\rho v_{\theta,t-1}
+
(1-\rho)(\nabla_\theta L_G)^2
}
$$

and:

$$
\boxed{
\phi_{t+1}
=
\phi_t
-
\eta
\frac{\nabla_\phi L_D^{min}}
{\sqrt{v_{\phi,t}}+\epsilon}
}
$$

followed by:

$$
\boxed{
\phi_{t+1}
=
\text{clip}
(\phi_{t+1},-0.01,0.01)
}
$$

with:

$$
\boxed{\eta=10^{-4}}
$$

---

# 22. Training Loop as a Mathematical Algorithm

```text
Input:
    malicious samples X
    normal samples X_normal
    black-box IDS B
    modification mask M

Initialize:
    generator parameters θ
    discriminator parameters φ

Repeat:

    1. Sample malicious x
       Sample noise n ~ U(0,1)^9

    2. Generate:
           x_tilde = Gθ(x,n)

    3. Apply restrictions:
           x_adv =
               (1-M) ⊙ x
               +
               M ⊙ x_tilde

    4. Generator:
           LG = Dφ(x_adv)

           gθ = ∇θ LG

           vθ = ρvθ + (1-ρ)gθ²

           θ ← θ - η gθ/(sqrt(vθ)+ε)

    5. Query black-box IDS:
           y = B(x_normal)
           y_adv = B(x_adv)

    6. Train discriminator:
           LD =
               E[Dφ(x_normal)]
               -
               E[Dφ(x_adv)]

           gφ = ∇φ LD

           vφ = ρvφ + (1-ρ)gφ²

           φ ← φ - η gφ/(sqrt(vφ)+ε)

    7. Clip discriminator weights:
           φ ← clip(φ,-0.01,0.01)

Until convergence.
```

---

# 23. Hyperparameters Reported by the Paper

| Parameter | Paper setting |
|---|---:|
| Batch size | 64 |
| Epochs | 100 |
| Generator learning rate | 0.0001 |
| Discriminator learning rate | 0.0001 |
| Noise dimension | 9 |
| Discriminator weight clipping | 0.01 |
| Optimizer | RMSProp |
| Deep-learning framework | PyTorch |
| Black-box IDS implementation | scikit-learn |

These settings are reported in the paper's experimental setup.

---

# 24. One-Slide Explanation for a Professor

The simplest way to explain the mathematical model is:

### Input

$$
x=\text{malicious traffic}
$$

### Generator

$$
\tilde{x}=G_\theta(x,n)
$$

### Feature restriction

$$
x_{adv}
=
(1-M)x+M\tilde{x}
$$

### Black-box IDS

$$
y=B(x_{adv})
$$

### Surrogate discriminator

$$
D_\phi(x_{adv})\approx B(x_{adv})
$$

### Generator loss

$$
L_G=E[D_\phi(x_{adv})]
$$

### Generator update

$$
\theta
\leftarrow
\theta-
\eta
\frac{\nabla_\theta L_G}
{\sqrt{v_\theta}+\epsilon}
$$

### Discriminator loss

$$
L_D
=
E[D(x_{normal})]
-
E[D(x_{adv})]
$$

### Discriminator update

$$
\phi
\leftarrow
\phi-
\eta
\frac{\nabla_\phi L_D}
{\sqrt{v_\phi}+\epsilon}
$$

### Weight clipping

$$
\phi\leftarrow
\text{clip}(\phi,-0.01,0.01)
$$

### Feedback loop

$$
\boxed{
B
\rightarrow
D
\rightarrow
G
\rightarrow
B
}
$$

That is the core mathematical idea of IDSGAN.

---

# 25. Key Intuition

The entire model can be understood as a three-player-style interaction:

$$
\boxed{
\text{Generator}
\quad\longleftrightarrow\quad
\text{Surrogate IDS}
\quad\longleftrightarrow\quad
\text{Black-box IDS}
}
$$

The **black-box IDS** gives predictions.

The **discriminator** learns those predictions.

The **generator** uses the discriminator's differentiable gradient to learn how to transform malicious records.

The **feature mask** prevents arbitrary modification of attack-functional features.

Thus:

$$
\boxed{
\text{Query black-box}
\rightarrow
\text{learn surrogate}
\rightarrow
\text{backpropagate}
\rightarrow
\text{update generator}
}
$$

This is the mathematical mechanism that makes IDSGAN a **black-box adversarial example generation framework** rather than a normal GAN.
