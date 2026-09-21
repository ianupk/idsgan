"""
IDSGAN - src package
====================
Implementation of Lin et al. "IDSGAN: Generative Adversarial Networks for
Attack Generation against Intrusion Detection" (arXiv:1809.02077v5).

Sub-modules
-----------
config              Hyperparameters and filesystem paths
data_preprocessing  NSL-KDD loading, encoding, and train/test splitting
feature_utils       Feature definitions and functional-feature masks
generator           PyTorch Generator network
discriminator       PyTorch Discriminator network (WGAN critic)
idsgan              IDSGAN training and inference wrapper
ids_models          Seven scikit-learn black-box IDS classifiers
metrics             Detection Rate (DR) and Evasion Increase Rate (EIR)
"""
