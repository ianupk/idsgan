import numpy as np

def detection_rate(y_true, y_pred):
    """
    Compute Detection Rate (DR) = correctly detected attacks / total attacks.
    Only considers samples where y_true == 1.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    attack_mask = (y_true == 1)
    if not np.any(attack_mask):
        return 0.0
        
    correctly_detected = np.sum(y_pred[attack_mask] == 1)
    total_attacks = np.sum(attack_mask)
    
    return float(correctly_detected) / total_attacks

def evasion_increase_rate(original_dr, adversarial_dr):
    """
    Compute Evasion Increase Rate (EIR) = 1 - (adversarial_dr / original_dr).
    Returns 0.0 if original_dr is 0.
    """
    if original_dr == 0.0:
        return 0.0
    return 1.0 - (adversarial_dr / original_dr)

def compute_all_metrics(original_labels, original_preds, adversarial_preds):
    """
    Computes original detection rate, adversarial detection rate, and evasion increase rate.
    
    Returns:
        dict: containing 'original_dr', 'adversarial_dr', and 'eir'
    """
    original_dr = detection_rate(original_labels, original_preds)
    adversarial_dr = detection_rate(original_labels, adversarial_preds)
    eir = evasion_increase_rate(original_dr, adversarial_dr)
    
    return {
        "original_dr": original_dr,
        "adversarial_dr": adversarial_dr,
        "eir": eir
    }
