"""
Feature utilities for NSL-KDD dataset.

Defines:
- 41 feature names and their types
- Feature set groupings (intrinsic, content, time-based, host-based)
- Attack type to category mapping
- Functional feature masks per attack category (Table 1 of the paper)
- Utility functions for the restricted modification mechanism
"""

import numpy as np

# 41 Feature Names (in order)
FEATURE_NAMES = [
    # Intrinsic features (1-9)
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent",
    # Content features (10-22)
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login",
    # Time-based traffic features (23-31)
    "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate",
    # Host-based traffic features (32-41)
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

# Feature indices by set (0-indexed, raw features before encoding)
INTRINSIC_INDICES = list(range(0, 9))      # Features 1-9
CONTENT_INDICES = list(range(9, 22))       # Features 10-22
TIME_BASED_INDICES = list(range(22, 31))   # Features 23-31
HOST_BASED_INDICES = list(range(31, 41))   # Features 32-41

# Categorical features (to be one-hot encoded)
CATEGORICAL_FEATURES = {
    "protocol_type": {
        "index": 1,
        "values": ["icmp", "tcp", "udp"],
    },
    "service": {
        "index": 2,
        "values": [
            "IRC", "X11", "Z39_50", "aol", "auth", "bgp", "courier",
            "csnet_ns", "ctf", "daytime", "discard", "domain",
            "domain_u", "echo", "eco_i", "ecr_i", "efs", "exec", "finger",
            "ftp", "ftp_data", "gopher", "harvest", "hostnames",
            "http", "http_2784", "http_443", "http_8001", "imap4",
            "iso_tsap", "klogin", "kshell", "ldap", "link", "login",
            "mtp", "name", "netbios_dgm", "netbios_ns", "netbios_ssn",
            "netstat", "nnsp", "nntp", "ntp_u", "other", "pm_dump",
            "pop_2", "pop_3", "printer", "private", "red_i",
            "remote_job", "rje", "shell", "smtp", "sql_net", "ssh",
            "sunrpc", "supdup", "systat", "telnet", "tftp_u", "tim_i",
            "time", "urh_i", "urp_i", "uucp", "uucp_path", "vmnet",
            "whois",
        ],
    },
    "flag": {
        "index": 3,
        "values": [
            "OTH", "REJ", "RSTO", "RSTOS0", "RSTR", "S0", "S1",
            "S2", "S3", "SF", "SH",
        ],
    },
}

# Binary features (0-indexed in raw features)
BINARY_FEATURE_INDICES_RAW = [
    6,   # land
    11,  # logged_in
    13,  # root_shell
    14,  # su_attempted
    20,  # is_host_login
    21,  # is_guest_login
]

# Attack type to category mapping
ATTACK_TYPE_TO_CATEGORY = {
    # Normal
    "normal": "Normal",
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS",
    "udpstorm": "DoS", "processtable": "DoS", "mailbomb": "DoS",
    # Probe
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "httptunnel": "U2R", "ps": "U2R",
    "sqlattack": "U2R", "xterm": "U2R",
    # R2L
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L",
    "warezclient": "R2L", "warezmaster": "R2L", "xlock": "R2L",
    "xsnoop": "R2L", "snmpgetattack": "R2L", "named": "R2L",
    "sendmail": "R2L", "snmpguess": "R2L", "worm": "R2L",
}

# Functional features per attack category (Table 1)
#
# From the paper:
# - Checkmark means the feature set IS functional (must NOT be modified)
# - No checkmark means the feature set is NOT functional (CAN be modified)
#
# Table 1:
#   Attack  | Intrinsic | Content | Time-based | Host-based
#   Probe   |     ✓     |         |     ✓      |     ✓
#   DoS     |     ✓     |         |            |
#   U2R     |     ✓     |    ✓    |            |
#   R2L     |     ✓     |    ✓    |            |

# Raw feature indices (0-indexed) that are FUNCTIONAL (unmodifiable)
FUNCTIONAL_FEATURES_RAW = {
    "DoS": INTRINSIC_INDICES,
    "Probe": INTRINSIC_INDICES + TIME_BASED_INDICES + HOST_BASED_INDICES,
    "U2R": INTRINSIC_INDICES + CONTENT_INDICES,
    "R2L": INTRINSIC_INDICES + CONTENT_INDICES,
}

# Raw feature indices that are NON-FUNCTIONAL (modifiable)
MODIFIABLE_FEATURES_RAW = {
    "DoS": CONTENT_INDICES + TIME_BASED_INDICES + HOST_BASED_INDICES,
    "Probe": CONTENT_INDICES,
    "U2R": TIME_BASED_INDICES + HOST_BASED_INDICES,
    "R2L": TIME_BASED_INDICES + HOST_BASED_INDICES,
}


def build_encoded_feature_index_map():
    """
    Build a mapping from raw feature indices to encoded feature indices.

    After one-hot encoding, the feature vector expands from 41 to 122 dimensions.
    This function returns a dict mapping each raw feature index to a list of
    encoded feature indices it maps to.

    Returns:
        dict: {raw_index: [encoded_indices]}
        int: total number of encoded features
    """
    raw_to_encoded = {}
    encoded_idx = 0
    categorical_indices = {v["index"] for v in CATEGORICAL_FEATURES.values()}

    for raw_idx in range(len(FEATURE_NAMES)):
        if raw_idx in categorical_indices:
            # Find which categorical feature this is
            for cat_name, cat_info in CATEGORICAL_FEATURES.items():
                if cat_info["index"] == raw_idx:
                    num_values = len(cat_info["values"])
                    raw_to_encoded[raw_idx] = list(
                        range(encoded_idx, encoded_idx + num_values)
                    )
                    encoded_idx += num_values
                    break
        else:
            raw_to_encoded[raw_idx] = [encoded_idx]
            encoded_idx += 1

    return raw_to_encoded, encoded_idx


# Build the mapping at module level
RAW_TO_ENCODED_MAP, TOTAL_ENCODED_FEATURES = build_encoded_feature_index_map()
assert TOTAL_ENCODED_FEATURES == 122, (
    f"Expected 122 encoded features, got {TOTAL_ENCODED_FEATURES}"
)


def get_functional_feature_mask(attack_category, num_features=122):
    """
    Get a boolean mask indicating functional (unmodifiable) features
    for a given attack category in the encoded feature space.

    Args:
        attack_category: One of "DoS", "Probe", "U2R", "R2L"
        num_features: Total number of encoded features (default 122)

    Returns:
        np.ndarray: Boolean mask of shape (num_features,).
                    True = functional (unmodifiable), False = modifiable
    """
    mask = np.zeros(num_features, dtype=bool)
    raw_indices = FUNCTIONAL_FEATURES_RAW[attack_category]

    for raw_idx in raw_indices:
        encoded_indices = RAW_TO_ENCODED_MAP[raw_idx]
        for enc_idx in encoded_indices:
            mask[enc_idx] = True

    return mask


def get_modifiable_feature_mask(attack_category, num_features=122):
    """
    Get a boolean mask indicating modifiable (non-functional) features
    for a given attack category in the encoded feature space.

    This is the complement of the functional feature mask.

    Args:
        attack_category: One of "DoS", "Probe", "U2R", "R2L"
        num_features: Total number of encoded features (default 122)

    Returns:
        np.ndarray: Boolean mask of shape (num_features,).
                    True = modifiable, False = unmodifiable
    """
    return ~get_functional_feature_mask(attack_category, num_features)


def get_categorical_feature_indices_encoded():
    """
    Get the encoded feature indices that correspond to categorical features.
    These should never be modified (they are always part of intrinsic features).

    Returns:
        list: List of encoded feature indices for categorical features
    """
    indices = []
    for cat_info in CATEGORICAL_FEATURES.values():
        raw_idx = cat_info["index"]
        indices.extend(RAW_TO_ENCODED_MAP[raw_idx])
    return sorted(indices)


def get_binary_feature_indices_encoded():
    """
    Get the encoded feature indices that correspond to binary features.
    Modified binary features should be rounded using the 0.5 threshold.

    Returns:
        list: List of encoded feature indices for binary features
    """
    indices = []
    for raw_idx in BINARY_FEATURE_INDICES_RAW:
        indices.extend(RAW_TO_ENCODED_MAP[raw_idx])
    return sorted(indices)


def get_added_unmodified_features(attack_category, fraction=0.5, seed=42):
    """
    For robustness evaluation (Section 4.3): randomly pick a fraction of
    non-functional features to add to the unmodified set.

    Args:
        attack_category: One of "DoS", "Probe", "U2R", "R2L"
        fraction: Fraction of non-functional features to add (default 0.5)
        seed: Random seed for reproducibility

    Returns:
        np.ndarray: Boolean mask of shape (num_features,).
                    True = unmodifiable (functional + added), False = modifiable
    """
    rng = np.random.RandomState(seed)
    functional_mask = get_functional_feature_mask(attack_category)
    modifiable_indices = np.where(~functional_mask)[0]

    # Pick fraction of modifiable features to add as unmodified
    n_add = int(len(modifiable_indices) * fraction)
    added_indices = rng.choice(modifiable_indices, size=n_add, replace=False)

    # Create new mask with added unmodified features
    new_mask = functional_mask.copy()
    new_mask[added_indices] = True

    return new_mask
