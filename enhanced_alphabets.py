"""
Enhanced Biochemical Alphabets for SCFD Protein Analysis
======================================================

Multiple alphabet encodings for different levels of biochemical detail.
"""

import numpy as np

# Original ternary alphabet (working baseline)
TERNARY_ALPHABET = {
    'ALA': 0, 'VAL': 0, 'ILE': 0, 'LEU': 0, 'MET': 0, 'PHE': 0, 'TYR': 0, 'TRP': 0,
    'GLY': 1, 'SER': 1, 'THR': 1, 'CYS': 1, 'ASN': 1, 'GLN': 1, 'PRO': 1,
    'ASP': 2, 'GLU': 2, 'LYS': 2, 'ARG': 2, 'HIS': 2,
}

# Enhanced biochemical alphabet (12 symbols)
BIOCHEMICAL_12_ALPHABET = {
    # Hydrophobic - grouped by properties
    'LEU': 0, 'ILE': 0,                    # Branched aliphatic
    'VAL': 1,                              # Small branched
    'ALA': 2,                              # Small nonpolar
    'MET': 3,                              # Sulfur-containing hydrophobic
    'PHE': 4, 'TYR': 4,                    # Aromatic (Tyr can H-bond but mostly hydrophobic)
    'TRP': 5,                              # Large aromatic
    
    # Polar - grouped by H-bonding capacity  
    'SER': 6, 'THR': 6,                    # Hydroxyl groups
    'ASN': 7, 'GLN': 7,                    # Amide groups
    'CYS': 8,                              # Sulfhydryl (disulfide potential)
    
    # Charged - by charge type
    'LYS': 9, 'ARG': 9,                    # Positive charges
    'ASP': 10, 'GLU': 10,                  # Negative charges  
    'HIS': 11,                             # Titratable (context-dependent charge)
    
    # Special structural roles
    'GLY': 12,                             # High flexibility
    'PRO': 13,                             # Rigid, helix breaker
}

# Full amino acid alphabet (20 symbols - one per amino acid)
FULL_20_ALPHABET = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
    'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
}

# Alphabet configurations
ALPHABET_CONFIGS = {
    'ternary': {
        'map': TERNARY_ALPHABET,
        'size': 3,
        'max_entropy': np.log2(3),
        'description': 'Hydrophobic/Polar/Charged grouping'
    },
    'biochemical_12': {
        'map': BIOCHEMICAL_12_ALPHABET, 
        'size': 14,  # 0-13 plus -1 for empty
        'max_entropy': np.log2(14),
        'description': 'Enhanced biochemical properties'
    },
    'full_20': {
        'map': FULL_20_ALPHABET,
        'size': 20,
        'max_entropy': np.log2(20),
        'description': 'One symbol per amino acid'
    }
}

def get_alphabet_config(alphabet_type='ternary'):
    """Get alphabet configuration by name."""
    if alphabet_type not in ALPHABET_CONFIGS:
        available = list(ALPHABET_CONFIGS.keys())
        raise ValueError(f"Unknown alphabet '{alphabet_type}'. Available: {available}")
    return ALPHABET_CONFIGS[alphabet_type]

def convert_residue_to_symbol(residue_name, alphabet_type='ternary'):
    """Convert amino acid residue name to symbol using specified alphabet."""
    config = get_alphabet_config(alphabet_type)
    return config['map'].get(residue_name.strip().upper(), -1)

def get_symbol_description(symbol, alphabet_type='ternary'):
    """Get human-readable description of symbol meaning."""
    descriptions = {
        'ternary': {
            0: 'Hydrophobic',
            1: 'Polar', 
            2: 'Charged',
            -1: 'Empty'
        },
        'biochemical_12': {
            0: 'Branched aliphatic (L,I)',
            1: 'Small branched (V)',
            2: 'Small nonpolar (A)', 
            3: 'Sulfur hydrophobic (M)',
            4: 'Aromatic (F,Y)',
            5: 'Large aromatic (W)',
            6: 'Hydroxyl polar (S,T)',
            7: 'Amide polar (N,Q)',
            8: 'Disulfide-forming (C)',
            9: 'Positive charge (K,R)',
            10: 'Negative charge (D,E)',
            11: 'Titratable (H)',
            12: 'High flexibility (G)',
            13: 'Rigid/helix-breaker (P)',
            -1: 'Empty'
        },
        'full_20': {i: f'Amino acid {i}' for i in range(20)}
    }
    descriptions['full_20'][-1] = 'Empty'
    
    return descriptions.get(alphabet_type, {}).get(symbol, f'Unknown symbol {symbol}')

def validate_alphabet_coverage():
    """Verify all 20 standard amino acids are covered in each alphabet."""
    standard_aa = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
                   'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
                   'THR', 'TRP', 'TYR', 'VAL']
    
    for name, config in ALPHABET_CONFIGS.items():
        missing = [aa for aa in standard_aa if aa not in config['map']]
        if missing:
            print(f"WARNING: {name} alphabet missing: {missing}")
        else:
            print(f"OK - {name} alphabet covers all 20 amino acids")

if __name__ == '__main__':
    print("Enhanced Alphabets for SCFD Protein Analysis")
    print("=" * 50)
    
    validate_alphabet_coverage()
    
    print("\nAlphabet Configurations:")
    for name, config in ALPHABET_CONFIGS.items():
        print(f"\n{name.upper()}:")
        print(f"  Size: {config['size']} symbols")
        print(f"  Max entropy: {config['max_entropy']:.2f}")
        print(f"  Description: {config['description']}")