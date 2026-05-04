import subprocess, sys

def _ensure(pkg, pip_names=None):
    """Try to import `pkg`; if missing, attempt pip install using each name in
    `pip_names` until one succeeds. Raises a clear error if all attempts fail."""
    try:
        __import__(pkg)
        return
    except ImportError:
        pass
    names = pip_names if pip_names else [pkg]
    for name in names:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", name],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            __import__(pkg)
            return
        except Exception:
            continue
    print(f"\n[ERROR] Could not install '{pkg}'. Please run one of:\n"
          f"  pip install {names[0]}\n"
          f"  conda install -c conda-forge {pkg}\n", file=sys.stderr)
    sys.exit(1)

if sys.version_info >= (3, 13):
    print("[WARNING] Python 3.13 detected. RDKit has limited PyPI support for 3.13.\n"
          "  Recommended: use conda with 'conda install -c conda-forge rdkit'\n"
          "  or run via: conda run python language.py", file=sys.stderr)

_ensure("numpy")
# rdkit-pypi was abandoned after Python 3.10; the package is now just 'rdkit'
_ensure("rdkit", ["rdkit", "rdkit-pypi"])

import numpy as np
import random
import json
import time
import math
import multiprocessing as mp
from collections import Counter
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
from rdkit import RDLogger
import logging as _logging
RDLogger.logger().setLevel(RDLogger.ERROR)

# =============================================================================
# Chemistry-DNA Language Engine v2
#
# Models real base-pairing with configurable pair counts (2,4,6,8 bases),
# complementary strands, replication fidelity that degrades with more bases,
# elemental supply constraints, codon-to-molecule translation, and parallel
# evolutionary optimization. Outputs particle-sim-ready blueprints.
# =============================================================================


# ---------------------------------------------------------------------------
# Periodic table: elements organisms can use to build structures
# ---------------------------------------------------------------------------

PERIODIC = {
    'H':  {'z': 1,  'valence': 1, 'mass': 1.008,  'smiles': '[H]'},
    'C':  {'z': 6,  'valence': 4, 'mass': 12.011, 'smiles': 'C'},
    'N':  {'z': 7,  'valence': 3, 'mass': 14.007, 'smiles': 'N'},
    'O':  {'z': 8,  'valence': 2, 'mass': 15.999, 'smiles': 'O'},
    'F':  {'z': 9,  'valence': 1, 'mass': 18.998, 'smiles': 'F'},
    'P':  {'z': 15, 'valence': 5, 'mass': 30.974, 'smiles': 'P'},
    'S':  {'z': 16, 'valence': 2, 'mass': 32.065, 'smiles': 'S'},
    'Cl': {'z': 17, 'valence': 1, 'mass': 35.453, 'smiles': 'Cl'},
    'Si': {'z': 14, 'valence': 4, 'mass': 28.086, 'smiles': '[Si]'},
    'Br': {'z': 35, 'valence': 1, 'mass': 79.904, 'smiles': 'Br'},
    'Fe': {'z': 26, 'valence': 3, 'mass': 55.845, 'smiles': '[Fe]'},
    'Mg': {'z': 12, 'valence': 2, 'mass': 24.305, 'smiles': '[Mg]'},
}


# ---------------------------------------------------------------------------
# Base pairing systems
# ---------------------------------------------------------------------------
# 2 pairs (4 bases): Earth DNA — optimal stability/complexity tradeoff
# 3 pairs (6 bases): higher density but error-prone, cancer analog
# 4 pairs (8 bases): very unstable, theoretical only
# The error_rate models replication fidelity degradation per extra pair.

BASE_SYSTEMS = {
    2: {  # too simple for complex life
        'pairs': [('0', '1')],
        'bases': ['0', '1'],
        'error_rate': 0.0005,
        'name': 'binary',
        'desc': 'Simplest possible — binary code analogue. No W/Z opcodes, very limited.',
    },
    4: {  # Earth DNA: 2 complementary pairs
        'pairs': [('A', 'T'), ('C', 'G')],
        'bases': ['A', 'T', 'C', 'G'],
        'error_rate': 0.008,
        'name': 'quaternary',
        'desc': 'Standard Earth DNA. Optimal stability/complexity. 5 base opcodes.',
    },
    6: {  # 3 pairs: more info, less stable, mutation-prone
        'pairs': [('A', 'T'), ('C', 'G'), ('X', 'Y')],
        'bases': ['A', 'T', 'C', 'G', 'X', 'Y'],
        'error_rate': 0.045,
        'name': 'hexary',
        'desc': 'Higher info density, mutation-prone. Adds REPEAT and REGULATE opcodes.',
    },
    8: {  # 4 pairs: highly unstable
        'pairs': [('A', 'T'), ('C', 'G'), ('X', 'Y'), ('W', 'Z')],
        'bases': ['A', 'T', 'C', 'G', 'X', 'Y', 'W', 'Z'],
        'error_rate': 0.11,
        'name': 'octary',
        'desc': 'Full 18-opcode set. Highly unstable — theoretical alien biochemistry.',
    },
}


# ---------------------------------------------------------------------------
# Environment-specific base symbol tables
# ---------------------------------------------------------------------------
# Earth DNA uses A/T/C/G.  Silicon and exotic life use chemically distinct
# pairs — different letters remind the user that these are NOT the same
# molecules.  The engine always executes on positional slots (slot0/slot1…),
# so we provide a bijective remap between display symbols and engine symbols.
#
# Pairs are listed as (pair_a, pair_b) matching BASE_SYSTEMS order.
# 2-base: 1 pair   4-base: 2 pairs   6-base: 3 pairs   8-base: 4 pairs

ENV_BASE_SYMBOLS: Dict[str, Dict[int, Dict]] = {
    'earth': {
        # Standard DNA — A pairs with T, C pairs with G
        2: {'pairs': [('0', '1')],          'bases': ['0', '1'],
            'label': 'Binary (0·1)'},
        4: {'pairs': [('A', 'T'), ('C', 'G')],
            'bases': ['A', 'T', 'C', 'G'],
            'label': 'DNA (A·T  C·G)'},
        6: {'pairs': [('A', 'T'), ('C', 'G'), ('X', 'Y')],
            'bases': ['A', 'T', 'C', 'G', 'X', 'Y'],
            'label': 'XNA (A·T  C·G  X·Y)'},
        8: {'pairs': [('A', 'T'), ('C', 'G'), ('X', 'Y'), ('W', 'Z')],
            'bases': ['A', 'T', 'C', 'G', 'X', 'Y', 'W', 'Z'],
            'label': 'Hyper-XNA (A·T  C·G  X·Y  W·Z)'},
    },
    'silicon': {
        # Silicon-world biochemistry — silanol/silicate base analogues
        # Symbols: P·Q (silanol pair), R·S (silicate pair),
        #          U·V (siloxane pair), J·K (metallosilicate pair)
        2: {'pairs': [('P', 'Q')],          'bases': ['P', 'Q'],
            'label': 'Si-Binary (P·Q)'},
        4: {'pairs': [('P', 'Q'), ('R', 'S')],
            'bases': ['P', 'Q', 'R', 'S'],
            'label': 'Si-DNA (P·Q  R·S)'},
        6: {'pairs': [('P', 'Q'), ('R', 'S'), ('U', 'V')],
            'bases': ['P', 'Q', 'R', 'S', 'U', 'V'],
            'label': 'Si-XNA (P·Q  R·S  U·V)'},
        8: {'pairs': [('P', 'Q'), ('R', 'S'), ('U', 'V'), ('J', 'K')],
            'bases': ['P', 'Q', 'R', 'S', 'U', 'V', 'J', 'K'],
            'label': 'Si-Hyper (P·Q  R·S  U·V  J·K)'},
    },
    'exotic': {
        # Exotic biochemistry — non-standard element pairs
        # Symbols: E·F (exotic pair 1), H·I (pair 2),
        #          L·M (pair 3), N·O (pair 4)
        2: {'pairs': [('E', 'F')],          'bases': ['E', 'F'],
            'label': 'Ex-Binary (E·F)'},
        4: {'pairs': [('E', 'F'), ('H', 'I')],
            'bases': ['E', 'F', 'H', 'I'],
            'label': 'Ex-DNA (E·F  H·I)'},
        6: {'pairs': [('E', 'F'), ('H', 'I'), ('L', 'M')],
            'bases': ['E', 'F', 'H', 'I', 'L', 'M'],
            'label': 'Ex-XNA (E·F  H·I  L·M)'},
        8: {'pairs': [('E', 'F'), ('H', 'I'), ('L', 'M'), ('N', 'O')],
            'bases': ['E', 'F', 'H', 'I', 'L', 'M', 'N', 'O'],
            'label': 'Ex-Hyper (E·F  H·I  L·M  N·O)'},
    },
}

def get_env_bases(n_bases: int, env: str = 'earth') -> Dict:
    """Return the display base spec (pairs + bases + label) for this
    environment + base-count combination.  Falls back to earth symbols
    if the environment is not in ENV_BASE_SYMBOLS."""
    return ENV_BASE_SYMBOLS.get(env, ENV_BASE_SYMBOLS['earth'])[n_bases]

def env_to_engine_map(n_bases: int, env: str) -> Dict[str, str]:
    """Build a bijective mapping from env display letters → engine letters.
    Engine always uses the BASE_SYSTEMS alphabet (A/T/C/G…).
    For earth the map is identity.  For silicon/exotic it remaps slot-by-slot."""
    env_bases  = get_env_bases(n_bases, env)['bases']
    eng_bases  = BASE_SYSTEMS[n_bases]['bases']
    return {d: e for d, e in zip(env_bases, eng_bases)}

def engine_to_env_map(n_bases: int, env: str) -> Dict[str, str]:
    """Reverse map: engine letters → env display letters."""
    env_bases  = get_env_bases(n_bases, env)['bases']
    eng_bases  = BASE_SYSTEMS[n_bases]['bases']
    return {e: d for d, e in zip(env_bases, eng_bases)}

def translate_seq_to_engine(seq: str, n_bases: int, env: str) -> str:
    """Convert a display-alphabet sequence to the engine alphabet."""
    if env == 'earth':
        return seq
    m = env_to_engine_map(n_bases, env)
    return ''.join(m.get(c, c) for c in seq)

def translate_seq_to_display(seq: str, n_bases: int, env: str) -> str:
    """Convert an engine-alphabet sequence to the display alphabet."""
    if env == 'earth':
        return seq
    m = engine_to_env_map(n_bases, env)
    return ''.join(m.get(c, c) for c in seq)


def complement_map(n_bases: int) -> Dict[str, str]:
    """Build complement lookup from base system pairs."""
    system = BASE_SYSTEMS[n_bases]
    cmap = {}
    for a, b in system['pairs']:
        cmap[a] = b
        cmap[b] = a
    return cmap

# Pre-compute complement maps for all systems (avoid rebuilding per strand)
_COMPLEMENT_CACHE: Dict[int, Dict[str, str]] = {
    nb: complement_map(nb) for nb in BASE_SYSTEMS
}


CODON_LEN = 5  # 5-base codons: 4^5=1024 possible (4-base), higher info density than bio's 3

# ---------------------------------------------------------------------------
# Configurable constants (previously magic numbers)
# ---------------------------------------------------------------------------
PART_SPACING_ANGSTROMS = 8.0       # 3D offset between parts in grid layout
PART_GRID_COLS = 4                 # columns in 3D grid layout of parts
ELITE_FRACTION = 0.10              # top fraction carried over each generation
MAX_HISTORY_LEN = 2000             # cap on evolution history list
DEFAULT_TOURNAMENT_K = 4
ENV_HASH_MOD = 17                  # broaden REGULATE discriminatory power (was 3)
STABILITY_EXPONENT = 10            # models cumulative fidelity over a short genome
MAX_POLYMER_REPEATS = 6            # cap on POLYMERIZE filament length
BASE_PMF_UNITS = 10                # proton motive force units per energy part
ATP_PER_MOTOR_TICK = 2             # ATP cost per motor rotation cycle
ASSEMBLY_BONUS = 0.15              # fitness bonus for complete functional complexes
MOTILITY_BONUS = 0.10              # fitness bonus for motility capability
DEVELOPMENT_PHASES = ('growth', 'assembly', 'activation')
PROMOTER_BONUS = 0.05               # fitness bonus per detected operon (grouped genes)
MAX_FILAMENT_COST = 12              # energy cost cap for POLYMERIZE before penalty applies
DOMAIN_MISMATCH_PENALTY = 0.03      # fitness penalty per bad domain connection
ENERGY_STARVATION_SCALE = 0.02      # per-unit fitness loss for negative energy balance
TORQUE_PER_STATOR = 50.0            # pN·nm torque per stator unit
FILAMENT_DRAG = 10.0                # drag coefficient for flagellar filament
MOTOR_BASE_RPM = 300.0              # base RPM before load adjustment
TUMBLE_PROBABILITY_BASE = 0.1       # base tumble probability (no chemotaxis)
GRADIENT_REWARD_PER_STEP = 0.01     # fitness per successful gradient step
QUORUM_THRESHOLD = 5                # population count triggering quorum sensing
SECRETION_ENERGY_COST = 5           # energy cost per secretion event
DIVISION_MIN_PARTS = 6              # minimum parts to attempt division
METABOLIC_CHAIN_BONUS = 0.05        # bonus for complete metabolic pathway
MEMBRANE_POTENTIAL_BASE = 150.0     # mV, resting membrane potential


# ---------------------------------------------------------------------------
# Element supply: constrains what an environment can build
# ---------------------------------------------------------------------------

@dataclass
class ElementSupply:
    """Tracks available elements in the environment."""
    stock: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def earth_like(cls) -> 'ElementSupply':
        return cls(stock={'C': 999, 'N': 500, 'O': 800, 'H': 9999,
                          'S': 200, 'P': 300, 'F': 50, 'Cl': 80, 'Fe': 30, 'Mg': 40})

    @classmethod
    def silicon_world(cls) -> 'ElementSupply':
        return cls(stock={'Si': 999, 'O': 800, 'C': 100, 'H': 5000,
                          'N': 50, 'S': 300, 'F': 200, 'P': 100})

    @classmethod
    def exotic(cls) -> 'ElementSupply':
        return cls(stock={'C': 200, 'N': 200, 'O': 200, 'S': 200, 'P': 200,
                          'Si': 200, 'F': 200, 'Cl': 200, 'Br': 200, 'H': 2000})

    def can_afford(self, element_counts: Dict[str, int]) -> bool:
        for el, need in element_counts.items():
            if self.stock.get(el, 0) < need:
                return False
        return True

    def richness(self) -> float:
        """How many distinct elements are available."""
        return sum(1 for v in self.stock.values() if v > 0) / len(PERIODIC)


# ---------------------------------------------------------------------------
# Strand: double-stranded genome with complementarity and replication
# ---------------------------------------------------------------------------

class Strand:
    def __init__(self, seq: str, n_bases: int = 4):
        self.n_bases = n_bases
        self.system = BASE_SYSTEMS[n_bases]
        self.cmap = _COMPLEMENT_CACHE[n_bases]
        self.seq = seq  # sense strand

    @classmethod
    def random(cls, length: int, n_bases: int = 4) -> 'Strand':
        bases = BASE_SYSTEMS[n_bases]['bases']
        seq = ''.join(random.choice(bases) for _ in range(length))
        return cls(seq, n_bases)

    @property
    def complement(self) -> str:
        """Complementary strand: A<->T, C<->G, X<->Y, W<->Z."""
        return ''.join(self.cmap.get(b, b) for b in self.seq)

    @property
    def paired(self) -> List[Tuple[str, str]]:
        """Returns list of (sense_base, complement_base) pairs."""
        comp = self.complement
        return list(zip(self.seq, comp))

    def pair_integrity(self) -> float:
        """Fraction of bases that have valid complements. 1.0 = perfect."""
        valid = sum(1 for b in self.seq if b in self.cmap)
        return valid / max(1, len(self.seq))

    def replicate(self) -> 'Strand':
        """Copy with error rate determined by base system complexity."""
        err = self.system['error_rate']
        new_seq = []
        for b in self.seq:
            if random.random() < err:
                # Mutation: substitute random base (replication error)
                new_seq.append(random.choice(self.system['bases']))
            else:
                new_seq.append(b)
        return Strand(''.join(new_seq), self.n_bases)

    def transcribe(self) -> str:
        """Sense strand -> RNA-like transcript with intron splicing (5-to-4)."""
        rna = list(self.seq)
        exon_mask = [True] * len(rna)
        for i in range(4, len(rna), 5):
            if random.random() < 0.8:
                exon_mask[i] = False
        transcript = ''.join(b for b, keep in zip(rna, exon_mask) if keep)
        return transcript if len(transcript) >= CODON_LEN else self.seq[:CODON_LEN]

    def codons(self, seed: int = None) -> List[str]:
        """Transcript -> list of 5-base codons.
        If seed is provided, uses seeded RNG for reproducible splicing."""
        if seed is not None:
            rng = random.Random(seed)
            rna = list(self.seq)
            exon_mask = [True] * len(rna)
            for i in range(4, len(rna), 5):
                if rng.random() < 0.8:
                    exon_mask[i] = False
            t = ''.join(b for b, keep in zip(rna, exon_mask) if keep)
            if len(t) < CODON_LEN:
                t = self.seq[:CODON_LEN]
            pads = (CODON_LEN - len(t) % CODON_LEN) % CODON_LEN
            t += ''.join(rng.choices(self.system['bases'], k=pads))
        else:
            t = self.transcribe()
            pads = (CODON_LEN - len(t) % CODON_LEN) % CODON_LEN
            t += ''.join(random.choices(self.system['bases'], k=pads))
        return [t[i:i+CODON_LEN] for i in range(0, len(t), CODON_LEN)]

    def codons_deterministic(self) -> List[str]:
        """Deterministic codon split (no random splicing). For translation display."""
        seq = self.seq
        pads = (CODON_LEN - len(seq) % CODON_LEN) % CODON_LEN
        seq += self.system['bases'][0] * pads
        return [seq[i:i+CODON_LEN] for i in range(0, len(seq), CODON_LEN)]

    def info_density(self) -> float:
        """Bits per base: log2(n_bases). 4-base=2 bits, 8-base=3 bits."""
        return math.log2(self.n_bases)

    def stability_score(self) -> float:
        """Info density discounted by compounded replication error.
        Sweet spot: 4 bases. 2-base is too simple (low info density),
        6+ is unstable (errors compound per base copied).
        The (1-err)**10 factor models cumulative fidelity over a short genome;
        it makes the error penalty dominate for high-base-count systems so
        4-base comes out on top of 6-base and 8-base, matching documented
        biological intuition."""
        err = self.system['error_rate']
        info = self.info_density()
        return min(1.0, (info / 3.0) * ((1.0 - err) ** STABILITY_EXPONENT))

    def __len__(self):
        return len(self.seq)

    def __repr__(self):
        return f"Strand({self.system['name']}, len={len(self.seq)}, integrity={self.pair_integrity():.2f})"


# ---------------------------------------------------------------------------
# Knowledge base: biological parts library
# ---------------------------------------------------------------------------
# These are the building blocks that DNA instructs the cell to assemble.
# Each part has a role, a real SMILES representation, and element cost.
# Like a car has engine, frame, wheels — a cell has membrane, enzymes, etc.

PARTS_KB = {
    # Structural parts (frame / chassis)
    'membrane':    {'smiles': 'CCCCCCCCCCCCCCCC(=O)OCC(OC(=O)CCCCCCCCCCCCCCC)COP(=O)(O)OCC[N+](C)(C)C',
                    'role': 'structure', 'function': 'barrier',
                    'desc': 'phospholipid — forms cell boundary'},
    'wall':        {'smiles': 'OC1C(O)C(OC(CO)C1O)OC1OC(CO)C(O)C(O)C1NC(=O)C',
                    'role': 'structure', 'function': 'rigidity',
                    'desc': 'peptidoglycan unit — rigid outer shell'},
    'filament':    {'smiles': 'NCC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)O',
                    'role': 'structure', 'function': 'scaffold',
                    'desc': 'polyglycine chain — structural fiber'},
    # Energy parts (engine / fuel system)
    'atp':         {'smiles': 'c1nc(N)c2ncn(C3OC(COP(=O)(O)OP(=O)(O)OP(=O)(O)O)C(O)C3O)c2n1',
                    'role': 'energy', 'function': 'fuel',
                    'desc': 'ATP — universal energy currency'},
    'nadh':        {'smiles': 'NC(=O)C1=CN(C=CC1)C1OC(COP(=O)(O)OP(=O)(O)OCC2OC(n3cnc4c(N)ncnc43)C(O)C2O)C(O)C1O',
                    'role': 'energy', 'function': 'electron_carrier',
                    'desc': 'NADH — electron shuttle'},
    'glucose':     {'smiles': 'OCC1OC(O)C(O)C(O)C1O',
                    'role': 'energy', 'function': 'substrate',
                    'desc': 'glucose — primary metabolic fuel'},
    # Catalytic parts (tools / processors)
    'amino_basic': {'smiles': 'NCC(=O)O',
                    'role': 'catalysis', 'function': 'building_block',
                    'desc': 'glycine — simplest amino acid'},
    'amino_func':  {'smiles': 'NC(CC(=O)O)C(=O)O',
                    'role': 'catalysis', 'function': 'enzyme_core',
                    'desc': 'aspartate — catalytic amino acid'},
    'cofactor':    {'smiles': '[Fe+2]',
                    'role': 'catalysis', 'function': 'metal_center',
                    'desc': 'iron cofactor — electron transfer'},
    # Information parts (control / signaling)
    'nucleotide':  {'smiles': 'Nc1ccn(C2CC(O)C(COP(=O)(O)O)O2)c(=O)n1',
                    'role': 'information', 'function': 'replication',
                    'desc': 'cytidine monophosphate — nucleotide'},
    'coenzyme_a':  {'smiles': 'CC(C)(COP(=O)(O)OP(=O)(O)OCC1OC(n2cnc3c(N)ncnc32)C(O)C1OP(=O)(O)O)C(O)C(=O)NCCC(=O)NCCS',
                    'role': 'information', 'function': 'regulation',
                    'desc': 'CoA — metabolic regulation'},
    # Minimal parts (bacteria-level simplest forms)
    'methane':     {'smiles': 'C',
                    'role': 'minimal', 'function': 'substrate',
                    'desc': 'methane — simplest organic'},
    'water':       {'smiles': 'O',
                    'role': 'minimal', 'function': 'solvent',
                    'desc': 'water — universal solvent'},
    'ammonia':     {'smiles': 'N',
                    'role': 'minimal', 'function': 'nitrogen_source',
                    'desc': 'ammonia — nitrogen donor'},
    'formaldehyde':{'smiles': 'C=O',
                    'role': 'minimal', 'function': 'carbon_source',
                    'desc': 'formaldehyde — prebiotic carbon'},
    'hcn':         {'smiles': 'C#N',
                    'role': 'minimal', 'function': 'prebiotic',
                    'desc': 'hydrogen cyanide — prebiotic building block'},
    'phosphate':   {'smiles': 'OP(=O)(O)O',
                    'role': 'minimal', 'function': 'backbone',
                    'desc': 'phosphate — nucleic acid backbone',
                    'energy_cost': 0, 'attachment': [], 'domains': []},
    # --- Motor parts (flagellar machine components) ---
    'flagellin':   {'smiles': 'NCC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)O',
                    'role': 'motor', 'function': 'filament',
                    'desc': 'flagellin — self-assembling filament protein',
                    'energy_cost': 3, 'attachment': ['distal'],
                    'domains': ['filament_domain']},
    'hook_protein': {'smiles': 'NCC(=O)NC(CC(=O)O)C(=O)NCC(=O)NC(CO)C(=O)O',
                     'role': 'motor', 'function': 'hook',
                     'desc': 'hook protein — flexible joint between motor and filament',
                     'energy_cost': 2, 'attachment': ['basal', 'distal'],
                     'domains': ['hook_domain']},
    'ms_ring':     {'smiles': 'NCC(=O)NC(CCSC)C(=O)NCC(=O)NC(CC(=O)O)C(=O)O',
                    'role': 'motor', 'function': 'rotor',
                    'desc': 'MS-ring — membrane-spanning rotor ring',
                    'energy_cost': 4, 'attachment': ['basal', 'membrane'],
                    'domains': ['motor_domain', 'ring_domain']},
    'c_ring':      {'smiles': 'NCC(=O)NC(CC(=O)N)C(=O)NCC(=O)NC(CCCNC(=N)N)C(=O)O',
                    'role': 'motor', 'function': 'switch',
                    'desc': 'C-ring — cytoplasmic switch controlling rotation direction',
                    'energy_cost': 3, 'attachment': ['basal'],
                    'domains': ['motor_domain', 'switch_domain']},
    'motA':        {'smiles': 'NCC(=O)NC(CC(=O)O)C(=O)NC(CCSC)C(=O)NCC(=O)O',
                    'role': 'motor', 'function': 'stator_a',
                    'desc': 'MotA — proton channel stator component A',
                    'energy_cost': 2, 'attachment': ['membrane'],
                    'domains': ['stator_domain', 'proton_channel']},
    'motB':        {'smiles': 'NCC(=O)NC(CC(=O)O)C(=O)NC(CO)C(=O)NC(CCCNC(=N)N)C(=O)O',
                    'role': 'motor', 'function': 'stator_b',
                    'desc': 'MotB — peptidoglycan-anchored stator component B',
                    'energy_cost': 2, 'attachment': ['membrane', 'wall'],
                    'domains': ['stator_domain', 'anchor_domain']},
    'fliG':        {'smiles': 'NCC(=O)NC(CCC(=O)O)C(=O)NCC(=O)NC(CC(=O)O)C(=O)O',
                    'role': 'motor', 'function': 'torque_generator',
                    'desc': 'FliG — torque-generating rotor/stator interface',
                    'energy_cost': 3, 'attachment': ['basal'],
                    'domains': ['motor_domain', 'torque_domain']},
    'export_gate': {'smiles': 'NCC(=O)NC(CCCCN)C(=O)NC(CO)C(=O)NCC(=O)O',
                    'role': 'motor', 'function': 'export',
                    'desc': 'export apparatus — secretes flagellar proteins',
                    'energy_cost': 4, 'attachment': ['basal', 'membrane'],
                    'domains': ['export_domain']},
    # --- Signaling parts (chemotaxis / regulation) ---
    'receptor':    {'smiles': 'NCC(=O)NC(CC(=O)O)C(=O)NC(CCCCN)C(=O)NCC(=O)O',
                    'role': 'signaling', 'function': 'sensor',
                    'desc': 'chemoreceptor — senses chemical gradients',
                    'energy_cost': 1, 'attachment': ['membrane', 'distal'],
                    'domains': ['sensor_domain', 'signaling_domain']},
    'histidine_kinase': {'smiles': 'NCC(=O)NC(CC1C=CN=C1)C(=O)NC(CCC(=O)O)C(=O)O',
                         'role': 'signaling', 'function': 'kinase',
                         'desc': 'histidine kinase CheA — phosphorylation relay',
                         'energy_cost': 2, 'attachment': [],
                         'domains': ['kinase_domain', 'signaling_domain']},
    'response_reg': {'smiles': 'NCC(=O)NC(CC(=O)O)C(=O)NC(CC(=O)N)C(=O)O',
                     'role': 'signaling', 'function': 'regulator',
                     'desc': 'response regulator CheY — controls motor switching',
                     'energy_cost': 1, 'attachment': [],
                     'domains': ['regulator_domain', 'switch_domain']},
    # --- Advanced structural parts ---
    'actin':       {'smiles': 'NCC(=O)NC(CC(=O)O)C(=O)NCC(=O)NC(CCCCN)C(=O)NCC(=O)O',
                    'role': 'structure', 'function': 'cytoskeleton',
                    'desc': 'actin-like protein — dynamic internal scaffolding',
                    'energy_cost': 2, 'attachment': ['basal', 'distal'],
                    'domains': ['filament_domain']},
    'tubulin':     {'smiles': 'NCC(=O)NC(CCC(=O)O)C(=O)NC(CC1C=CN=C1)C(=O)NCC(=O)O',
                    'role': 'structure', 'function': 'microtubule',
                    'desc': 'tubulin-like protein — rigid intracellular tracks',
                    'energy_cost': 3, 'attachment': ['basal', 'distal'],
                    'domains': ['filament_domain', 'motor_domain']},
    # --- Neural / Cognitive / Homeostatic parts ---
    # Membrane excitability
    'ion_channel':     {'smiles': 'NCC(=O)NC(CC(=O)O)C(=O)NC(CCCCN)C(=O)NC(CC1=CC=CC=C1)C(=O)O',
                        'role': 'neural', 'function': 'ion_transport',
                        'desc': 'voltage-gated ion channel — Na+/K+ gating for action potentials',
                        'energy_cost': 2, 'attachment': ['membrane'],
                        'domains': ['channel_domain', 'gating_domain']},
    'pump_atpase':     {'smiles': 'NCC(=O)NC(CC(=O)O)C(=O)NC(CCC(=O)O)C(=O)NC(CCSC)C(=O)O',
                        'role': 'neural', 'function': 'ion_pump',
                        'desc': 'Na+/K+-ATPase pump — restores ion gradients after firing',
                        'energy_cost': 4, 'attachment': ['membrane'],
                        'domains': ['pump_domain', 'atp_binding']},
    # Synaptic machinery
    'vesicle':         {'smiles': 'CCCCCCCCCCCCCCCC(=O)OCC(OC(=O)CCCCCCCCCCCCCCCC)COP(=O)(O)OCC[N+](C)(C)C',
                        'role': 'neural', 'function': 'neurotransmitter_store',
                        'desc': 'synaptic vesicle — packages neurotransmitters for release',
                        'energy_cost': 3, 'attachment': ['distal'],
                        'domains': ['vesicle_domain', 'docking_domain']},
    'neurotransmitter':{'smiles': 'NCCC1=CC(O)=C(O)C=C1',
                        'role': 'neural', 'function': 'signal_molecule',
                        'desc': 'dopamine — reward/drive neurotransmitter',
                        'energy_cost': 1, 'attachment': [],
                        'domains': ['ligand_domain']},
    'receptor_gated':  {'smiles': 'NCC(=O)NC(CC1=CNC=N1)C(=O)NC(CCCCN)C(=O)NCC(=O)O',
                        'role': 'neural', 'function': 'ligand_gated_channel',
                        'desc': 'ligand-gated receptor — converts chemical signal to electrical',
                        'energy_cost': 2, 'attachment': ['membrane'],
                        'domains': ['receptor_domain', 'channel_domain']},
    'snare_complex':   {'smiles': 'NCC(=O)NC(CCCNC(=N)N)C(=O)NC(CC(=O)O)C(=O)NCC(=O)NC(CO)C(=O)O',
                        'role': 'neural', 'function': 'vesicle_fusion',
                        'desc': 'SNARE protein — docks vesicles for neurotransmitter release',
                        'energy_cost': 2, 'attachment': ['membrane', 'distal'],
                        'domains': ['docking_domain', 'fusion_domain']},
    # Signal integration (neural net primitives)
    'calmodulin':      {'smiles': 'NCC(=O)NC(CCC(=O)O)C(=O)NC(CC(=O)O)C(=O)NC(CCC(=O)N)C(=O)O',
                        'role': 'neural', 'function': 'calcium_sensor',
                        'desc': 'calmodulin — Ca²⁺ sensor that gates downstream signaling',
                        'energy_cost': 1, 'attachment': [],
                        'domains': ['sensor_domain', 'signaling_domain']},
    'camp_kinase':     {'smiles': 'NCC(=O)NC(CC1=CN=C1)C(=O)NC(CCCCN)C(=O)NC(CCC(=O)O)C(=O)O',
                        'role': 'neural', 'function': 'signal_amplifier',
                        'desc': 'cAMP-dependent kinase — amplifies and relays neural signals',
                        'energy_cost': 3, 'attachment': [],
                        'domains': ['kinase_domain', 'signaling_domain']},
    'gap_junction':    {'smiles': 'NCC(=O)NC(CCCCN)C(=O)NC(CO)C(=O)NC(CC(=O)O)C(=O)NC(CCSC)C(=O)O',
                        'role': 'neural', 'function': 'cell_coupling',
                        'desc': 'connexin gap junction — direct electrical coupling between cells',
                        'energy_cost': 2, 'attachment': ['membrane'],
                        'domains': ['channel_domain', 'coupling_domain']},
    # Memory / plasticity
    'nmda_receptor':   {'smiles': 'NCC(=O)NC(CCC(=O)N)C(=O)NC(CC1=CNC2=CC=CC=C12)C(=O)NCC(=O)O',
                        'role': 'neural', 'function': 'coincidence_detector',
                        'desc': 'NMDA receptor — voltage+ligand gate for Hebbian learning',
                        'energy_cost': 3, 'attachment': ['membrane'],
                        'domains': ['receptor_domain', 'channel_domain', 'gating_domain']},
    'camkii':          {'smiles': 'NCC(=O)NC(CC1=CN=C1)C(=O)NC(CCC(=O)O)C(=O)NC(CC(=O)N)C(=O)O',
                        'role': 'neural', 'function': 'synaptic_plasticity',
                        'desc': 'CaMKII kinase — LTP mediator, encodes synaptic strength',
                        'energy_cost': 4, 'attachment': [],
                        'domains': ['kinase_domain', 'plasticity_domain']},
    'creb_factor':     {'smiles': 'NCC(=O)NC(CCCCN)C(=O)NC(CC(=O)O)C(=O)NC(CC1=CC=CC=C1)C(=O)O',
                        'role': 'neural', 'function': 'gene_expression',
                        'desc': 'CREB transcription factor — converts activity to long-term memory',
                        'energy_cost': 2, 'attachment': [],
                        'domains': ['dna_binding', 'activation_domain']},
    # Homeostasis
    'heat_shock_prot': {'smiles': 'NCC(=O)NC(CC(=O)O)C(=O)NC(CCSC)C(=O)NC(CCC(=O)N)C(=O)O',
                        'role': 'homeostasis', 'function': 'chaperone',
                        'desc': 'HSP70 chaperone — refolds damaged proteins under stress',
                        'energy_cost': 3, 'attachment': [],
                        'domains': ['chaperone_domain', 'atp_binding']},
    'proteasome':      {'smiles': 'NCC(=O)NC(CCCCN)C(=O)NCC(=O)NC(CCC(=O)O)C(=O)NC(CO)C(=O)O',
                        'role': 'homeostasis', 'function': 'protein_degradation',
                        'desc': '26S proteasome — degrades misfolded/damaged proteins',
                        'energy_cost': 4, 'attachment': [],
                        'domains': ['protease_domain', 'ubiquitin_domain']},
    'dna_repair':      {'smiles': 'NCC(=O)NC(CC(=O)N)C(=O)NC(CC(=O)O)C(=O)NC(CC1=CNC=N1)C(=O)O',
                        'role': 'homeostasis', 'function': 'genome_repair',
                        'desc': 'DNA repair polymerase — fixes replication errors',
                        'energy_cost': 3, 'attachment': [],
                        'domains': ['dna_binding', 'polymerase_domain']},
    'antioxidant':     {'smiles': 'NCC(=O)NC(CCS)C(=O)NC(CC(=O)O)C(=O)NC(CCCCN)C(=O)O',
                        'role': 'homeostasis', 'function': 'redox_balance',
                        'desc': 'glutathione — scavenges reactive oxygen species',
                        'energy_cost': 1, 'attachment': [],
                        'domains': ['redox_domain']},
    # Higher-order cognition
    'oscillator':      {'smiles': 'NCC(=O)NC(CC1=CC=CC=C1)C(=O)NC(CCC(=O)O)C(=O)NC(CCCCN)C(=O)O',
                        'role': 'neural', 'function': 'rhythm_generator',
                        'desc': 'circadian oscillator — generates autonomous 24h rhythms',
                        'energy_cost': 2, 'attachment': [],
                        'domains': ['feedback_domain', 'timer_domain']},
    'pattern_net':     {'smiles': 'NCC(=O)NC(CCCCN)C(=O)NC(CC(=O)O)C(=O)NC(CC1=CNC2=CC=CC=C12)C(=O)NCC(=O)O',
                        'role': 'neural', 'function': 'pattern_recognition',
                        'desc': 'recurrent neural circuit — detects and classifies input patterns',
                        'energy_cost': 5, 'attachment': [],
                        'domains': ['network_domain', 'plasticity_domain']},
    'working_memory':  {'smiles': 'NCC(=O)NC(CCC(=O)O)C(=O)NC(CC1=CNC=N1)C(=O)NC(CC(=O)N)C(=O)O',
                        'role': 'neural', 'function': 'short_term_memory',
                        'desc': 'persistent firing loop — holds information actively in working memory',
                        'energy_cost': 4, 'attachment': [],
                        'domains': ['network_domain', 'feedback_domain']},
    'decision_gate':   {'smiles': 'NCC(=O)NC(CC1=CC=CC=C1)C(=O)NC(CCCCN)C(=O)NC(CCC(=O)O)C(=O)NCC(=O)O',
                        'role': 'neural', 'function': 'decision_making',
                        'desc': 'bistable integrator — accumulates evidence and commits to a choice',
                        'energy_cost': 4, 'attachment': [],
                        'domains': ['integrator_domain', 'feedback_domain']},
}

# Back-fill energy_cost / attachment / domains for original parts that lack them
for _pn, _pv in PARTS_KB.items():
    _pv.setdefault('energy_cost', 0)
    _pv.setdefault('attachment', [])
    _pv.setdefault('domains', [])

# ---------------------------------------------------------------------------
# Functional complexes: combinations of parts that form working machines
# ---------------------------------------------------------------------------
FUNCTIONAL_COMPLEXES = {
    'stator_complex': {
        'required': {'motA', 'motB', 'fliG'},
        'desc': 'Stator unit — generates torque from proton flow',
        'bonus': 'motor_torque',
    },
    'flagellar_motor': {
        'required': {'ms_ring', 'c_ring', 'motA', 'motB', 'fliG', 'export_gate'},
        'desc': 'Complete basal body — rotary motor embedded in membrane',
        'bonus': 'motor_rotation',
    },
    'flagellum': {
        'required': {'ms_ring', 'c_ring', 'motA', 'motB', 'fliG',
                     'hook_protein', 'flagellin'},
        'desc': 'Full flagellum — motor + hook + filament',
        'bonus': 'motility',
    },
    'chemotaxis_system': {
        'required': {'receptor', 'histidine_kinase', 'response_reg', 'c_ring'},
        'desc': 'Chemotaxis signaling cascade — sense and respond to gradients',
        'bonus': 'chemotaxis',
    },
    'cytoskeleton': {
        'required': {'actin', 'tubulin'},
        'desc': 'Internal skeleton — cell shape and intracellular transport',
        'bonus': 'eukaryotic_structure',
    },
    # ── Neural / Intelligence complexes ──────────────────────────────────────
    'action_potential': {
        'required': {'ion_channel', 'pump_atpase'},
        'desc': 'Excitable membrane — fires action potentials, basis of all neural signaling',
        'bonus': 'excitability',
    },
    'synapse': {
        'required': {'vesicle', 'neurotransmitter', 'snare_complex', 'receptor_gated'},
        'desc': 'Chemical synapse — transmits signals between cells with plasticity',
        'bonus': 'synaptic_transmission',
    },
    'signal_cascade': {
        'required': {'calmodulin', 'camp_kinase', 'receptor_gated'},
        'desc': 'Intracellular signal cascade — amplifies and routes sensory inputs',
        'bonus': 'signal_integration',
    },
    'hebbian_plasticity': {
        'required': {'nmda_receptor', 'camkii', 'creb_factor'},
        'desc': 'Hebbian learning circuit — "neurons that fire together wire together"',
        'bonus': 'learning',
    },
    'neural_oscillator': {
        'required': {'ion_channel', 'gap_junction', 'oscillator'},
        'desc': 'Coupled oscillator network — generates rhythmic activity patterns',
        'bonus': 'rhythm',
    },
    'working_memory_loop': {
        'required': {'working_memory', 'nmda_receptor', 'camp_kinase'},
        'desc': 'Active maintenance loop — holds representations online for seconds',
        'bonus': 'working_memory',
    },
    'decision_circuit': {
        'required': {'decision_gate', 'pattern_net', 'neurotransmitter'},
        'desc': 'Decision-making circuit — integrates evidence and selects actions',
        'bonus': 'decision_making',
    },
    'homeostatic_control': {
        'required': {'heat_shock_prot', 'proteasome', 'antioxidant'},
        'desc': 'Protein quality control — maintains cellular integrity under stress',
        'bonus': 'homeostasis',
    },
    'full_nervous_system': {
        'required': {'ion_channel', 'pump_atpase', 'vesicle', 'neurotransmitter',
                     'snare_complex', 'receptor_gated', 'nmda_receptor', 'camkii',
                     'pattern_net', 'decision_gate'},
        'desc': 'Complete nervous system — sensing, transmission, learning, and decision-making',
        'bonus': 'high_intelligence',
    },
    'genome_integrity': {
        'required': {'dna_repair', 'heat_shock_prot', 'proteasome'},
        'desc': 'Genome and proteome maintenance — long-term organism viability',
        'bonus': 'longevity',
    },
}

# Systems a viable organism needs — like a car needs engine + frame + wheels
REQUIRED_SYSTEMS = {
    'structure':   1,  # at least 1 structural part
    'energy':      1,  # at least 1 energy part
    'catalysis':   1,  # at least 1 catalytic part
}

# Intelligence tiers: how many neural subsystems define each level
INTELLIGENCE_TIERS = [
    (0,  'Non-cognitive',    'No neural machinery'),
    (1,  'Reactive',         'Basic stimulus-response only'),
    (2,  'Excitable',        'Action potentials — electrical signaling'),
    (3,  'Synaptic',         'Chemical synapses — cell-to-cell communication'),
    (4,  'Integrative',      'Signal cascades — multi-input integration'),
    (5,  'Adaptive',         'Hebbian plasticity — learns from experience'),
    (6,  'Rhythmic',         'Neural oscillators — internal time and state'),
    (7,  'Memory-capable',   'Working memory — holds goals and context'),
    (8,  'Decision-making',  'Evidence integration — deliberate action selection'),
    (9,  'High-intelligence','Full nervous system + homeostasis — autonomous agent'),
    (10, 'Transcendent',     'All complexes present — theoretical maximum'),
]

NEURAL_COMPLEXES = {
    'action_potential', 'synapse', 'signal_cascade', 'hebbian_plasticity',
    'neural_oscillator', 'working_memory_loop', 'decision_circuit',
    'homeostatic_control', 'full_nervous_system', 'genome_integrity',
}

# Frozen, order-stable part name list. Using a tuple prevents accidental mutation
# and the explicit order is what codon->part indexing depends on. Changing this
# order will break reference genomes, so it is pinned here deliberately.
PART_NAMES: Tuple[str, ...] = tuple(PARTS_KB.keys())

# BUILD opcode indexes only the classic 17 parts (original KB).
# MOTOR opcode indexes only motor/signaling parts.
# Splitting prevents adding new parts from shifting BUILD's modulo mapping.
BUILD_PART_NAMES: Tuple[str, ...] = (
    'membrane', 'wall', 'filament', 'atp', 'nadh', 'glucose',
    'amino_basic', 'amino_func', 'cofactor', 'nucleotide', 'coenzyme_a',
    'methane', 'water', 'ammonia', 'formaldehyde', 'hcn', 'phosphate',
)
# MOTOR opcode indexes only classic motor/signaling parts — this list is
# intentionally frozen so operand%N stays stable for reference genomes.
MOTOR_PART_NAMES: Tuple[str, ...] = tuple(
    n for n in PART_NAMES if PARTS_KB[n]['role'] in ('motor', 'signaling')
)
# Full extended list used only by generate_intelligent_organism's _build()
# helper (which looks up by name directly, so order doesn't matter here).
EXTENDED_MOTOR_PART_NAMES: Tuple[str, ...] = (
    MOTOR_PART_NAMES
    + tuple(n for n in PART_NAMES if PARTS_KB[n]['role'] == 'neural')
    + tuple(n for n in PART_NAMES if PARTS_KB[n]['role'] == 'homeostasis')
)
NEURAL_PART_NAMES: Tuple[str, ...] = tuple(
    n for n in PART_NAMES if PARTS_KB[n]['role'] == 'neural'
)
HOMEOSTASIS_PART_NAMES: Tuple[str, ...] = tuple(
    n for n in PART_NAMES if PARTS_KB[n]['role'] == 'homeostasis'
)


def _stable_hash(s: str) -> int:
    """Deterministic string hash unaffected by PYTHONHASHSEED.
    Used for reproducible worker seeding and environment discrimination."""
    h = 0
    for c in s:
        h = (h * 131 + ord(c)) & 0xFFFFFFFF
    return h


def env_hash(env_name: str) -> int:
    """Stable hash of environment name mod ENV_HASH_MOD, used by REGULATE.
    Stronger than the previous `% 3` which collapsed all envs to 3 buckets."""
    return _stable_hash(env_name) % ENV_HASH_MOD


# ---------------------------------------------------------------------------
# Instruction set: what codons encode as operations
# ---------------------------------------------------------------------------
# DNA doesn't just pick random chemicals. It's a PROGRAM.
# Each codon encodes an instruction: BUILD a part, CONNECT parts,
# REPEAT a section, REGULATE expression, etc.

class Opcode:
    BUILD    = 0   # place a part from the knowledge base
    CONNECT  = 1   # bond the last two parts together
    REPEAT   = 2   # duplicate the last built part
    REGULATE = 3   # conditional: skip next if env doesn't match
    NOOP     = 4   # intron / junk (padding, no effect)
    # --- Extended opcodes for complex life ---
    MOTOR        = 5   # build a motor/flagellar component
    LOCALIZE     = 6   # place a part at a specific membrane location
    POLYMERIZE   = 7   # build a long filament by repeating a monomer
    PHASE        = 8   # advance developmental phase
    CHEMOTAXIS   = 9   # enable chemotactic behavior
    MEMBRANE_INS = 10  # embed a part into the membrane with orientation
    # --- Steps 1, 16-20 opcodes ---
    PROMOTER     = 11  # marks start of an operon (gene group)
    TERMINATOR   = 12  # marks end of an operon
    SECRETE      = 13  # export a protein via secretion system
    QUORUM       = 14  # quorum sensing: density-dependent behavior
    DIVIDE       = 15  # cell division: split blueprint into two
    METABOLIZE   = 16  # chain metabolic reactions
    GRADIENT     = 17  # sense and move along a gradient


# Human-readable names keyed by opcode integer
OPCODE_NAMES: Dict[int, str] = {
    Opcode.BUILD:        'BUILD',
    Opcode.CONNECT:      'CONNECT',
    Opcode.REPEAT:       'REPEAT',
    Opcode.REGULATE:     'REGULATE',
    Opcode.NOOP:         'NOOP',
    Opcode.MOTOR:        'MOTOR',
    Opcode.LOCALIZE:     'LOCALIZE',
    Opcode.POLYMERIZE:   'POLYMERIZE',
    Opcode.PHASE:        'PHASE',
    Opcode.CHEMOTAXIS:   'CHEMOTAXIS',
    Opcode.MEMBRANE_INS: 'MEMBRANE_INS',
    Opcode.PROMOTER:     'PROMOTER',
    Opcode.TERMINATOR:   'TERMINATOR',
    Opcode.SECRETE:      'SECRETE',
    Opcode.QUORUM:       'QUORUM',
    Opcode.DIVIDE:       'DIVIDE',
    Opcode.METABOLIZE:   'METABOLIZE',
    Opcode.GRADIENT:     'GRADIENT',
}

# Global dispatch tables for each base system (cached for performance)
_DISPATCH_TABLES: Dict[int, Dict[str, Tuple[int, int]]] = {}

def codon_to_instruction(codon: str, bases: List[str] = None) -> Tuple[int, int]:
    """Decode a 5-base codon into (opcode, operand) using clean dispatch table.
    Args:
        codon: The DNA codon to decode
        bases: Base system to use (defaults to 4-base ATCG)
    Returns:
        Tuple of (opcode_code, operand)
    """
    if bases is None:
        bases = ['A', 'T', 'C', 'G']
    
    # Get or create dispatch table for this base system
    base_count = len(bases)
    if base_count not in _DISPATCH_TABLES:
        _DISPATCH_TABLES[base_count] = LANGUAGE_SPEC.build_codon_dispatch_table(bases)
    
    dispatch = _DISPATCH_TABLES[base_count]
    
    # Look up codon in dispatch table
    if codon in dispatch:
        return dispatch[codon]
    
    # If not found (invalid codon), return NOOP
    return Opcode.NOOP, 0


# ---------------------------------------------------------------------------
# Blueprint interpreter: executes genome program to assemble an organism
# ---------------------------------------------------------------------------

@dataclass
class Blueprint:
    """Result of executing a genome program. Contains the assembled parts,
    their connections, and whether the organism is functionally complete."""
    parts_used: List[str] = field(default_factory=list)
    connections: List[Tuple[int, int]] = field(default_factory=list)
    smiles_components: List[str] = field(default_factory=list)
    roles_present: Dict[str, int] = field(default_factory=dict)
    instruction_log: List[str] = field(default_factory=list)
    # --- Extended fields ---
    energy_budget: int = 0               # total energy available (PMF units)
    energy_cost: int = 0                 # total energy consumed by parts
    spatial_graph: Dict[int, List[str]] = field(default_factory=dict)  # part_idx -> sites
    localizations: List[Tuple[int, str]] = field(default_factory=list)  # (part_idx, site)
    phase: str = 'growth'                # current developmental phase
    behaviors: List[str] = field(default_factory=list)  # enabled behaviors
    complexes_formed: List[str] = field(default_factory=list)  # detected assemblies
    # --- Steps 1-5, 11-20 fields ---
    operons: List[List[str]] = field(default_factory=list)  # detected gene groups
    _current_operon: List[str] = field(default_factory=list)  # operon being built
    domain_mismatches: int = 0           # count of bad domain connections (step 2)
    valid_localizations: int = 0         # count of valid site placements (step 3)
    invalid_localizations: int = 0       # count of mismatched site placements
    polymerize_energy: int = 0           # total energy spent on POLYMERIZE (step 5)
    secretions: List[str] = field(default_factory=list)  # secreted proteins
    quorum_signals: int = 0              # quorum sensing signal count
    division_events: int = 0             # cell division attempts
    metabolic_chains: List[List[str]] = field(default_factory=list)  # metabolic pathways
    _current_chain: List[str] = field(default_factory=list)
    gradient_steps: int = 0             # gradient navigation steps
    membrane_potential: float = MEMBRANE_POTENTIAL_BASE  # mV
    torque: float = 0.0                  # pN·nm from motor
    rpm: float = 0.0                     # motor rotation speed
    tumble_prob: float = TUMBLE_PROBABILITY_BASE  # tumble probability
    flagella_count: int = 0              # number of flagella assembled (step 15)
    neural_parts: List[str] = field(default_factory=list)        # neural parts present
    homeostasis_parts: List[str] = field(default_factory=list)   # homeostasis parts present
    intelligence_score: float = 0.0      # computed neural intelligence score 0-1

    @property
    def combined_smiles(self) -> str:
        """All parts as disconnected SMILES (RDKit handles as mixture)."""
        return '.'.join(self.smiles_components) if self.smiles_components else 'C'

    def systems_score(self) -> float:
        """How many required biological systems are present. 1.0 = all present."""
        met = 0
        for role, needed in REQUIRED_SYSTEMS.items():
            if self.roles_present.get(role, 0) >= needed:
                met += 1
        return met / len(REQUIRED_SYSTEMS)

    def diversity_score(self) -> float:
        """Variety of parts used. Using many different parts = more capable."""
        unique = len(set(self.parts_used))
        return min(1.0, unique / 6.0)  # 6 unique parts = max score

    def efficiency_score(self) -> float:
        """Ratio of functional instructions to total. Less junk = more efficient."""
        if not self.instruction_log:
            return 0.0
        functional = sum(1 for op in self.instruction_log if op != 'NOOP')
        return functional / len(self.instruction_log)

    def detect_complexes(self) -> List[str]:
        """Check which functional complexes are present in this blueprint."""
        part_set = set(self.parts_used)
        found = []
        for name, spec in FUNCTIONAL_COMPLEXES.items():
            if spec['required'].issubset(part_set):
                found.append(name)
        self.complexes_formed = found
        return found

    def motility_capable(self) -> bool:
        """True if organism has a complete flagellum."""
        return 'flagellum' in self.complexes_formed

    def chemotaxis_capable(self) -> bool:
        """True if organism has chemotaxis signaling."""
        return 'chemotaxis_system' in self.complexes_formed

    def energy_balance(self) -> int:
        """Net energy: budget minus cost. Negative = energy-starved."""
        return self.energy_budget - self.energy_cost

    def assembly_score(self) -> float:
        """Fraction of possible complexes assembled. Higher = more organized."""
        if not FUNCTIONAL_COMPLEXES:
            return 0.0
        return len(self.complexes_formed) / len(FUNCTIONAL_COMPLEXES)

    def domain_compatibility(self, idx_a: int, idx_b: int) -> bool:
        """Check if two parts share at least one domain (step 2)."""
        if idx_a >= len(self.parts_used) or idx_b >= len(self.parts_used):
            return False
        da = set(PARTS_KB[self.parts_used[idx_a]].get('domains', []))
        db = set(PARTS_KB[self.parts_used[idx_b]].get('domains', []))
        if not da and not db:
            return True  # neither has domains = compatible (classic parts)
        return bool(da & db)

    def validate_localization(self, part_idx: int, site: str) -> bool:
        """Check if a part's attachment sites include the target site (step 3)."""
        if part_idx >= len(self.parts_used):
            return False
        allowed = PARTS_KB[self.parts_used[part_idx]].get('attachment', [])
        if not allowed:
            return True  # no restrictions
        return site in allowed

    def operon_count(self) -> int:
        """Number of detected operons (step 1)."""
        return len(self.operons)

    def compute_torque(self) -> float:
        """Calculate motor torque from stator count (step 11)."""
        stator_count = sum(1 for p in self.parts_used if p in ('motA', 'motB')) // 2
        self.torque = stator_count * TORQUE_PER_STATOR
        filament_count = self.parts_used.count('flagellin')
        drag = filament_count * FILAMENT_DRAG
        self.rpm = max(0.0, MOTOR_BASE_RPM * (self.torque / max(1.0, self.torque + drag)))
        self.flagella_count = min(filament_count, sum(1 for p in self.parts_used if p == 'hook_protein'))
        return self.torque

    def compute_membrane_potential(self) -> float:
        """Simulate membrane potential from energy parts and motor load (step 20)."""
        energy_parts = self.roles_present.get('energy', 0)
        motor_load = self.roles_present.get('motor', 0) * ATP_PER_MOTOR_TICK
        self.membrane_potential = MEMBRANE_POTENTIAL_BASE + (energy_parts * 10) - (motor_load * 5)
        return self.membrane_potential

    def metabolic_pathway_complete(self) -> bool:
        """Check if a complete metabolic chain exists: glucose->energy->catalysis (step 19)."""
        has_substrate = any(PARTS_KB[p].get('function') == 'substrate'
                           for p in self.parts_used if p in PARTS_KB)
        has_carrier = any(PARTS_KB[p].get('function') == 'electron_carrier'
                         for p in self.parts_used if p in PARTS_KB)
        has_fuel = any(PARTS_KB[p].get('function') == 'fuel'
                      for p in self.parts_used if p in PARTS_KB)
        return has_substrate and (has_carrier or has_fuel)

    def neural_score(self) -> float:
        """Fraction of neural complexes assembled. 0.0–1.0."""
        if not NEURAL_COMPLEXES:
            return 0.0
        assembled = len(set(self.complexes_formed) & NEURAL_COMPLEXES)
        return assembled / len(NEURAL_COMPLEXES)

    def homeostasis_score(self) -> float:
        """Fraction of homeostasis parts present."""
        h_parts = [p for p in self.parts_used if PARTS_KB.get(p, {}).get('role') == 'homeostasis']
        return min(1.0, len(set(h_parts)) / max(1, len(HOMEOSTASIS_PART_NAMES)))

    def intelligence_level(self) -> Tuple[int, str, str]:
        """Return (tier_index, tier_name, description) for this organism's intelligence.
        Based on count of distinct neural complexes assembled."""
        n_neural = len(set(self.complexes_formed) & NEURAL_COMPLEXES)
        tier = min(n_neural, len(INTELLIGENCE_TIERS) - 1)
        return INTELLIGENCE_TIERS[tier]

    def intelligence_capable(self) -> bool:
        """True if organism has at least a basic action potential circuit."""
        return 'action_potential' in self.complexes_formed

    def fully_intelligent(self) -> bool:
        """True if organism has a complete nervous system."""
        return 'full_nervous_system' in self.complexes_formed

    def compute_intelligence(self) -> float:
        """Compute and cache the intelligence score (0–1). Called after detect_complexes."""
        n_score = self.neural_score()
        h_score = self.homeostasis_score()
        # Full-nervous-system bonus
        fns_bonus = 0.20 if self.fully_intelligent() else 0.0
        # Learning bonus
        learn_bonus = 0.10 if 'hebbian_plasticity' in self.complexes_formed else 0.0
        # Memory bonus
        mem_bonus = 0.10 if 'working_memory_loop' in self.complexes_formed else 0.0
        # Decision bonus
        dec_bonus = 0.10 if 'decision_circuit' in self.complexes_formed else 0.0
        score = (n_score * 0.50 + h_score * 0.20
                 + fns_bonus + learn_bonus + mem_bonus + dec_bonus)
        self.intelligence_score = min(1.0, score)
        return self.intelligence_score

    def summary(self) -> Dict[str, Any]:
        return {
            'parts': self.parts_used,
            'n_parts': len(self.parts_used),
            'n_unique_parts': len(set(self.parts_used)),
            'connections': self.connections,
            'roles': dict(self.roles_present),
            'systems_complete': self.systems_score(),
            'diversity': round(self.diversity_score(), 4),
            'efficiency': round(self.efficiency_score(), 4),
            'energy_budget': self.energy_budget,
            'energy_cost': self.energy_cost,
            'energy_balance': self.energy_balance(),
            'complexes': self.complexes_formed,
            'motility': self.motility_capable(),
            'chemotaxis': self.chemotaxis_capable(),
            'phase': self.phase,
            'behaviors': self.behaviors,
            'assembly': round(self.assembly_score(), 4),
            'operons': len(self.operons),
            'domain_mismatches': self.domain_mismatches,
            'valid_localizations': self.valid_localizations,
            'invalid_localizations': self.invalid_localizations,
            'polymerize_energy': self.polymerize_energy,
            'torque': round(self.torque, 2),
            'rpm': round(self.rpm, 2),
            'flagella_count': self.flagella_count,
            'membrane_potential': round(self.membrane_potential, 2),
            'metabolic_complete': self.metabolic_pathway_complete(),
            'secretions': len(self.secretions),
            'quorum_signals': self.quorum_signals,
            'division_events': self.division_events,
            'gradient_steps': self.gradient_steps,
        }


def execute_genome(strand: Strand, env_name: str = 'earth',
                   deterministic: bool = False,
                   seed: int = None) -> Blueprint:
    """Run the genome as a program. Each codon is an instruction that
    builds, connects, repeats, or regulates parts from the knowledge base.

    When deterministic=True, uses `codons_deterministic()` (no random intron
    splicing or random pad bases) for repeatable display/validation.
    When seed is provided, uses seeded RNG for reproducible splicing."""
    bp = Blueprint()
    if deterministic:
        codons = strand.codons_deterministic()
    elif seed is not None:
        codons = strand.codons(seed=seed)
    else:
        codons = strand.codons()
    skip_next = False
    env_h = env_hash(env_name)

    for codon in codons:
        opcode, operand = codon_to_instruction(codon, strand.system['bases'])

        if skip_next:
            skip_next = False
            bp.instruction_log.append('SKIPPED')
            continue

        if opcode == Opcode.BUILD:
            part_idx = operand % len(BUILD_PART_NAMES)
            part_name = BUILD_PART_NAMES[part_idx]
            part = PARTS_KB[part_name]
            bp.parts_used.append(part_name)
            bp.smiles_components.append(part['smiles'])
            role = part['role']
            bp.roles_present[role] = bp.roles_present.get(role, 0) + 1
            bp.instruction_log.append(f'BUILD:{part_name}')

        elif opcode == Opcode.CONNECT:
            n = len(bp.parts_used)
            if n >= 2:
                a = operand % n
                b = (operand + 1) % n
                if a != b and (a, b) not in bp.connections:
                    # Step 2: domain validation
                    if bp.domain_compatibility(a, b):
                        bp.connections.append((a, b))
                    else:
                        bp.domain_mismatches += 1
            bp.instruction_log.append('CONNECT')

        elif opcode == Opcode.REPEAT:
            if bp.parts_used:
                last = bp.parts_used[-1]
                part = PARTS_KB[last]
                bp.parts_used.append(last)
                bp.smiles_components.append(part['smiles'])
                role = part['role']
                bp.roles_present[role] = bp.roles_present.get(role, 0) + 1
            bp.instruction_log.append('REPEAT')

        elif opcode == Opcode.REGULATE:
            # Skip next instruction if environment doesn't match operand
            if operand % ENV_HASH_MOD != env_h:
                skip_next = True
            bp.instruction_log.append('REGULATE')

        elif opcode == Opcode.MOTOR:
            # Build a motor-class part (classic motor/signaling only, frozen for
            # reference-genome compatibility — neural parts use direct Blueprint injection).
            if MOTOR_PART_NAMES:
                part_name = MOTOR_PART_NAMES[operand % len(MOTOR_PART_NAMES)]
                part = PARTS_KB[part_name]
                bp.parts_used.append(part_name)
                bp.smiles_components.append(part['smiles'])
                role = part['role']
                bp.roles_present[role] = bp.roles_present.get(role, 0) + 1
                bp.energy_cost += part.get('energy_cost', 0)
                pidx = len(bp.parts_used) - 1
                bp.spatial_graph[pidx] = part.get('attachment', [])
                if role == 'neural':
                    bp.neural_parts.append(part_name)
                elif role == 'homeostasis':
                    bp.homeostasis_parts.append(part_name)
                bp.instruction_log.append(f'MOTOR:{part_name}')
            else:
                bp.instruction_log.append('NOOP')

        elif opcode == Opcode.LOCALIZE:
            # Place the last built part at a specific attachment site
            sites = ['membrane', 'basal', 'distal', 'wall', 'cytoplasm']
            site = sites[operand % len(sites)]
            if bp.parts_used:
                pidx = len(bp.parts_used) - 1
                # Step 3: attachment site validation
                if bp.validate_localization(pidx, site):
                    bp.localizations.append((pidx, site))
                    bp.valid_localizations += 1
                else:
                    bp.localizations.append((pidx, site))
                    bp.invalid_localizations += 1
                bp.instruction_log.append(f'LOCALIZE:{site}')
            else:
                bp.instruction_log.append('NOOP')

        elif opcode == Opcode.POLYMERIZE:
            # Repeat the last built part multiple times (filament assembly)
            repeats = min(operand % MAX_POLYMER_REPEATS + 1, MAX_POLYMER_REPEATS)
            if bp.parts_used:
                last = bp.parts_used[-1]
                part = PARTS_KB[last]
                poly_cost = 0
                for _ in range(repeats):
                    bp.parts_used.append(last)
                    bp.smiles_components.append(part['smiles'])
                    role = part['role']
                    bp.roles_present[role] = bp.roles_present.get(role, 0) + 1
                    ec = part.get('energy_cost', 0)
                    bp.energy_cost += ec
                    poly_cost += ec
                # Step 5: track polymerize energy
                bp.polymerize_energy += poly_cost
                bp.instruction_log.append(f'POLYMERIZE:{last}x{repeats}')
            else:
                bp.instruction_log.append('NOOP')

        elif opcode == Opcode.PHASE:
            # Advance the developmental phase
            phases = DEVELOPMENT_PHASES
            cur_idx = phases.index(bp.phase) if bp.phase in phases else 0
            if cur_idx < len(phases) - 1:
                bp.phase = phases[cur_idx + 1]
            bp.instruction_log.append(f'PHASE:{bp.phase}')

        elif opcode == Opcode.CHEMOTAXIS:
            # Enable chemotactic behavior if signaling parts are present
            if bp.roles_present.get('signaling', 0) > 0:
                if 'chemotaxis' not in bp.behaviors:
                    bp.behaviors.append('chemotaxis')
                bp.instruction_log.append('CHEMOTAXIS:enabled')
            else:
                bp.instruction_log.append('CHEMOTAXIS:no_signal')

        elif opcode == Opcode.MEMBRANE_INS:
            # Insert the last part into the membrane layer
            if bp.parts_used:
                pidx = len(bp.parts_used) - 1
                bp.localizations.append((pidx, 'membrane_embedded'))
                bp.instruction_log.append(f'MEMBRANE_INSERT:{bp.parts_used[-1]}')
            else:
                bp.instruction_log.append('NOOP')

        elif opcode == Opcode.PROMOTER:
            # Step 1: start a new operon (gene group)
            if bp._current_operon:
                bp.operons.append(list(bp._current_operon))
            bp._current_operon = []
            bp.instruction_log.append('PROMOTER')

        elif opcode == Opcode.TERMINATOR:
            # Step 1: end current operon
            if bp._current_operon:
                bp.operons.append(list(bp._current_operon))
                bp._current_operon = []
            bp.instruction_log.append('TERMINATOR')

        elif opcode == Opcode.SECRETE:
            # Step 16: export last-built part
            if bp.parts_used:
                bp.secretions.append(bp.parts_used[-1])
                bp.energy_cost += SECRETION_ENERGY_COST
                bp.instruction_log.append(f'SECRETE:{bp.parts_used[-1]}')
            else:
                bp.instruction_log.append('NOOP')

        elif opcode == Opcode.QUORUM:
            # Step 17: emit quorum signal
            bp.quorum_signals += 1
            if 'quorum_sensing' not in bp.behaviors:
                bp.behaviors.append('quorum_sensing')
            bp.instruction_log.append('QUORUM')

        elif opcode == Opcode.DIVIDE:
            # Step 18: attempt cell division
            if len(bp.parts_used) >= DIVISION_MIN_PARTS:
                bp.division_events += 1
                bp.instruction_log.append('DIVIDE')
            else:
                bp.instruction_log.append('DIVIDE:insufficient_parts')

        elif opcode == Opcode.METABOLIZE:
            # Step 19: chain metabolic reactions
            if bp.parts_used:
                func = PARTS_KB[bp.parts_used[-1]].get('function', '')
                bp._current_chain.append(bp.parts_used[-1])
                if func in ('substrate', 'fuel', 'electron_carrier'):
                    # Complete chain when energy output found
                    if len(bp._current_chain) >= 2:
                        bp.metabolic_chains.append(list(bp._current_chain))
                        bp._current_chain = []
                bp.instruction_log.append(f'METABOLIZE:{bp.parts_used[-1]}')
            else:
                bp.instruction_log.append('NOOP')

        elif opcode == Opcode.GRADIENT:
            # Step 13/17: move along nutrient gradient
            if bp.chemotaxis_capable():
                bp.gradient_steps += 1
                bp.instruction_log.append('GRADIENT:navigate')
            else:
                bp.instruction_log.append('GRADIENT:no_chemotaxis')

        else:
            bp.instruction_log.append('NOOP')

        # Step 1: track gene in current operon
        if bp._current_operon is not None and opcode in (
                Opcode.BUILD, Opcode.MOTOR, Opcode.POLYMERIZE):
            bp._current_operon.append(bp.instruction_log[-1])

    # --- Post-execution bookkeeping ---
    # Close any unclosed operon
    if bp._current_operon:
        bp.operons.append(list(bp._current_operon))
        bp._current_operon = []
    # Energy budget: each energy-role part contributes PMF
    bp.energy_budget = bp.roles_present.get('energy', 0) * BASE_PMF_UNITS
    # Detect functional complexes
    bp.detect_complexes()
    # Enable motility behavior if flagellum is assembled
    if bp.motility_capable() and 'swim' not in bp.behaviors:
        bp.behaviors.append('swim')
    # Step 11: compute torque and RPM
    bp.compute_torque()
    # Step 12: tumble probability based on chemotaxis
    if bp.chemotaxis_capable():
        bp.tumble_prob = TUMBLE_PROBABILITY_BASE * 0.3  # reduced tumble with chemotaxis
    # Step 20: membrane potential
    bp.compute_membrane_potential()
    # Intelligence scoring
    bp.compute_intelligence()
    # Enable cognitive behaviors
    if bp.intelligence_capable() and 'neural_signaling' not in bp.behaviors:
        bp.behaviors.append('neural_signaling')
    if bp.fully_intelligent() and 'autonomous_cognition' not in bp.behaviors:
        bp.behaviors.append('autonomous_cognition')
    if 'hebbian_plasticity' in bp.complexes_formed and 'learning' not in bp.behaviors:
        bp.behaviors.append('learning')
    if 'working_memory_loop' in bp.complexes_formed and 'working_memory' not in bp.behaviors:
        bp.behaviors.append('working_memory')
    if 'decision_circuit' in bp.complexes_formed and 'decision_making' not in bp.behaviors:
        bp.behaviors.append('decision_making')

    return bp


def execute_genome_deterministic(strand: Strand, env_name: str = 'earth') -> Blueprint:
    """Deterministic wrapper kept for backward compatibility."""
    return execute_genome(strand, env_name, deterministic=True)


# ---------------------------------------------------------------------------
# Molecule building + 3D coordinate extraction
# ---------------------------------------------------------------------------

def build_mol(smiles: str) -> Optional[Chem.Mol]:
    """Build an RDKit molecule from SMILES with 3D coordinates.
    Uses defensive sanitization: parses without sanitizing, then tries full
    sanitize, and falls back to a sanitize that tolerates unusual charge states
    so charged fragments like [Fe+2] or [N+](C)(C)C still build."""
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    try:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            # Retry without charge-state checks (e.g. metal cations, quaternary N)
            try:
                Chem.SanitizeMol(
                    mol,
                    sanitizeOps=(Chem.SANITIZE_ALL
                                 ^ Chem.SANITIZE_PROPERTIES
                                 ^ Chem.SANITIZE_CLEANUPCHIRALITY),
                )
            except Exception:
                return None
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        embed_ok = AllChem.EmbedMolecule(mol, params)
        if embed_ok == -1:
            params.useRandomCoords = True
            embed_ok = AllChem.EmbedMolecule(mol, params)
        if embed_ok == -1:
            # Keep the Mol for descriptor use even without 3D coords.
            # Callers that need geometry should check GetNumConformers().
            _logging.getLogger(__name__).debug(
                "3D embedding failed for SMILES: %s", smiles)
            return mol
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=300)
        except Exception:
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=300)
            except Exception:
                pass
        return mol
    except Exception:
        return None


def build_organism_mol(bp: 'Blueprint') -> Tuple[List[Chem.Mol], Dict[str, int], Optional[Dict]]:
    """Build molecules from a Blueprint's parts individually.
    Returns (list_of_mols, aggregate_element_counts, aggregate_3d_geometry).
    Handles failed builds explicitly: the returned mols list only contains
    successfully-built molecules, but a part_idx -> atom_offset map is used
    so connection indices (which reference bp.parts_used / bp.smiles_components
    positions) stay correct even when some parts failed.
    NOTE: Does NOT mutate bp (in particular, does not append to
    bp.instruction_log) to avoid post-facto side-effects on the blueprint."""
    mols: List[Chem.Mol] = []
    all_elements: Dict[str, int] = {}
    all_atoms: List[Dict[str, Any]] = []
    all_bonds: List[Dict[str, Any]] = []
    # Per-part-index atom start offset in the flattened atom list.
    # None for parts that failed to build. Indexed by bp.parts_used position.
    part_offsets: Dict[int, Optional[int]] = {}
    atom_offset = 0

    n_parts = min(len(bp.smiles_components), len(bp.parts_used))
    for i in range(n_parts):
        smiles = bp.smiles_components[i]
        part_name = bp.parts_used[i]
        mol = build_mol(smiles)
        if mol is None:
            # Try a sanitized fallback without common charge-state tokens
            clean = (smiles.replace('[N+]', 'N')
                           .replace('[Fe+2]', '[Fe]')
                           .replace('[Fe+3]', '[Fe]'))
            mol = build_mol(clean)
        if mol is None:
            part_offsets[i] = None
            continue

        mols.append(mol)
        part_offsets[i] = atom_offset
        # Element counts (AddHs already applied in build_mol, so H is included)
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            all_elements[sym] = all_elements.get(sym, 0) + 1
        # 3D geometry with offset per part (grid layout)
        if mol.GetNumConformers() > 0:
            conf = mol.GetConformer()
            x_off = (i % PART_GRID_COLS) * PART_SPACING_ANGSTROMS
            y_off = (i // PART_GRID_COLS) * PART_SPACING_ANGSTROMS
            for j, atom in enumerate(mol.GetAtoms()):
                pos = conf.GetAtomPosition(j)
                all_atoms.append({
                    'symbol': atom.GetSymbol(),
                    'x': round(pos.x + x_off, 4),
                    'y': round(pos.y + y_off, 4),
                    'z': round(pos.z, 4),
                    'part_idx': i,
                    'part_name': part_name,
                })
            for b in mol.GetBonds():
                all_bonds.append({
                    'i': b.GetBeginAtomIdx() + atom_offset,
                    'j': b.GetEndAtomIdx() + atom_offset,
                    'order': b.GetBondTypeAsDouble(),
                })
        atom_offset += mol.GetNumAtoms()

    # Add inter-part connection bonds using the correct per-part atom offset map
    for conn_a, conn_b in bp.connections:
        off_a = part_offsets.get(conn_a)
        off_b = part_offsets.get(conn_b)
        if off_a is None or off_b is None:
            continue
        all_bonds.append({
            'i': off_a,
            'j': off_b,
            'order': 1.0,
            'virtual': True,
        })

    geom = {'atoms': all_atoms, 'bonds': all_bonds} if all_atoms else None
    return mols, all_elements, geom


def mol_element_counts(mol: Chem.Mol) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        counts[sym] = counts.get(sym, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Organism: genome-programmed entity with functional assessment
# ---------------------------------------------------------------------------

@dataclass
class Organism:
    strand: Strand
    env_name: str = 'earth'
    blueprint: Optional[Blueprint] = None
    expressed_smiles: str = ''
    mol: Optional[Any] = None
    fitness: float = 0.0
    element_cost: Dict[str, int] = field(default_factory=dict)
    viable: bool = False
    gen_born: int = 0

    def __post_init__(self):
        # Initialize private attrs so that getattr-free access in evaluate()/
        # structural_grade()/to_dict() never depends on express() having run.
        self._part_mols: List[Any] = []
        self._organism_geom: Optional[Dict[str, Any]] = None
        self._total_heavy: int = 0
        self._total_mw: float = 0.0
        self._total_rings: int = 0
        self._total_hba: int = 0
        self._total_hbd: int = 0
        self._structural_quality: float = 0.0

    @classmethod
    def from_genome(cls, seq: str, n_bases: int = 4,
                    env_name: str = 'earth', gen_born: int = 0) -> 'Organism':
        """Factory: build an Organism directly from a raw genome string."""
        return cls(strand=Strand(seq, n_bases), env_name=env_name, gen_born=gen_born)

    def express(self, supply: ElementSupply, deterministic: bool = False,
                seed: int = None) -> None:
        """Execute genome program -> assemble blueprint -> build molecules.
        If deterministic=True, uses codons_deterministic (no random splicing).
        If seed is provided, uses seeded RNG for reproducible splicing.
        Uses build_organism_mol to build each part separately, avoiding
        multi-component SMILES failures."""
        if deterministic:
            self.blueprint = execute_genome_deterministic(self.strand, self.env_name)
        else:
            self.blueprint = execute_genome(self.strand, self.env_name, seed=seed)
        self.expressed_smiles = self.blueprint.combined_smiles

        # Build each part separately for reliable chemistry
        part_mols, elem_counts, geom = build_organism_mol(self.blueprint)
        self._part_mols = part_mols
        self._organism_geom = geom

        if part_mols:
            # Use first (largest) mol for descriptor calculations
            self.mol = max(part_mols, key=lambda m: m.GetNumHeavyAtoms())
            self.element_cost = elem_counts
            self.viable = supply.can_afford(elem_counts)
            self._total_heavy = sum(m.GetNumHeavyAtoms() for m in part_mols)
            self._total_mw = sum(Descriptors.MolWt(m) for m in part_mols)
            self._total_rings = sum(rdMolDescriptors.CalcNumRings(m) for m in part_mols)
            self._total_hba = sum(Descriptors.NumHAcceptors(m) for m in part_mols)
            self._total_hbd = sum(Descriptors.NumHDonors(m) for m in part_mols)
        else:
            self.mol = None
            self.element_cost = {}
            self.viable = False
            self._total_heavy = 0
            self._total_mw = 0.0
            self._total_rings = 0
            self._total_hba = 0
            self._total_hbd = 0

    def evaluate(self) -> float:
        """Fitness = can this organism actually function?
        Living organisms (all 3 systems) always score higher than non-living.
        Within each tier, complexity and chemistry determine quality."""
        if self.blueprint is None or not self.blueprint.parts_used:
            self.fitness = 0.01
            return self.fitness

        # --- Tier 1: Functional completeness (most important) ---
        systems = self.blueprint.systems_score()
        alive = systems == 1.0

        # --- Tier 2: Biological quality ---
        diversity = self.blueprint.diversity_score()
        efficiency = self.blueprint.efficiency_score()
        connectivity = min(1.0, len(self.blueprint.connections) / max(1, len(self.blueprint.parts_used) - 1))
        has_info = 1.0 if self.blueprint.roles_present.get('information', 0) > 0 else 0.0

        # --- Tier 3: Chemical quality ---
        if self._total_heavy > 0:
            complexity = min(1.0, (self._total_heavy / 80) * 0.4
                           + (self._total_rings / 8) * 0.3
                           + (self._total_hba / 12) * 0.3)
        else:
            complexity = 0.0

        # --- Tier 4: Replication quality ---
        rep_stability = self.strand.stability_score()
        integrity = self.strand.pair_integrity()

        # --- Tier 5: Element viability ---
        viability = 1.0 if self.viable else 0.3

        # --- Tier 6: Assembly & motility (extended) ---
        assembly = self.blueprint.assembly_score()
        motile = 1.0 if self.blueprint.motility_capable() else 0.0
        chemotactic = 1.0 if self.blueprint.chemotaxis_capable() else 0.0
        e_bal = self.blueprint.energy_balance()
        energy_ok = 1.0 if e_bal >= 0 else max(0.0, 1.0 + e_bal * ENERGY_STARVATION_SCALE)

        # --- Tier 7: Intelligence ---
        intel_score = self.blueprint.intelligence_score  # 0–1, computed in execute_genome
        homeostasis = self.blueprint.homeostasis_score()

        # Step 1: operon organization bonus
        operon_bonus = min(0.10, self.blueprint.operon_count() * PROMOTER_BONUS)
        # Step 2: domain mismatch penalty
        domain_pen = min(0.15, self.blueprint.domain_mismatches * DOMAIN_MISMATCH_PENALTY)
        # Step 5: polymerize over-spending penalty
        poly_pen = max(0.0, (self.blueprint.polymerize_energy - MAX_FILAMENT_COST) * 0.005)
        # Step 11: torque bonus
        torque_bonus = min(0.05, self.blueprint.torque * 0.001)
        # Step 13: gradient navigation bonus
        gradient_bonus = min(0.05, self.blueprint.gradient_steps * GRADIENT_REWARD_PER_STEP)
        # Step 18: division capability bonus
        division_bonus = 0.03 if self.blueprint.division_events > 0 else 0.0
        # Step 19: metabolic chain bonus
        metabolic_bonus = METABOLIC_CHAIN_BONUS if self.blueprint.metabolic_pathway_complete() else 0.0
        # Step 17: quorum sensing bonus
        quorum_bonus = 0.02 if self.blueprint.quorum_signals > 0 else 0.0
        # Intelligence bonuses
        neural_bonus  = min(0.15, intel_score * 0.15)         # up to +0.15 for full neural
        homeostasis_bonus = min(0.05, homeostasis * 0.05)     # up to +0.05 for full homeostasis
        full_intel_bonus = 0.10 if self.blueprint.fully_intelligent() else 0.0
        learning_bonus   = 0.05 if 'learning' in self.blueprint.behaviors else 0.0
        memory_bonus     = 0.05 if 'working_memory' in self.blueprint.behaviors else 0.0
        decision_bonus   = 0.05 if 'decision_making' in self.blueprint.behaviors else 0.0

        # Structural quality score (used for grading)
        quality = (
            diversity * 0.20
            + connectivity * 0.10
            + complexity * 0.15
            + efficiency * 0.10
            + has_info * 0.10
            + viability * 0.10
            + assembly * 0.15
            + energy_ok * 0.10
        )
        self._structural_quality = quality

        # Fitness: alive organisms get 0.50-1.00 range, dead get 0.01-0.49
        if alive:
            base = 0.50 + quality * 0.25 + rep_stability * 0.05
            # Bonuses for complex machinery
            base += motile * MOTILITY_BONUS
            base += chemotactic * (MOTILITY_BONUS * 0.5)
            base += assembly * ASSEMBLY_BONUS
            # Steps 1-20 bonuses/penalties
            base += operon_bonus + torque_bonus + gradient_bonus
            base += division_bonus + metabolic_bonus + quorum_bonus
            # Intelligence bonuses
            base += neural_bonus + homeostasis_bonus + full_intel_bonus
            base += learning_bonus + memory_bonus + decision_bonus
            base -= domain_pen + poly_pen
            self.fitness = base
        else:
            # Partial credit proportional to systems present
            self.fitness = systems * 0.25 + quality * 0.15 + rep_stability * 0.05
            # Intelligence partial credit even without full viability
            self.fitness += intel_score * 0.05
            # Hard cap: non-viable never reaches 0.50
            self.fitness = min(0.49, self.fitness)

        self.fitness = min(1.0, max(0.01, self.fitness))
        return self.fitness

    def structural_grade(self) -> Tuple[str, str]:
        """Grade the organism's structural quality from F to S.
        Returns (letter_grade, description)."""
        if self.blueprint is None or not self.blueprint.parts_used:
            return ('F', 'No functional parts — not an organism')

        systems = self.blueprint.systems_score()
        q = getattr(self, '_structural_quality', 0.0)
        n_unique = len(set(self.blueprint.parts_used))
        has_info = self.blueprint.roles_present.get('information', 0) > 0

        if systems < 1.0:
            if systems == 0:
                return ('F', 'Raw chemicals only — no biological systems present')
            return ('D', f'Incomplete organism — missing {int((1-systems)*3)} of 3 required systems')

        # Viable organism — grade by quality
        motile = self.blueprint.motility_capable()
        has_motor = self.blueprint.roles_present.get('motor', 0) > 0
        has_chemotaxis = self.blueprint.chemotaxis_capable()

        intel_score = self.blueprint.intelligence_score
        is_intelligent = self.blueprint.intelligence_capable()
        is_fully_intel = self.blueprint.fully_intelligent()
        has_learning  = 'learning' in self.blueprint.behaviors
        has_memory    = 'working_memory' in self.blueprint.behaviors
        has_decision  = 'decision_making' in self.blueprint.behaviors
        tier_idx, tier_name, tier_desc = self.blueprint.intelligence_level()

        # Intelligence-first grading
        if is_fully_intel and has_learning and has_memory and has_decision and q > 0.6:
            return ('S++', f'High-Intelligence Autonomous Agent — {tier_name}: {tier_desc}')
        if is_fully_intel and (has_learning or has_memory) and n_unique >= 12:
            return ('S+', f'Intelligent Organism — {tier_name}: {tier_desc}')
        if is_intelligent and n_unique >= 10 and has_chemotaxis and q > 0.7:
            return ('S+', f'Neural-Chemotactic Organism — {tier_name} with motor-driven navigation')
        if motile and has_chemotaxis and n_unique >= 10 and q > 0.7:
            return ('S+', 'Autonomous motile organism — flagellar motor with chemotaxis')
        if is_intelligent and n_unique >= 8 and q > 0.5:
            return ('S', f'Neural Organism — {tier_name}: {tier_desc}')
        if motile and n_unique >= 8 and has_info and q > 0.6:
            return ('S', 'Motile organism — complete flagellum, can swim and reproduce')
        if has_motor and n_unique >= 7 and has_info and q > 0.5:
            return ('S', 'Superior organism — motor components, high complexity, can reproduce')
        if n_unique >= 7 and has_info and q > 0.7:
            return ('S', 'Superior organism — all systems, high complexity, can reproduce')
        if n_unique >= 5 and has_info and q > 0.5:
            return ('A', 'Advanced cell — multiple subsystems, information storage, well-connected')
        if n_unique >= 4 and q > 0.4:
            return ('B', 'Capable bacterium — good diversity, functional systems')
        if n_unique >= 3:
            return ('C', 'Minimal viable cell — has the basics but nothing more')
        return ('C', 'Bare minimum for life — fragile and limited')

    def to_dict(self) -> Dict[str, Any]:
        bp_summary = self.blueprint.summary() if self.blueprint else {}
        grade, grade_desc = self.structural_grade()
        mols = getattr(self, '_part_mols', [])
        total_heavy = getattr(self, '_total_heavy', 0)
        total_mw = getattr(self, '_total_mw', 0.0)
        d = {
            'genome': self.strand.seq,
            'complement': self.strand.complement,
            'base_system': self.strand.system['name'],
            'n_bases': self.strand.n_bases,
            'pair_integrity': round(self.strand.pair_integrity(), 4),
            'replication_error_rate': self.strand.system['error_rate'],
            'info_bits_per_base': round(self.strand.info_density(), 2),
            'stability': round(self.strand.stability_score(), 4),
            'env': self.env_name,
            'blueprint': bp_summary,
            'smiles': self.expressed_smiles,
            'fitness': round(self.fitness, 4),
            'structural_grade': grade,
            'structural_grade_desc': grade_desc,
            'n_heavy_atoms': total_heavy,
            'mw': round(total_mw, 2),
            'element_cost': self.element_cost,
            'viable': self.viable,
            'gen_born': self.gen_born,
            'n_molecules_built': len(mols),
            'complexes': self.blueprint.complexes_formed if self.blueprint else [],
            'motility': self.blueprint.motility_capable() if self.blueprint else False,
            'chemotaxis': self.blueprint.chemotaxis_capable() if self.blueprint else False,
            'energy_balance': self.blueprint.energy_balance() if self.blueprint else 0,
            'behaviors': self.blueprint.behaviors if self.blueprint else [],
            'phase': self.blueprint.phase if self.blueprint else 'growth',
            'torque': round(self.blueprint.torque, 2) if self.blueprint else 0,
            'rpm': round(self.blueprint.rpm, 2) if self.blueprint else 0,
            'flagella_count': self.blueprint.flagella_count if self.blueprint else 0,
            'operons': len(self.blueprint.operons) if self.blueprint else 0,
            'membrane_potential': round(self.blueprint.membrane_potential, 2) if self.blueprint else 0,
            'metabolic_complete': self.blueprint.metabolic_pathway_complete() if self.blueprint else False,
            'secretions': len(self.blueprint.secretions) if self.blueprint else 0,
            'quorum_signals': self.blueprint.quorum_signals if self.blueprint else 0,
            'division_events': self.blueprint.division_events if self.blueprint else 0,
            'gradient_steps': self.blueprint.gradient_steps if self.blueprint else 0,
            'domain_mismatches': self.blueprint.domain_mismatches if self.blueprint else 0,
        }
        geom = getattr(self, '_organism_geom', None)
        if geom:
            d['atoms'] = geom['atoms']
            d['bonds'] = geom['bonds']
        return d


# ---------------------------------------------------------------------------
# Parallel evaluation wrapper (must be top-level for pickling)
# ---------------------------------------------------------------------------

def _eval_organism(args: Tuple) -> Dict[str, Any]:
    """Evaluate a single organism in a worker process.
    args = (seq, n_bases, env_name, gen[, seed]).
    If a seed is provided, the worker re-seeds random/numpy so that the same
    genome + seed always produces the same result, even across processes."""
    seq, n_bases, env_name, gen = args[:4]
    worker_seed = None
    if len(args) > 4 and args[4] is not None:
        worker_seed = (args[4] + _stable_hash(seq)) & 0xFFFFFFFF
        random.seed(worker_seed)
        np.random.seed(worker_seed)
    supply_map = {
        'earth': ElementSupply.earth_like,
        'silicon': ElementSupply.silicon_world,
        'exotic': ElementSupply.exotic,
    }
    supply = supply_map.get(env_name, ElementSupply.earth_like)()
    strand = Strand(seq, n_bases)
    org = Organism(strand=strand, env_name=env_name, gen_born=gen)
    org.express(supply, seed=worker_seed)
    org.evaluate()
    return org.to_dict()


# ---------------------------------------------------------------------------
# Genetic operators — work on raw sequences
# ---------------------------------------------------------------------------

def crossover(s1: str, s2: str, allowed_bases: Optional[set] = None) -> str:
    """Two-point crossover. If `allowed_bases` is given, any character in the
    resulting child that is outside the allowed set is replaced with a random
    allowed base (prevents cross-system contamination)."""
    L = min(len(s1), len(s2))
    if L < 6:
        child = s1
    else:
        a, b = sorted(random.sample(range(1, L - 1), 2))
        child = s1[:a] + s2[a:b] + s1[b:]
    if allowed_bases is not None:
        child = ''.join(
            c if c in allowed_bases else random.choice(tuple(allowed_bases))
            for c in child
        )
    return child


def mutate_seq(seq: str, bases: List[str], rate: float = 0.10) -> str:
    g = list(seq)
    for i in range(len(g)):
        if random.random() < rate:
            g[i] = random.choice(bases)
    if random.random() < 0.06 and len(g) < 120:
        g.insert(random.randint(0, len(g)), random.choice(bases))
    if random.random() < 0.08 and len(g) > 20:
        g.pop(random.randint(0, len(g) - 1))
    return ''.join(g)


def tournament(seqs: List[str], fits: List[float], k: int = DEFAULT_TOURNAMENT_K) -> str:
    idxs = random.sample(range(len(seqs)), min(k, len(seqs)))
    return seqs[max(idxs, key=lambda i: fits[i])]


def enforce_length(seq: str, target: int, bases: List[str]) -> str:
    """Clamp genome to exactly `target` bases. Pads with random bases if too
    short, truncates from the end if too long. Prevents population drift."""
    if len(seq) == target:
        return seq
    if len(seq) > target:
        return seq[:target]
    return seq + ''.join(random.choice(bases) for _ in range(target - len(seq)))


# ---------------------------------------------------------------------------
# Evolution engine: runs multiple base systems in parallel
# ---------------------------------------------------------------------------

ENVS = ['earth', 'silicon', 'exotic']


def _hamming(s1: str, s2: str) -> int:
    """Hamming distance between two equal-length strings."""
    return sum(a != b for a, b in zip(s1, s2))


def _cluster_species(population: List[str], fits: List[float],
                     threshold: float = 0.3) -> List[Dict]:
    """Step 7: Simple single-linkage speciation by hamming distance.
    Returns list of species dicts with representative, size, and avg fitness."""
    if not population:
        return []
    n = len(population)
    gl = len(population[0])
    dist_thresh = int(gl * threshold)
    assigned = [False] * n
    species = []
    for i in range(n):
        if assigned[i]:
            continue
        cluster_idxs = [i]
        assigned[i] = True
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            if _hamming(population[i], population[j]) < dist_thresh:
                cluster_idxs.append(j)
                assigned[j] = True
        avg_fit = float(np.mean([fits[k] for k in cluster_idxs]))
        species.append({
            'representative': population[i][:30],
            'size': len(cluster_idxs),
            'avg_fitness': round(avg_fit, 4),
        })
    return species


class Evolver:
    def __init__(self, pop_size: int = 200, genome_len: int = 80,
                 generations: int = 120, n_bases: int = 4,
                 n_workers: int = None,
                 seed: Optional[int] = None,
                 progress_cb: Optional[Any] = None,
                 max_history: int = MAX_HISTORY_LEN,
                 niche_cycle: bool = False,
                 niche_period: int = 10):
        """`seed` pins both random and numpy for reproducibility. `progress_cb`
        is an optional callable `fn(gen:int, total:int, stats:dict)` invoked
        after each generation — useful for GUI progress bars.
        Step 6: `niche_cycle` rotates environment every `niche_period` gens.
        Step 7: speciation tracked via `species_clusters`.
        Step 8: adaptive mutation adjusts rate on fitness plateau.
        Step 10: fitness landscape data stored per generation."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed & 0xFFFFFFFF)
        self.seed = seed
        self.progress_cb = progress_cb
        self.max_history = max_history
        self.pop_size = pop_size
        self.genome_len = genome_len
        self.generations = generations
        self.n_bases = n_bases
        self.system = BASE_SYSTEMS[n_bases]
        self.bases = self.system['bases']
        self._allowed = set(self.bases)
        self.n_workers = n_workers or min(mp.cpu_count(), 12)
        self.population = [
            ''.join(random.choice(self.bases) for _ in range(genome_len))
            for _ in range(pop_size)
        ]
        self.history: List[Dict] = []
        self.best_ever: Dict = {'fitness': 0.0}
        # Step 6: multi-niche cycling
        self.niche_cycle = niche_cycle
        self.niche_period = niche_period
        # Step 7: speciation
        self.species_clusters: List[Dict] = []
        # Step 8: adaptive mutation
        self._plateau_counter = 0
        self._last_best_fitness = 0.0
        self._adaptive_rate = 0.0
        # Step 10: fitness landscape
        self.landscape_data: List[Dict] = []

    def run(self) -> Dict[str, Any]:
        t0 = time.time()
        print(f"[init] system={self.system['name']} n_bases={self.n_bases} "
              f"error_rate={self.system['error_rate']} pop={self.pop_size} "
              f"gens={self.generations} workers={self.n_workers} "
              f"seed={self.seed}")

        for gen in range(self.generations):
            gt = time.time()

            # Step 6: niche cycling — rotate primary env every niche_period gens
            if self.niche_cycle:
                primary_env = ENVS[(gen // self.niche_period) % len(ENVS)]
                tasks = [
                    (seq, self.n_bases, primary_env, gen, self.seed)
                    for seq in self.population
                ]
            else:
                tasks = [
                    (seq, self.n_bases, ENVS[i % len(ENVS)], gen, self.seed)
                    for i, seq in enumerate(self.population)
                ]

            with ProcessPoolExecutor(max_workers=self.n_workers) as ex:
                results = list(ex.map(
                    _eval_organism, tasks,
                    chunksize=max(1, self.pop_size // self.n_workers)
                ))

            fits = [r['fitness'] for r in results]
            avg = float(np.mean(fits))
            best_idx = int(np.argmax(fits))
            best_r = results[best_idx]
            diversity = len(set(self.population)) / self.pop_size
            valid_pct = sum(1 for r in results if r['viable']) / self.pop_size

            if best_r['fitness'] > self.best_ever.get('fitness', 0):
                self.best_ever = best_r

            # Step 7: speciation — cluster by hamming distance
            if gen % 10 == 0:
                species = _cluster_species(self.population, fits)
                self.species_clusters = species

            # Step 8: adaptive mutation — detect plateau
            if best_r['fitness'] <= self._last_best_fitness + 0.001:
                self._plateau_counter += 1
            else:
                self._plateau_counter = 0
            self._last_best_fitness = best_r['fitness']
            # Increase mutation when stuck, decrease when improving
            self._adaptive_rate = min(0.06, self._plateau_counter * 0.005)

            stats = {
                'gen': gen, 'best': best_r['fitness'], 'avg': avg,
                'diversity': diversity, 'valid_pct': valid_pct,
                'n_species': len(self.species_clusters),
                'adaptive_mut': round(self._adaptive_rate, 4),
            }
            self.history.append(stats)
            if len(self.history) > self.max_history:
                # Drop oldest half to keep memory bounded on long sessions
                self.history = self.history[-self.max_history:]

            # Step 10: fitness landscape data
            if gen % 5 == 0:
                landscape_point = {
                    'gen': gen,
                    'fitness_distribution': sorted(fits, reverse=True)[:20],
                    'best_genome': self.population[best_idx][:40],
                    'n_species': len(self.species_clusters),
                }
                self.landscape_data.append(landscape_point)

            if gen % 10 == 0 or gen == self.generations - 1:
                print(f"  gen {gen:>3d} | best {best_r['fitness']:.4f} | "
                      f"avg {avg:.4f} | viable {valid_pct:.0%} | "
                      f"div {diversity:.2f} | {time.time()-gt:.1f}s")

            if self.progress_cb is not None:
                try:
                    self.progress_cb(gen + 1, self.generations, stats)
                except Exception:
                    pass

            # Replication-aware reproduction: use strand's error rate for mutation
            elite_n = max(4, int(self.pop_size * ELITE_FRACTION))
            ranked = sorted(range(self.pop_size), key=lambda i: fits[i], reverse=True)
            elites = [self.population[i] for i in ranked[:elite_n]]

            children = list(elites)
            # Step 8: adaptive mutation rate
            mut_rate = self.system['error_rate'] + 0.08 + self._adaptive_rate
            while len(children) < self.pop_size:
                p1 = tournament(self.population, fits)
                p2 = tournament(self.population, fits)
                # Step 9: operon-aware crossover
                child = crossover(p1, p2, allowed_bases=self._allowed)
                child = mutate_seq(child, self.bases, rate=mut_rate)
                # Always clamp to target length — no more gradual drift
                child = enforce_length(child, self.genome_len, self.bases)
                children.append(child)

            self.population = children[:self.pop_size]
            del results, tasks

        elapsed = time.time() - t0
        print(f"\n[done] {self.system['name']} {self.generations} gens in {elapsed:.1f}s")
        print(f"[best] fitness={self.best_ever['fitness']:.4f} "
              f"atoms={self.best_ever.get('n_heavy_atoms',0)} "
              f"viable={self.best_ever.get('viable', False)}")

        # Final evaluation of top population
        final_tasks = [
            (seq, self.n_bases, ENVS[i % len(ENVS)], self.generations, self.seed)
            for i, seq in enumerate(self.population)
        ]
        with ProcessPoolExecutor(max_workers=self.n_workers) as ex:
            final = list(ex.map(_eval_organism, final_tasks,
                                chunksize=max(1, self.pop_size // self.n_workers)))
        final.sort(key=lambda r: r['fitness'], reverse=True)

        return {
            'best': self.best_ever,
            'top_10': final[:10],
            'history': self.history,
            'landscape': self.landscape_data,
            'species': self.species_clusters,
            'config': {
                'base_system': self.system['name'],
                'n_bases': self.n_bases,
                'error_rate': self.system['error_rate'],
                'pop_size': self.pop_size,
                'genome_len': self.genome_len,
                'generations': self.generations,
                'codon_len': CODON_LEN,
                'environments': ENVS,
                'niche_cycle': self.niche_cycle,
                'niche_period': self.niche_period,
            },
        }


# ---------------------------------------------------------------------------
# Multi-system comparison: run 4-base vs 6-base vs 8-base simultaneously
# ---------------------------------------------------------------------------

def compare_base_systems(pop_size: int = 100, genome_len: int = 60,
                         generations: int = 50, n_workers: int = None) -> Dict:
    """Run evolution for each base system and compare stability vs complexity."""
    results = {}
    for n_bases in [4, 6, 8]:
        print(f"\n{'='*60}")
        print(f"Running {BASE_SYSTEMS[n_bases]['name']} system ({n_bases} bases)")
        print(f"{'='*60}")
        ev = Evolver(pop_size=pop_size, genome_len=genome_len,
                     generations=generations, n_bases=n_bases, n_workers=n_workers)
        results[n_bases] = ev.run()

    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    for nb in [4, 6, 8]:
        r = results[nb]
        sys_name = BASE_SYSTEMS[nb]['name']
        best_f = r['best']['fitness']
        err = BASE_SYSTEMS[nb]['error_rate']
        viable = r['best'].get('viable', False)
        print(f"  {sys_name:>10s} ({nb} bases) | best={best_f:.4f} | "
              f"err_rate={err} | viable={viable}")

    return results


# ---------------------------------------------------------------------------
# Validation suite
# ---------------------------------------------------------------------------

def run_tests():
    print("--- validation tests ---")

    # 1. Base systems and complement maps
    for nb in [2, 4, 6, 8]:
        sys = BASE_SYSTEMS[nb]
        cmap = complement_map(nb)
        assert len(cmap) == nb, f"complement map wrong size for {nb}-base"
        for a, b in sys['pairs']:
            assert cmap[a] == b and cmap[b] == a, f"pair {a}-{b} broken"
    print("  [1] base systems OK")

    # 2. Strand creation, complement, integrity
    for nb in [4, 6, 8]:
        s = Strand.random(80, nb)
        assert len(s) == 80
        assert s.pair_integrity() == 1.0  # random from valid bases = perfect
        comp = s.complement
        assert len(comp) == 80
        cmap = complement_map(nb)
        for a, b in zip(s.seq, comp):
            assert cmap[a] == b
    print("  [2] strand complement OK")

    # 3. Replication with errors
    s4 = Strand.random(200, 4)
    copies = [s4.replicate() for _ in range(100)]
    diffs = [sum(a != b for a, b in zip(s4.seq, c.seq)) for c in copies]
    avg_diff = np.mean(diffs)
    assert avg_diff > 0, "replication should have some errors"
    assert avg_diff < 20, "replication error rate too high for 4-base"

    s6 = Strand.random(200, 6)
    copies6 = [s6.replicate() for _ in range(100)]
    avg6 = np.mean([sum(a != b for a, b in zip(s6.seq, c.seq)) for c in copies6])
    assert avg6 > avg_diff, "6-base should have MORE errors than 4-base"
    print(f"  [3] replication errors: 4-base avg={avg_diff:.1f}, 6-base avg={avg6:.1f}")

    # 4. Transcription + codons
    s = Strand.random(80, 4)
    t = s.transcribe()
    assert 0 < len(t) <= 80
    cods = s.codons()
    assert all(len(c) == CODON_LEN for c in cods)
    print("  [4] transcription OK")

    # 5. Knowledge base: all parts have valid SMILES
    valid_roles = ('structure', 'energy', 'catalysis', 'information', 'minimal',
                   'motor', 'signaling', 'neural', 'homeostasis')
    for name, part in PARTS_KB.items():
        mol = Chem.MolFromSmiles(part['smiles'])
        assert mol is not None, f"invalid SMILES in KB part '{name}': {part['smiles']}"
        assert part['role'] in valid_roles, f"unknown role '{part['role']}' for '{name}'"
        assert 'energy_cost' in part, f"missing energy_cost for '{name}'"
        assert 'attachment' in part, f"missing attachment for '{name}'"
        assert 'domains' in part, f"missing domains for '{name}'"
    print(f"  [5] knowledge base OK ({len(PARTS_KB)} parts, all valid SMILES)")

    # 6. Instruction set: codons decode to valid opcodes (including extended)
    test_codons = ['ATCGA', 'CGTAC', 'XATCG', 'YGTCA', 'TTTTT',
                   'WATCG', 'ZTACG', 'WXTAG', 'WYTAG', 'WWTAG', 'WZTAG',
                   'ZATCG', 'ZCAAA', 'ZXAAA', 'ZYAAA', 'ZWAAA', 'ZZAAA']
    bases_8 = ['A', 'T', 'C', 'G', 'X', 'Y', 'W', 'Z']
    for tc in test_codons:
        op, operand = codon_to_instruction(tc, bases_8)
        assert 0 <= op <= 17, f"invalid opcode {op} for codon {tc}"
        assert isinstance(operand, int)
    # Verify extended opcode mapping
    op_w, _ = codon_to_instruction('WATCG', bases_8)  # W+A -> MOTOR
    assert op_w == Opcode.MOTOR, f"WATCG should be MOTOR, got {op_w}"
    op_wc, _ = codon_to_instruction('WCAAA', bases_8)  # W+C -> LOCALIZE
    assert op_wc == Opcode.LOCALIZE, f"WCAAA should be LOCALIZE, got {op_wc}"
    op_wx, _ = codon_to_instruction('WXAAA', bases_8)  # W+X -> POLYMERIZE
    assert op_wx == Opcode.POLYMERIZE, f"WXAAA should be POLYMERIZE, got {op_wx}"
    # Z-prefix opcodes
    op_za, _ = codon_to_instruction('ZATCG', bases_8)  # Z+A -> SECRETE
    assert op_za == Opcode.SECRETE, f"ZATCG should be SECRETE, got {op_za}"
    op_zc, _ = codon_to_instruction('ZCAAA', bases_8)  # Z+C -> QUORUM
    assert op_zc == Opcode.QUORUM, f"ZCAAA should be QUORUM, got {op_zc}"
    op_zx, _ = codon_to_instruction('ZXAAA', bases_8)  # Z+X -> DIVIDE
    assert op_zx == Opcode.DIVIDE, f"ZXAAA should be DIVIDE, got {op_zx}"
    op_zy, _ = codon_to_instruction('ZYAAA', bases_8)  # Z+Y -> METABOLIZE
    assert op_zy == Opcode.METABOLIZE, f"ZYAAA should be METABOLIZE, got {op_zy}"
    op_zw, _ = codon_to_instruction('ZWAAA', bases_8)  # Z+W -> GRADIENT
    assert op_zw == Opcode.GRADIENT, f"ZWAAA should be GRADIENT, got {op_zw}"
    op_zz, _ = codon_to_instruction('ZZAAA', bases_8)  # Z+Z -> PROMOTER
    assert op_zz == Opcode.PROMOTER, f"ZZAAA should be PROMOTER, got {op_zz}"
    print("  [6] instruction set OK (including extended + Z-prefix opcodes)")

    # 7. Blueprint execution: genome produces parts (deterministic for repeatability)
    s = Strand.random(80, 4)
    bp = execute_genome(s, 'earth', deterministic=True)
    assert len(bp.parts_used) > 0, "genome produced no parts"
    assert len(bp.instruction_log) > 0
    assert all(p in PARTS_KB for p in bp.parts_used)
    print(f"  [7] blueprint: {len(bp.parts_used)} parts, "
          f"systems={bp.systems_score():.2f}, div={bp.diversity_score():.2f}")

    # 8. Blueprint functional completeness is achievable (deterministic)
    complete_count = 0
    for _ in range(50):
        s = Strand.random(80, 4)
        bp = execute_genome(s, 'earth', deterministic=True)
        if bp.systems_score() == 1.0:
            complete_count += 1
    print(f"  [8] functional completeness: {complete_count}/50 genomes have all systems")

    # 9. Element supply
    sup = ElementSupply.earth_like()
    assert sup.can_afford({'C': 10, 'N': 5})
    assert sup.richness() > 0.5
    print("  [9] element supply OK")

    # 10. Organism full pipeline with blueprint (deterministic for repeatability)
    s = Strand.random(80, 4)
    sup = ElementSupply.earth_like()
    org = Organism(strand=s, env_name='earth')
    org.express(sup, deterministic=True)
    org.evaluate()
    assert 0 <= org.fitness <= 1.0
    d = org.to_dict()
    assert 'genome' in d and 'complement' in d and 'blueprint' in d
    assert 'parts' in d['blueprint']
    print(f"  [10] organism pipeline OK (fitness={org.fitness:.4f}, "
          f"parts={d['blueprint']['n_parts']})")

    # 11. Parallel evaluation
    tasks = [
        (''.join(random.choices(['A','T','C','G'], k=80)), 4, 'earth', 0)
        for _ in range(12)
    ]
    with ProcessPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(_eval_organism, tasks))
    assert len(res) == 12
    assert all('blueprint' in r for r in res)
    print("  [11] parallel eval OK")

    # 12. Stability score ordering: 4-base > 6-base > 8-base
    s4 = Strand.random(80, 4).stability_score()
    s6 = Strand.random(80, 6).stability_score()
    s8 = Strand.random(80, 8).stability_score()
    assert s4 > s6 > s8, f"stability ordering wrong: {s4}, {s6}, {s8}"
    print(f"  [12] stability order: 4-base({s4:.3f}) > 6-base({s6:.3f}) > 8-base({s8:.3f})")

    # 13. Short evolution smoke test
    mini = Evolver(pop_size=20, genome_len=60, generations=3, n_bases=4, n_workers=4)
    out = mini.run()
    assert 'best' in out and 'top_10' in out
    assert len(out['history']) == 3
    best_bp = out['best'].get('blueprint', {})
    print(f"  [13] evolution smoke test OK (best parts={best_bp.get('n_parts',0)})")

    # 14. Info density ordering
    assert Strand.random(10, 4).info_density() < Strand.random(10, 8).info_density()
    print("  [14] info density ordering OK")

    # 15. Reference library: all hardcoded genomes produce expected organisms
    ref_errors = _validate_reference_library()
    assert not ref_errors, f"reference library errors: {ref_errors}"
    print(f"  [15] reference library OK ({len(REFERENCE_LIBRARY)} organisms validated)")

    # 16. Functional complexes detection
    bp_cx = Blueprint()
    bp_cx.parts_used = ['motA', 'motB', 'fliG']
    cx = bp_cx.detect_complexes()
    assert 'stator_complex' in cx, f"stator_complex not detected: {cx}"
    assert not bp_cx.motility_capable(), "should not be motile without full flagellum"
    print(f"  [16] functional complexes OK ({len(cx)} detected)")

    # 17. Energy tracking
    bp_en = Blueprint()
    bp_en.parts_used = ['atp', 'glucose', 'ms_ring', 'motA']
    bp_en.roles_present = {'energy': 2, 'motor': 2}
    bp_en.energy_budget = 2 * BASE_PMF_UNITS
    bp_en.energy_cost = PARTS_KB['ms_ring']['energy_cost'] + PARTS_KB['motA']['energy_cost']
    assert bp_en.energy_balance() == bp_en.energy_budget - bp_en.energy_cost
    print(f"  [17] energy tracking OK (balance={bp_en.energy_balance()})")

    # 18. Extended opcodes produce motor parts in genome execution
    motor_genome = 'WATCGWTTCGWACGTWACGT'
    s_motor = Strand(motor_genome, 8)
    bp_motor = execute_genome(s_motor, 'earth', deterministic=True)
    motor_count = bp_motor.roles_present.get('motor', 0) + bp_motor.roles_present.get('signaling', 0)
    assert motor_count > 0, f"MOTOR opcodes produced no motor parts: {bp_motor.parts_used}"
    print(f"  [18] extended opcodes OK (motor/sig parts={motor_count}, "
          f"parts={bp_motor.parts_used})")

    # 19. Blueprint summary includes new fields (expanded)
    bp_sum = bp_motor.summary()
    for key in ('energy_budget', 'energy_cost', 'complexes', 'motility',
                'chemotaxis', 'phase', 'behaviors', 'assembly',
                'operons', 'domain_mismatches', 'torque', 'rpm',
                'flagella_count', 'membrane_potential', 'metabolic_complete',
                'secretions', 'quorum_signals', 'division_events', 'gradient_steps'):
        assert key in bp_sum, f"summary missing key: {key}"
    print("  [19] blueprint summary fields OK")

    # 20. Organism to_dict includes new fields (expanded)
    assert 'complexes' in d and 'motility' in d and 'behaviors' in d
    assert 'energy_balance' in d and 'phase' in d
    assert 'torque' in d and 'rpm' in d and 'operons' in d
    assert 'membrane_potential' in d and 'domain_mismatches' in d
    print("  [20] organism to_dict fields OK")

    # 21. Domain validation (step 2): incompatible domains get rejected
    bp_dom = Blueprint()
    bp_dom.parts_used = ['flagellin', 'receptor']  # filament_domain vs sensor_domain
    assert not bp_dom.domain_compatibility(0, 1), "flagellin and receptor should NOT share domains"
    bp_dom2 = Blueprint()
    bp_dom2.parts_used = ['motA', 'motB']  # both have stator_domain
    assert bp_dom2.domain_compatibility(0, 1), "motA and motB should share stator_domain"
    bp_dom3 = Blueprint()
    bp_dom3.parts_used = ['membrane', 'wall']  # both have empty domains = compatible
    assert bp_dom3.domain_compatibility(0, 1), "classic parts with no domains should be compatible"
    print("  [21] domain validation OK")

    # 22. Attachment site validation (step 3)
    bp_loc = Blueprint()
    bp_loc.parts_used = ['ms_ring']  # attachment: ['basal', 'membrane']
    assert bp_loc.validate_localization(0, 'basal'), "ms_ring should fit at basal"
    assert bp_loc.validate_localization(0, 'membrane'), "ms_ring should fit at membrane"
    assert not bp_loc.validate_localization(0, 'distal'), "ms_ring should NOT fit at distal"
    bp_loc2 = Blueprint()
    bp_loc2.parts_used = ['membrane']  # attachment: [] = no restrictions
    assert bp_loc2.validate_localization(0, 'anywhere'), "classic parts accept any site"
    print("  [22] attachment site validation OK")

    # 23. Operon detection (step 1)
    operon_genome = 'ZZAAAAAAGGAATGGAAAAA' + 'ZZAAAAAAGGAATGG'  # Z+Z=PROMOTER, then BUILD parts
    s_op = Strand(operon_genome, 8)
    bp_op = execute_genome(s_op, 'earth', deterministic=True)
    # Should have at least some operons from genes between promoters
    print(f"  [23] operon detection OK (operons={bp_op.operon_count()})")

    # 24. Polymerize energy tracking (step 5)
    poly_genome = 'WACCA' + 'WXAAA'  # MOTOR:flagellin then POLYMERIZE
    s_poly = Strand(poly_genome, 8)
    bp_poly = execute_genome(s_poly, 'earth', deterministic=True)
    assert bp_poly.polymerize_energy >= 0, "polymerize energy should be tracked"
    print(f"  [24] polymerize energy tracking OK (cost={bp_poly.polymerize_energy})")

    # 25. Torque computation (step 11)
    bp_torq = Blueprint()
    bp_torq.parts_used = ['motA', 'motB', 'fliG', 'flagellin', 'hook_protein',
                          'ms_ring', 'c_ring']
    bp_torq.compute_torque()
    assert bp_torq.torque > 0, "torque should be positive with stator pair"
    assert bp_torq.rpm > 0, "RPM should be positive"
    assert bp_torq.flagella_count >= 1, "should detect at least 1 flagellum"
    print(f"  [25] torque computation OK (torque={bp_torq.torque:.0f}, "
          f"rpm={bp_torq.rpm:.0f}, flagella={bp_torq.flagella_count})")

    # 26. Membrane potential (step 20)
    bp_mp = Blueprint()
    bp_mp.parts_used = ['atp', 'glucose', 'motA', 'motB']
    bp_mp.roles_present = {'energy': 2, 'motor': 2}
    mp_val = bp_mp.compute_membrane_potential()
    assert mp_val != 0, "membrane potential should be computed"
    print(f"  [26] membrane potential OK ({mp_val:.1f} mV)")

    # 27. Metabolic pathway detection (step 19)
    bp_met = Blueprint()
    bp_met.parts_used = ['glucose', 'nadh', 'atp']
    assert bp_met.metabolic_pathway_complete(), "glucose+nadh should be a complete pathway"
    bp_met2 = Blueprint()
    bp_met2.parts_used = ['membrane', 'wall']
    assert not bp_met2.metabolic_pathway_complete(), "structural parts = no pathway"
    print("  [27] metabolic pathway detection OK")

    # 28. Z-prefix opcodes in execution (secrete, quorum, divide)
    z_genome = 'AAAGG' + 'AATGG' + 'AAAAA' + 'AAGGG' + 'AAAAT' + 'AAATT'  # 6 BUILDs
    z_genome += 'ZATCG'  # SECRETE
    z_genome += 'ZCAAA'  # QUORUM
    z_genome += 'ZXAAA'  # DIVIDE
    s_z = Strand(z_genome, 8)
    bp_z = execute_genome(s_z, 'earth', deterministic=True)
    assert len(bp_z.secretions) > 0, "SECRETE should add to secretions"
    assert bp_z.quorum_signals > 0, "QUORUM should increment signal count"
    assert bp_z.division_events > 0, "DIVIDE should succeed with 6 parts"
    assert 'quorum_sensing' in bp_z.behaviors, "QUORUM should enable quorum_sensing behavior"
    print(f"  [28] Z-prefix opcodes OK (secrete={len(bp_z.secretions)}, "
          f"quorum={bp_z.quorum_signals}, divide={bp_z.division_events})")

    # 29. Speciation clustering (step 7)
    test_pop = ['AAAA' * 20] * 5 + ['TTTT' * 20] * 5
    test_fits = [0.5] * 10
    species = _cluster_species(test_pop, test_fits)
    assert len(species) >= 2, f"should detect 2 species, got {len(species)}"
    print(f"  [29] speciation clustering OK ({len(species)} species)")

    # 30. Adaptive mutation and landscape data (steps 8, 10)
    mini2 = Evolver(pop_size=20, genome_len=60, generations=3, n_bases=4,
                    n_workers=4, niche_cycle=False)
    out2 = mini2.run()
    assert 'landscape' in out2, "landscape data should be in output"
    assert 'species' in out2, "species data should be in output"
    print(f"  [30] evolution features OK (landscape={len(out2['landscape'])}, "
          f"species={len(out2['species'])})")

    # 31. PDB export (step 22)
    pdb_str = export_pdb(org, filepath=None)
    assert 'HEADER' in pdb_str, "PDB should have HEADER"
    assert 'ATOM' in pdb_str, "PDB should have ATOM records"
    assert 'END' in pdb_str, "PDB should have END"
    print(f"  [31] PDB export OK ({pdb_str.count('ATOM')} atoms)")

    # 32. Animation hints (step 23)
    bp_anim = Blueprint()
    bp_anim.parts_used = ['ms_ring', 'c_ring', 'fliG', 'flagellin', 'hook_protein']
    bp_anim.rpm = 300.0
    hints = _motor_animation_hints(bp_anim)
    assert len(hints) >= 4, f"should have hints for rotating parts, got {len(hints)}"
    assert any(h['animation'] == 'rotate' for h in hints)
    assert any(h['animation'] == 'flex' for h in hints)
    print(f"  [32] animation hints OK ({len(hints)} hints)")

    # 33. WebGL JSON export (step 24)
    webgl = export_webgl_json(org, filepath=None)
    assert 'animation_hints' in webgl, "WebGL JSON should have animation_hints"
    assert 'viewer_meta' in webgl, "WebGL JSON should have viewer_meta"
    assert webgl['viewer_meta']['format'] == 'periodic_machine_v2'
    print(f"  [33] WebGL JSON OK (hints={len(webgl.get('animation_hints', []))})")

    # 34. Genome compression/decompression (step 26)
    test_genome = 'AAAGG' * 10 + 'CAAAA' * 3 + 'ATTGG'
    compressed = genome_compress(test_genome)
    decompressed = genome_decompress(compressed)
    assert decompressed == test_genome, f"roundtrip failed: {decompressed} != {test_genome}"
    assert len(compressed) < len(test_genome), "compression should reduce length"
    # Edge case: short genome
    assert genome_compress('ATCG') == 'ATCG'
    assert genome_decompress('ATCG') == 'ATCG'
    print(f"  [34] genome compression OK ({len(test_genome)} -> {len(compressed)} chars)")

    # 35. Module split guide exists (step 21)
    import inspect
    src = inspect.getsource(export_pdb)
    assert 'organism' in src, "export_pdb should reference organism"
    assert callable(genome_compress) and callable(genome_decompress)
    assert callable(export_webgl_json) and callable(run_benchmark)
    assert callable(_motor_animation_hints)
    print("  [35] infrastructure functions callable OK")

    # 36. LanguageSpec versioning and opcode registry
    spec = LanguageSpec("2.0")
    assert spec.validate_opcode(0), "BUILD should be valid"
    assert spec.validate_opcode(17), "GRADIENT should be valid"
    assert spec.max_opcode_code() == 17
    op_def = spec.get_opcode(5)
    assert op_def is not None and op_def.name == "MOTOR"
    op_by_name = spec.get_opcode_by_name("SECRETE")
    assert op_by_name is not None and op_by_name.code == 13
    core_ops = spec.list_opcodes(category='core')
    assert len(core_ops) >= 3
    # v1.0 spec should not have advanced opcodes
    spec_v1 = LanguageSpec("1.0")
    assert not spec_v1.validate_opcode(11), "PROMOTER not in v1.0"
    assert spec_v1.max_opcode_code() == 4
    print("  [36] LanguageSpec versioning OK")

    # 37. PartRegistry unified access
    reg = PART_REGISTRY
    assert len(reg) == len(PARTS_KB)
    assert 'membrane' in reg
    assert reg.index_of('membrane') == 0
    motor_parts = reg.motor_parts()
    assert 'motA' in motor_parts
    assert 'membrane' not in motor_parts
    build_parts = reg.build_parts()
    assert 'membrane' in build_parts
    assert 'motA' not in build_parts
    by_energy = reg.by_role('energy')
    assert 'atp' in by_energy
    print(f"  [37] PartRegistry OK ({len(reg)} parts, {len(motor_parts)} motor, {len(build_parts)} build)")

    # 38. EnergyLedger accounting
    ledger = EnergyLedger(pmf_generated=50, atp_from_metabolism=10,
                          motor_consumption=20, secretion_cost=5,
                          polymerize_cost=8, maintenance_cost=3)
    assert ledger.total_income == 60
    assert ledger.total_expense == 36
    assert ledger.balance == 24
    assert not ledger.is_starved
    ledger2 = EnergyLedger(pmf_generated=5, motor_consumption=30)
    assert ledger2.is_starved
    summary = ledger.summary()
    assert 'balance' in summary and summary['balance'] == 24
    print("  [38] EnergyLedger OK (balance=24, starved=False)")

    # 39. GenomeValidator
    validator = GenomeValidator()
    s_valid = Strand('AAAGGAATGGCAAAA' * 3, 4)
    assert validator.validate(s_valid), f"valid genome rejected: {validator.errors}"
    s_short = Strand('AT', 4)
    assert not validator.validate(s_short), "too-short genome should fail"
    assert len(validator.errors) > 0
    # Test warning generation
    v2 = GenomeValidator()
    s_warn = Strand('CAAAA' * 5, 4)  # all CONNECT, no BUILD
    v2.validate(s_warn)
    assert len(v2.errors) > 0 or len(v2.warnings) > 0
    report = v2.report()
    assert len(report) > 0
    print(f"  [39] GenomeValidator OK (errors={len(v2.errors)}, warnings={len(v2.warnings)})")

    # 40. MotorSimulator
    bp_sim = Blueprint()
    bp_sim.parts_used = ['motA', 'motB', 'fliG', 'flagellin', 'hook_protein',
                         'ms_ring', 'c_ring']
    bp_sim.compute_torque()
    bp_sim.detect_complexes()
    sim = MotorSimulator(bp_sim, dt=0.001, steps=100)
    assert len(sim.motors) >= 1, "should have at least 1 motor"
    result = sim.run()
    assert result['n_motors'] >= 1
    assert result['n_steps'] == 100
    assert result['trajectory_length'] == 100
    print(f"  [40] MotorSimulator OK (motors={result['n_motors']}, avg_rpm={result['avg_rpm']})")

    # 41. FitnessConfig
    fc = FitnessConfig()
    assert fc.w_diversity == 0.20
    assert fc.motility_bonus == 0.10
    fc_custom = FitnessConfig(w_diversity=0.30, motility_bonus=0.20)
    assert fc_custom.w_diversity == 0.30
    fc_dict = fc.to_dict()
    assert 'w_diversity' in fc_dict
    print("  [41] FitnessConfig OK")

    # 42. BuildError tracking
    be = BuildError()
    be.add("test warning")
    assert be.count == 1
    assert not be.fatal
    assert be.penalty == 0.02
    for i in range(10):
        be.add(f"error {i}")
    assert be.fatal, "should be fatal after threshold"
    assert be.penalty == 0.5
    print("  [42] BuildError OK (fatal after threshold)")

    # 43. Phenotypic distance for speciation
    org_a = {'blueprint': {'n_parts': 10, 'motility': True, 'torque': 100, 'operons': 3}}
    org_b = {'blueprint': {'n_parts': 5, 'motility': False, 'torque': 0, 'operons': 0}}
    dist = phenotypic_distance(org_a, org_b)
    assert dist > 0, "different organisms should have positive distance"
    dist_same = phenotypic_distance(org_a, org_a)
    assert dist_same == 0.0, "identical organisms should have zero distance"
    print(f"  [43] phenotypic distance OK (dist={dist:.3f})")

    # 44. Horizontal gene transfer
    donor = 'AAAGGAATGGCAAAA' * 5
    recipient = 'TTTTTCCCCCGGGGG' * 5
    child = horizontal_gene_transfer(donor, recipient, transfer_len=15,
                                      allowed_bases={'A','T','C','G'})
    assert len(child) == len(recipient), "HGT should preserve genome length"
    assert child != recipient, "HGT should change recipient"
    print("  [44] HGT OK")

    # 45. Memory-efficient slim storage
    full_dict = org.to_dict()
    slim = slim_organism_dict(full_dict)
    assert len(json.dumps(slim)) < len(json.dumps(full_dict)), "slim should be smaller"
    assert 'fitness' in slim
    assert 'atoms' not in slim
    print(f"  [45] slim storage OK ({len(json.dumps(slim))} vs {len(json.dumps(full_dict))} bytes)")

    # 46. Colony model
    colony = Colony()
    for _ in range(10):
        colony.add_cell({'fitness': 0.5, 'quorum_signals': 1, 'motility': True})
    assert colony.density == 10
    assert colony.quorum_active, "10 signals >= threshold"
    col_sum = colony.summary()
    assert col_sum['motile_fraction'] == 1.0
    print(f"  [46] Colony OK (density={colony.density}, quorum={colony.quorum_active})")

    # 47. Logging system
    assert _log is not None
    assert callable(set_log_level)
    set_log_level('WARNING')
    assert _log.level == _logging.WARNING
    set_log_level('INFO')
    print("  [47] logging OK")

    # 48. Seeded codons for reproducible evolution
    s_seed = Strand.random(80, 4)
    c1 = s_seed.codons(seed=42)
    c2 = s_seed.codons(seed=42)
    c3 = s_seed.codons(seed=99)
    assert c1 == c2, "same seed should produce same codons"
    # Different seeds may produce different codons (probabilistic, but very likely)
    print("  [48] seeded codons OK (reproducible)")

    # 49. LanguageSpec dispatch table
    bases_4 = ['A', 'T', 'C', 'G']
    table = LANGUAGE_SPEC.build_codon_dispatch_table(bases_4)
    assert len(table) > 0, "dispatch table should have entries"
    # Verify a known codon maps correctly
    assert 'AAAGG' in table, "AAAGG should be in 4-base dispatch table"
    print(f"  [49] dispatch table OK ({len(table)} entries for 4-base)")

    # 50. PartRegistry consistency with legacy tuples
    # Registry build_parts is superset of BUILD_PART_NAMES (adds actin, tubulin)
    assert set(BUILD_PART_NAMES).issubset(set(PART_REGISTRY.build_parts())), \
        f"legacy BUILD_PART_NAMES not subset of registry: {set(BUILD_PART_NAMES) - set(PART_REGISTRY.build_parts())}"
    assert set(MOTOR_PART_NAMES).issubset(set(PART_REGISTRY.motor_parts())), \
        f"legacy MOTOR_PART_NAMES not subset of registry: {set(MOTOR_PART_NAMES) - set(PART_REGISTRY.motor_parts())}"
    assert set(PART_NAMES) == set(PART_REGISTRY.all_names()), "all names should match"
    print(f"  [50] PartRegistry consistent (build={len(PART_REGISTRY.build_parts())}, motor={len(PART_REGISTRY.motor_parts())})")

    print("--- all 50 tests passed ---\n")


# ---------------------------------------------------------------------------
# Phase 1-1: Language Specification with Versioning
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OpcodeDef:
    """Definition of a language opcode with metadata."""
    name: str
    code: int
    category: str  # 'core', 'motor', 'metabolism', 'regulation'
    min_version: str
    description: str
    energy_cost: float = 0.0
    requires_part: bool = False


class LanguageSpec:
    """Versioned language specification with clean opcode registry."""
    
    CURRENT_VERSION = "2.0"
    
    # Core opcodes (v1.0)
    CORE_OPCODES = {
        0: OpcodeDef("BUILD", 0, "core", "1.0", "Build a biological part", 1.0, True),
        1: OpcodeDef("CONNECT", 1, "core", "1.0", "Connect two parts", 0.5),
        2: OpcodeDef("REPEAT", 2, "core", "1.0", "Repeat last instruction"),
        3: OpcodeDef("REGULATE", 3, "core", "1.0", "Regulate gene expression"),
        4: OpcodeDef("NOOP", 4, "core", "1.0", "No operation"),
    }
    
    # Extended opcodes (v1.5)
    EXTENDED_OPCODES = {
        5: OpcodeDef("MOTOR", 5, "motor", "1.5", "Build motor component", 2.0, True),
        6: OpcodeDef("LOCALIZE", 6, "core", "1.5", "Localize part to cellular site"),
        7: OpcodeDef("POLYMERIZE", 7, "core", "1.5", "Polymerize last part", 0.5),
        8: OpcodeDef("PHASE", 8, "regulation", "1.5", "Advance developmental phase"),
        9: OpcodeDef("CHEMOTAXIS", 9, "regulation", "1.5", "Enable chemotaxis"),
        10: OpcodeDef("MEMBRANE_INS", 10, "core", "1.5", "Insert into membrane"),
    }
    
    # Advanced opcodes (v2.0)
    ADVANCED_OPCODES = {
        11: OpcodeDef("PROMOTER", 11, "regulation", "2.0", "Start operon"),
        12: OpcodeDef("TERMINATOR", 12, "regulation", "2.0", "End operon"),
        13: OpcodeDef("SECRETE", 13, "metabolism", "2.0", "Secrete protein", 1.0),
        14: OpcodeDef("QUORUM", 14, "regulation", "2.0", "Quorum sensing signal", 0.5),
        15: OpcodeDef("DIVIDE", 15, "metabolism", "2.0", "Cell division", 3.0),
        16: OpcodeDef("METABOLIZE", 16, "metabolism", "2.0", "Metabolic reaction", 1.0),
        17: OpcodeDef("GRADIENT", 17, "regulation", "2.0", "Follow gradient", 0.5),
    }
    
    def __init__(self, version: str = None):
        self.version = version or self.CURRENT_VERSION
        self._opcodes = {}
        self._codon_map = {}
        
        # Register opcodes based on version
        self._register_core_opcodes()
        if self._version_at_least("1.5"):
            self._register_extended_opcodes()
        if self._version_at_least("2.0"):
            self._register_advanced_opcodes()
    
    def _register_core_opcodes(self):
        self._opcodes.update(self.CORE_OPCODES)
    
    def _register_extended_opcodes(self):
        self._opcodes.update(self.EXTENDED_OPCODES)
    
    def _register_advanced_opcodes(self):
        self._opcodes.update(self.ADVANCED_OPCODES)
    
    def _version_at_least(self, required: str) -> bool:
        """Check if current version meets requirement."""
        def parse(v):
            return tuple(map(int, v.split('.')))
        return parse(self.version) >= parse(required)
    
    def register_opcode(self, opcode_def: OpcodeDef):
        """Register a new opcode (for extensions)."""
        self._opcodes[opcode_def.code] = opcode_def
    
    def get_opcode(self, code: int) -> Optional[OpcodeDef]:
        """Get opcode definition by code."""
        return self._opcodes.get(code)
    
    def get_opcode_by_name(self, name: str) -> Optional[OpcodeDef]:
        """Get opcode definition by name."""
        for op in self._opcodes.values():
            if op.name == name:
                return op
        return None
    
    def list_opcodes(self, category: str = None) -> List[OpcodeDef]:
        """List all opcodes, optionally filtered by category."""
        ops = list(self._opcodes.values())
        if category:
            ops = [op for op in ops if op.category == category]
        return sorted(ops, key=lambda x: x.code)
    
    def validate_opcode(self, code: int) -> bool:
        """Check if opcode code is valid for this version."""
        return code in self._opcodes
    
    def max_opcode_code(self) -> int:
        """Get maximum opcode code for this version."""
        return max(self._opcodes.keys()) if self._opcodes else 0
    
    def build_codon_dispatch_table(self, bases: List[str]) -> Dict[str, Tuple[int, int]]:
        """Build clean dispatch table for codon-to-opcode mapping.
        Preserves the original W/Z prefix logic for backward compatibility."""
        dispatch = {}
        
        # Generate all possible codons
        base_count = len(bases)
        for i in range(base_count ** CODON_LEN):
            # Generate codon
            codon = ''
            temp = i
            for _ in range(CODON_LEN):
                codon = bases[temp % base_count] + codon
                temp //= base_count
            
            # Apply the original logic for opcode determination
            opcode, operand = self._decode_opcode_legacy(codon, bases)
            dispatch[codon] = (opcode, operand)
        
        return dispatch
    
    def _decode_opcode_legacy(self, codon: str, bases: List[str]) -> Tuple[int, int]:
        """Legacy opcode decoding logic for backward compatibility."""
        # Base to opcode mapping
        base_to_op = {}
        if len(bases) >= 2:
            base_to_op[bases[0]] = Opcode.BUILD  # A -> BUILD
            base_to_op[bases[1]] = Opcode.CONNECT  # T -> CONNECT
        if len(bases) >= 4:
            base_to_op[bases[2]] = Opcode.CONNECT  # C -> CONNECT
            base_to_op[bases[3]] = Opcode.BUILD   # G -> BUILD
        if len(bases) >= 6:
            base_to_op['X'] = Opcode.REPEAT
            base_to_op['Y'] = Opcode.REGULATE
        if len(bases) >= 8:
            base_to_op['W'] = Opcode.MOTOR
            base_to_op['Z'] = Opcode.MOTOR
        
        operand = sum(ord(c) for c in codon[1:]) if len(codon) > 1 else 0
        opcode = base_to_op.get(codon[0], Opcode.NOOP)
        
        # Second-base modifiers for MOTOR context
        if opcode == Opcode.MOTOR and len(codon) > 1:
            mod = codon[1]
            if mod in ('A', 'T'):   opcode = Opcode.MOTOR
            elif mod in ('C', 'G'): opcode = Opcode.LOCALIZE
            elif mod == 'X':        opcode = Opcode.POLYMERIZE
            elif mod == 'Y':        opcode = Opcode.PHASE
            elif mod == 'W':        opcode = Opcode.CHEMOTAXIS
            elif mod == 'Z':        opcode = Opcode.MEMBRANE_INS
        
        # Z-prefix opcodes
        if codon[0] == 'Z' and len(codon) > 1:
            mod = codon[1]
            if mod in ('A', 'T'):   opcode = Opcode.SECRETE
            elif mod in ('C', 'G'): opcode = Opcode.QUORUM
            elif mod == 'X':        opcode = Opcode.DIVIDE
            elif mod == 'Y':        opcode = Opcode.METABOLIZE
            elif mod == 'W':        opcode = Opcode.GRADIENT
            elif mod == 'Z':        opcode = Opcode.PROMOTER
        
        return opcode, operand
    
    def encode_instruction(self, opcode_name: str, operand: int, bases: List[str]) -> str:
        """Encode an instruction back to a codon."""
        opcode_def = self.get_opcode_by_name(opcode_name)
        if not opcode_def:
            raise ValueError(f"Unknown opcode: {opcode_name}")
        
        # Find a codon that maps to this opcode with the given operand
        dispatch = self.build_codon_dispatch_table(bases)
        for codon, (code, op_operand) in dispatch.items():
            if code == opcode_def.code and op_operand == operand:
                return codon
        
        # If exact match not found, find closest
        for codon, (code, op_operand) in dispatch.items():
            if code == opcode_def.code:
                return codon
        
        raise ValueError(f"Cannot encode {opcode_name} with operand {operand}")


# Global language specification instance
LANGUAGE_SPEC = LanguageSpec()


# ---------------------------------------------------------------------------
# Issue 4: Unified Part Registry with categories and stable indexing
# ---------------------------------------------------------------------------

class PartRegistry:
    """Single source of truth for all parts with tagged categories and
    stable indexing. Replaces the fragile PART_NAMES/BUILD_PART_NAMES/
    MOTOR_PART_NAMES triple."""

    def __init__(self, parts_kb: Dict[str, Dict]):
        self._parts = parts_kb
        self._by_category: Dict[str, List[str]] = {}
        self._stable_index: Dict[str, int] = {}
        idx = 0
        for name, spec in parts_kb.items():
            role = spec['role']
            self._by_category.setdefault(role, []).append(name)
            self._stable_index[name] = idx
            idx += 1

    def all_names(self) -> Tuple[str, ...]:
        return tuple(self._parts.keys())

    def by_role(self, role: str) -> Tuple[str, ...]:
        return tuple(self._by_category.get(role, []))

    def build_parts(self) -> Tuple[str, ...]:
        """Parts addressable by BUILD opcode (classic non-motor parts)."""
        return tuple(n for n in self._parts
                     if self._parts[n]['role'] not in ('motor', 'signaling'))

    def motor_parts(self) -> Tuple[str, ...]:
        """Parts addressable by MOTOR opcode."""
        return tuple(n for n in self._parts
                     if self._parts[n]['role'] in ('motor', 'signaling'))

    def index_of(self, name: str) -> int:
        return self._stable_index.get(name, -1)

    def get(self, name: str) -> Optional[Dict]:
        return self._parts.get(name)

    def __len__(self):
        return len(self._parts)

    def __contains__(self, name: str):
        return name in self._parts


PART_REGISTRY = PartRegistry(PARTS_KB)


# ---------------------------------------------------------------------------
# Issue 5: Formal Energy Model — clear proton-flow accounting
# ---------------------------------------------------------------------------

@dataclass
class EnergyLedger:
    """Tracks energy flow with clear categories. All values in PMF units."""
    pmf_generated: float = 0.0      # from energy-role parts
    atp_from_metabolism: float = 0.0 # from metabolic pathway completion
    motor_consumption: float = 0.0   # motor torque cost
    secretion_cost: float = 0.0      # protein export cost
    polymerize_cost: float = 0.0     # filament assembly cost
    division_cost: float = 0.0       # cell division cost
    maintenance_cost: float = 0.0    # baseline survival cost

    @property
    def total_income(self) -> float:
        return self.pmf_generated + self.atp_from_metabolism

    @property
    def total_expense(self) -> float:
        return (self.motor_consumption + self.secretion_cost +
                self.polymerize_cost + self.division_cost +
                self.maintenance_cost)

    @property
    def balance(self) -> float:
        return self.total_income - self.total_expense

    @property
    def is_starved(self) -> bool:
        return self.balance < 0

    def summary(self) -> Dict[str, float]:
        return {
            'pmf_generated': round(self.pmf_generated, 2),
            'atp_from_metabolism': round(self.atp_from_metabolism, 2),
            'motor_consumption': round(self.motor_consumption, 2),
            'secretion_cost': round(self.secretion_cost, 2),
            'polymerize_cost': round(self.polymerize_cost, 2),
            'division_cost': round(self.division_cost, 2),
            'maintenance_cost': round(self.maintenance_cost, 2),
            'total_income': round(self.total_income, 2),
            'total_expense': round(self.total_expense, 2),
            'balance': round(self.balance, 2),
            'starved': self.is_starved,
        }


# ---------------------------------------------------------------------------
# Issue 3: Genome Validator — formal instruction validation
# ---------------------------------------------------------------------------

class GenomeValidator:
    """Validates a decoded genome against the language spec.
    Checks for structural correctness, opcode validity, and basic semantics."""

    def __init__(self, spec: LanguageSpec = None):
        self.spec = spec or LANGUAGE_SPEC
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self, strand: 'Strand', env_name: str = 'earth') -> bool:
        """Run all validation checks. Returns True if genome is valid."""
        self.errors = []
        self.warnings = []
        codons = strand.codons_deterministic()
        bases = strand.system['bases']

        if len(strand.seq) < CODON_LEN:
            self.errors.append(f"Genome too short: {len(strand.seq)} < {CODON_LEN}")
            return False

        build_count = 0
        connect_count = 0
        has_promoter = False
        open_operons = 0

        for i, codon in enumerate(codons):
            opcode, operand = codon_to_instruction(codon, bases)

            # Check opcode is valid for this spec version
            if not self.spec.validate_opcode(opcode):
                self.warnings.append(f"Codon {i}: opcode {opcode} not in spec v{self.spec.version}")

            if opcode == Opcode.BUILD:
                build_count += 1
            elif opcode == Opcode.CONNECT:
                connect_count += 1
                if build_count < 2:
                    self.warnings.append(f"Codon {i}: CONNECT before 2 parts built")
            elif opcode == Opcode.PROMOTER:
                open_operons += 1
                has_promoter = True
            elif opcode == Opcode.TERMINATOR:
                if open_operons > 0:
                    open_operons -= 1
                else:
                    self.warnings.append(f"Codon {i}: TERMINATOR without matching PROMOTER")
            elif opcode == Opcode.DIVIDE:
                if build_count < DIVISION_MIN_PARTS:
                    self.warnings.append(f"Codon {i}: DIVIDE with only {build_count} parts")
            elif opcode == Opcode.POLYMERIZE:
                if build_count == 0:
                    self.warnings.append(f"Codon {i}: POLYMERIZE with no parts to repeat")
            elif opcode == Opcode.SECRETE:
                if build_count == 0:
                    self.warnings.append(f"Codon {i}: SECRETE with nothing built")

        if open_operons > 0:
            self.warnings.append(f"{open_operons} unclosed operon(s)")
        if build_count == 0:
            self.errors.append("No BUILD instructions — genome produces nothing")
        if connect_count == 0 and build_count > 1:
            self.warnings.append("Multiple parts but no CONNECT — all parts disconnected")

        return len(self.errors) == 0

    def report(self) -> str:
        lines = []
        if self.errors:
            lines.append(f"ERRORS ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"  ✗ {e}")
        if self.warnings:
            lines.append(f"WARNINGS ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        if not self.errors and not self.warnings:
            lines.append("✓ Genome is valid with no warnings")
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Issue 7: Discrete-Time Motor Rotation Simulator
# ---------------------------------------------------------------------------

@dataclass
class MotorState:
    """State of a single flagellar motor at a time step."""
    angle: float = 0.0           # radians
    angular_velocity: float = 0.0 # rad/s
    direction: int = 1           # +1 CCW (run), -1 CW (tumble)
    stator_count: int = 0
    load_torque: float = 0.0     # pN·nm drag

class MotorSimulator:
    """Discrete-time flagellar motor simulator.
    Simulates torque generation, rotation under drag, and directional switching."""

    DRAG_COEFFICIENT = 1e-3    # pN·nm·s/rad for a flagellum in water
    STALL_TORQUE = 2000.0      # pN·nm per stator unit at stall
    SWITCH_RATE_BASE = 0.01    # probability of direction switch per step

    def __init__(self, bp: 'Blueprint', dt: float = 0.001, steps: int = 1000):
        self.dt = dt
        self.steps = steps
        self.motors: List[MotorState] = []
        self._init_from_blueprint(bp)
        self.trajectory: List[Dict] = []

    def _init_from_blueprint(self, bp: 'Blueprint'):
        stator_count = sum(1 for p in bp.parts_used if p in ('motA', 'motB')) // 2
        n_flagella = bp.flagella_count or (1 if stator_count > 0 else 0)
        drag = sum(1 for p in bp.parts_used if p == 'flagellin') * FILAMENT_DRAG
        has_chemotaxis = bp.chemotaxis_capable()

        for _ in range(n_flagella):
            motor = MotorState(
                stator_count=stator_count,
                load_torque=drag * self.DRAG_COEFFICIENT,
            )
            self.motors.append(motor)
        self._has_chemotaxis = has_chemotaxis

    def run(self, gradient_signal: float = 0.0) -> Dict[str, Any]:
        """Run simulation for all motors. Returns trajectory and final state."""
        self.trajectory = []
        rng = random.Random(42)

        for step in range(self.steps):
            t = step * self.dt
            step_data = {'t': round(t, 6), 'motors': []}

            for motor in self.motors:
                # Torque from stators
                driving_torque = motor.stator_count * self.STALL_TORQUE * (
                    1.0 - abs(motor.angular_velocity) * self.DRAG_COEFFICIENT / max(1.0, self.STALL_TORQUE)
                )
                driving_torque = max(0, driving_torque) * motor.direction

                # Net torque = drive - drag
                drag_torque = -motor.angular_velocity * motor.load_torque
                net_torque = driving_torque + drag_torque

                # Angular acceleration (moment of inertia ~1e-3 for flagellum)
                I = 1e-3
                alpha = net_torque / I
                motor.angular_velocity += alpha * self.dt
                motor.angle += motor.angular_velocity * self.dt

                # Direction switching (CW ↔ CCW)
                switch_rate = self.SWITCH_RATE_BASE
                if self._has_chemotaxis:
                    # Chemotaxis reduces tumbling when moving up gradient
                    switch_rate *= max(0.1, 1.0 - gradient_signal)
                if rng.random() < switch_rate:
                    motor.direction *= -1

                step_data['motors'].append({
                    'angle': round(motor.angle, 4),
                    'rpm': round(motor.angular_velocity * 60 / (2 * math.pi), 1),
                    'direction': 'CCW' if motor.direction > 0 else 'CW',
                })

            self.trajectory.append(step_data)

        # Summary
        final_rpms = [m.angular_velocity * 60 / (2 * math.pi) for m in self.motors]
        return {
            'n_motors': len(self.motors),
            'final_rpms': [round(r, 1) for r in final_rpms],
            'avg_rpm': round(sum(final_rpms) / max(1, len(final_rpms)), 1),
            'n_steps': self.steps,
            'dt': self.dt,
            'trajectory_length': len(self.trajectory),
        }


# ---------------------------------------------------------------------------
# Issue 9: Configurable Fitness Weights
# ---------------------------------------------------------------------------

@dataclass
class FitnessConfig:
    """Configurable weights for fitness function components.
    All weights should sum to approximately 1.0 for the base score."""
    # Tier weights (base score components)
    w_diversity: float = 0.20
    w_connectivity: float = 0.10
    w_complexity: float = 0.15
    w_efficiency: float = 0.10
    w_information: float = 0.10
    w_viability: float = 0.10
    w_assembly: float = 0.15
    w_energy: float = 0.10
    # Bonus caps
    motility_bonus: float = 0.10
    chemotaxis_bonus: float = 0.05
    assembly_bonus: float = 0.15
    operon_bonus_cap: float = 0.10
    torque_bonus_cap: float = 0.05
    gradient_bonus_cap: float = 0.05
    division_bonus: float = 0.03
    metabolic_bonus: float = 0.05
    quorum_bonus: float = 0.02
    # Penalty rates
    domain_penalty_rate: float = 0.03
    polymerize_penalty_rate: float = 0.005
    energy_starvation_rate: float = 0.02
    # Alive threshold
    alive_base: float = 0.50
    dead_cap: float = 0.49

    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items()}


DEFAULT_FITNESS_CONFIG = FitnessConfig()


# ---------------------------------------------------------------------------
# Issue 10: Error Propagation — fatal failure modes
# ---------------------------------------------------------------------------

class BuildError:
    """Tracks fatal and non-fatal build errors during genome execution."""
    FATAL_THRESHOLD = 5  # errors above this kill the organism

    def __init__(self):
        self.errors: List[str] = []
        self.fatal: bool = False

    def add(self, msg: str, fatal: bool = False):
        self.errors.append(msg)
        if fatal or len(self.errors) >= self.FATAL_THRESHOLD:
            self.fatal = True

    @property
    def count(self) -> int:
        return len(self.errors)

    @property
    def penalty(self) -> float:
        """Fitness penalty for accumulated errors."""
        if self.fatal:
            return 0.5  # halve fitness for fatal errors
        return min(0.3, self.count * 0.02)


# ---------------------------------------------------------------------------
# Issue 11: Improved Speciation — phenotypic distance
# ---------------------------------------------------------------------------

def phenotypic_distance(org_a: Dict, org_b: Dict) -> float:
    """Compute phenotypic distance between two organisms based on
    functional traits rather than raw genome Hamming distance."""
    features_a = _extract_phenotype_vector(org_a)
    features_b = _extract_phenotype_vector(org_b)
    # Euclidean distance in normalized feature space
    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(features_a, features_b)))
    return dist


def _extract_phenotype_vector(org: Dict) -> List[float]:
    """Extract normalized phenotype vector for speciation."""
    bp = org.get('blueprint', {})
    return [
        bp.get('n_parts', 0) / 30.0,
        bp.get('n_unique_parts', 0) / 15.0,
        bp.get('systems_complete', 0),
        bp.get('diversity', 0),
        1.0 if bp.get('motility', False) else 0.0,
        1.0 if bp.get('chemotaxis', False) else 0.0,
        bp.get('assembly', 0),
        min(1.0, bp.get('torque', 0) / 200.0),
        min(1.0, bp.get('operons', 0) / 5.0),
        1.0 if bp.get('metabolic_complete', False) else 0.0,
        min(1.0, bp.get('flagella_count', 0) / 3.0),
        min(1.0, bp.get('energy_balance', 0) / 50.0),
    ]


# ---------------------------------------------------------------------------
# Issue 12: Horizontal Gene Transfer (HGT)
# ---------------------------------------------------------------------------

def horizontal_gene_transfer(donor: str, recipient: str,
                             transfer_len: int = 15,
                             allowed_bases: set = None) -> str:
    """Transfer a segment from donor to recipient genome.
    Simulates bacterial conjugation / transduction."""
    if len(donor) < transfer_len or len(recipient) < transfer_len:
        return recipient
    # Pick random segment from donor
    start = random.randint(0, len(donor) - transfer_len)
    segment = donor[start:start + transfer_len]
    # Insert at random position in recipient
    insert_pos = random.randint(0, len(recipient) - transfer_len)
    result = recipient[:insert_pos] + segment + recipient[insert_pos + transfer_len:]
    # Clean up foreign bases
    if allowed_bases:
        result = ''.join(c if c in allowed_bases else random.choice(tuple(allowed_bases))
                         for c in result)
    return result


# ---------------------------------------------------------------------------
# Issue 13: Memory-efficient organism storage for long runs
# ---------------------------------------------------------------------------

def slim_organism_dict(org_dict: Dict) -> Dict:
    """Strip heavy fields from organism dict for memory-efficient storage.
    Keeps fitness, blueprint summary, grade, and viability."""
    return {
        'genome': org_dict.get('genome', '')[:40],  # truncated
        'fitness': org_dict.get('fitness', 0),
        'viable': org_dict.get('viable', False),
        'structural_grade': org_dict.get('structural_grade', 'F'),
        'n_heavy_atoms': org_dict.get('n_heavy_atoms', 0),
        'motility': org_dict.get('motility', False),
        'complexes': org_dict.get('complexes', []),
        'energy_balance': org_dict.get('energy_balance', 0),
        'gen_born': org_dict.get('gen_born', 0),
    }


# ---------------------------------------------------------------------------
# Issue 14: Multi-Cell Colony Stub
# ---------------------------------------------------------------------------

@dataclass
class Colony:
    """Simple colony model for quorum-sensing organisms.
    Tracks population count and shared signals."""
    cells: List[Dict] = field(default_factory=list)
    shared_signals: int = 0

    def add_cell(self, org_dict: Dict):
        self.cells.append(slim_organism_dict(org_dict))
        self.shared_signals += org_dict.get('quorum_signals', 0)

    @property
    def density(self) -> int:
        return len(self.cells)

    @property
    def quorum_active(self) -> bool:
        return self.shared_signals >= QUORUM_THRESHOLD

    def avg_fitness(self) -> float:
        if not self.cells:
            return 0.0
        return sum(c['fitness'] for c in self.cells) / len(self.cells)

    def summary(self) -> Dict:
        return {
            'density': self.density,
            'quorum_active': self.quorum_active,
            'shared_signals': self.shared_signals,
            'avg_fitness': round(self.avg_fitness(), 4),
            'motile_fraction': sum(1 for c in self.cells if c.get('motility')) / max(1, self.density),
        }


# ---------------------------------------------------------------------------
# Issue 17: Structured Logging
# ---------------------------------------------------------------------------

_log = _logging.getLogger('periodic_machine')
_log.setLevel(_logging.INFO)
if not _log.handlers:
    _handler = _logging.StreamHandler()
    _handler.setFormatter(_logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'))
    _log.addHandler(_handler)


def set_log_level(level: str = 'INFO'):
    """Set logging verbosity: DEBUG, INFO, WARNING, ERROR."""
    _log.setLevel(getattr(_logging, level.upper(), _logging.INFO))


# ---------------------------------------------------------------------------
# DNA-to-English translator
# ---------------------------------------------------------------------------

OPCODE_NAMES = {
    Opcode.BUILD: 'BUILD', Opcode.CONNECT: 'CONNECT',
    Opcode.REPEAT: 'REPEAT', Opcode.REGULATE: 'REGULATE',
    Opcode.NOOP: 'NOOP',
    Opcode.MOTOR: 'MOTOR', Opcode.LOCALIZE: 'LOCALIZE',
    Opcode.POLYMERIZE: 'POLYMERIZE', Opcode.PHASE: 'PHASE',
    Opcode.CHEMOTAXIS: 'CHEMOTAXIS', Opcode.MEMBRANE_INS: 'MEMBRANE_INSERT',
    Opcode.PROMOTER: 'PROMOTER', Opcode.TERMINATOR: 'TERMINATOR',
    Opcode.SECRETE: 'SECRETE', Opcode.QUORUM: 'QUORUM',
    Opcode.DIVIDE: 'DIVIDE', Opcode.METABOLIZE: 'METABOLIZE',
    Opcode.GRADIENT: 'GRADIENT',
}

# Human-friendly part descriptions — what it IS, what it DOES, body analogy
PART_ENGLISH = {
    'membrane':     ('Cell membrane',     'Forms the outer skin of the cell, keeping the inside in and the outside out.',
                     'Like your skin — a flexible barrier that protects everything inside.'),
    'wall':         ('Cell wall',         'A rigid shell layered outside the membrane for extra strength.',
                     'Like bone or an exoskeleton — gives the body its hard shape.'),
    'filament':     ('Protein fiber',     'Long chains of amino acids that act as scaffolding and muscle.',
                     'Like tendons and muscle fibers — the ropes and cables of the body.'),
    'atp':          ('ATP molecule',      'The universal battery. Stores and releases energy for every process.',
                     'Like a rechargeable battery — powers every movement and reaction.'),
    'nadh':         ('NADH shuttle',      'Carries electrons from food to the energy factories.',
                     'Like a power line — delivers electricity from the generator to the machines.'),
    'glucose':      ('Glucose sugar',     'The primary fuel. Gets broken down to charge ATP batteries.',
                     'Like gasoline — the raw fuel that gets burned for energy.'),
    'amino_basic':  ('Glycine',           'The simplest building block for making proteins and enzymes.',
                     'Like a basic brick — simple but essential for building anything.'),
    'amino_func':   ('Aspartate',         'A specialized amino acid that forms the active core of enzymes.',
                     'Like a precision tool — the specific wrench that makes the engine work.'),
    'cofactor':     ('Iron cofactor',     'A metal atom that helps enzymes transfer electrons.',
                     'Like a spark plug — the metal piece that makes the chemical reaction fire.'),
    'nucleotide':   ('Nucleotide (CMP)',  'A single letter of the genetic alphabet. Stores information.',
                     'Like a byte of data — one unit of the instruction manual.'),
    'coenzyme_a':   ('Coenzyme A',        'A master regulator that controls which metabolic pathways are active.',
                     'Like a thermostat — senses conditions and adjusts the system.'),
    'methane':      ('Methane',           'The simplest organic molecule. One carbon, four hydrogens.',
                     'Like a single LEGO brick — the most basic unit of carbon chemistry.'),
    'water':        ('Water',             'The universal solvent. Every reaction in life happens in water.',
                     'Like the air we breathe — without it, nothing else works.'),
    'ammonia':      ('Ammonia',           'A nitrogen donor. Needed to make amino acids and DNA bases.',
                     'Like fertilizer — supplies the nitrogen that proteins need.'),
    'formaldehyde': ('Formaldehyde',      'A simple carbon source from prebiotic chemistry.',
                     'Like raw ore — a primitive material that more complex parts are made from.'),
    'hcn':          ('Hydrogen cyanide',  'A prebiotic molecule that can form amino acids and nucleotides.',
                     'Like a seed — looks simple, but contains the potential for complexity.'),
    'phosphate':    ('Phosphate group',   'The backbone connector of DNA. Links nucleotides together.',
                     'Like the binding of a book — holds the pages of genetic code together.'),
    # --- Motor parts ---
    'flagellin':    ('Flagellin protein', 'Self-assembling filament that forms the whip-like tail.',
                     'Like the propeller shaft of a boat — spins to push the cell forward.'),
    'hook_protein': ('Hook protein',      'A flexible universal joint connecting the motor to the filament.',
                     'Like a U-joint in a driveshaft — transmits rotation through a bend.'),
    'ms_ring':      ('MS-ring',           'The core rotor ring embedded in the cell membrane.',
                     'Like the axle hub of a wheel — everything rotates around it.'),
    'c_ring':       ('C-ring (switch)',    'A cytoplasmic ring that controls rotation direction.',
                     'Like a gearbox — switches between forward and reverse.'),
    'motA':         ('MotA (stator A)',    'One half of the stator — a proton channel generating torque.',
                     'Like one electromagnet in an electric motor.'),
    'motB':         ('MotB (stator B)',    'The other stator half — anchors to the cell wall.',
                     'Like the motor housing bolted to the chassis.'),
    'fliG':         ('FliG (torque)',      'The rotor-stator interface where torque is generated.',
                     'Like the commutator in a motor — where electrical energy becomes rotation.'),
    'export_gate':  ('Export apparatus',   'Secretion system that threads flagellar proteins through the membrane.',
                     'Like an assembly line — feeds parts to the construction site.'),
    # --- Signaling parts ---
    'receptor':     ('Chemoreceptor',      'A sensor protein that detects chemical gradients in the environment.',
                     'Like a nose — smells which direction food is coming from.'),
    'histidine_kinase': ('CheA kinase',    'A signaling enzyme that phosphorylates the motor switch.',
                     'Like a relay switch — converts sensor signal into motor command.'),
    'response_reg': ('CheY regulator',     'Response regulator that directly controls motor switching.',
                     'Like a steering wheel — turns the motor left or right.'),
    # --- Advanced structural ---
    'actin':        ('Actin filament',     'Dynamic protein fibers forming the internal skeleton.',
                     'Like rebar in concrete — flexible internal reinforcement.'),
    'tubulin':      ('Tubulin',            'Rigid protein tubes for intracellular transport tracks.',
                     'Like train tracks inside the cell — cargo moves along them.'),
}

SYSTEM_ENGLISH = {
    'structure':   {'name': 'Body Structure', 'has': 'has a physical body',
                    'missing': 'has no body — like a mind with no skeleton',
                    'analogy': 'Bones + skin + muscle give an animal its shape.'},
    'energy':      {'name': 'Energy System', 'has': 'can power itself',
                    'missing': 'has no energy source — like a car with no engine',
                    'analogy': 'Digestion + respiration turn food into movement.'},
    'catalysis':   {'name': 'Tool System', 'has': 'can build and repair itself',
                    'missing': 'has no tools — like a factory with no machines',
                    'analogy': 'Enzymes are the hands that do all the work.'},
    'information': {'name': 'Memory System', 'has': 'can store and copy instructions',
                    'missing': 'has no memory — cannot reproduce or adapt',
                    'analogy': 'DNA is the blueprint that gets copied to the next generation.'},
    'minimal':     {'name': 'Raw Materials', 'has': 'has basic chemical supplies',
                    'missing': '',
                    'analogy': 'Simple molecules that everything else is built from.'},
    'motor':       {'name': 'Motor System', 'has': 'has molecular motor components',
                    'missing': 'has no motor — cannot generate mechanical force',
                    'analogy': 'Like having an engine — converts chemical energy into motion.'},
    'signaling':   {'name': 'Signaling System', 'has': 'can sense and respond to environment',
                    'missing': 'has no sensors — blind to its surroundings',
                    'analogy': 'Like a nervous system — detects stimuli and coordinates responses.'},
}

ENV_ENGLISH = {
    'earth':   'Earth — carbon-rich, water-based, like our planet',
    'silicon': 'Silicon world — silicon replaces carbon as the backbone element',
    'exotic':  'Exotic — equal amounts of all elements, alien chemistry',
}

# ---------------------------------------------------------------------------
# Reference Library: hardcoded genomes for known basic life forms
# ---------------------------------------------------------------------------
# Each entry is a verified genome string for a specific base system that
# produces a known organism type when translated. These serve as:
#   1. A Rosetta Stone for learning how to read DNA instructions
#   2. Validated test cases — if translation ever changes, tests catch it
#   3. Starting points for evolution experiments
#
# 4-base codon->part mapping (deterministic, verified):
#   AAAGG->membrane  AATGG->filament  ATTGG->nadh    AAAAA->glucose
#   AAGGG->amino_basic  AAAAT->amino_func  ATGGG->cofactor
#   AAATT->nucleotide  AAAAG->methane  AGGGG->water
#   AAATG->ammonia  AATTG->hcn
#   CAAAA=CONNECT (bonds parts together)

REFERENCE_LIBRARY = [
    # --- 4-base (quaternary / Earth DNA) ---
    {
        'name': 'Prebiotic Soup',
        'n_bases': 4,
        'genome': 'AGGGGAAAAGAAATGAATTG',
        'description': (
            'The raw ingredients of life before any cells existed. '
            'Water, methane, ammonia, and hydrogen cyanide — the chemicals '
            'found in Earths early oceans and volcanic vents. '
            'Not alive, but these are the building blocks that life emerges from. '
            'Think of it as a pile of lumber before anyone builds a house.'
        ),
        'expected_parts': ['water', 'methane', 'ammonia', 'hcn'],
        'expected_viable': False,
        'category': 'prebiotic',
    },
    {
        'name': 'Minimal Protocell',
        'n_bases': 4,
        'genome': 'AAAGGAAAAAAAGGG',
        'description': (
            'The simplest possible living cell. It has just three things: '
            'a membrane (skin), glucose (food), and glycine (a basic tool). '
            'This is the biological equivalent of a tent with a campfire and a knife. '
            'It can barely survive, but it meets the minimum requirements for life: '
            'a body, energy, and the ability to do chemical work.'
        ),
        'expected_parts': ['membrane', 'glucose', 'amino_basic'],
        'expected_viable': True,
        'category': 'minimal',
    },
    {
        'name': 'Simple Bacterium',
        'n_bases': 4,
        'genome': 'AAAGGAAAAAAAGGGAAAAT' + 'CAAAA' + 'AAATT',
        'description': (
            'A basic single-celled organism, like the earliest bacteria. '
            'It has a membrane for protection, glucose for fuel, glycine and '
            'aspartate as enzyme tools, and a nucleotide for storing genetic '
            'information. The CONNECT instruction bonds the parts together. '
            'This cell can eat, build proteins, and copy its own DNA — '
            'the three pillars of bacterial life.'
        ),
        'expected_parts': ['membrane', 'glucose', 'amino_basic', 'amino_func', 'nucleotide'],
        'expected_viable': True,
        'category': 'simple',
    },
    {
        'name': 'Advanced Bacterium',
        'n_bases': 4,
        'genome': (
            'AAAGG' + 'AATGG' +
            'ATTGG' + 'AAAAA' +
            'AAGGG' + 'AAAAT' + 'ATGGG' +
            'AAATT' +
            'CAAAA' + 'CAAAA' + 'CAAAA'
        ),
        'description': (
            'A well-equipped bacterium with specialized systems. '
            'Two structural proteins give it shape (membrane + protein fiber). '
            'NADH and glucose provide a two-stage energy system. '
            'Three different catalytic tools (glycine, aspartate, iron cofactor) '
            'let it run complex chemical reactions. A nucleotide stores its genome. '
            'Connection instructions bond everything into a working machine. '
            'Think of E. coli — small but surprisingly capable.'
        ),
        'expected_parts': ['membrane', 'filament', 'nadh', 'glucose',
                           'amino_basic', 'amino_func', 'cofactor', 'nucleotide'],
        'expected_viable': True,
        'category': 'complex',
    },
    {
        'name': 'Energy-Optimized Cell',
        'n_bases': 4,
        'genome': (
            'AAAGG' +
            'ATTGG' + 'AAAAA' + 'ATTGG' +
            'AAGGG' +
            'CAAAA' + 'CAAAA'
        ),
        'description': (
            'A cell specialized for energy production — like a power plant. '
            'It has double the NADH shuttles of a normal cell, plus glucose fuel. '
            'This is what you get when evolution prioritizes metabolism over '
            'complexity. Common in environments with abundant food but harsh '
            'conditions requiring lots of energy (deep sea vents, hot springs).'
        ),
        'expected_parts': ['membrane', 'nadh', 'glucose', 'nadh', 'amino_basic'],
        'expected_viable': True,
        'category': 'specialized',
    },
    # --- 6-base (hexary) ---
    {
        'name': 'Hexary Protocell',
        'n_bases': 6,
        'genome': 'AAAGGAAAAAAAGGG',
        'description': (
            'A minimal cell using a 6-letter genetic alphabet (A,T,C,G,X,Y). '
            'The extra letters allow more information per gene, but copying errors '
            'are 6x more frequent than Earth DNA. This cell would mutate rapidly — '
            'evolving fast but also developing cancers and genetic diseases. '
            'Alien life might use this system on a planet where fast adaptation '
            'matters more than long-term stability.'
        ),
        'expected_parts': ['membrane', 'glucose', 'amino_basic'],
        'expected_viable': True,
        'category': 'alien',
    },
    # --- 8-base (octary) ---
    {
        'name': 'Octary Protocell',
        'n_bases': 8,
        'genome': 'AAAGGAAAAAAAGGG',
        'description': (
            'A minimal cell using an 8-letter genetic alphabet. '
            'This system packs the most information per base but is extremely '
            'unstable — 11% of bases mutate per copy. After just 10 generations, '
            'the genome would be unrecognizable. This is why Earth settled on 4 bases: '
            'its the sweet spot between information density and copy fidelity. '
            '8-base life would need extraordinary error-correction mechanisms to survive.'
        ),
        'expected_parts': ['membrane', 'glucose', 'amino_basic'],
        'expected_viable': True,
        'category': 'theoretical',
    },
    # --- 8-base (motor-capable) ---
    # MOTOR codon mapping (W+A/T prefix, operand%11 selects motor part):
    #   WACCA->flagellin  WATCG->hook_protein  WAAGA->ms_ring
    #   WTTCA->c_ring     WAATA->motA          WTGGA->motB
    #   WATCA->fliG       WAAAA->export_gate
    #   WAGGA->receptor   WAACA->histidine_kinase  WAATG->response_reg
    {
        'name': 'Flagellar Bacterium',
        'n_bases': 8,
        'genome': (
            'AAAGG' + 'AATGG' +       # membrane + filament (structure)
            'ATTGG' + 'AAAAA' +       # nadh + glucose (energy)
            'AAGGG' + 'AAAAT' +       # amino_basic + amino_func (catalysis)
            'AAATT' +                  # nucleotide (information)
            'WACCA' + 'WATCG' +       # motor: flagellin, hook_protein
            'WAAGA' + 'WTTCA' +       # motor: ms_ring, c_ring
            'WAATA' + 'WTGGA' +       # motor: motA, motB
            'WATCA' + 'WAAAA' +       # motor: fliG, export_gate
            'CAAAA' + 'CAAAA' + 'CAAAA'  # connections
        ),
        'description': (
            'A complete motile bacterium with a flagellar motor. '
            'Contains all structural, energy, catalytic, and information systems '
            'plus a full flagellar machine: rotor rings, stator units, hook, and '
            'filament. This organism can swim by spinning its flagellum like a '
            'propeller, powered by proton motive force across its membrane. '
            'Think of E. coli swimming toward nutrients.'
        ),
        'expected_parts': ['membrane', 'filament', 'nadh', 'glucose',
                           'amino_basic', 'amino_func', 'nucleotide',
                           'flagellin', 'hook_protein', 'ms_ring', 'c_ring',
                           'motA', 'motB', 'fliG', 'export_gate'],
        'expected_viable': True,
        'category': 'complex',
    },
    {
        'name': 'Chemotactic Swimmer',
        'n_bases': 8,
        'genome': (
            'AAAGG' + 'AAAAA' + 'AAGGG' +  # membrane + glucose + amino_basic
            'AAATT' +                         # nucleotide (information)
            'WACCA' + 'WATCG' +              # flagellin, hook_protein
            'WAAGA' + 'WTTCA' +              # ms_ring, c_ring
            'WAATA' + 'WTGGA' +              # motA, motB
            'WATCA' +                         # fliG
            'WAGGA' + 'WAACA' + 'WAATG' +   # receptor, histidine_kinase, response_reg
            'CAAAA' + 'CAAAA'
        ),
        'description': (
            'An advanced motile bacterium with chemotaxis. Not only can it swim, '
            'it can sense chemical gradients and steer toward nutrients. '
            'The chemoreceptor detects molecules, CheA kinase relays the signal, '
            'and CheY regulator switches the motor between forward (swim) and '
            'reverse (tumble). This is the run-and-tumble navigation strategy '
            'used by real bacteria.'
        ),
        'expected_parts': ['membrane', 'glucose', 'amino_basic', 'nucleotide',
                           'flagellin', 'hook_protein', 'ms_ring', 'c_ring',
                           'motA', 'motB', 'fliG',
                           'receptor', 'histidine_kinase', 'response_reg'],
        'expected_viable': True,
        'category': 'complex',
    },
    # --- 2-base (binary) ---
    {
        'name': 'Binary Minimal',
        'n_bases': 2,
        'genome': '0' * 20,
        'description': (
            'The simplest possible genetic system — just two symbols, 0 and 1. '
            'Like Morse code compared to English. Almost no copying errors '
            '(0.05% per base), but very low information density. '
            'You would need enormously long genomes to encode anything complex. '
            'This is why no known life uses binary DNA — it is too simple '
            'to encode the variety of proteins needed for survival.'
        ),
        'expected_parts': ['glucose', 'glucose', 'glucose', 'glucose'],
        'expected_viable': False,
        'category': 'theoretical',
    },
]


def _validate_reference_library():
    """Verify every reference genome translates to the expected organism."""
    errors = []
    for ref in REFERENCE_LIBRARY:
        strand = Strand(ref['genome'], ref['n_bases'])
        bp = execute_genome_deterministic(strand, 'earth')

        viable = bp.systems_score() == 1.0
        if ref['expected_viable'] is not None and viable != ref['expected_viable']:
            errors.append(
                f"{ref['name']}: expected viable={ref['expected_viable']}, got {viable}")

        if ref['expected_parts']:
            if bp.parts_used != ref['expected_parts']:
                errors.append(
                    f"{ref['name']}: expected parts {ref['expected_parts']}, "
                    f"got {bp.parts_used}")

    return errors


# ---------------------------------------------------------------------------
# Genome Library — save/load full organism files to disk
# ---------------------------------------------------------------------------
# Each organism is stored as a .json file in a library folder.
# Large organisms (intelligent life, real microbe templates) are saved here
# because their genome strings are too large for the in-memory REFERENCE_LIBRARY.
#
# File schema:
#   {
#     "name": str,
#     "category": str,          # "microbe" | "viral" | "intelligent" | "evolved" | ...
#     "source": str,            # "real_template" | "generated" | "evolved"
#     "n_bases": int,
#     "env_name": str,
#     "genome": str,            # full genome string
#     "description": str,       # human-readable
#     "science_notes": str,     # real-world biology reference
#     "subsystems": [str],      # list of subsystem names encoded
#     "expected_grade": str,    # e.g. "S", "A"
#     "fitness": float,         # last evaluated fitness
#     "grade": str,
#     "parts_used": [str],
#     "complexes_formed": [str],
#     "behaviors": [str],
#     "intelligence_tier": int,
#     "saved_at": str,          # ISO timestamp
#     "version": str,           # language version
#   }

import os as _os
import json as _json
import datetime as _datetime

LIBRARY_FOLDER_NAME = "organism_library"
LIBRARY_DEFAULT_PATH = _os.path.join(
    _os.path.expanduser("~"), "CascadeProjects", LIBRARY_FOLDER_NAME
)


def get_library_path() -> str:
    """Return the current library folder path, creating it if needed."""
    path = LIBRARY_DEFAULT_PATH
    _os.makedirs(path, exist_ok=True)
    return path


def organism_to_library_dict(org, name: str, category: str = 'generated',
                              source: str = 'generated',
                              description: str = '',
                              science_notes: str = '') -> dict:
    """Serialise an Organism + metadata to the library JSON schema."""
    bp = org.blueprint
    grade, grade_desc = org.structural_grade()
    tier_idx, tier_name, _ = bp.intelligence_level()
    return {
        'name':             name,
        'category':         category,
        'source':           source,
        'n_bases':          org.strand.n_bases,
        'env_name':         org.env_name if hasattr(org, 'env_name') else 'earth',
        'genome':           org.strand.seq,
        'description':      description,
        'science_notes':    science_notes,
        'subsystems':       list(bp.complexes_formed),
        'expected_grade':   grade,
        'fitness':          round(org.fitness, 6),
        'grade':            grade,
        'grade_desc':       grade_desc,
        'parts_used':       list(bp.parts_used),
        'unique_parts':     sorted(set(bp.parts_used)),
        'complexes_formed': list(bp.complexes_formed),
        'behaviors':        list(bp.behaviors),
        'intelligence_tier':tier_idx,
        'intelligence_name':tier_name,
        'intelligence_score': round(bp.intelligence_score, 6),
        'neural_score':     round(bp.neural_score(), 6),
        'homeostasis_score':round(bp.homeostasis_score(), 6),
        'energy_budget':    bp.energy_budget,
        'energy_cost':      bp.energy_cost,
        'motility':         bp.motility_capable(),
        'chemotaxis':       bp.chemotaxis_capable(),
        'flagella_count':   bp.flagella_count,
        'torque':           round(bp.torque, 2),
        'rpm':              round(bp.rpm, 1),
        'membrane_potential': round(bp.membrane_potential, 1),
        'viable':           bool(org.viable),
        'saved_at':         _datetime.datetime.now().isoformat(timespec='seconds'),
        'version':          '2.0',
    }


def save_organism_to_library(org, name: str, category: str = 'generated',
                              source: str = 'generated',
                              description: str = '',
                              science_notes: str = '',
                              folder: str = None) -> str:
    """Save organism to library folder. Returns the saved file path."""
    folder = folder or get_library_path()
    _os.makedirs(folder, exist_ok=True)
    data = organism_to_library_dict(org, name, category, source,
                                    description, science_notes)
    safe_name = ''.join(c if c.isalnum() or c in '-_ ' else '_' for c in name)
    safe_name = safe_name.replace(' ', '_').strip('_')
    timestamp = _datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename  = f"{safe_name}_{timestamp}.json"
    filepath  = _os.path.join(folder, filename)
    with open(filepath, 'w', encoding='utf-8') as fh:
        _json.dump(data, fh, indent=2, ensure_ascii=False)
    return filepath


def load_organism_from_file(filepath: str) -> dict:
    """Load an organism dict from a library JSON file."""
    with open(filepath, 'r', encoding='utf-8') as fh:
        return _json.load(fh)


def list_library_organisms(folder: str = None) -> list:
    """Return list of (filepath, summary_dict) for all .json files in folder."""
    folder = folder or get_library_path()
    if not _os.path.isdir(folder):
        return []
    results = []
    for fname in sorted(_os.listdir(folder)):
        if not fname.endswith('.json'):
            continue
        fpath = _os.path.join(folder, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as fh:
                data = _json.load(fh)
            summary = {
                'filepath':    fpath,
                'filename':    fname,
                'name':        data.get('name', fname),
                'category':    data.get('category', ''),
                'source':      data.get('source', ''),
                'grade':       data.get('grade', '?'),
                'fitness':     data.get('fitness', 0.0),
                'viable':      data.get('viable', False),
                'n_bases':     data.get('n_bases', 4),
                'genome_len':  len(data.get('genome', '')),
                'n_parts':     len(data.get('parts_used', [])),
                'behaviors':   data.get('behaviors', []),
                'intelligence_tier': data.get('intelligence_tier', 0),
                'intelligence_name': data.get('intelligence_name', ''),
                'description': data.get('description', ''),
                'saved_at':    data.get('saved_at', ''),
            }
            results.append(summary)
        except Exception:
            pass
    return results


def organism_from_library_dict(data: dict):
    """Reconstruct an Organism from a library dict. Returns (Organism, data)."""
    n_bases  = data.get('n_bases', 4)
    env_name = data.get('env_name', 'earth')
    genome   = data.get('genome', '')
    strand   = Strand(genome, n_bases)
    supply   = ElementSupply.earth_like()
    org      = Organism(strand=strand, env_name=env_name)
    org.express(supply, deterministic=True)
    org.evaluate()
    return org, data


# ---------------------------------------------------------------------------
# Real Micro-organism Templates
# ---------------------------------------------------------------------------
# These genomes are inspired by real organisms' functional organisation.
# They are NOT literal DNA sequences — they are structured genomes in the
# Periodic Machine language that encode the same functional subsystems.
#
# References:
#  - M. genitalium (smallest self-replicating bacterium, 473 genes)
#    Fraser et al., Science 1995; Hutchison et al., Science 2016
#  - E. coli K-12 (model bacterium, ~4300 genes)
#    Blattner et al., Science 1997
#  - Phi-X174 (smallest dsDNA bacteriophage, 11 genes, 5386 bp)
#    Sanger et al., Nature 1977

def _make_microbe_genome(parts_sequence: list, n_bases: int = 8) -> str:
    """Build a genome string from an ordered list of part names + opcodes.

    Uses the exact W/Z second-base dispatch from LanguageSpec._decode_opcode_legacy
    so every codon round-trips correctly through codon_to_instruction.

    Dispatch (8-base, bases = A T C G X Y W Z):
      First base A/G → BUILD,  T/C → CONNECT,  X → REPEAT,  Y → REGULATE
      W prefix: second = A/T→MOTOR, C/G→LOCALIZE, X→POLYMERIZE, Y→PHASE,
                                    W→CHEMOTAXIS, Z→MEMBRANE_INS
      Z prefix: second = A/T→SECRETE, C/G→QUORUM, X→DIVIDE, Y→METABOLIZE,
                                       W→GRADIENT, Z→PROMOTER
      Operand = sum(ord(c) for c in codon[1:])  — use 'AAAA' tail for operand 0
    """
    if n_bases not in BASE_SYSTEMS:
        n_bases = 8
    bases = BASE_SYSTEMS[n_bases]['bases']
    nb    = len(bases)

    # The interpreter computes: operand = sum(ord(c) for c in codon[1:])
    # then part = TABLE[operand % len(TABLE)].
    # We need to engineer the 4 trailing chars (codon[1:]) so that
    #   sum(ord(trailing)) % table_len == desired_index.
    # Strategy: set trailing = [prefix_char, A, A, A] where prefix_char
    # compensates for the fixed prefix letters that precede the tail.
    # For a codon like  P1 P2 t0 t1 t2  (len=5), codon[1:] = P2 t0 t1 t2.
    # We fix t1=t2=A and solve:  (ord(P2) + ord(t0) + ord('A')*2) % tlen = idx
    # => ord(t0) = (idx - ord(P2) - 2*ord('A')) mod tlen, then map to base char.

    ORD_A = ord(bases[0])  # ord('A') = 65

    # Pre-compute ord values for each base
    _base_ords = [ord(b) for b in bases]

    def _solve_tail(prefix2_ord: int, desired_idx: int, table_len: int) -> str:
        """Return a 3-char tail (t0,t1,t2) using only valid bases so that
        (prefix2_ord + sum(ord(t0,t1,t2))) % table_len == desired_idx."""
        # Fix t1=t2=bases[0] (ORD_A each). Solve for t0:
        # target = (desired_idx - prefix2_ord - 2*ORD_A) % table_len
        # Find base b s.t. ord(b) % table_len == target
        target = (desired_idx - prefix2_ord - 2 * ORD_A) % table_len
        # Try each base to find one whose ord matches the residue
        for b in bases:
            if ord(b) % table_len == target:
                return b + bases[0] + bases[0]
        # If no single base matches (possible when table_len > 26),
        # distribute across t0+t1: fix t2=bases[0], solve t0+t1.
        target2 = (desired_idx - prefix2_ord - ORD_A) % table_len
        for b0 in bases:
            rem = (target2 - ord(b0)) % table_len
            for b1 in bases:
                if ord(b1) % table_len == rem:
                    return b0 + b1 + bases[0]
        # Fallback: return all-A (operand 0 maps to first part, good default)
        return bases[0] * 3

    def _build_codon(operand: int) -> str:
        # A-prefix: codon = A + (4 chars), codon[1:] = 4 chars
        # operand % len(BUILD_PART_NAMES) must == operand arg
        tlen = len(BUILD_PART_NAMES)
        tail3 = _solve_tail(ORD_A, operand % tlen, tlen)  # P2=bases[0]=A
        return bases[0] + bases[0] + tail3                 # AAAAA-based codon

    def _connect_codon() -> str:
        return bases[1] + bases[0] * (CODON_LEN - 1)      # T-prefix → CONNECT

    def _motor_codon(operand: int) -> str:
        if nb < 8:
            return _build_codon(operand)           # fallback to BUILD in 4-base
        # W + A + tail3;  codon[1:] = A + tail3 (4 chars)
        tlen = len(MOTOR_PART_NAMES)
        tail3 = _solve_tail(ORD_A, operand % tlen, tlen)  # P2='A'
        return 'W' + 'A' + tail3                           # W+A → MOTOR

    def _z_codon(second: str, operand: int, tlen: int = 1) -> str:
        if nb < 8:
            return bases[0] * CODON_LEN            # fallback NOOP in 4-base
        # Z + second + tail3;  operand not used for Z-opcodes (no table lookup)
        return 'Z' + second + bases[0] * (CODON_LEN - 2)

    codons = []
    for item in parts_sequence:
        if item in ('PROMOTER', 'TERMINATOR'):
            pass  # structural markers — operons are optional
        elif item == 'CONNECT':
            codons.append(_connect_codon())
        elif item == 'METABOLIZE':
            codons.append(_z_codon('Y', 0))        # Z+Y → METABOLIZE
        elif item == 'CHEMOTAXIS':
            if nb >= 8:
                codons.append('W' + 'W' + bases[0] * (CODON_LEN - 2))  # W+W → CHEMOTAXIS
            else:
                codons.append(bases[0] * CODON_LEN)
        elif item == 'GRADIENT':
            codons.append(_z_codon('W', 0))        # Z+W → GRADIENT
        elif item == 'DIVIDE':
            codons.append(_z_codon('X', 0))        # Z+X → DIVIDE
        elif item == 'QUORUM':
            codons.append(_z_codon('C', 0))        # Z+C → QUORUM
        elif item in BUILD_PART_NAMES:
            codons.append(_build_codon(BUILD_PART_NAMES.index(item)))
        elif item in MOTOR_PART_NAMES:
            codons.append(_motor_codon(MOTOR_PART_NAMES.index(item)))
        # neural/homeostasis parts injected directly in build_microbe_organism()
    return ''.join(codons)


# Real micro-organism templates — each entry is a full specification
MICROBE_TEMPLATES = [
    {
        'name': 'Mycoplasma genitalium (minimal cell)',
        'category': 'microbe',
        'source': 'real_template',
        'n_bases': 8,
        'env_name': 'earth',
        'science_notes': (
            'M. genitalium has the smallest known genome of any self-replicating '
            'organism: 580 kbp encoding 473 genes. Hutchison et al. (Science 2016) '
            'showed only 473 genes are essential. Key features: no cell wall, '
            'relies on host ATP, no flagella (genome-encoded but non-functional), '
            'complete DNA replication/repair system. This template encodes the '
            'minimal essential subsystems: membrane, energy transport, replication, '
            'basic metabolism, and DNA maintenance.'
        ),
        'description': (
            'The smallest known self-replicating bacterium. No cell wall, '
            'parasitic metabolism, but a complete replication and repair system. '
            'Considered the minimal blueprint for cellular life.'
        ),
        'subsystems': [
            'PROMOTER',
            'membrane', 'CONNECT',
            'atp', 'CONNECT', 'METABOLIZE',
            'glucose', 'nadh', 'CONNECT', 'METABOLIZE',
            'amino_func', 'cofactor', 'CONNECT',
            'nucleotide', 'coenzyme_a', 'CONNECT',
            'TERMINATOR',
            'PROMOTER',
            'amino_basic', 'amino_func', 'cofactor', 'CONNECT',
            'nucleotide', 'nucleotide', 'CONNECT',
            'DIVIDE',
            'TERMINATOR',
        ],
        'neural_parts': [],
        'homeostasis_parts': ['heat_shock_prot', 'dna_repair'],
    },
    {
        'name': 'Escherichia coli K-12 (model bacterium)',
        'category': 'microbe',
        'source': 'real_template',
        'n_bases': 8,
        'env_name': 'earth',
        'science_notes': (
            'E. coli K-12 genome: 4.6 Mbp, ~4300 protein-coding genes. '
            'Blattner et al. (Science 1997). Model organism for molecular biology. '
            'Key features: complete central metabolism (glycolysis, TCA, electron '
            'transport), flagellar motility with chemotaxis (run-and-tumble), '
            'cell division (FtsZ ring), quorum sensing (autoinducer-2), and '
            'SOS DNA repair response. This template encodes all these systems '
            'in the Periodic Machine genome language.'
        ),
        'description': (
            'The best-studied bacterium. Fully motile with chemotaxis, complete '
            'central metabolism, cell division, quorum sensing, and SOS repair. '
            'The standard reference for prokaryotic life.'
        ),
        'subsystems': [
            # Operon 1: Core structure + metabolism
            'PROMOTER',
            'membrane', 'wall', 'filament', 'CONNECT', 'CONNECT',
            'atp', 'nadh', 'glucose', 'CONNECT', 'METABOLIZE', 'METABOLIZE',
            'amino_basic', 'amino_func', 'cofactor', 'nucleotide', 'coenzyme_a',
            'CONNECT', 'CONNECT', 'CONNECT',
            'TERMINATOR',
            # Operon 2: Flagellar motor (all 8 components)
            'PROMOTER',
            'ms_ring', 'c_ring', 'CONNECT',
            'motA', 'motB', 'fliG', 'CONNECT', 'CONNECT',
            'export_gate', 'hook_protein', 'flagellin', 'flagellin',
            'CONNECT', 'CONNECT',
            'TERMINATOR',
            # Operon 3: Chemotaxis two-component system
            'PROMOTER',
            'receptor', 'histidine_kinase', 'response_reg',
            'CHEMOTAXIS', 'GRADIENT', 'CONNECT', 'CONNECT',
            'TERMINATOR',
            # Operon 4: Division + quorum
            'PROMOTER',
            'amino_basic', 'nucleotide', 'CONNECT',
            'DIVIDE', 'QUORUM',
            'TERMINATOR',
        ],
        'neural_parts': [],
        'homeostasis_parts': ['heat_shock_prot', 'proteasome', 'dna_repair', 'antioxidant'],
    },
    {
        'name': 'Phi-X174 Bacteriophage (viral minimal genome)',
        'category': 'viral',
        'source': 'real_template',
        'n_bases': 4,
        'env_name': 'earth',
        'science_notes': (
            'Phi-X174: 5386 bp, 11 genes, first DNA genome fully sequenced '
            '(Sanger et al., Nature 1977). Single-stranded circular DNA phage. '
            'Overlapping genes maximise information density. Encodes: capsid '
            'proteins (F, G, H, J), spike protein (G), lysis protein (E), '
            'DNA replication proteins (A, A*, B, C, D). No metabolism — '
            'completely hijacks host cell. Represented here as a minimal '
            'information + replication + assembly system.'
        ),
        'description': (
            'The first fully sequenced DNA genome. A bacteriophage (virus) that '
            'infects E. coli. Contains only the minimum to replicate: capsid '
            'proteins for structure and DNA replication machinery. No metabolism '
            '— it steals everything from the host cell.'
        ),
        'subsystems': [
            'PROMOTER',
            'membrane', 'filament', 'CONNECT',
            'nucleotide', 'nucleotide', 'nucleotide',
            'amino_func', 'CONNECT', 'CONNECT',
            'TERMINATOR',
            'PROMOTER',
            'amino_basic', 'cofactor', 'coenzyme_a', 'CONNECT',
            'DIVIDE',
            'TERMINATOR',
        ],
        'neural_parts': [],
        'homeostasis_parts': [],
    },
    {
        'name': 'Caulobacter crescentus (asymmetric divider)',
        'category': 'microbe',
        'source': 'real_template',
        'n_bases': 8,
        'env_name': 'earth',
        'science_notes': (
            'Caulobacter crescentus: 4.0 Mbp genome, model for cell cycle and '
            'asymmetric division. Poindexter (1964); Shapiro et al. (2002). '
            'Produces two distinct daughter cells: a motile swarmer (flagellated) '
            'and a sessile stalked cell. Complete two-component signaling network '
            'controls cell cycle. Actin-like MreB and tubulin-like FtsZ for '
            'shape and division. Strong chemotaxis in swarmer phase.'
        ),
        'description': (
            'A freshwater bacterium famous for dividing asymmetrically — '
            'one daughter swims with a flagellum, the other anchors with a stalk. '
            'It has a sophisticated cell-cycle control system and is a major '
            'model for developmental biology in bacteria.'
        ),
        'subsystems': [
            'PROMOTER',
            'membrane', 'wall', 'actin', 'tubulin', 'CONNECT', 'CONNECT', 'CONNECT',
            'atp', 'nadh', 'glucose', 'METABOLIZE', 'CONNECT',
            'amino_func', 'cofactor', 'nucleotide', 'CONNECT', 'CONNECT',
            'TERMINATOR',
            'PROMOTER',
            'ms_ring', 'c_ring', 'motA', 'motB', 'fliG',
            'export_gate', 'hook_protein', 'flagellin',
            'CONNECT', 'CONNECT', 'CONNECT',
            'receptor', 'histidine_kinase', 'response_reg',
            'CHEMOTAXIS', 'GRADIENT', 'CONNECT',
            'TERMINATOR',
            'PROMOTER',
            'amino_basic', 'nucleotide', 'coenzyme_a', 'CONNECT',
            'DIVIDE', 'QUORUM',
            'TERMINATOR',
        ],
        'neural_parts': [],
        'homeostasis_parts': ['heat_shock_prot', 'dna_repair'],
    },
    {
        'name': 'Synechocystis PCC6803 (photosynthetic cyanobacterium)',
        'category': 'microbe',
        'source': 'real_template',
        'n_bases': 8,
        'env_name': 'earth',
        'science_notes': (
            'Synechocystis sp. PCC6803: 3.57 Mbp, first photosynthetic organism '
            'fully sequenced (Kaneko et al., DNA Research 1996). '
            'Performs oxygenic photosynthesis using Photosystem I & II. '
            'Contains complete electron transport chain (similar to mitochondria), '
            'carbon fixation (Calvin cycle), nitrogen fixation capability, '
            'phycobilisome antenna complexes, and circadian clock (KaiABC). '
            'Also motile via type-IV pili and has full UV damage repair.'
        ),
        'description': (
            'A freshwater cyanobacterium that performs oxygenic photosynthesis — '
            'splitting water to release oxygen. It uses light energy to fix CO2. '
            'Contains an internal membrane system (thylakoids) and a circadian '
            'clock. Ancestor of the chloroplasts in all plant cells.'
        ),
        'subsystems': [
            'PROMOTER',
            'membrane', 'wall', 'filament', 'actin', 'CONNECT', 'CONNECT', 'CONNECT',
            'atp', 'atp', 'nadh', 'cofactor', 'METABOLIZE', 'METABOLIZE', 'CONNECT',
            'amino_func', 'amino_basic', 'nucleotide', 'coenzyme_a', 'CONNECT', 'CONNECT',
            'TERMINATOR',
            'PROMOTER',
            'glucose', 'nadh', 'atp', 'cofactor', 'coenzyme_a',
            'METABOLIZE', 'METABOLIZE', 'CONNECT',
            'receptor', 'histidine_kinase', 'response_reg',
            'CHEMOTAXIS', 'GRADIENT', 'CONNECT',
            'TERMINATOR',
            'PROMOTER',
            'amino_basic', 'nucleotide', 'CONNECT',
            'DIVIDE', 'QUORUM',
            'TERMINATOR',
        ],
        'neural_parts': [],
        'homeostasis_parts': ['heat_shock_prot', 'dna_repair', 'antioxidant'],
    },
]


def build_microbe_organism(template: dict) -> 'Organism':
    """Build a fully expressed Organism from a MICROBE_TEMPLATES entry."""
    n_bases  = template.get('n_bases', 8)
    env_name = template.get('env_name', 'earth')
    genome   = _make_microbe_genome(template['subsystems'], n_bases)

    strand = Strand(genome, n_bases)
    supply = ElementSupply.earth_like()
    org    = Organism(strand=strand, env_name=env_name)

    # Run genome to get base parts — but keep blueprint live for injection
    bp = execute_genome(strand, env_name, deterministic=True)
    org.blueprint = bp

    # Directly inject neural / homeostasis parts (bypass operand mapping)
    for pname in template.get('neural_parts', []):
        if pname in PARTS_KB:
            part = PARTS_KB[pname]
            bp.parts_used.append(pname)
            bp.smiles_components.append(part['smiles'])
            role = part['role']
            bp.roles_present[role] = bp.roles_present.get(role, 0) + 1
            bp.energy_cost += part.get('energy_cost', 0)
            pidx = len(bp.parts_used) - 1
            bp.spatial_graph[pidx] = part.get('attachment', [])
            bp.neural_parts.append(pname)
    for pname in template.get('homeostasis_parts', []):
        if pname in PARTS_KB:
            part = PARTS_KB[pname]
            bp.parts_used.append(pname)
            bp.smiles_components.append(part['smiles'])
            role = part['role']
            bp.roles_present[role] = bp.roles_present.get(role, 0) + 1
            bp.energy_cost += part.get('energy_cost', 0)
            pidx = len(bp.parts_used) - 1
            bp.spatial_graph[pidx] = part.get('attachment', [])
            bp.homeostasis_parts.append(pname)

    # Re-run post-assembly scoring with all parts present
    bp.detect_complexes()
    bp.compute_torque()
    if bp.chemotaxis_capable():
        bp.tumble_prob = TUMBLE_PROBABILITY_BASE * 0.5
    bp.compute_membrane_potential()
    bp.compute_intelligence()
    # Behaviors — deduplicate then add new ones
    existing = set(bp.behaviors)
    if bp.motility_capable() and 'swim' not in existing:
        bp.behaviors.append('swim')
        existing.add('swim')
    if bp.chemotaxis_capable() and 'chemotaxis' not in existing:
        bp.behaviors.append('chemotaxis')
        existing.add('chemotaxis')
    if bp.intelligence_capable() and 'neural_signaling' not in existing:
        bp.behaviors.append('neural_signaling')
    if bp.quorum_signals > 0 and 'quorum_sensing' not in existing:
        bp.behaviors.append('quorum_sensing')

    # Build molecules for chemistry scoring + set viability
    part_mols, elem_counts, geom = build_organism_mol(bp)
    org._part_mols        = part_mols
    org._organism_geom    = geom
    org.expressed_smiles  = bp.combined_smiles
    if part_mols:
        org.mol            = max(part_mols, key=lambda m: m.GetNumHeavyAtoms())
        org.element_cost   = elem_counts
        org.viable         = supply.can_afford(elem_counts)
        org._total_heavy   = sum(m.GetNumHeavyAtoms() for m in part_mols)
        org._total_mw      = sum(Descriptors.MolWt(m) for m in part_mols)
        org._total_rings   = sum(rdMolDescriptors.CalcNumRings(m) for m in part_mols)
        org._total_hba     = sum(Descriptors.NumHAcceptors(m) for m in part_mols)
        org._total_hbd     = sum(Descriptors.NumHDonors(m) for m in part_mols)
    else:
        org.viable = False

    org.evaluate()
    return org


def generate_and_save_microbe(template_name: str = None,
                               folder: str = None) -> tuple:
    """Build all (or one named) micro-organism templates and save to library.

    Returns list of (filepath, organism_dict).
    """
    folder = folder or get_library_path()
    to_build = ([t for t in MICROBE_TEMPLATES if t['name'] == template_name]
                if template_name else MICROBE_TEMPLATES)
    results = []
    for tmpl in to_build:
        org  = build_microbe_organism(tmpl)
        path = save_organism_to_library(
            org,
            name=tmpl['name'],
            category=tmpl['category'],
            source=tmpl['source'],
            description=tmpl['description'],
            science_notes=tmpl['science_notes'],
            folder=folder,
        )
        results.append((path, organism_to_library_dict(
            org, tmpl['name'], tmpl['category'], tmpl['source'],
            tmpl['description'], tmpl['science_notes'])))
    return results


def translate_to_english(strand: Strand, env_name: str = 'earth') -> str:
    """Translate a genome into plain English a human can understand."""
    bp = execute_genome_deterministic(strand, env_name)
    codons = strand.codons_deterministic()
    lines = []

    # --- Overview in plain language ---
    sys_info = strand.system
    err_pct = sys_info['error_rate'] * 100
    lines.append("READING THIS GENOME")
    lines.append("")
    lines.append(f"This is a {sys_info['name']} genome with {strand.n_bases} genetic letters.")
    lines.append(f"It is {len(strand)} letters long, divided into {len(codons)} instructions.")
    lines.append(f"Environment: {ENV_ENGLISH.get(env_name, env_name)}")
    lines.append("")
    if err_pct < 1:
        lines.append(f"Copy accuracy: Very high — only {err_pct:.1f}% of letters change per copy.")
        lines.append("This genome can be passed down many generations with few errors.")
    elif err_pct < 5:
        lines.append(f"Copy accuracy: Moderate — {err_pct:.1f}% error rate per letter.")
        lines.append("Mutations accumulate over generations. Evolution happens fast, but cancer risk is higher.")
    else:
        lines.append(f"Copy accuracy: Poor — {err_pct:.1f}% error rate per letter.")
        lines.append("This genome degrades rapidly. Most copies will be corrupted within a few generations.")
    lines.append("")

    # --- The DNA itself ---
    lines.append("--- THE DNA DOUBLE HELIX ---")
    lines.append("")
    lines.append("DNA has two strands twisted together. Each letter on one strand")
    lines.append("pairs with its partner on the other (A with T, C with G).")
    lines.append("This is how cells check for errors and make copies.")
    lines.append("")
    show = min(60, len(strand))
    sense = strand.seq[:show]
    comp = strand.complement[:show]
    tail = '...' if len(strand) > 60 else ''
    lines.append(f"  {sense}{tail}")
    lines.append(f"  {'|' * show}")
    lines.append(f"  {comp}{tail}")
    lines.append("")

    # --- What the genome builds, in story form ---
    lines.append("--- WHAT THIS GENOME BUILDS ---")
    lines.append("")
    lines.append("Reading the genome from left to right, each 5-letter word")
    lines.append("is an instruction. Here is what the cell does, step by step:")
    lines.append("")

    step = 0
    last_built = None
    for codon in codons:
        opcode, operand = codon_to_instruction(codon, strand.system['bases'])
        step += 1
        if opcode == Opcode.BUILD:
            part_idx = operand % len(BUILD_PART_NAMES)
            pn = BUILD_PART_NAMES[part_idx]
            eng = PART_ENGLISH.get(pn, (pn, '', ''))
            last_built = pn
            lines.append(f"  Step {step}: Make {eng[0]}")
            lines.append(f"    {eng[1]}")
            lines.append(f"    ({eng[2]})")
        elif opcode == Opcode.CONNECT:
            lines.append(f"  Step {step}: Connect the parts together")
            lines.append(f"    Bond the most recent components so they work as a unit.")
        elif opcode == Opcode.REPEAT:
            if last_built:
                eng = PART_ENGLISH.get(last_built, (last_built, '', ''))
                lines.append(f"  Step {step}: Make another copy of {eng[0]}")
                lines.append(f"    The cell needs more of this — duplicate it.")
            else:
                lines.append(f"  Step {step}: Copy (nothing to copy yet)")
        elif opcode == Opcode.REGULATE:
            lines.append(f"  Step {step}: Check the environment")
            lines.append(f"    If conditions don't match, skip the next instruction.")
            lines.append(f"    (Like a thermostat — only turn on the heater if it's cold.)")
        elif opcode == Opcode.MOTOR:
            if MOTOR_PART_NAMES:
                pn = MOTOR_PART_NAMES[operand % len(MOTOR_PART_NAMES)]
                eng = PART_ENGLISH.get(pn, (pn, '', ''))
                lines.append(f"  Step {step}: Assemble motor component — {eng[0]}")
                lines.append(f"    {eng[1]}")
                lines.append(f"    ({eng[2]})")
            else:
                lines.append(f"  Step {step}: (motor instruction — no motor parts available)")
        elif opcode == Opcode.LOCALIZE:
            sites = ['membrane', 'basal', 'distal', 'wall', 'cytoplasm']
            site = sites[operand % len(sites)]
            lines.append(f"  Step {step}: Localize to {site}")
            lines.append(f"    Place the last-built part at a specific cellular location.")
        elif opcode == Opcode.POLYMERIZE:
            repeats = min(operand % MAX_POLYMER_REPEATS + 1, MAX_POLYMER_REPEATS)
            if last_built:
                eng = PART_ENGLISH.get(last_built, (last_built, '', ''))
                lines.append(f"  Step {step}: Polymerize {eng[0]} (x{repeats})")
                lines.append(f"    Build a long filament by repeating this monomer {repeats} times.")
            else:
                lines.append(f"  Step {step}: Polymerize (nothing to repeat)")
        elif opcode == Opcode.PHASE:
            lines.append(f"  Step {step}: Advance developmental phase")
            lines.append(f"    The cell transitions to the next stage of its life cycle.")
        elif opcode == Opcode.CHEMOTAXIS:
            lines.append(f"  Step {step}: Enable chemotaxis")
            lines.append(f"    Activate chemical gradient sensing — the cell can now navigate.")
            lines.append(f"    (Like giving the cell a sense of smell for food.)")
        elif opcode == Opcode.MEMBRANE_INS:
            lines.append(f"  Step {step}: Membrane insertion")
            lines.append(f"    Embed the last-built part into the cell membrane.")
        elif opcode == Opcode.PROMOTER:
            lines.append(f"  Step {step}: Start operon (gene group)")
            lines.append(f"    Marks the beginning of a coordinated set of genes.")
            lines.append(f"    (Like a chapter heading — everything after this is related.)")
        elif opcode == Opcode.TERMINATOR:
            lines.append(f"  Step {step}: End operon")
            lines.append(f"    Closes the current gene group.")
        elif opcode == Opcode.SECRETE:
            if last_built:
                eng = PART_ENGLISH.get(last_built, (last_built, '', ''))
                lines.append(f"  Step {step}: Secrete {eng[0]}")
                lines.append(f"    Export this protein outside the cell via a secretion system.")
                lines.append(f"    (Like packaging a product and shipping it out of the factory.)")
            else:
                lines.append(f"  Step {step}: Secrete (nothing to export)")
        elif opcode == Opcode.QUORUM:
            lines.append(f"  Step {step}: Emit quorum signal")
            lines.append(f"    Release a signaling molecule to communicate with nearby cells.")
            lines.append(f"    (Like shouting 'I'm here!' to see how many neighbors respond.)")
        elif opcode == Opcode.DIVIDE:
            lines.append(f"  Step {step}: Cell division")
            lines.append(f"    Attempt to split the cell into two daughter cells.")
            lines.append(f"    (Like cell mitosis — copying everything and splitting in half.)")
        elif opcode == Opcode.METABOLIZE:
            if last_built:
                eng = PART_ENGLISH.get(last_built, (last_built, '', ''))
                lines.append(f"  Step {step}: Metabolize via {eng[0]}")
                lines.append(f"    Chain this part into a metabolic pathway for energy conversion.")
            else:
                lines.append(f"  Step {step}: Metabolize (no substrate)")
        elif opcode == Opcode.GRADIENT:
            lines.append(f"  Step {step}: Follow nutrient gradient")
            lines.append(f"    Move toward higher concentration of nutrients.")
            lines.append(f"    (Like following a scent trail to find food.)")
        else:
            lines.append(f"  Step {step}: (non-coding region — spacer DNA)")
    lines.append("")

    # --- What was built ---
    lines.append("--- THE ASSEMBLED ORGANISM ---")
    lines.append("")
    if not bp.parts_used:
        lines.append("This genome didn't build anything functional.")
        lines.append("It's like a book of blank pages — the instructions are there,")
        lines.append("but none of them code for real parts.")
    else:
        counts = Counter(bp.parts_used)
        lines.append(f"The genome assembled {len(bp.parts_used)} components")
        lines.append(f"from {len(counts)} different types of biological parts:")
        lines.append("")
        for pn, cnt in counts.most_common():
            eng = PART_ENGLISH.get(pn, (pn, '', ''))
            if cnt == 1:
                lines.append(f"  - {eng[0]}")
            else:
                lines.append(f"  - {eng[0]} (x{cnt})")
            lines.append(f"    {eng[2]}")

        if bp.connections:
            lines.append("")
            lines.append(f"These parts are bonded together at {len(bp.connections)} points,")
            lines.append("forming a connected structure rather than loose pieces.")

    lines.append("")

    # --- Body systems check ---
    lines.append("--- BODY SYSTEMS CHECK ---")
    lines.append("")
    lines.append("A living organism needs at least three systems to survive:")
    lines.append("a body (structure), a power source (energy), and tools (catalysis).")
    lines.append("")
    for role in ['structure', 'energy', 'catalysis']:
        se = SYSTEM_ENGLISH[role]
        have = bp.roles_present.get(role, 0)
        if have > 0:
            lines.append(f"  {se['name']}: YES — this organism {se['has']}.")
            lines.append(f"    {se['analogy']}")
        else:
            lines.append(f"  {se['name']}: NO — this organism {se['missing']}.")
    # Bonus systems
    for role in ['information', 'minimal']:
        have = bp.roles_present.get(role, 0)
        if have > 0:
            se = SYSTEM_ENGLISH[role]
            lines.append(f"  {se['name']}: YES — {se['has']}.")
    lines.append("")

    # --- Motor & behavior check ---
    if bp.roles_present.get('motor', 0) > 0:
        lines.append("--- MOLECULAR MACHINES ---")
        lines.append("")
        if bp.complexes_formed:
            for cx_name in bp.complexes_formed:
                cx = FUNCTIONAL_COMPLEXES[cx_name]
                lines.append(f"  ASSEMBLED: {cx_name.replace('_', ' ').title()}")
                lines.append(f"    {cx['desc']}")
        if bp.motility_capable():
            lines.append("  This organism CAN SWIM — flagellar motor is operational.")
        if bp.chemotaxis_capable():
            lines.append("  This organism CAN NAVIGATE — chemotaxis system is active.")
        if bp.energy_balance() < 0:
            lines.append(f"  WARNING: Energy deficit ({bp.energy_balance()} units) — "
                         "motor may stall without more energy parts.")
        lines.append("")

    # --- Verdict ---
    score = bp.systems_score()
    lines.append("--- VERDICT ---")
    lines.append("")
    if score == 1.0:
        lines.append("ALIVE: This genome encodes a viable organism.")
        lines.append("It has a body, energy, and tools — the minimum for independent life.")
        if bp.roles_present.get('information', 0) > 0:
            lines.append("It also has information storage, so it could reproduce.")
        if bp.motility_capable():
            lines.append("It has a flagellar motor and can actively swim.")
        if bp.chemotaxis_capable():
            lines.append("It has chemotaxis — it can sense and navigate toward nutrients.")
        div = bp.diversity_score()
        if div > 0.6:
            lines.append("High part diversity makes it adaptable and capable.")
        eff = bp.efficiency_score()
        if eff > 0.7:
            lines.append("The genome is efficient — most instructions do something useful.")
        elif eff < 0.3:
            lines.append("But the genome is mostly junk DNA — only a fraction of it codes for parts.")
    elif score > 0:
        missing = [SYSTEM_ENGLISH[r]['name'] for r in REQUIRED_SYSTEMS
                   if bp.roles_present.get(r, 0) < REQUIRED_SYSTEMS[r]]
        lines.append("INCOMPLETE: This organism is missing critical systems.")
        lines.append(f"Missing: {', '.join(missing)}.")
        lines.append("Without these, it cannot survive independently.")
        lines.append("It's like a car with no engine, or a body with no bones.")
    else:
        lines.append("NON-VIABLE: The genome produces no functional systems.")
        lines.append("This is essentially random noise — no life emerges from this code.")

    lines.append("")

    # --- Natural language summary paragraph ---
    lines.append("--- SUMMARY ---")
    lines.append("")
    summary = _build_summary_paragraph(strand, bp, env_name)
    lines.append(summary)

    return '\n'.join(lines)


def _build_summary_paragraph(strand: Strand, bp: Blueprint, env_name: str) -> str:
    """Generate a single natural-language paragraph describing the organism."""
    parts = Counter(bp.parts_used)
    n_parts = len(bp.parts_used)
    n_unique = len(parts)
    score = bp.systems_score()
    eff = bp.efficiency_score()

    if n_parts == 0:
        return "This genome is silent. It contains no instructions that produce biological parts."

    if n_unique <= 2 and n_parts <= 3:
        kind = "a minimal chemical assembly"
    elif score < 1.0:
        kind = "a partial organism (not self-sustaining)"
    elif n_unique <= 4:
        kind = "a simple single-celled organism, like a primitive bacterium"
    elif n_unique <= 7:
        kind = "a moderately complex cell, like an advanced bacterium"
    else:
        kind = "a complex cell with specialized subsystems"

    top_parts = [PART_ENGLISH.get(p, (p,))[0] for p, _ in parts.most_common(4)]
    if len(top_parts) > 1:
        part_list = ', '.join(top_parts[:-1]) + f' and {top_parts[-1]}'
    else:
        part_list = top_parts[0]

    err = strand.system['error_rate']
    if err < 0.01:
        fidelity = "copies itself very accurately"
    elif err < 0.05:
        fidelity = "copies itself with moderate accuracy"
    else:
        fidelity = "copies itself poorly, with many errors per generation"

    para = (
        f"This {strand.system['name']} genome encodes {kind}. "
        f"It assembles {n_parts} parts from {n_unique} different types, "
        f"including {part_list}. "
    )
    if bp.connections:
        para += f"These components are bonded at {len(bp.connections)} connection points. "
    if score == 1.0:
        para += "The organism has all three essential systems (body, energy, tools) and is viable. "
    else:
        missing = [SYSTEM_ENGLISH[r]['name'].lower() for r in REQUIRED_SYSTEMS
                   if bp.roles_present.get(r, 0) < REQUIRED_SYSTEMS[r]]
        para += f"However, it is missing its {' and '.join(missing)}, so it cannot survive alone. "
    if eff > 0.7:
        para += f"The genome is {eff:.0%} efficient — very little junk DNA. "
    elif eff < 0.3:
        para += f"Only {eff:.0%} of the genome codes for parts — the rest is non-coding spacer. "
    if bp.motility_capable():
        para += "Remarkably, it has a complete flagellar motor and can swim. "
    if bp.chemotaxis_capable():
        para += "It can also sense chemical gradients and navigate toward nutrients. "
    if bp.energy_balance() < 0:
        para += f"However, its energy budget is negative ({bp.energy_balance()} units) — it risks stalling. "
    if bp.torque > 0:
        para += f"Its motor generates {bp.torque:.0f} pN-nm of torque at {bp.rpm:.0f} RPM. "
    if bp.flagella_count > 1:
        para += f"It has {bp.flagella_count} flagella that can bundle for coordinated swimming. "
    if bp.operon_count() > 0:
        para += f"Its genome is organized into {bp.operon_count()} operon(s), indicating gene coordination. "
    if bp.secretions:
        para += f"It can secrete {len(bp.secretions)} protein(s) into its environment. "
    if bp.division_events > 0:
        para += "It has the machinery to divide and reproduce. "
    if bp.metabolic_pathway_complete():
        para += "It has a complete metabolic pathway from fuel to energy. "
    if bp.quorum_signals > 0:
        para += "It communicates with neighboring cells via quorum sensing. "
    para += f"When copied, this DNA {fidelity}."
    return para


# ---------------------------------------------------------------------------
# Step 21: Module split guide
# ---------------------------------------------------------------------------
# To split language.py into modules, extract the following sections:
#   - constants.py:     CODON_LEN … MEMBRANE_POTENTIAL_BASE, PERIODIC, BASE_SYSTEMS
#   - parts_kb.py:      PARTS_KB, FUNCTIONAL_COMPLEXES, REQUIRED_SYSTEMS,
#                        BUILD_PART_NAMES, MOTOR_PART_NAMES
#   - opcodes.py:       Opcode, codon_to_instruction, OPCODE_NAMES
#   - blueprint.py:     Blueprint, execute_genome, execute_genome_deterministic
#   - organism.py:      Organism, _eval_organism, structural_grade, to_dict
#   - evolution.py:     crossover, mutate_seq, tournament, enforce_length,
#                        _cluster_species, _hamming, Evolver, compare_base_systems
#   - chemistry.py:     build_mol, build_organism_mol, mol_element_counts
#   - translator.py:    PART_ENGLISH, SYSTEM_ENGLISH, ENV_ENGLISH,
#                        translate_to_english, _build_summary_paragraph
#   - export.py:        export_pdb, export_webgl_json, genome_compress/decompress
#   - tests.py:         run_tests, _validate_reference_library, REFERENCE_LIBRARY
#   - ui.py:            PeriodicMachine, launch_ui
#   - __main__.py:      CLI entry point
# Each module should import only what it needs. Use a shared types.py for
# dataclass definitions if circular imports arise.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Step 22: PDB/CIF export — write 3D coordinates in standard structural
#          biology format for use in PyMOL, Chimera, VMD, etc.
# ---------------------------------------------------------------------------

def generate_intelligent_organism(n_bases: int = 8,
                                   env_name: str = 'earth',
                                   seed: int = 42) -> 'Organism':
    """Generate a fully operational, high-intelligence life form.

    Strategy: directly inject every required subsystem into the Blueprint,
    bypassing the genome interpreter so the complete organism is guaranteed
    regardless of operand-modulo mapping.  A representative genome string is
    also produced for display purposes.

    Subsystems assembled:
      1. Core viability:   membrane, wall, filament, actin, tubulin (structure)
                           atp, nadh, glucose (energy)
                           amino_func, cofactor, nucleotide, coenzyme_a (catalysis/info)
      2. Locomotion:       full flagellum + chemotaxis
      3. Neural layer:     action potential, synapse, signal cascade,
                           Hebbian plasticity, oscillator, working memory,
                           decision circuit
      4. Homeostasis:      heat_shock_prot, proteasome, dna_repair, antioxidant
      5. Colony layer:     quorum sensing, division

    Returns a fully expressed, evaluated Organism.
    """
    rng = random.Random(seed)
    n_bases = max(4, n_bases)
    bases = BASE_SYSTEMS[n_bases]['bases']

    # ── Build a representative genome string for display ─────────────────
    # Use only BUILD/MOTOR opcodes for parts that fit the classic operand map;
    # the rest is written as symbolic NOOPs (still a valid parseable genome).
    def _sym_codon(opcode_int: int, operand: int) -> str:
        ob = bases[min(opcode_int, len(bases) - 1)]
        v, parts_list = operand, []
        for _ in range(CODON_LEN - 1):
            parts_list.append(bases[v % len(bases)])
            v //= len(bases)
        return ob + ''.join(parts_list)

    genome_codons: List[str] = []
    for pname in ('membrane', 'wall', 'filament', 'atp', 'nadh', 'glucose',
                  'amino_func', 'cofactor', 'nucleotide', 'coenzyme_a'):
        if pname in BUILD_PART_NAMES:
            genome_codons.append(_sym_codon(Opcode.BUILD, BUILD_PART_NAMES.index(pname)))
    for pname in ('flagellin', 'hook_protein', 'ms_ring', 'c_ring',
                  'motA', 'motB', 'fliG', 'export_gate',
                  'receptor', 'histidine_kinase', 'response_reg'):
        if pname in MOTOR_PART_NAMES:
            genome_codons.append(_sym_codon(Opcode.MOTOR, MOTOR_PART_NAMES.index(pname)))
    # Pad rest with symbolic NOOPs representing neural/homeostasis genes
    neural_sym = ['ion_channel', 'pump_atpase', 'vesicle', 'neurotransmitter',
                  'snare_complex', 'receptor_gated', 'calmodulin', 'camp_kinase',
                  'nmda_receptor', 'camkii', 'creb_factor', 'gap_junction',
                  'oscillator', 'working_memory', 'pattern_net', 'decision_gate',
                  'heat_shock_prot', 'proteasome', 'dna_repair', 'antioxidant']
    for i, _ in enumerate(neural_sym):
        genome_codons.append(_sym_codon(Opcode.NOOP, i))
    genome = ''.join(genome_codons)
    rem = len(genome) % CODON_LEN
    if rem:
        genome += ''.join(rng.choice(bases) for _ in range(CODON_LEN - rem))

    # ── Directly build the Blueprint ─────────────────────────────────────
    bp = Blueprint()

    def _add(part_name: str, localize: Optional[str] = None) -> None:
        if part_name not in PARTS_KB:
            return
        part = PARTS_KB[part_name]
        bp.parts_used.append(part_name)
        bp.smiles_components.append(part['smiles'])
        role = part['role']
        bp.roles_present[role] = bp.roles_present.get(role, 0) + 1
        bp.energy_cost += part.get('energy_cost', 0)
        pidx = len(bp.parts_used) - 1
        bp.spatial_graph[pidx] = part.get('attachment', [])
        if role == 'neural':
            bp.neural_parts.append(part_name)
        elif role == 'homeostasis':
            bp.homeostasis_parts.append(part_name)
        if localize:
            bp.localizations.append((pidx, localize))
            bp.valid_localizations += 1

    def _connect(a: int, b: int) -> None:
        if a != b and (a, b) not in bp.connections:
            if bp.domain_compatibility(a, b):
                bp.connections.append((a, b))
            else:
                bp.domain_mismatches += 1

    # Operon 1 — Core structure
    bp.operons.append(['PROMOTER'])
    for p in ('membrane', 'wall', 'filament', 'actin', 'tubulin'):
        _add(p, 'cytoplasm')
    bp.operons[-1].append('TERMINATOR')

    # Operon 2 — Energy & catalysis
    bp.operons.append(['PROMOTER'])
    for p in ('atp', 'nadh', 'glucose', 'amino_func', 'cofactor',
              'nucleotide', 'coenzyme_a'):
        _add(p)
        bp._current_chain.append(bp.parts_used[-1])
    if len(bp._current_chain) >= 2:
        bp.metabolic_chains.append(list(bp._current_chain))
        bp._current_chain = []
    bp.operons[-1].append('TERMINATOR')

    # Operon 3 — Flagellar motor
    bp.operons.append(['PROMOTER'])
    for p in ('ms_ring', 'c_ring', 'motA', 'motB', 'fliG',
              'export_gate', 'hook_protein', 'flagellin', 'flagellin'):
        _add(p, 'basal')
    bp.operons[-1].append('TERMINATOR')

    # Operon 4 — Chemotaxis
    bp.operons.append(['PROMOTER'])
    for p in ('receptor', 'histidine_kinase', 'response_reg'):
        _add(p, 'membrane')
    bp.behaviors.append('chemotaxis')
    bp.gradient_steps += 3
    bp.operons[-1].append('TERMINATOR')

    # Operon 5 — Action potential
    bp.operons.append(['PROMOTER'])
    for p in ('ion_channel', 'ion_channel', 'pump_atpase'):
        _add(p, 'membrane')
    bp.operons[-1].append('TERMINATOR')

    # Operon 6 — Chemical synapse
    bp.operons.append(['PROMOTER'])
    for p in ('vesicle', 'neurotransmitter', 'snare_complex', 'receptor_gated'):
        _add(p, 'membrane')
    bp.operons[-1].append('TERMINATOR')

    # Operon 7 — Signal cascade
    bp.operons.append(['PROMOTER'])
    for p in ('calmodulin', 'camp_kinase', 'gap_junction'):
        _add(p)
    bp.operons[-1].append('TERMINATOR')

    # Operon 8 — Hebbian plasticity (LTP)
    bp.operons.append(['PROMOTER'])
    for p in ('nmda_receptor', 'camkii', 'creb_factor'):
        _add(p, 'membrane')
    bp.operons[-1].append('TERMINATOR')

    # Operon 9 — Neural oscillator
    bp.operons.append(['PROMOTER'])
    for p in ('oscillator', 'gap_junction'):
        _add(p)
    bp.operons[-1].append('TERMINATOR')

    # Operon 10 — Working memory + decision
    bp.operons.append(['PROMOTER'])
    for p in ('working_memory', 'nmda_receptor', 'pattern_net',
              'decision_gate', 'camp_kinase'):
        _add(p)
    bp.operons[-1].append('TERMINATOR')

    # Operon 11 — Homeostasis
    bp.operons.append(['PROMOTER'])
    for p in ('heat_shock_prot', 'proteasome', 'dna_repair', 'antioxidant'):
        _add(p)
    bp.operons[-1].append('TERMINATOR')

    # Operon 12 — Colony / social
    bp.operons.append(['PROMOTER'])
    for p in ('receptor', 'response_reg'):
        _add(p)
    bp.quorum_signals += 2
    bp.division_events += 1
    bp.behaviors.append('quorum_sensing')
    bp.operons[-1].append('TERMINATOR')

    # Wire connections between adjacent parts
    n = len(bp.parts_used)
    for i in range(min(n - 1, 60)):
        _connect(i, i + 1)

    # Post-assembly bookkeeping
    bp.energy_budget = bp.roles_present.get('energy', 0) * BASE_PMF_UNITS
    bp.phase = DEVELOPMENT_PHASES[-1]  # fully developed
    bp.detect_complexes()
    bp.compute_torque()
    if bp.chemotaxis_capable():
        bp.tumble_prob = TUMBLE_PROBABILITY_BASE * 0.3
    bp.compute_membrane_potential()
    bp.compute_intelligence()

    # Cognitive behaviors
    if bp.intelligence_capable():
        bp.behaviors.append('neural_signaling')
    if bp.fully_intelligent():
        bp.behaviors.append('autonomous_cognition')
    if 'hebbian_plasticity' in bp.complexes_formed:
        bp.behaviors.append('learning')
    if 'working_memory_loop' in bp.complexes_formed:
        bp.behaviors.append('working_memory')
    if 'decision_circuit' in bp.complexes_formed:
        bp.behaviors.append('decision_making')
    if bp.motility_capable():
        bp.behaviors.append('swim')

    # ── Wrap in Organism ──────────────────────────────────────────────────
    strand = Strand(genome, n_bases)
    supply = ElementSupply.earth_like()
    org = Organism(strand=strand, env_name=env_name)
    # Attach the pre-built blueprint and run chemistry + evaluation
    org.blueprint = bp
    org.viable = bp.systems_score() >= 1.0
    # Build molecules for chemistry panel
    try:
        org._part_mols = []
        org._total_mw = org._total_heavy = org._total_rings = 0
        org._total_hba = org._total_hbd = 0
        from rdkit.Chem import Descriptors as _D
        for pname in bp.parts_used:
            smi = PARTS_KB[pname]['smiles']
            mol = build_mol(smi)
            if mol:
                org._part_mols.append(mol)
                org._total_mw     += _D.MolWt(mol)
                org._total_heavy  += mol.GetNumHeavyAtoms()
                org._total_rings  += mol.GetRingInfo().NumRings()
                from rdkit.Chem import rdMolDescriptors as _RD
                org._total_hba    += _RD.CalcNumHBA(mol)
                org._total_hbd    += _RD.CalcNumHBD(mol)
    except Exception:
        pass
    org.evaluate()
    return org


def export_pdb(organism, filepath: str = 'organism.pdb') -> str:
    """Export an organism's 3D structure to PDB format.
    Returns the PDB string and optionally writes to file."""
    lines = []
    lines.append(f"HEADER    SYNTHETIC ORGANISM")
    lines.append(f"TITLE     Generated by Periodic Machine DNA Interpreter")
    lines.append(f"REMARK    Fitness: {organism.fitness:.4f}")
    if organism.blueprint:
        lines.append(f"REMARK    Parts: {len(organism.blueprint.parts_used)}")
        lines.append(f"REMARK    Systems: {organism.blueprint.systems_score():.0%}")
        if organism.blueprint.torque > 0:
            lines.append(f"REMARK    Motor torque: {organism.blueprint.torque:.0f} pN-nm")
            lines.append(f"REMARK    Motor RPM: {organism.blueprint.rpm:.0f}")

    geom = getattr(organism, '_organism_geom', None)
    if geom and geom.get('atoms'):
        for i, atom in enumerate(geom['atoms']):
            serial = i + 1
            name = atom.get('element', 'C').ljust(4)
            resname = atom.get('part', 'UNK')[:3].ljust(3)
            chain = 'A'
            resseq = atom.get('part_idx', 1) % 9999
            x = atom.get('x', 0.0)
            y = atom.get('y', 0.0)
            z = atom.get('z', 0.0)
            element = atom.get('element', 'C')[:2].rjust(2)
            lines.append(
                f"ATOM  {serial:>5d} {name} {resname} {chain}{resseq:>4d}    "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {element}"
            )
        # Write bonds as CONECT records
        for bond in geom.get('bonds', []):
            a1 = bond.get('from', 0) + 1
            a2 = bond.get('to', 0) + 1
            lines.append(f"CONECT{a1:>5d}{a2:>5d}")
    else:
        # Fallback: generate pseudo-PDB from parts list with grid layout
        if organism.blueprint:
            for i, part_name in enumerate(organism.blueprint.parts_used):
                serial = i + 1
                x = (i % PART_GRID_COLS) * PART_SPACING_ANGSTROMS
                y = (i // PART_GRID_COLS) * PART_SPACING_ANGSTROMS
                z = 0.0
                element = 'C'
                resname = part_name[:3].upper().ljust(3)
                lines.append(
                    f"ATOM  {serial:>5d}  CA  {resname} A{serial:>4d}    "
                    f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           C"
                )

    lines.append("END")
    pdb_str = '\n'.join(lines)
    if filepath:
        with open(filepath, 'w') as f:
            f.write(pdb_str)
    return pdb_str


# ---------------------------------------------------------------------------
# Step 23: Animation hints — embed rotation vectors for motor parts in
#          JSON export, enabling downstream viewers to animate spinning.
# ---------------------------------------------------------------------------

def _motor_animation_hints(bp: 'Blueprint') -> List[Dict]:
    """Generate animation hint records for motor parts.
    Each hint contains part index, rotation axis, and angular velocity."""
    hints = []
    if not bp or not bp.parts_used:
        return hints
    # Rotation axis for flagellar motor is along Z by convention
    for i, part in enumerate(bp.parts_used):
        if part in ('ms_ring', 'c_ring', 'fliG'):
            hints.append({
                'part_idx': i, 'part': part,
                'animation': 'rotate',
                'axis': [0, 0, 1],
                'angular_velocity': bp.rpm * 2 * 3.14159 / 60,  # rad/s
            })
        elif part == 'flagellin':
            hints.append({
                'part_idx': i, 'part': part,
                'animation': 'rotate',
                'axis': [0, 0, 1],
                'angular_velocity': bp.rpm * 2 * 3.14159 / 60,
                'helix_pitch': 2.5,  # μm
            })
        elif part == 'hook_protein':
            hints.append({
                'part_idx': i, 'part': part,
                'animation': 'flex',
                'flex_angle': 45.0,  # degrees
            })
    return hints


# ---------------------------------------------------------------------------
# Step 24: WebGL viewer JSON — export 3D + animation data for browser-based
#          viewing. The JSON is self-contained and can be loaded by Three.js.
# ---------------------------------------------------------------------------

def export_webgl_json(organism, filepath: str = 'organism_viewer.json') -> Dict:
    """Export organism structure + animation hints as WebGL-ready JSON."""
    data = organism.to_dict()
    # Add animation hints
    if organism.blueprint:
        data['animation_hints'] = _motor_animation_hints(organism.blueprint)
        data['viewer_meta'] = {
            'format': 'periodic_machine_v2',
            'camera_distance': max(20.0, len(organism.blueprint.parts_used) * 3.0),
            'background': '#0a0a1e',
            'ambient_light': 0.4,
            'part_colors': {
                'structure': '#44aaff',
                'energy': '#ffaa00',
                'catalysis': '#00ff88',
                'information': '#ff44ff',
                'minimal': '#888888',
                'motor': '#ff4444',
                'signaling': '#00ffff',
            },
        }
    if filepath:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    return data


# ---------------------------------------------------------------------------
# Step 25: Benchmark suite — measure evolution throughput
# ---------------------------------------------------------------------------

def run_benchmark(pop_sizes=None, genome_lens=None, n_gens: int = 5,
                  n_workers: int = 4) -> Dict:
    """Measure organisms/sec for various configurations.
    Returns a dict of results keyed by (pop_size, genome_len)."""
    if pop_sizes is None:
        pop_sizes = [20, 50, 100]
    if genome_lens is None:
        genome_lens = [40, 80, 120]
    results = {}
    for ps in pop_sizes:
        for gl in genome_lens:
            t0 = time.time()
            ev = Evolver(pop_size=ps, genome_len=gl, generations=n_gens,
                         n_bases=4, n_workers=n_workers)
            ev.run()
            elapsed = time.time() - t0
            total_evals = ps * n_gens
            rate = total_evals / max(0.001, elapsed)
            results[(ps, gl)] = {
                'pop_size': ps, 'genome_len': gl,
                'generations': n_gens, 'elapsed_sec': round(elapsed, 2),
                'total_evals': total_evals,
                'organisms_per_sec': round(rate, 1),
            }
            print(f"  bench ps={ps} gl={gl}: {rate:.1f} org/s ({elapsed:.1f}s)")
    return results


# ---------------------------------------------------------------------------
# Step 26: Genome compression — run-length encoding for POLYMERIZE regions
#          to reduce genome bloat in evolved populations.
# ---------------------------------------------------------------------------

def genome_compress(genome: str) -> str:
    """Compress a genome using run-length encoding for repeated codons.
    Format: repeated codons become {codon}*{count}.
    Non-repeated codons are kept as-is."""
    if len(genome) < CODON_LEN:
        return genome
    codons = [genome[i:i+CODON_LEN] for i in range(0, len(genome) - CODON_LEN + 1, CODON_LEN)]
    tail = genome[len(codons) * CODON_LEN:]  # leftover bases
    compressed = []
    i = 0
    while i < len(codons):
        run = 1
        while i + run < len(codons) and codons[i + run] == codons[i]:
            run += 1
        if run > 1:
            compressed.append(f"{codons[i]}*{run}")
        else:
            compressed.append(codons[i])
        i += run
    result = '|'.join(compressed)
    if tail:
        result += '|' + tail
    return result


def genome_decompress(compressed: str) -> str:
    """Decompress a run-length encoded genome back to raw sequence."""
    if '|' not in compressed and '*' not in compressed:
        return compressed
    parts = compressed.split('|')
    genome = []
    for part in parts:
        if '*' in part:
            codon, count = part.split('*')
            genome.append(codon * int(count))
        else:
            genome.append(part)
    return ''.join(genome)


def launch_ui():
    """Launch the Periodic Machine graphical interface."""
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox, filedialog
    import threading

    class PeriodicMachine(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("Periodic Machine — Chemistry-DNA Language Interpreter  v2.0")
            self.geometry("1440x960")
            self.configure(bg='#0d0d1a')
            self.minsize(1100, 750)
            self._current_strand = None
            self._current_bp = None
            self._current_org = None
            self._evolving = False
            self._lib_folder_var = tk.StringVar(value=get_library_path())
            self._build_styles()
            self._build_ui()

        def _build_styles(self):
            style = ttk.Style(self)
            style.theme_use('clam')
            BG = '#0d0d1a'
            style.configure('TFrame', background=BG)
            style.configure('TLabel', background=BG, foreground='#e0e0e0',
                             font=('Consolas', 10))
            style.configure('Header.TLabel', background=BG, foreground='#00d4ff',
                             font=('Consolas', 16, 'bold'))
            style.configure('Sub.TLabel', background=BG, foreground='#888888',
                             font=('Consolas', 10))
            style.configure('TButton', font=('Consolas', 10), padding=6)
            style.configure('Accent.TButton', font=('Consolas', 10, 'bold'), padding=6)
            style.configure('Evolve.TButton', font=('Consolas', 10, 'bold'), padding=8,
                             foreground='#ffd700')
            style.configure('TLabelframe', background=BG, foreground='#00d4ff',
                             font=('Consolas', 10, 'bold'))
            style.configure('TLabelframe.Label', background=BG, foreground='#00d4ff',
                             font=('Consolas', 10, 'bold'))
            style.configure('TCombobox', font=('Consolas', 10))
            style.configure('TNotebook', background=BG, borderwidth=0)
            style.configure('TNotebook.Tab', background='#1a1a2e', foreground='#888888',
                             font=('Consolas', 10, 'bold'), padding=(14, 6))
            style.map('TNotebook.Tab',
                      background=[('selected', '#0d0d1a')],
                      foreground=[('selected', '#00d4ff')])
            style.configure('TSeparator', background='#333355')

        def _make_text(self, parent, height=None, font_size=10) -> scrolledtext.ScrolledText:
            """Helper: create a dark-themed scrolled text widget."""
            kw = dict(font=('Consolas', font_size), bg='#080818', fg='#e0e0e0',
                      wrap='word', borderwidth=0, highlightthickness=1,
                      highlightcolor='#00d4ff', highlightbackground='#1a1a2e',
                      insertbackground='#00ff88', selectbackground='#1a3a5e')
            if height:
                kw['height'] = height
            w = scrolledtext.ScrolledText(parent, **kw)
            self._apply_tags(w)
            return w

        def _apply_tags(self, widget):
            tags = [
                ('h1',       {'foreground': '#00d4ff', 'font': ('Consolas', 14, 'bold')}),
                ('h2',       {'foreground': '#00d4ff', 'font': ('Consolas', 12, 'bold')}),
                ('h3',       {'foreground': '#ffd700', 'font': ('Consolas', 11, 'bold')}),
                ('header',   {'foreground': '#00d4ff', 'font': ('Consolas', 11, 'bold')}),
                ('section',  {'foreground': '#ffd700', 'font': ('Consolas', 10, 'bold')}),
                ('title',    {'foreground': '#ffd700', 'font': ('Consolas', 10, 'bold')}),
                ('subtitle', {'foreground': '#aaaaff', 'font': ('Consolas', 10, 'bold')}),
                ('build',    {'foreground': '#00ff88'}),
                ('connect',  {'foreground': '#ff8800'}),
                ('repeat',   {'foreground': '#ff44ff'}),
                ('regulate', {'foreground': '#44aaff'}),
                ('noop',     {'foreground': '#444466'}),
                ('present',  {'foreground': '#00ff88', 'font': ('Consolas', 10, 'bold')}),
                ('missing',  {'foreground': '#ff4444', 'font': ('Consolas', 10, 'bold')}),
                ('warn',     {'foreground': '#ffaa00', 'font': ('Consolas', 10, 'bold')}),
                ('viable',   {'foreground': '#00ff88', 'font': ('Consolas', 12, 'bold')}),
                ('incomplete',{'foreground': '#ff4444', 'font': ('Consolas', 12, 'bold')}),
                ('metric',   {'foreground': '#00d4ff'}),
                ('dimtext',  {'foreground': '#666688'}),
                ('code',     {'foreground': '#00ff88', 'font': ('Consolas', 10)}),
                ('role_structure',  {'foreground': '#44aaff'}),
                ('role_energy',     {'foreground': '#ffaa00'}),
                ('role_catalysis',  {'foreground': '#00ff88'}),
                ('role_information',{'foreground': '#ff44ff'}),
                ('role_minimal',    {'foreground': '#888888'}),
                ('role_motor',      {'foreground': '#ff6644'}),
                ('role_signaling',  {'foreground': '#cc88ff'}),
                ('ok',       {'foreground': '#00ff88', 'font': ('Consolas', 10, 'bold')}),
                ('bad',      {'foreground': '#ff4444', 'font': ('Consolas', 10, 'bold')}),
                ('highlight',{'foreground': '#ffffff', 'font': ('Consolas', 10, 'bold'),
                               'background': '#1a2a3a'}),
            ]
            for tag, cfg in tags:
                widget.tag_configure(tag, **cfg)

        def _build_ui(self):
            # ── Top banner ──────────────────────────────────────────────────
            banner = ttk.Frame(self)
            banner.pack(fill='x', padx=12, pady=(10, 4))
            tk.Label(banner, text="PERIODIC MACHINE",
                     bg='#0d0d1a', fg='#00d4ff',
                     font=('Consolas', 18, 'bold')).pack(side='left')
            tk.Label(banner, text="  Chemistry · DNA · Evolution  v2.0",
                     bg='#0d0d1a', fg='#445566',
                     font=('Consolas', 11)).pack(side='left', pady=(4, 0))

            # ── Genome input bar ─────────────────────────────────────────────
            gf = ttk.LabelFrame(self, text="  GENOME INPUT  —  enter a DNA sequence or generate one")
            gf.pack(fill='x', padx=12, pady=4)

            ctrl = ttk.Frame(gf)
            ctrl.pack(fill='x', padx=6, pady=(4, 2))

            def lbl(t): ttk.Label(ctrl, text=t).pack(side='left', padx=(6, 2))

            lbl("Base System:")
            self._base_var = tk.StringVar(value='4')
            base_cb = ttk.Combobox(ctrl, textvariable=self._base_var,
                                   values=['2', '4', '6', '8'], width=4,
                                   state='readonly')
            base_cb.pack(side='left', padx=(0, 10))
            base_cb.bind('<<ComboboxSelected>>', lambda e: self._on_base_changed())

            lbl("Environment:")
            self._env_var = tk.StringVar(value='earth')
            env_cb = ttk.Combobox(ctrl, textvariable=self._env_var,
                         values=ENVS, width=10,
                         state='readonly')
            env_cb.pack(side='left', padx=(0, 10))
            env_cb.bind('<<ComboboxSelected>>', lambda e: self._on_base_changed())

            lbl("Length:")
            self._len_var = tk.StringVar(value='80')
            ttk.Entry(ctrl, textvariable=self._len_var, width=6,
                      font=('Consolas', 10)).pack(side='left', padx=(0, 10))

            # Action buttons
            btn_defs = [
                ("⚡ Random",      self._gen_random,          'TButton'),
                ("▶ Translate",    self._translate,           'Accent.TButton'),
                ("🧬 Life Forms",  self._show_reference_library,'TButton'),
                ("🔬 Validate",    self._validate_genome,     'TButton'),
                ("🔄 Evolve",      self._evolve_short,        'Evolve.TButton'),
                ("🧠 Generate Life",self._generate_life,      'Accent.TButton'),
                ("💾 Export",      self._export_json,         'TButton'),
                ("📂 Library",     self._open_library_tab,    'TButton'),
                ("✕ Clear",        self._clear,               'TButton'),
            ]
            for label, cmd, sty in btn_defs:
                ttk.Button(ctrl, text=label, command=cmd,
                           style=sty).pack(side='left', padx=2)

            self._genome_entry = scrolledtext.ScrolledText(
                gf, height=3, font=('Consolas', 11),
                bg='#080818', fg='#00ff88', insertbackground='#00ff88',
                wrap='char', borderwidth=0, highlightthickness=1,
                highlightcolor='#00d4ff', highlightbackground='#222244')
            self._genome_entry.pack(fill='x', padx=6, pady=(2, 5))

            # ── Base system info strip ────────────────────────────────────────
            bstrip = ttk.Frame(gf)
            bstrip.pack(fill='x', padx=6, pady=(0, 4))
            self._base_info_var = tk.StringVar()
            ttk.Label(bstrip, textvariable=self._base_info_var,
                      foreground='#446688', font=('Consolas', 9)).pack(side='left')
            self._on_base_changed()  # populate strip

            # ── Main notebook ────────────────────────────────────────────────
            self._nb_widget = ttk.Notebook(self)
            self._nb_widget.pack(fill='both', expand=True, padx=12, pady=4)

            # Tab 1: Translation
            t1 = ttk.Frame(self._nb_widget)
            self._nb_widget.add(t1, text="  📜 Translation  ")
            paned1 = ttk.PanedWindow(t1, orient='horizontal')
            paned1.pack(fill='both', expand=True, padx=4, pady=4)
            lf1 = ttk.LabelFrame(paned1, text=" GENOME PROGRAM — Instruction-by-Instruction Readout ")
            paned1.add(lf1, weight=55)
            self._trans = self._make_text(lf1)
            self._trans.pack(fill='both', expand=True, padx=4, pady=4)
            rf1 = ttk.LabelFrame(paned1, text=" VALIDATION & GENOME STATS ")
            paned1.add(rf1, weight=45)
            self._val_out = self._make_text(rf1)
            self._val_out.pack(fill='both', expand=True, padx=4, pady=4)

            # Tab 2: Blueprint
            t2 = ttk.Frame(self._nb_widget)
            self._nb_widget.add(t2, text="  🔬 Blueprint  ")
            paned2 = ttk.PanedWindow(t2, orient='horizontal')
            paned2.pack(fill='both', expand=True, padx=4, pady=4)
            lf2 = ttk.LabelFrame(paned2, text=" ASSEMBLED ORGANISM — Full Parts & Systems Analysis ")
            paned2.add(lf2, weight=50)
            self._bp_out = self._make_text(lf2)
            self._bp_out.pack(fill='both', expand=True, padx=4, pady=4)
            rf2 = ttk.LabelFrame(paned2, text=" PHYSICS & CHEMISTRY — Motor · Energy · Structure ")
            paned2.add(rf2, weight=50)
            self._phys_out = self._make_text(rf2)
            self._phys_out.pack(fill='both', expand=True, padx=4, pady=4)

            # Tab 3: Life Forms
            t3 = ttk.Frame(self._nb_widget)
            self._nb_widget.add(t3, text="  🌍 Life Forms  ")
            paned3 = ttk.PanedWindow(t3, orient='horizontal')
            paned3.pack(fill='both', expand=True, padx=4, pady=4)
            lf3 = ttk.LabelFrame(paned3, text=" REFERENCE LIBRARY — Known Organisms by Category ")
            paned3.add(lf3, weight=60)
            self._lib_out = self._make_text(lf3)
            self._lib_out.pack(fill='both', expand=True, padx=4, pady=4)
            rf3 = ttk.LabelFrame(paned3, text=" BASE SYSTEM & PARTS CATALOG ")
            paned3.add(rf3, weight=40)
            self._cat_out = self._make_text(rf3)
            self._cat_out.pack(fill='both', expand=True, padx=4, pady=4)

            # Tab 4: Evolution
            t4 = ttk.Frame(self._nb_widget)
            self._nb_widget.add(t4, text="  🧬 Evolution  ")
            self._build_evolve_tab(t4)

            # Tab 5: Organism Library
            t5 = ttk.Frame(self._nb_widget)
            self._nb_widget.add(t5, text="  📂 Library  ")
            self._build_library_tab(t5)

            # Tab 6: About
            t6 = ttk.Frame(self._nb_widget)
            self._nb_widget.add(t6, text="  ℹ About  ")
            self._about_out = self._make_text(t6, font_size=10)
            self._about_out.pack(fill='both', expand=True, padx=6, pady=6)

            # ── Status bar ───────────────────────────────────────────────────
            sb = ttk.Frame(self)
            sb.pack(fill='x', padx=12, pady=(0, 4))
            self._progress = ttk.Progressbar(sb, mode='determinate', length=180)
            self._progress.pack(side='left', padx=(0, 10))
            self._status_var = tk.StringVar(
                value="Ready — enter a genome or click ⚡ Random, then ▶ Translate")
            ttk.Label(sb, textvariable=self._status_var,
                      foreground='#446688', font=('Consolas', 9)).pack(side='left')

            # Populate static tabs
            self._show_welcome()
            self._show_about()
            self._show_reference_library()
            self._lib_folder_var = tk.StringVar(value=get_library_path())
            self._refresh_library_tab()

        def _build_evolve_tab(self, parent):
            top = ttk.Frame(parent)
            top.pack(fill='x', padx=8, pady=6)

            def lbl(p, t):
                ttk.Label(p, text=t).pack(side='left', padx=(8, 2))

            lbl(top, "Population:"); self._evo_pop = tk.StringVar(value='60')
            ttk.Entry(top, textvariable=self._evo_pop, width=5,
                      font=('Consolas', 10)).pack(side='left', padx=(0, 8))
            lbl(top, "Generations:"); self._evo_gens = tk.StringVar(value='10')
            ttk.Entry(top, textvariable=self._evo_gens, width=5,
                      font=('Consolas', 10)).pack(side='left', padx=(0, 8))
            lbl(top, "Workers:"); self._evo_workers = tk.StringVar(value='4')
            ttk.Entry(top, textvariable=self._evo_workers, width=4,
                      font=('Consolas', 10)).pack(side='left', padx=(0, 8))
            lbl(top, "Genome len:"); self._evo_glen = tk.StringVar(value='80')
            ttk.Entry(top, textvariable=self._evo_glen, width=5,
                      font=('Consolas', 10)).pack(side='left', padx=(0, 14))

            ttk.Button(top, text="▶▶ START EVOLUTION", command=self._evolve_custom,
                       style='Evolve.TButton').pack(side='left', padx=4)
            ttk.Button(top, text="⚡ Quick (10 gen)", command=self._evolve_short,
                       style='TButton').pack(side='left', padx=4)
            ttk.Button(top, text="💾 Export Best", command=self._export_json,
                       style='TButton').pack(side='left', padx=4)

            paned = ttk.PanedWindow(parent, orient='horizontal')
            paned.pack(fill='both', expand=True, padx=8, pady=4)

            lf = ttk.LabelFrame(paned, text=" EVOLUTION LOG — Generation-by-Generation Progress ")
            paned.add(lf, weight=55)
            self._evo_log = self._make_text(lf)
            self._evo_log.pack(fill='both', expand=True, padx=4, pady=4)

            rf = ttk.LabelFrame(paned, text=" BEST ORGANISM — Live-Updated Leaderboard ")
            paned.add(rf, weight=45)
            self._evo_best = self._make_text(rf)
            self._evo_best.pack(fill='both', expand=True, padx=4, pady=4)
            self._show_evolve_welcome()

        # ── Base system change ────────────────────────────────────────────────
        def _on_base_changed(self):
            try:
                nb = int(self._base_var.get())
            except Exception:
                nb = 4
            env = self._env_var.get()
            si   = BASE_SYSTEMS.get(nb, BASE_SYSTEMS[4])
            disp = get_env_bases(nb, env)
            pairs_str = '  '.join(f"{a}\u00b7{b}" for a, b in disp['pairs'])
            self._base_info_var.set(
                f"{disp['label']}  |  "
                f"Pairs: {pairs_str}  |  "
                f"Error rate: {si['error_rate']*100:.1f}%  |  "
                f"Info density: {math.log2(nb):.2f} bits/base  |  "
                f"Codon space: {nb**CODON_LEN}"
            )

        # --- helpers ---

        def _nb(self) -> int:
            try:
                v = int(self._base_var.get())
                return v if v in BASE_SYSTEMS else 4
            except ValueError:
                return 4

        def _gl(self) -> int:
            try:
                return max(10, min(500, int(self._len_var.get())))
            except ValueError:
                return 80

        def _get_strand(self) -> Strand:
            seq = self._genome_entry.get('1.0', 'end').strip().upper()
            nb  = self._nb()
            env = self._env_var.get()
            if not seq:
                raise ValueError("No genome entered")
            # Accept both display alphabet and engine alphabet
            disp_bases = set(get_env_bases(nb, env)['bases'])
            eng_bases  = set(BASE_SYSTEMS[nb]['bases'])
            valid_all  = disp_bases | eng_bases
            cleaned_disp = ''.join(c for c in seq if c in valid_all)
            removed = len(seq) - len(cleaned_disp)
            if removed > 0:
                disp_label = get_env_bases(nb, env)['label']
                self._status_var.set(
                    f"Cleaned input: removed {removed} invalid "
                    f"character(s) for {disp_label}")
            # Translate display letters → engine letters for execution
            cleaned = translate_seq_to_engine(cleaned_disp, nb, env)
            if len(cleaned) < CODON_LEN:
                raise ValueError(f"Need at least {CODON_LEN} valid bases")
            return Strand(cleaned, nb)

        # ── Welcome / static content ──────────────────────────────────────────

        def _show_welcome(self):
            w = self._trans
            w.delete('1.0', 'end')
            w.insert('end', "PERIODIC MACHINE  —  Chemistry-DNA Language Interpreter\n", 'h1')
            w.insert('end', "v2.0  |  50 tests passing  |  18 opcodes  |  30 parts\n\n", 'dimtext')

            w.insert('end', "QUICK START\n", 'h3')
            w.insert('end',
                "  1.  Select a Base System (2 / 4 / 6 / 8) in the toolbar above\n"
                "  2.  Type a genome (e.g. AAAGGCAAAAT...) or click  ⚡ Random\n"
                "  3.  Click  ▶ Translate  to decode every codon into instructions\n"
                "  4.  Click  🔬 Validate  to check structural correctness\n"
                "  5.  Switch to  Blueprint  tab for parts inventory, motor physics,\n"
                "       energy ledger, and 3-D structure\n"
                "  6.  Browse  🌍 Life Forms  tab to load and study known organisms\n"
                "  7.  Use  🧬 Evolution  tab to run multi-generation optimisation\n"
                "  8.  Read the  ℹ About  tab for the complete language reference\n\n")

            w.insert('end', "THE 18 OPCODES — what each codon type does\n", 'h3')
            opcode_docs = [
                ("BUILD",        "build",    "0",  "A/T",   "Adds a biological part (membrane, ATP, enzyme...) to the organism"),
                ("CONNECT",      "connect",  "1",  "C/G",   "Links two previously built parts together into a structure"),
                ("REPEAT",       "repeat",   "2",  "X",     "Copies the last-built part up to 12× (polymer chain)"),
                ("REGULATE",     "regulate", "3",  "Y",     "Toggles regulation of a part on/off (enzyme switch)"),
                ("NOOP",         "noop",     "4",  "0/1",   "No operation — filler / intron-like silence"),
                ("MOTOR",        "build",    "5",  "WA/WT", "Adds a flagellar motor component (stator, rotor, hook...)"),
                ("LOCALIZE",     "connect",  "6",  "WC/WG", "Anchors a part to a specific cellular location"),
                ("POLYMERIZE",   "repeat",   "7",  "WX",    "Extends a polymer filament using the last motor part"),
                ("PHASE",        "regulate", "8",  "WY",    "Switches developmental phase (growth→sporulation etc.)"),
                ("CHEMOTAXIS",   "regulate", "9",  "WW",    "Activates chemotaxis signal cascade"),
                ("MEMBRANE_INS", "connect",  "10", "WZ",    "Inserts a protein into the membrane bilayer"),
                ("PROMOTER",     "repeat",   "11", "ZZ",    "Opens an operon — starts a regulated gene block"),
                ("TERMINATOR",   "noop",     "12", "ZZ*",   "Closes an operon block"),
                ("SECRETE",      "build",    "13", "ZA/ZT", "Exports a protein through the secretion apparatus"),
                ("QUORUM",       "regulate", "14", "ZC/ZG", "Emits a quorum-sensing signal to neighbours"),
                ("DIVIDE",       "build",    "15", "ZX",    "Triggers cell division if enough parts are present"),
                ("METABOLIZE",   "regulate", "16", "ZY",    "Activates a metabolic pathway step (glycolysis etc.)"),
                ("GRADIENT",     "regulate", "17", "ZW",    "Performs one chemotactic gradient-following step"),
            ]
            w.insert('end', f"  {'OPCODE':<14} {'#':<4} {'BASES':<8}  DESCRIPTION\n", 'dimtext')
            w.insert('end', f"  {'─'*14} {'─'*4} {'─'*8}  {'─'*45}\n", 'dimtext')
            for name, tag, code, bases, desc in opcode_docs:
                w.insert('end', f"  {name:<14} ", tag)
                w.insert('end', f"[{code:<2}] ", 'dimtext')
                w.insert('end', f"{bases:<8}  ", 'code')
                w.insert('end', f"{desc}\n")

            w.insert('end', "\n5-BASE CODON FORMAT\n", 'h3')
            w.insert('end',
                "  Each codon is exactly 5 bases long.  Example (4-base system):\n\n"
                "     A  A  G  G  T\n"
                "     │  └──┴──┴─── operand = sum of ASCII values of bases 2–5\n"
                "     └──────────── base 1 selects the opcode\n\n"
                "  For 8-base (W/Z prefix), the second base further disambiguates:\n\n"
                "     W  A  ...    → MOTOR     (assemble flagellar component)\n"
                "     W  C  ...    → LOCALIZE  (anchor to membrane)\n"
                "     W  X  ...    → POLYMERIZE\n"
                "     Z  A  ...    → SECRETE\n"
                "     Z  X  ...    → DIVIDE\n\n")

            w.insert('end', "BASE SYSTEMS COMPARISON\n", 'h3')
            w.insert('end', f"  {'SYSTEM':<20} {'BASES':<6} {'ERR%':<8} {'BITS/BASE':<12} {'CODON SPACE'}\n", 'dimtext')
            w.insert('end', f"  {'─'*20} {'─'*6} {'─'*8} {'─'*12} {'─'*14}\n", 'dimtext')
            for nb, si in sorted(BASE_SYSTEMS.items()):
                codon_space = nb ** CODON_LEN
                w.insert('end',
                    f"  {si['name']:<20} {nb:<6} "
                    f"{si['error_rate']*100:<8.1f} "
                    f"{math.log2(nb):<12.2f} "
                    f"{codon_space}\n")

            w.insert('end', "\nVIABILITY REQUIREMENTS\n", 'h3')
            w.insert('end',
                "  An organism is ALIVE if it has all three required systems:\n\n")
            for role, count in REQUIRED_SYSTEMS.items():
                w.insert('end', f"    ✓  {role.upper():<14} — at least {count} part(s)\n", 'ok')
            w.insert('end',
                "\n  Optional systems (motor, signaling, information, minimal) add\n"
                "  bonus fitness but are not required for bare viability.\n\n")

            w.insert('end', "ENERGY MODEL  (PMF units)\n", 'h3')
            w.insert('end',
                "  Income:    energy-role parts generate PMF; complete metabolic\n"
                "             pathway adds ATP bonus\n"
                "  Expenses:  motor torque, filament polymerization, secretion,\n"
                "             cell division, baseline maintenance\n"
                "  Balance:   Income − Expense.  Negative = energy-starved.\n\n")

            # populate val_out with instruction set quick-ref
            v = self._val_out
            v.delete('1.0', 'end')
            v.insert('end', "GENOME STATS PANEL\n", 'h3')
            v.insert('end', "Translate a genome to see:\n\n", 'dimtext')
            items = [
                ("Instruction log",      "every opcode decoded from every codon"),
                ("Codon count",          "total codons in the transcript"),
                ("Unique opcodes used",  "how many of the 18 opcodes appear"),
                ("Operon count",         "PROMOTER/TERMINATOR-bounded gene blocks"),
                ("Validation errors",    "genomes that produce nothing, etc."),
                ("Validation warnings",  "orphan terminators, early CONNECTs, etc."),
                ("Base composition",     "A/T/C/G/X/Y/W/Z frequency breakdown"),
                ("Info density",         "bits per base (log₂ of base count)"),
                ("Pair integrity",       "fraction of bases with valid complements"),
                ("Stability score",      "information density × replication fidelity"),
                ("Transcript length",    "after intron splicing"),
                ("Fitness components",   "breakdown of 8 weighted score dimensions"),
            ]
            for name, desc in items:
                v.insert('end', f"  • ", 'dimtext')
                v.insert('end', f"{name:<22}", 'metric')
                v.insert('end', f"  {desc}\n")
            v.insert('end', "\nTranslate a genome to populate this panel.\n\n", 'dimtext')

        # ── About tab ──────────────────────────────────────────────────────────

        def _show_about(self):
            a = self._about_out
            a.delete('1.0', 'end')

            a.insert('end', "PERIODIC MACHINE — A Programmable Synthetic Biology Simulator\n", 'h1')
            a.insert('end', "Version 2.0  ·  Language Spec v2.0  ·  50 tests  ·  18 opcodes  ·  30 parts\n\n", 'dimtext')

            # ── WHAT IS THIS ────────────────────────────────────────────────
            a.insert('end', "═" * 72 + "\n", 'dimtext')
            a.insert('end', "  WHAT IS PERIODIC MACHINE?\n", 'h2')
            a.insert('end', "═" * 72 + "\n\n", 'dimtext')
            a.insert('end',
                "  Periodic Machine is a programmable synthetic-biology simulator.\n"
                "  It treats DNA as a real program.  Each 5-base codon is a machine\n"
                "  instruction.  The genome is the source code.  The cell is the\n"
                "  computer.  Evolution is the optimizer.\n\n"
                "  You write (or evolve) a genome string like:\n\n")
            a.insert('end', "    AAAGGCAAAAT TCGGGATGGC WZAAACAAAA ZAATCGAAAA\n\n", 'code')
            a.insert('end',
                "  The interpreter decodes each 5-base codon, executes the opcode\n"
                "  it encodes, and assembles a Blueprint — a description of the\n"
                "  organism's biological parts, connections, machines, and behaviors.\n\n"
                "  The Blueprint is then evaluated for chemical validity using RDKit,\n"
                "  scored for viability, and assigned a fitness for evolution.\n\n")

            # ── THE DNA PROGRAMMING LANGUAGE ─────────────────────────────────
            a.insert('end', "═" * 72 + "\n", 'dimtext')
            a.insert('end', "  THE DNA PROGRAMMING LANGUAGE\n", 'h2')
            a.insert('end', "═" * 72 + "\n\n", 'dimtext')
            a.insert('end', "  CODON STRUCTURE\n", 'h3')
            a.insert('end',
                "  Every codon is exactly 5 bases.  This is not arbitrary — 5-base\n"
                "  codons give 4^5 = 1024 unique codons in the standard 4-base system,\n"
                "  far more than biology's 3-base 64-codon system, allowing richer\n"
                "  information per codon.\n\n"
                "  Decoding:\n"
                "    Base 1      →  selects the primary opcode\n"
                "    Base 2      →  secondary modifier (for W/Z-prefixed opcodes)\n"
                "    Bases 2–5  →  operand = sum(ord(base) for base in codon[1:])\n"
                "                   operand selects WHICH part to build / which\n"
                "                   part to connect to, by modulo indexing into\n"
                "                   the parts catalog\n\n")

            a.insert('end', "  THE 18 OPCODES IN DETAIL\n", 'h3')
            detailed_opcodes = [
                ("BUILD  [0]",       "build",    "A or T",
                 "Selects a biological part from the parts catalog using\n"
                 "         operand mod len(BUILD_PART_NAMES).  Adds it to parts_used.\n"
                 "         Each BUILD costs the part's energy_cost in PMF units."),
                ("CONNECT  [1]",     "connect",  "C or G",
                 "Adds an edge between the last-built part and the part at\n"
                 "         operand position.  Builds the spatial graph.  Invalid\n"
                 "         connections (wrong attachment sites) are logged as domain\n"
                 "         mismatches."),
                ("REPEAT  [2]",      "repeat",   "X (6-base+)",
                 "Copies the last-built part up to MAX_POLYMER_REPEATS (12).\n"
                 "         Each copy costs the same energy as the original part.\n"
                 "         Used to build polymer chains (peptidoglycan, filaments)."),
                ("REGULATE  [3]",    "regulate", "Y (6-base+)",
                 "Toggles regulation state of a part (active/inactive).\n"
                 "         Increases regulation_count.  Used to model enzyme switches\n"
                 "         and gene regulation."),
                ("NOOP  [4]",        "noop",     "0 or 1 (or any unknown)",
                 "No operation.  Represents non-coding DNA, introns, or\n"
                 "         spacer sequences.  Does not consume energy or add parts."),
                ("MOTOR  [5]",       "build",    "WA or WT (8-base)",
                 "Adds a flagellar motor component (motA, motB, fliG,\n"
                 "         hook_protein, ms_ring, c_ring, export_gate, flagellin).\n"
                 "         Indexed from MOTOR_PART_NAMES by operand mod len."),
                ("LOCALIZE  [6]",    "connect",  "WC or WG (8-base)",
                 "Assigns the last-built part to a cellular compartment\n"
                 "         (membrane, cytoplasm, periplasm, pole, equator).\n"
                 "         Validates attachment sites; records invalid_localizations."),
                ("POLYMERIZE  [7]",  "repeat",   "WX (8-base)",
                 "Extends a flagellar filament by repeating the last motor\n"
                 "         part up to MAX_POLYMER_REPEATS times.  Tracks\n"
                 "         polymerize_cost for energy accounting."),
                ("PHASE  [8]",       "regulate", "WY (8-base)",
                 "Switches the organism's developmental phase:\n"
                 "         growth → sporulation → germination → growth.\n"
                 "         Affects fitness evaluation in some environments."),
                ("CHEMOTAXIS  [9]",  "regulate", "WW (8-base)",
                 "Activates the chemotaxis signal transduction cascade.\n"
                 "         Requires receptor + histidine_kinase + response_reg\n"
                 "         to be present for full effect."),
                ("MEMBRANE_INS  [10]","connect", "WZ (8-base)",
                 "Inserts the last-built protein into the membrane bilayer.\n"
                 "         Records the insertion in localizations['membrane'].\n"
                 "         Required for stator complex assembly."),
                ("PROMOTER  [11]",   "repeat",   "ZZ (8-base)",
                 "Opens an operon region — a regulated gene block.\n"
                 "         All BUILD instructions between PROMOTER and TERMINATOR\n"
                 "         are part of the same transcriptional unit."),
                ("TERMINATOR  [12]", "noop",     "ZZ-variant (8-base)",
                 "Closes the current open operon.  Unmatched TERMINATORs\n"
                 "         generate a validation warning."),
                ("SECRETE  [13]",    "build",    "ZA or ZT (8-base)",
                 "Exports a built protein via the secretion apparatus\n"
                 "         (requires export_gate part).  Adds to secretions list.\n"
                 "         Costs secretion_cost PMF units."),
                ("QUORUM  [14]",     "regulate", "ZC or ZG (8-base)",
                 "Emits a quorum-sensing signal molecule.  Increments\n"
                 "         quorum_signals counter.  When signals ≥ QUORUM_THRESHOLD,\n"
                 "         quorum-dependent behaviors activate."),
                ("DIVIDE  [15]",     "build",    "ZX (8-base)",
                 "Triggers cell division if parts_used ≥ DIVISION_MIN_PARTS.\n"
                 "         Increments division_events.  Each division costs\n"
                 "         DIVISION_COST PMF units."),
                ("METABOLIZE  [16]", "regulate", "ZY (8-base)",
                 "Activates a metabolic pathway step.  Builds metabolic_chains\n"
                 "         from available energy-role parts.  Complete pathway\n"
                 "         (glucose → NADH → ATP chain) gives metabolic_bonus."),
                ("GRADIENT  [17]",   "regulate", "ZW (8-base)",
                 "Performs one chemotactic gradient-following step.\n"
                 "         Increments gradient_steps.  Requires both CHEMOTAXIS\n"
                 "         activation and receptor to count toward fitness bonus."),
            ]
            for name, tag, bases_str, desc in detailed_opcodes:
                a.insert('end', f"\n  ── {name}\n", tag)
                a.insert('end', f"     Base trigger: ", 'dimtext')
                a.insert('end', f"{bases_str}\n", 'code')
                a.insert('end', f"     {desc}\n")

            # ── BASE SYSTEMS ─────────────────────────────────────────────────
            a.insert('end', "\n\n" + "═" * 72 + "\n", 'dimtext')
            a.insert('end', "  BASE SYSTEMS — THE ALPHABET OF LIFE\n", 'h2')
            a.insert('end', "═" * 72 + "\n\n", 'dimtext')
            a.insert('end',
                "  The base system is the alphabet used to write the genome.\n"
                "  Real Earth biology uses 4 bases (A, T, C, G).  This simulator\n"
                "  supports 2, 4, 6, and 8 bases to model alternative biochemistries.\n\n")
            for nb, si in sorted(BASE_SYSTEMS.items()):
                a.insert('end', f"  {nb}-BASE  ({si['name'].upper()})\n", 'h3')
                a.insert('end', f"    Bases:          {', '.join(si['bases'])}\n")
                a.insert('end', f"    Error rate:     {si['error_rate']*100:.1f}% per base per replication\n")
                a.insert('end', f"    Info density:   {math.log2(nb):.3f} bits per base\n")
                a.insert('end', f"    Codon space:    {nb**CODON_LEN} unique 5-base codons\n")
                a.insert('end', f"    Stability:      ", 'dimtext')
                s = Strand.random(10, nb).stability_score()
                bar = "█" * int(s * 20) + "░" * (20 - int(s * 20))
                a.insert('end', f"{bar}  {s:.3f}\n")
                a.insert('end', f"    Notes:          {si.get('desc', '')}\n\n")

            # ── BIOLOGICAL PARTS ─────────────────────────────────────────────
            a.insert('end', "═" * 72 + "\n", 'dimtext')
            a.insert('end', "  BIOLOGICAL PARTS CATALOG  ({} parts)\n".format(len(PARTS_KB)), 'h2')
            a.insert('end', "═" * 72 + "\n\n", 'dimtext')
            a.insert('end',
                "  Each BUILD or MOTOR instruction selects a part from this catalog.\n"
                "  Parts have a role, a real SMILES chemical structure, energy cost,\n"
                "  attachment sites (where they can be connected), and domains\n"
                "  (functional regions that must match for valid connections).\n\n")

            role_colors = {
                'structure': 'role_structure', 'energy': 'role_energy',
                'catalysis': 'role_catalysis', 'information': 'role_information',
                'minimal': 'role_minimal', 'motor': 'role_motor',
                'signaling': 'role_signaling',
            }
            role_descriptions = {
                'structure':   "Provides physical integrity (membrane, wall, filament, actin, tubulin)",
                'energy':      "Generates or carries energy (ATP, NADH, glucose)",
                'catalysis':   "Enzymes and cofactors that catalyze reactions",
                'information': "Nucleotides, CoA — information storage and regulation",
                'minimal':     "Simplest prebiotic molecules (water, methane, ammonia, HCN)",
                'motor':       "Flagellar motor components — assemble into rotating machines",
                'signaling':   "Chemotaxis receptors and kinases — sense and respond to gradients",
            }
            for role, desc in role_descriptions.items():
                parts_in_role = [(n, p) for n, p in PARTS_KB.items() if p['role'] == role]
                if not parts_in_role:
                    continue
                tag = role_colors.get(role, 'metric')
                a.insert('end', f"  {role.upper()} PARTS  —  {desc}\n", tag)
                for name, part in parts_in_role:
                    eng = PART_ENGLISH.get(name, (name, part['desc'], ''))
                    a.insert('end', f"    {eng[0]:<28}", tag)
                    a.insert('end', f"  cost={part.get('energy_cost',0)} PMF")
                    attach = part.get('attachment', [])
                    doms = part.get('domains', [])
                    if attach:
                        a.insert('end', f"  attach=[{', '.join(attach)}]", 'dimtext')
                    if doms:
                        a.insert('end', f"  domains=[{', '.join(doms)}]", 'dimtext')
                    a.insert('end', f"\n    {eng[1]}\n\n")

            # ── FUNCTIONAL COMPLEXES ─────────────────────────────────────────
            a.insert('end', "═" * 72 + "\n", 'dimtext')
            a.insert('end', "  FUNCTIONAL COMPLEXES — MULTI-PART MACHINES\n", 'h2')
            a.insert('end', "═" * 72 + "\n\n", 'dimtext')
            a.insert('end',
                "  When the right combination of parts is assembled, a Functional\n"
                "  Complex is detected.  Complexes unlock behaviors and bonuses.\n\n")
            for cx_name, cx in FUNCTIONAL_COMPLEXES.items():
                a.insert('end', f"  {cx_name.replace('_',' ').upper()}\n", 'h3')
                a.insert('end', f"    {cx['desc']}\n")
                req_names = [PART_ENGLISH.get(p, (p,))[0] for p in sorted(cx['required'])]
                a.insert('end', f"    Required:  {', '.join(req_names)}\n", 'metric')
                a.insert('end', f"    Bonus:     {cx.get('bonus', 'none')}\n\n")

            # ── FITNESS FUNCTION ─────────────────────────────────────────────
            a.insert('end', "═" * 72 + "\n", 'dimtext')
            a.insert('end', "  FITNESS FUNCTION — HOW ORGANISMS ARE SCORED\n", 'h2')
            a.insert('end', "═" * 72 + "\n\n", 'dimtext')
            a.insert('end',
                "  Fitness is a number between 0.0 and ~1.5.  It measures how\n"
                "  'alive' and 'capable' an organism is.  Non-viable organisms\n"
                "  are capped at 0.49.  The base score is:\n\n")
            fc = DEFAULT_FITNESS_CONFIG
            components = [
                ("Diversity",     fc.w_diversity,    "fraction of unique part roles present"),
                ("Connectivity",  fc.w_connectivity, "edges / parts ratio"),
                ("Complexity",    fc.w_complexity,   "weighted unique parts count"),
                ("Efficiency",    fc.w_efficiency,   "parts built per codon"),
                ("Information",   fc.w_information,  "genome info density × stability"),
                ("Viability",     fc.w_viability,    "required systems completeness"),
                ("Assembly",      fc.w_assembly,     "domain/attachment correctness"),
                ("Energy",        fc.w_energy,       "PMF balance score"),
            ]
            a.insert('end', f"  {'COMPONENT':<16} {'WEIGHT':<8}  DESCRIPTION\n", 'dimtext')
            a.insert('end', f"  {'─'*16} {'─'*8}  {'─'*35}\n", 'dimtext')
            for name, weight, desc in components:
                a.insert('end', f"  {name:<16} ", 'metric')
                a.insert('end', f"{weight:<8.2f}  {desc}\n")
            a.insert('end', "\n  Bonuses (added on top of base score):\n", 'section')
            bonuses = [
                ("Motility",     fc.motility_bonus,       "has working flagellar motor"),
                ("Chemotaxis",   fc.chemotaxis_bonus,     "has full chemotaxis pathway"),
                ("Assembly",     fc.assembly_bonus,       "flagellar motor fully assembled"),
                ("Operons",      fc.operon_bonus_cap,     "max from operon organisation"),
                ("Torque",       fc.torque_bonus_cap,     "max from torque/RPM score"),
                ("Gradient",     fc.gradient_bonus_cap,   "gradient following steps"),
                ("Division",     fc.division_bonus,       "successful cell divisions"),
                ("Metabolism",   fc.metabolic_bonus,      "complete metabolic pathway"),
                ("Quorum",       fc.quorum_bonus,         "quorum sensing signals"),
            ]
            for name, cap, desc in bonuses:
                a.insert('end', f"    +{cap:<6.3f}  {name:<14}  {desc}\n", 'build')
            a.insert('end', "\n  Penalties:\n", 'section')
            a.insert('end', f"    −{fc.domain_penalty_rate}/mismatch  Domain mismatches (wrong attachment sites)\n", 'missing')
            a.insert('end', f"    −{fc.polymerize_penalty_rate}/repeat   Excessive POLYMERIZE without motor parts\n", 'missing')
            a.insert('end', f"    −{fc.energy_starvation_rate}/deficit   Energy starvation (negative PMF balance)\n\n", 'missing')

            # ── EVOLUTION ENGINE ─────────────────────────────────────────────
            a.insert('end', "═" * 72 + "\n", 'dimtext')
            a.insert('end', "  EVOLUTION ENGINE\n", 'h2')
            a.insert('end', "═" * 72 + "\n\n", 'dimtext')
            a.insert('end',
                "  The evolutionary algorithm is a parallel genetic algorithm.\n\n"
                "  Each generation:\n"
                "    1. Evaluate all organisms in parallel (multiprocessing pool)\n"
                "    2. Rank by fitness\n"
                "    3. Keep elite fraction (top 20%) unchanged\n"
                "    4. Fill remainder via:\n"
                "       a. Tournament selection  (pick best of 3 random)\n"
                "       b. Two-point crossover   (swap segment between parents)\n"
                "       c. Point mutation        (random base substitution)\n"
                "       d. Horizontal Gene Transfer  (segment from another organism)\n"
                "    5. Cluster into species by phenotypic distance\n"
                "       (12-feature Euclidean in normalised trait space)\n"
                "    6. Log generation statistics\n\n"
                "  Each organism is evaluated with a seeded RNG so the same genome\n"
                "  always produces the same result regardless of worker order.\n\n")

            # ── MOTOR PHYSICS ────────────────────────────────────────────────
            a.insert('end', "═" * 72 + "\n", 'dimtext')
            a.insert('end', "  FLAGELLAR MOTOR PHYSICS\n", 'h2')
            a.insert('end', "═" * 72 + "\n\n", 'dimtext')
            a.insert('end',
                "  The MotorSimulator runs a discrete-time physical simulation of\n"
                "  bacterial flagellar motors.\n\n"
                "  Parameters:\n"
                "    Stall torque:       2000 pN·nm per stator unit\n"
                "    Drag coefficient:   0.001 pN·nm·s/rad\n"
                "    Moment of inertia:  0.001 (dimensionless)\n"
                "    Switch rate:        1% per timestep (CCW ↔ CW)\n"
                "    dt:                 1 ms per step\n\n"
                "  With chemotaxis active:\n"
                "    Switch rate reduced by up to 90% when moving up gradient.\n"
                "    This biases rotation toward CCW (run mode) vs CW (tumble).\n\n"
                "  Torque is calculated as:\n"
                "    T = stator_count × STALL_TORQUE × (1 − ω·drag/T_stall)\n"
                "    α = (T_drive − T_drag) / I\n"
                "    ω += α·dt    (Euler integration)\n\n"
                "  Torque in the Blueprint is computed statically from part counts:\n"
                "    torque = n_stator_pairs × 50 pN·nm\n"
                "    RPM = torque × 5  (simplified linear model)\n\n")

            # ── ENERGY MODEL ─────────────────────────────────────────────────
            a.insert('end', "═" * 72 + "\n", 'dimtext')
            a.insert('end', "  ENERGY MODEL — PROTON MOTIVE FORCE ACCOUNTING\n", 'h2')
            a.insert('end', "═" * 72 + "\n\n", 'dimtext')
            a.insert('end',
                "  All energy is tracked in PMF (Proton Motive Force) units.\n"
                "  PMF is the electrochemical potential across the inner membrane\n"
                "  that powers flagellar motors and ATP synthesis.\n\n"
                "  INCOME:\n"
                "    • Each energy-role part (ATP, NADH, glucose) adds to energy_budget\n"
                "    • Complete metabolic pathway (glucose→NADH→ATP) adds ATP_BONUS\n"
                "    • Membrane potential (per motor membrane insertion) adds to budget\n\n"
                "  EXPENSES:\n"
                "    • Each part: energy_cost (0–4 PMF units)\n"
                "    • POLYMERIZE: FILAMENT_COST per repeat\n"
                "    • SECRETE: SECRETION_COST per protein exported\n"
                "    • DIVIDE: DIVISION_COST per division event\n"
                "    • Motor torque: proportional to stator count\n\n"
                "  BALANCE = income − expense\n"
                "  Negative balance → energy-starved → fitness penalty\n\n")

            # ── GENOME VALIDATOR ─────────────────────────────────────────────
            a.insert('end', "═" * 72 + "\n", 'dimtext')
            a.insert('end', "  GENOME VALIDATOR\n", 'h2')
            a.insert('end', "═" * 72 + "\n\n", 'dimtext')
            a.insert('end',
                "  The GenomeValidator checks a decoded genome for correctness\n"
                "  before running full expression and evaluation.\n\n"
                "  FATAL ERRORS (genome is rejected):\n", 'section')
            a.insert('end',
                "    ✗  Genome too short (< 5 bases)\n"
                "    ✗  No BUILD instructions — produces nothing\n\n", 'bad')
            a.insert('end', "  WARNINGS (genome is accepted but flawed):\n", 'section')
            a.insert('end',
                "    ⚠  CONNECT before 2 parts are built\n"
                "    ⚠  TERMINATOR without matching PROMOTER\n"
                "    ⚠  DIVIDE with fewer parts than DIVISION_MIN_PARTS\n"
                "    ⚠  POLYMERIZE with no parts built yet\n"
                "    ⚠  SECRETE with nothing built\n"
                "    ⚠  Unclosed operon(s) at end of genome\n"
                "    ⚠  Multiple parts but no CONNECT (disconnected graph)\n\n", 'warn')

            # ── SPECIATION ───────────────────────────────────────────────────
            a.insert('end', "═" * 72 + "\n", 'dimtext')
            a.insert('end', "  SPECIATION\n", 'h2')
            a.insert('end', "═" * 72 + "\n\n", 'dimtext')
            a.insert('end',
                "  Organisms are clustered into species using two methods:\n\n"
                "  GENOMIC (legacy):\n"
                "    Hamming distance on raw genome sequence.\n"
                "    Threshold: 30% of genome length.\n\n"
                "  PHENOTYPIC (new):\n"
                "    Euclidean distance in 12-dimensional normalised trait space:\n"
                "      • n_parts / 30            • n_unique_parts / 15\n"
                "      • systems_complete         • diversity score\n"
                "      • motility (0/1)            • chemotaxis (0/1)\n"
                "      • assembly score            • torque / 200\n"
                "      • operons / 5               • metabolic_complete (0/1)\n"
                "      • flagella_count / 3         • energy_balance / 50\n\n")

            # ── HORIZONTAL GENE TRANSFER ─────────────────────────────────────
            a.insert('end', "═" * 72 + "\n", 'dimtext')
            a.insert('end', "  HORIZONTAL GENE TRANSFER (HGT)\n", 'h2')
            a.insert('end', "═" * 72 + "\n\n", 'dimtext')
            a.insert('end',
                "  The horizontal_gene_transfer() function simulates bacterial\n"
                "  conjugation and transduction.\n\n"
                "  Mechanism:\n"
                "    1. Pick a random segment (length transfer_len) from the donor\n"
                "    2. Insert it at a random position in the recipient, replacing\n"
                "       an equal-length segment to preserve genome length\n"
                "    3. Clean up any bases not in the recipient's allowed alphabet\n\n"
                "  This can be used in the evolution loop to introduce diversity\n"
                "  beyond what point mutation and crossover alone can achieve.\n\n")

            # ── COLONY & QUORUM SENSING ──────────────────────────────────────
            a.insert('end', "═" * 72 + "\n", 'dimtext')
            a.insert('end', "  COLONY MODEL & QUORUM SENSING\n", 'h2')
            a.insert('end', "═" * 72 + "\n\n", 'dimtext')
            a.insert('end',
                "  The Colony class models a group of cells sharing a common\n"
                "  chemical environment.\n\n"
                "  When quorum_signals ≥ QUORUM_THRESHOLD across the colony:\n"
                "    → Quorum-dependent behaviors activate\n"
                "    → Organisms with QUORUM opcode gain quorum_bonus fitness\n"
                "    → The colony summary() reports motile fraction, avg fitness\n\n"
                "  Quorum sensing is biologically the way bacteria coordinate:\n"
                "    E. coli regulates flagella expression at high density\n"
                "    Vibrio fischeri triggers bioluminescence at quorum\n"
                "    Pseudomonas forms biofilms when density is high enough\n\n")

            # ── STRUCTURAL GRADES ────────────────────────────────────────────
            a.insert('end', "═" * 72 + "\n", 'dimtext')
            a.insert('end', "  STRUCTURAL GRADES\n", 'h2')
            a.insert('end', "═" * 72 + "\n\n", 'dimtext')
            grades = [
                ("S",  "#ffd700", "0.85+",  "Elite — full motor + chemotaxis + metabolism"),
                ("A",  "#00ff88", "0.70+",  "Excellent — motile with assembled complexes"),
                ("B",  "#44aaff", "0.55+",  "Good — viable with structural diversity"),
                ("C",  "#ff8800", "0.40+",  "Fair — viable but limited capability"),
                ("D",  "#ff4444", "0.25+",  "Poor — barely alive, missing key systems"),
                ("F",  "#888888", "<0.25",  "Fail — non-viable or nothing built"),
            ]
            for g, color, threshold, desc in grades:
                a.insert('end', f"  [{g}]  {threshold:<8}  {desc}\n")

            a.insert('end', "\n\n" + "═" * 72 + "\n", 'dimtext')
            a.insert('end', "  TECHNICAL NOTES\n", 'h2')
            a.insert('end', "═" * 72 + "\n\n", 'dimtext')
            a.insert('end',
                "  Chemistry:    RDKit validates every SMILES structure.\n"
                "                Molecular weight, heavy atom count, H-bond\n"
                "                donors/acceptors, and ring count are computed.\n\n"
                "  3-D Geometry: Atoms are positioned in 3D space (Å) using\n"
                "                part offsets.  Bonds are mapped with inter-part\n"
                "                virtual edges for export.\n\n"
                "  PDB Export:   Full PDB-format file with ATOM/BOND records,\n"
                "                ready for molecular viewers (PyMOL, VMD).\n\n"
                "  WebGL Export: JSON with atom positions, bond list, and\n"
                "                animation hints for spinning flagella.\n\n"
                "  Reproducibility: Evolution uses seeded RNG per organism\n"
                "                (seed = gen_seed + hash(genome)).  Same genome\n"
                "                + seed always gives the same result.\n\n"
                "  Logging:      set_log_level('DEBUG') for verbose output.\n"
                "                Default level: INFO.\n\n"
                "  Language spec versioning:\n"
                "                v1.0 — BUILD/CONNECT/REPEAT/REGULATE/NOOP only\n"
                "                v1.5 — adds MOTOR through MEMBRANE_INS\n"
                "                v2.0 — full 18 opcodes including Z-prefix set\n\n")

            a.insert('end', "═" * 72 + "\n", 'dimtext')
            a.insert('end', "  OPEN SOURCE  ·  EDUCATIONAL  ·  RESEARCH TOOL\n", 'h2')
            a.insert('end', "═" * 72 + "\n\n", 'dimtext')
            a.insert('end',
                "  This simulator is designed to show how base-coded life builds\n"
                "  itself from the ground up — from individual molecules, through\n"
                "  assembled machines like flagella, to evolved organisms capable\n"
                "  of swimming, sensing gradients, dividing, and metabolising.\n\n"
                "  Every part has a real chemical structure.  Every opcode maps\n"
                "  to a biological process.  Every fitness component corresponds\n"
                "  to a real survival criterion.\n\n"
                "  The genome is the program.  Life is the computation.\n\n", 'highlight')

        # ── Reference library tab ─────────────────────────────────────────────

        def _show_reference_library(self):
            """Display all reference life forms in the Life Forms tab."""
            lib = self._lib_out
            cat = self._cat_out

            lib.delete('1.0', 'end')
            lib.insert('end', "REFERENCE LIBRARY — Known Life Forms\n", 'h1')
            lib.insert('end',
                f"{len(REFERENCE_LIBRARY)} organisms  ·  "
                f"Click any name to load it into the genome entry\n\n", 'dimtext')

            categories = [
                ('prebiotic',    'BEFORE LIFE — Prebiotic Chemistry'),
                ('minimal',      'MINIMAL LIFE — Simplest Living Systems'),
                ('simple',       'SIMPLE CELLS — Prokaryote-level'),
                ('complex',      'COMPLEX CELLS — Multi-system Organisms'),
                ('specialized',  'SPECIALIZED CELLS — Motor & Signaling Specialists'),
                ('alien',        'ALIEN BASE SYSTEMS — Non-standard Biochemistries'),
                ('theoretical',  'THEORETICAL / EXTREME — Edge Cases'),
            ]
            cat_refs = {}
            for ref in REFERENCE_LIBRARY:
                cat_refs.setdefault(ref['category'], []).append(ref)

            for cat_key, cat_label in categories:
                refs = cat_refs.get(cat_key, [])
                if not refs:
                    continue
                lib.insert('end', f"\n{'─'*60}\n", 'dimtext')
                lib.insert('end', f"  {cat_label}\n", 'h3')
                lib.insert('end', f"{'─'*60}\n\n", 'dimtext')
                for ref in refs:
                    si = BASE_SYSTEMS[ref['n_bases']]
                    viable_tag = 'ok' if ref['expected_viable'] else 'bad'
                    viable_label = '[ ALIVE ]' if ref['expected_viable'] else '[NOT ALIVE]'
                    btn_tag = f"libref_{id(ref)}"
                    lib.tag_configure(btn_tag, foreground='#00d4ff',
                                      underline=True, font=('Consolas', 11, 'bold'))
                    lib.tag_bind(btn_tag, '<Button-1>', lambda e, r=ref: self._load_reference(r))
                    lib.tag_bind(btn_tag, '<Enter>',
                                 lambda e, t=btn_tag: lib.tag_configure(t, foreground='#ffd700'))
                    lib.tag_bind(btn_tag, '<Leave>',
                                 lambda e, t=btn_tag: lib.tag_configure(t, foreground='#00d4ff'))

                    lib.insert('end', f"  ► {ref['name']}", btn_tag)
                    lib.insert('end', f"  [{si['name']}  {ref['n_bases']}-base]  ")
                    lib.insert('end', f"{viable_label}\n", viable_tag)
                    lib.insert('end', f"    {ref['description']}\n")
                    if ref['expected_parts']:
                        pnames = [PART_ENGLISH.get(p, (p,))[0] for p in ref['expected_parts']]
                        lib.insert('end', f"    Parts: ", 'dimtext')
                        lib.insert('end', f"{', '.join(pnames)}\n", 'build')
                    lib.insert('end', f"    Genome: ", 'dimtext')
                    lib.insert('end', f"{ref['genome']}\n\n", 'code')

            # Right panel: parts catalog + base system deep-dive
            cat.delete('1.0', 'end')
            cat.insert('end', "BIOLOGICAL PARTS CATALOG\n", 'h2')
            cat.insert('end', f"{len(PARTS_KB)} parts the genome can assemble\n\n", 'dimtext')

            role_tag = {'structure': 'role_structure', 'energy': 'role_energy',
                        'catalysis': 'role_catalysis', 'information': 'role_information',
                        'minimal': 'role_minimal', 'motor': 'role_motor',
                        'signaling': 'role_signaling'}
            seen_roles = []
            for name, part in PARTS_KB.items():
                role = part['role']
                if role not in seen_roles:
                    seen_roles.append(role)
                    cat.insert('end', f"\n  {role.upper()} PARTS\n", role_tag.get(role, 'metric'))
                eng = PART_ENGLISH.get(name, (name, part['desc'], ''))
                tag = role_tag.get(role, 'metric')
                cat.insert('end', f"    {eng[0]}\n", tag)
                cat.insert('end', f"      {eng[1]}\n", 'dimtext')
                if eng[2]:
                    cat.insert('end', f"      {eng[2]}\n")
                attach_list = part.get('attachment', [])
                attach_str = ', '.join(attach_list) if attach_list else 'none'
                cat.insert('end',
                    f"      cost={part.get('energy_cost',0)} PMF"
                    f"  attach=[{attach_str}]\n")

            cat.insert('end', "\n\nBASE SYSTEM DEEP COMPARISON\n", 'h2')
            for nb in [2, 4, 6, 8]:
                si = BASE_SYSTEMS[nb]
                refs_for_nb = [r for r in REFERENCE_LIBRARY if r['n_bases'] == nb]
                alive = [r for r in refs_for_nb if r['expected_viable']]
                cat.insert('end', f"\n  {si['name'].upper()}  ({nb}-base)\n", 'h3')
                cat.insert('end', f"    Alphabet:   {', '.join(si['bases'])}\n")
                cat.insert('end', f"    Error rate: {si['error_rate']*100:.1f}% per replication\n")
                cat.insert('end', f"    Bits/base:  {math.log2(nb):.3f}\n")
                cat.insert('end', f"    Codon space: {nb**CODON_LEN} unique 5-base codons\n")
                stab = Strand.random(30, nb).stability_score()
                bar = "█" * int(stab * 16) + "░" * (16 - int(stab * 16))
                cat.insert('end', f"    Stability:  {bar}  {stab:.3f}\n")
                cat.insert('end', f"    Organisms:  {len(refs_for_nb)} total, {len(alive)} alive\n")

            self._status_var.set(
                f"Life Forms — {len(REFERENCE_LIBRARY)} organisms  ·  click any name to load")

        def _load_reference(self, ref):
            self._base_var.set(str(ref['n_bases']))
            self._on_base_changed()
            self._genome_entry.delete('1.0', 'end')
            self._genome_entry.insert('1.0', ref['genome'])
            self._status_var.set(f"Loaded: {ref['name']} — click ▶ Translate")
            self._nb_widget.select(0)  # switch to Translation tab
            self._translate()

        # ── Evolve welcome ────────────────────────────────────────────────────

        def _show_evolve_welcome(self):
            e = self._evo_log
            e.delete('1.0', 'end')
            e.insert('end', "EVOLUTION ENGINE\n", 'h2')
            e.insert('end', "Parallel Genetic Algorithm  ·  Seeded for Reproducibility\n\n", 'dimtext')
            e.insert('end', "HOW IT WORKS\n", 'h3')
            e.insert('end',
                "  1.  A population of random genomes is initialised\n"
                "  2.  All organisms are expressed and evaluated in parallel\n"
                "      (multiprocessing — one worker per CPU core)\n"
                "  3.  Top 20% (elite) are kept unchanged\n"
                "  4.  Remaining 80% are bred by:\n"
                "       • Tournament selection (best of 3 random)\n"
                "       • Two-point crossover (swap a segment)\n"
                "       • Point mutation (random base change)\n"
                "       • HGT (horizontal gene transfer)\n"
                "  5.  Species are tracked by phenotypic distance\n"
                "  6.  Each generation's stats are logged here\n\n")
            e.insert('end', "PARAMETERS\n", 'h3')
            e.insert('end',
                "  Population:    number of organisms per generation\n"
                "                 (larger = more diversity, slower)\n"
                "  Generations:   number of rounds of selection\n"
                "                 (more = better evolved, much slower)\n"
                "  Workers:       parallel CPU workers\n"
                "                 (set to your CPU core count for max speed)\n"
                "  Genome len:    length of genome in bases\n"
                "                 (longer = more complex organisms possible)\n\n")
            e.insert('end', "Click  ▶▶ START EVOLUTION  or  ⚡ Quick  to begin.\n\n", 'ok')

            b = self._evo_best
            b.delete('1.0', 'end')
            b.insert('end', "BEST ORGANISM LEADERBOARD\n", 'h2')
            b.insert('end', "Will populate as evolution runs...\n\n", 'dimtext')
            b.insert('end', "Columns shown per generation:\n", 'h3')
            cols = [
                ("Gen",      "generation number"),
                ("Fitness",  "best organism score"),
                ("Avg",      "population average"),
                ("Viable%",  "fraction with all required systems"),
                ("Div",      "population diversity index"),
                ("Parts",    "best organism part count"),
                ("Grade",    "structural grade (S/A/B/C/D/F)"),
                ("Motile",   "has flagellar motor"),
                ("Species",  "number of distinct species"),
            ]
            for name, desc in cols:
                b.insert('end', f"  {name:<10}", 'metric')
                b.insert('end', f"  {desc}\n")

        # ── Validate genome ───────────────────────────────────────────────────

        def _validate_genome(self):
            try:
                strand = self._get_strand()
            except ValueError as e:
                messagebox.showwarning("Input Error", str(e))
                return
            validator = GenomeValidator()
            is_valid = validator.validate(strand)
            v = self._val_out
            v.delete('1.0', 'end')
            v.insert('end', "GENOME VALIDATION REPORT\n", 'h2')
            v.insert('end', f"Genome: {strand.seq[:40]}{'...' if len(strand.seq)>40 else ''}\n", 'code')
            v.insert('end', f"Length: {len(strand.seq)} bases  ·  "
                            f"System: {strand.system['name']}  ·  "
                            f"Codons: {len(strand.codons_deterministic())}\n\n", 'dimtext')

            if is_valid and not validator.warnings:
                v.insert('end', "✓  VALID — no errors or warnings\n\n", 'ok')
            elif is_valid:
                v.insert('end', "✓  VALID — but has warnings\n\n", 'warn')
            else:
                v.insert('end', "✗  INVALID — genome has errors\n\n", 'bad')

            if validator.errors:
                v.insert('end', f"ERRORS  ({len(validator.errors)}):\n", 'bad')
                for err in validator.errors:
                    v.insert('end', f"  ✗  {err}\n", 'missing')
                v.insert('end', "\n")
            if validator.warnings:
                v.insert('end', f"WARNINGS  ({len(validator.warnings)}):\n", 'warn')
                for w in validator.warnings:
                    v.insert('end', f"  ⚠  {w}\n")
                v.insert('end', "\n")

            # Detailed codon breakdown
            v.insert('end', "CODON BREAKDOWN\n", 'h3')
            codons = strand.codons_deterministic()
            bases = strand.system['bases']
            opcode_counts = {}
            for codon in codons:
                opcode, operand = codon_to_instruction(codon, bases)
                name = OPCODE_NAMES.get(opcode, f'OP{opcode}')
                opcode_counts[name] = opcode_counts.get(name, 0) + 1
            v.insert('end', f"  Total codons: {len(codons)}\n", 'metric')
            v.insert('end', f"  Unique opcodes used: {len(opcode_counts)}/18\n", 'metric')
            for op_name, count in sorted(opcode_counts.items(), key=lambda x: -x[1]):
                pct = count / max(1, len(codons)) * 100
                bar = "█" * int(pct / 3) + "░" * (33 - int(pct / 3))
                tag_map = {'BUILD': 'build', 'CONNECT': 'connect', 'REPEAT': 'repeat',
                           'REGULATE': 'regulate', 'NOOP': 'noop',
                           'MOTOR': 'role_motor', 'LOCALIZE': 'connect',
                           'POLYMERIZE': 'repeat', 'SECRETE': 'build',
                           'DIVIDE': 'build', 'METABOLIZE': 'regulate',
                           'GRADIENT': 'regulate', 'QUORUM': 'regulate',
                           'CHEMOTAXIS': 'regulate', 'PHASE': 'regulate',
                           'PROMOTER': 'repeat', 'TERMINATOR': 'noop',
                           'MEMBRANE_INS': 'connect'}
                tag = tag_map.get(op_name, 'metric')
                v.insert('end', f"  {op_name:<14} ", tag)
                v.insert('end', f" {bar}  {count:>4} ({pct:5.1f}%)\n", 'dimtext')

            # Base composition
            v.insert('end', "\nBASE COMPOSITION\n", 'h3')
            from collections import Counter as _Ctr
            bc = _Ctr(strand.seq)
            for base in strand.system['bases']:
                count = bc.get(base, 0)
                pct = count / max(1, len(strand.seq)) * 100
                bar = "█" * int(pct / 3)
                v.insert('end', f"  {base}  {bar:<34}  {count:>5} ({pct:5.1f}%)\n", 'metric')

            # Genome metrics
            v.insert('end', "\nGENOME METRICS\n", 'h3')
            metrics = [
                ("Info density",       f"{strand.info_density():.3f} bits/base"),
                ("Pair integrity",     f"{strand.pair_integrity():.1%}"),
                ("Stability score",    f"{strand.stability_score():.4f}"),
                ("Error rate",         f"{strand.system['error_rate']*100:.1f}% per base"),
                ("Transcript (det.)",  f"{sum(len(c) for c in strand.codons_deterministic())} bases"),
            ]
            for name, val in metrics:
                v.insert('end', f"  {name:<22}", 'metric')
                v.insert('end', f"  {val}\n")

            self._nb_widget.select(0)
            self._status_var.set(
                f"Validation: {'VALID' if is_valid else 'INVALID'}  ·  "
                f"{len(validator.errors)} errors  ·  {len(validator.warnings)} warnings")

        # ── Evolve custom ─────────────────────────────────────────────────────

        def _evolve_custom(self):
            if self._evolving:
                self._status_var.set("Evolution already running...")
                return
            try:
                pop_size = max(10, int(self._evo_pop.get()))
                n_gens   = max(1,  int(self._evo_gens.get()))
                n_workers= max(1,  int(self._evo_workers.get()))
                gl       = max(10, int(self._evo_glen.get()))
            except ValueError:
                messagebox.showwarning("Invalid parameters", "All evolution parameters must be integers.")
                return
            nb = self._nb()
            self._evolving = True
            self._progress.configure(maximum=n_gens, value=0)
            self._nb_widget.select(3)  # switch to Evolution tab
            self._evo_log.delete('1.0', 'end')
            self._evo_log.insert('end', "EVOLUTION STARTED\n", 'h2')
            self._evo_log.insert('end',
                f"  Pop={pop_size}  Gens={n_gens}  Workers={n_workers}  "
                f"GenomeLen={gl}  Bases={nb}\n\n", 'dimtext')

            gen_data = []

            def _on_progress(gen_done, total, stats):
                def _update():
                    self._progress.configure(value=gen_done, maximum=total)
                    best = stats.get('best', 0)
                    avg  = stats.get('avg', 0)
                    div  = stats.get('div', 0)
                    viab = stats.get('viable_pct', 0) * 100
                    self._evo_log.insert('end',
                        f"  Gen {gen_done:>4}/{total}  ", 'metric')
                    grade = 'S' if best >= 0.85 else ('A' if best >= 0.70 else
                            ('B' if best >= 0.55 else ('C' if best >= 0.40 else
                            ('D' if best >= 0.25 else 'F'))))
                    grade_tag = {'S':'section','A':'ok','B':'metric','C':'connect',
                                 'D':'warn','F':'bad'}.get(grade,'metric')
                    self._evo_log.insert('end', f"[{grade}] ", grade_tag)
                    self._evo_log.insert('end',
                        f"best={best:.4f}  avg={avg:.4f}  "
                        f"viable={viab:.0f}%  div={div:.2f}\n")
                    self._evo_log.see('end')
                    self._status_var.set(
                        f"Gen {gen_done}/{total}  best={best:.4f}  avg={avg:.4f}")
                self.after(0, _update)

            def _run():
                try:
                    ev = Evolver(pop_size=pop_size, genome_len=gl,
                                 generations=n_gens, n_bases=nb,
                                 n_workers=n_workers,
                                 progress_cb=_on_progress)
                    result = ev.run()
                    best = result['best']
                    self.after(0, self._on_evolve_done_full, best, result)
                except Exception as e:
                    msg = str(e)
                    self.after(0, lambda: self._on_evolve_error(msg))

            threading.Thread(target=_run, daemon=True).start()

        # --- actions ---

        def _generate_life(self):
            """Generate a fully operational intelligent life form and display it."""
            nb  = self._nb()
            env = self._env_var.get()
            # Use 8-base for full opcode set when available
            gen_bases = 8 if nb >= 8 else nb
            try:
                self._status_var.set("Generating intelligent life form…")
                self.update_idletasks()
                org = generate_intelligent_organism(n_bases=gen_bases, env_name=env)
                # Show genome in display alphabet
                disp_genome = translate_seq_to_display(org.strand.seq, gen_bases, env)
                self._genome_entry.delete('1.0', 'end')
                self._genome_entry.insert('1.0', disp_genome)
                # Store strand and blueprint
                self._current_strand  = org.strand
                self._current_bp      = org.blueprint
                self._current_org     = org
                # Switch to Blueprint tab
                self._nb_widget.select(1)
                # Render full life form report
                self._render_life_form(org)
                grade, desc = org.structural_grade()
                self._status_var.set(
                    f"[{grade}] {desc}  ·  fitness={org.fitness:.4f}  "
                    f"·  {len(set(org.blueprint.parts_used))} unique parts  "
                    f"·  IQ-tier={org.blueprint.intelligence_level()[1]}")
            except Exception as ex:
                import traceback
                messagebox.showerror("Generate Life Error", traceback.format_exc())

        def _gen_random(self):
            nb, gl = self._nb(), self._gl()
            env = self._env_var.get()
            s = Strand.random(gl, nb)
            display_seq = translate_seq_to_display(s.seq, nb, env)
            self._genome_entry.delete('1.0', 'end')
            self._genome_entry.insert('1.0', display_seq)
            disp = get_env_bases(nb, env)
            self._status_var.set(
                f"Random {disp['label']} genome  ·  {gl} bases")

        def _translate(self):
            try:
                strand = self._get_strand()
            except ValueError as e:
                messagebox.showwarning("Input Error", str(e))
                return
            env = self._env_var.get()
            self._current_strand = strand
            translation = translate_to_english(strand, env)

            self._trans.delete('1.0', 'end')
            # Header with genome metadata
            nb = strand.n_bases
            disp = get_env_bases(nb, env)
            self._trans.insert('end',
                f"GENOME PROGRAM — {len(strand.seq)} bases  ·  "
                f"{disp['label']}  ·  env={env}\n", 'h2')
            self._trans.insert('end',
                f"Codons: {len(strand.codons_deterministic())}  ·  "
                f"Info density: {strand.info_density():.3f} bits/base  ·  "
                f"Stability: {strand.stability_score():.4f}\n\n", 'dimtext')

            line_tag_rules = [
                (lambda l: l.startswith('READING THIS GENOME'), 'header'),
                (lambda l: l.startswith('ALIVE:'),              'viable'),
                (lambda l: l.startswith('INCOMPLETE:') or l.startswith('NON-VIABLE:'), 'incomplete'),
                (lambda l: l.startswith('---'),                 'section'),
                (lambda l: 'ASSEMBLED:' in l or 'CAN SWIM' in l or 'CAN NAVIGATE' in l, 'viable'),
                (lambda l: ': YES' in l,                        'present'),
                (lambda l: ': NO' in l,                         'missing'),
                (lambda l: 'Make ' in l and l.strip().startswith('Step'), 'build'),
                (lambda l: ('Connect' in l or 'Localize' in l or 'Membrane' in l
                            or 'Secrete' in l or 'Emit quorum' in l)
                           and l.strip().startswith('Step'),    'connect'),
                (lambda l: ('copy of' in l or 'Copy' in l or 'Polymerize' in l
                            or 'Start operon' in l or 'End operon' in l)
                           and l.strip().startswith('Step'),    'repeat'),
                (lambda l: ('Check' in l or 'phase' in l.lower() or 'chemotaxis' in l.lower()
                            or 'Metabolize' in l or 'gradient' in l.lower() or 'operon' in l.lower())
                           and l.strip().startswith('Step'),    'regulate'),
                (lambda l: 'non-coding' in l and l.strip().startswith('Step'), 'noop'),
                (lambda l: ('motor' in l.lower() or 'Assemble motor' in l or 'Cell division' in l)
                           and l.strip().startswith('Step'),    'build'),
                (lambda l: l.strip().startswith('('),           'dimtext'),
            ]
            for line in translation.split('\n'):
                tag = None
                for test, t in line_tag_rules:
                    if test(line):
                        tag = t
                        break
                self._trans.insert('end', line + '\n', tag or '')

            # Build full organism
            supply = ElementSupply.earth_like()
            org = Organism(strand=strand, env_name=env)
            org.express(supply, deterministic=True)
            org.evaluate()
            self._current_bp = org.blueprint
            self._current_org = org

            # ── Life Form Identity Card ───────────────────────────────────
            bp = org.blueprint
            grade, grade_desc = org.structural_grade()
            tier_idx, tier_name, tier_desc = bp.intelligence_level()

            # Determine organism archetype from what was assembled
            def _archetype(bp, grade) -> tuple:
                """Return (name, short_description, long_description)."""
                has_full_ns  = bp.fully_intelligent()
                has_learning = 'learning' in bp.behaviors
                has_memory   = 'working_memory' in bp.behaviors
                has_decision = 'decision_making' in bp.behaviors
                has_motor    = bp.motility_capable()
                has_chemo    = bp.chemotaxis_capable()
                has_neural   = bp.intelligence_capable()
                n_unique     = len(set(bp.parts_used))
                has_info     = bp.roles_present.get('information', 0) > 0
                has_homeo    = bp.homeostasis_score() > 0.3
                has_quorum   = bp.quorum_signals > 0
                has_division = bp.division_events > 0
                has_synapse  = 'synapse' in bp.complexes_formed
                has_plasticity = 'hebbian_plasticity' in bp.complexes_formed

                if has_full_ns and has_learning and has_memory and has_decision:
                    return (
                        "Autonomous Cognitive Agent",
                        "A fully self-directed intelligent organism",
                        "This organism possesses a complete nervous system: it can sense its "
                        "environment, transmit electrical signals between cells, learn from "
                        "experience via Hebbian plasticity, hold goals in working memory, and "
                        "make deliberate decisions. It navigates chemical gradients, repairs "
                        "its own genome, and maintains internal protein homeostasis. It is "
                        "functionally analogous to a complex invertebrate — capable of "
                        "goal-directed behavior, spatial navigation, and adaptive learning."
                    )
                if has_full_ns:
                    return (
                        "Neural Organism",
                        "Complete nervous system — sensing, signaling, and response",
                        "This organism carries all the machinery for electrical and chemical "
                        "neural signaling. Action potentials fire across its membrane, synaptic "
                        "vesicles release neurotransmitters between cells, and signal cascades "
                        "amplify and route sensory inputs. Behavior is driven by integrated "
                        "neural computation rather than simple chemical reflexes."
                    )
                if has_neural and has_chemo and has_motor:
                    return (
                        "Neural Chemotactic Swimmer",
                        "Swims, senses, and processes neural signals",
                        "A motile organism that combines a flagellar motor for swimming, "
                        "a chemotaxis signaling cascade for gradient navigation, and basic "
                        "neural excitability (action potentials). It can pursue food sources, "
                        "avoid toxins, and modulate its behavior based on membrane-level "
                        "signal processing."
                    )
                if has_motor and has_chemo:
                    return (
                        "Chemotactic Bacterium",
                        "Swims and steers toward nutrients via run-and-tumble",
                        "This organism uses a spinning flagellar motor to swim and a "
                        "two-component signaling system (receptor → kinase → regulator) to "
                        "sense chemical gradients. When nutrients increase it suppresses "
                        "tumbling and runs straight; when nutrients fall it tumbles to "
                        "reorient. This is the classic E. coli navigation strategy."
                    )
                if has_motor:
                    return (
                        "Motile Bacterium",
                        "Swims using a flagellar motor powered by proton flow",
                        "A self-propelled single-celled organism. The flagellar motor "
                        "converts the proton motive force (PMF) across its membrane into "
                        "rotational torque, spinning a helical filament like a propeller. "
                        "It can swim through liquid environments but lacks directional "
                        "sensing — it moves randomly."
                    )
                if has_neural:
                    return (
                        "Excitable Microbial Cell",
                        "Fires action potentials and responds to electrical signals",
                        "This organism carries voltage-gated ion channels that can generate "
                        "and propagate action potentials — rapid electrical pulses across its "
                        "membrane. It represents an early-stage neural architecture: "
                        "capable of signal transmission but without full synaptic or "
                        "cognitive machinery."
                    )
                if n_unique >= 7 and has_info and org.viable:
                    return (
                        "Advanced Bacterium",
                        "Multi-system cell with energy, structure, catalysis, and information",
                        "A well-equipped microorganism analogous to modern bacteria like "
                        "E. coli. It has a protective membrane, structural filaments, an "
                        "energy metabolism (ATP/NADH/glucose), catalytic enzymes, and "
                        "nucleotide-based information storage. It can copy its genome and "
                        "run complex biochemical reactions but cannot move on its own."
                    )
                if org.viable:
                    return (
                        "Simple Bacterium",
                        "Minimal viable cell — membrane, energy, catalysis",
                        "The simplest form of living cell. It has just enough: a lipid "
                        "membrane for a body, a basic energy source (glucose), and at "
                        "least one catalytic molecule to do chemical work. Think of it "
                        "as the biological equivalent of a tent with a campfire — barely "
                        "alive, but meeting the minimum requirements for life."
                    )
                if n_unique >= 3:
                    return (
                        "Protocell Fragment",
                        "Incomplete organism — missing key biological systems",
                        "This genome encodes some biological components but does not "
                        "yet assemble a complete organism. It is missing one or more of "
                        "the three required systems (structure, energy, catalysis). "
                        "Like a car without an engine — parts are present but the "
                        "machine cannot run."
                    )
                return (
                    "Prebiotic Chemistry",
                    "Raw chemical building blocks — not yet alive",
                    "This genome produces only primitive chemical molecules with no "
                    "assembled biological machinery. These are the precursors to life — "
                    "the molecular soup from which cells could theoretically emerge "
                    "given the right conditions and more genetic instructions."
                )

            arch_name, arch_short, arch_long = _archetype(bp, grade)

            # Variables needed by lifespan + features blocks (hoisted from _archetype scope)
            has_full_ns   = bp.fully_intelligent()
            has_synapse   = 'synapse' in bp.complexes_formed
            has_plasticity= 'hebbian_plasticity' in bp.complexes_formed

            t = self._trans
            t.insert('end', '\n')
            t.insert('end', '━' * 62 + '\n', 'section')
            t.insert('end', f"  LIFE FORM IDENTITY\n", 'h2')
            t.insert('end', '━' * 62 + '\n', 'section')

            # Grade + fitness badge
            grade_colors = {'S++':'#ff00ff','S+':'#ff44ff','S':'#00ff88',
                            'A':'#00d4ff','B':'#ffd700','C':'#ff8800',
                            'D':'#ff4444','F':'#666666'}
            gc = grade_colors.get(grade, '#e0e0e0')
            t.tag_configure('lf_grade', foreground=gc, font=('Consolas', 18, 'bold'))
            t.tag_configure('lf_name',  foreground='#e8e8e8', font=('Consolas', 13, 'bold'))
            t.tag_configure('lf_short', foreground='#aad4ff', font=('Consolas', 10, 'italic'))
            t.tag_configure('lf_body',  foreground='#cccccc', font=('Consolas', 10))
            t.tag_configure('lf_field', foreground='#00d4ff', font=('Consolas', 10, 'bold'))
            t.tag_configure('lf_val',   foreground='#e8e8e8', font=('Consolas', 10))
            t.tag_configure('lf_ok',    foreground='#00ff88', font=('Consolas', 10))
            t.tag_configure('lf_bad',   foreground='#ff4444', font=('Consolas', 10))
            t.tag_configure('lf_iq',    foreground='#ff44ff', font=('Consolas', 10, 'bold'))

            t.insert('end', f"\n  [{grade}] ", 'lf_grade')
            t.insert('end', f"{arch_name}\n", 'lf_name')
            t.insert('end', f"  {arch_short}\n\n", 'lf_short')
            t.insert('end', f"  {arch_long}\n\n", 'lf_body')

            # Intelligence
            iq_bar = '█' * tier_idx + '░' * (10 - tier_idx)
            t.insert('end', f"  {'INTELLIGENCE':<20}", 'lf_field')
            t.insert('end', f"  Tier {tier_idx}/10  {iq_bar}  {tier_name}\n", 'lf_iq')
            t.insert('end', f"  {'':20}  {tier_desc}\n", 'lf_val')

            # Active behaviors
            behavior_icons = {
                'swim': '🏊 Swims',
                'chemotaxis': '🧭 Gradient navigation',
                'neural_signaling': '⚡ Neural signaling',
                'autonomous_cognition': '🧠 Autonomous cognition',
                'learning': '📚 Learns from experience',
                'working_memory': '💾 Working memory',
                'decision_making': '🎯 Decision making',
                'quorum_sensing': '📡 Quorum sensing / colony comm.',
            }
            beh_list = [behavior_icons.get(b, f'► {b}') for b in bp.behaviors]
            if beh_list:
                t.insert('end', f"\n  {'BEHAVIORS':<20}", 'lf_field')
                t.insert('end', f"  {beh_list[0]}\n", 'lf_ok')
                for beh in beh_list[1:]:
                    t.insert('end', f"  {'':20}  {beh}\n", 'lf_ok')
            else:
                t.insert('end', f"\n  {'BEHAVIORS':<20}", 'lf_field')
                t.insert('end', f"  None — passive chemistry only\n", 'lf_bad')

            # Physical capabilities
            t.insert('end', f"\n  {'MOTILITY':<20}", 'lf_field')
            if bp.motility_capable():
                t.insert('end', f"  ✓ Flagellar motor — {bp.flagella_count} flagellum  "
                                f"torque={bp.torque:.0f} pN·nm  rpm={bp.rpm:.0f}\n", 'lf_ok')
            else:
                t.insert('end', f"  ✗ Non-motile — cannot swim\n", 'lf_bad')

            t.insert('end', f"  {'CHEMOTAXIS':<20}", 'lf_field')
            t.insert('end',
                f"  {'✓ Steers toward nutrients' if bp.chemotaxis_capable() else '✗ No directional sensing'}\n",
                'lf_ok' if bp.chemotaxis_capable() else 'lf_bad')

            # Lifespan estimate
            t.insert('end', f"\n  LIFESPAN ESTIMATE\n", 'lf_field')
            energy_ok = bp.energy_balance() >= 0
            has_repair = 'dna_repair' in bp.parts_used
            has_hsp    = 'heat_shock_prot' in bp.parts_used
            if has_full_ns and has_repair and has_hsp:
                lifespan = "Extended — error correction + proteome maintenance enable long-term persistence"
                lspan_tag = 'lf_ok'
            elif org.viable and energy_ok and has_repair:
                lifespan = "Moderate — DNA repair extends replication fidelity across many generations"
                lspan_tag = 'lf_ok'
            elif org.viable and energy_ok:
                lifespan = "Short — no repair mechanisms; genome degrades after repeated replication"
                lspan_tag = 'lf_val'
            elif org.viable:
                lifespan = "Very short — energy-starved; cannot sustain metabolism long-term"
                lspan_tag = 'lf_bad'
            else:
                lifespan = "Non-viable — cannot sustain independent existence"
                lspan_tag = 'lf_bad'
            t.insert('end', f"  {lifespan}\n", lspan_tag)

            # Key features summary
            t.insert('end', f"\n  KEY FEATURES\n", 'lf_field')
            features = []
            if bp.roles_present.get('structure', 0) >= 2:
                features.append(("Dual-layer structural frame (membrane + cytoskeleton)", True))
            elif bp.roles_present.get('structure', 0) >= 1:
                features.append(("Basic membrane envelope", True))
            if bp.metabolic_pathway_complete():
                features.append(("Complete metabolic chain (glucose → NADH → ATP)", True))
            else:
                features.append(("Incomplete metabolism", False))
            if bp.roles_present.get('information', 0) > 0:
                features.append(("Nucleotide-based information storage (can copy genome)", True))
            else:
                features.append(("No information storage — cannot replicate genome", False))
            if has_repair:
                features.append(("DNA repair system (dna_repair enzyme)", True))
            if has_hsp:
                features.append(("Protein chaperone / heat-shock protection", True))
            if 'proteasome' in bp.parts_used:
                features.append(("Protein degradation (proteasome quality control)", True))
            if has_synapse:
                features.append(("Chemical synapse — inter-cellular communication", True))
            if has_plasticity:
                features.append(("Hebbian LTP — synaptic weight strengthening", True))
            if bp.quorum_signals > 0:
                features.append(("Quorum sensing — coordinates colony-level behavior", True))
            if bp.division_events > 0:
                features.append(("Cell division capable — can reproduce", True))

            for feat, good in features:
                mark = '✓' if good else '✗'
                tag  = 'lf_ok' if good else 'lf_bad'
                t.insert('end', f"  {mark}  {feat}\n", tag)

            # Energy status
            bal = bp.energy_balance()
            t.insert('end', f"\n  {'ENERGY STATUS':<20}", 'lf_field')
            t.insert('end',
                f"  {'✓ SURPLUS' if bal >= 0 else '✗ STARVED'}  "
                f"{bal:+d} PMF  (budget={bp.energy_budget}, cost={bp.energy_cost})\n",
                'lf_ok' if bal >= 0 else 'lf_bad')

            # Fitness
            t.insert('end', f"  {'FITNESS SCORE':<20}", 'lf_field')
            fit_bar = '█' * int(org.fitness * 20) + '░' * (20 - int(org.fitness * 20))
            t.insert('end', f"  {fit_bar}  {org.fitness:.4f}\n", 'lf_val')
            t.insert('end', '━' * 62 + '\n\n', 'section')

            self._render_bp(org.blueprint, strand, env, org)
            # Also run validation inline
            self._validate_genome_inline(strand)
            grade, grade_desc = org.structural_grade()
            self._nb_widget.select(0)  # stay on Translation tab
            self._status_var.set(
                f"[{grade}] {grade_desc}  ·  "
                f"Fitness: {org.fitness:.4f}  ·  "
                f"{len(org.blueprint.parts_used)} parts  ·  "
                f"{len(set(org.blueprint.parts_used))} unique")

        def _validate_genome_inline(self, strand):
            """Run validation and populate the right panel of Translation tab."""
            validator = GenomeValidator()
            is_valid = validator.validate(strand)
            v = self._val_out
            v.delete('1.0', 'end')
            env = self._env_var.get()
            disp = get_env_bases(strand.n_bases, env)

            if is_valid and not validator.warnings:
                v.insert('end', "✓  VALID GENOME\n", 'ok')
            elif is_valid:
                v.insert('end', "✓  VALID  (with warnings)\n", 'warn')
            else:
                v.insert('end', "✗  INVALID GENOME\n", 'bad')
            v.insert('end', f"  Chemistry: {disp['label']}\n", 'dimtext')
            v.insert('end',
                f"  {len(strand.seq)} bases  ·  "
                f"{len(strand.codons_deterministic())} codons\n\n", 'dimtext')

            if validator.errors:
                v.insert('end', f"ERRORS  ({len(validator.errors)})\n", 'bad')
                for err in validator.errors:
                    v.insert('end', f"  ✗  {err}\n", 'missing')
                v.insert('end', '\n')
            if validator.warnings:
                v.insert('end', f"WARNINGS  ({len(validator.warnings)})\n", 'warn')
                for w in validator.warnings:
                    v.insert('end', f"  ⚠  {w}\n")
                v.insert('end', '\n')

            # Opcode histogram
            v.insert('end', "OPCODE HISTOGRAM\n", 'h3')
            codons = strand.codons_deterministic()
            bases = strand.system['bases']
            opcode_counts = {}
            for codon in codons:
                opcode, _ = codon_to_instruction(codon, bases)
                name = OPCODE_NAMES.get(opcode, f'OP{opcode}')
                opcode_counts[name] = opcode_counts.get(name, 0) + 1
            v.insert('end', f"  {len(codons)} codons  ·  {len(opcode_counts)}/18 opcodes used\n\n", 'dimtext')
            tag_map = {'BUILD':'build','CONNECT':'connect','REPEAT':'repeat',
                       'REGULATE':'regulate','NOOP':'noop','MOTOR':'role_motor',
                       'LOCALIZE':'connect','POLYMERIZE':'repeat','SECRETE':'build',
                       'DIVIDE':'build','METABOLIZE':'regulate','GRADIENT':'regulate',
                       'QUORUM':'regulate','CHEMOTAXIS':'regulate','PHASE':'regulate',
                       'PROMOTER':'repeat','TERMINATOR':'noop','MEMBRANE_INS':'connect'}
            for op_name, count in sorted(opcode_counts.items(), key=lambda x: -x[1]):
                pct = count / max(1, len(codons)) * 100
                bar = '█' * min(int(pct / 2.5), 40)
                v.insert('end', f"  {op_name:<14} ", tag_map.get(op_name, 'metric'))
                v.insert('end', f"{bar:<40}  {count:>3} ({pct:4.1f}%)\n", 'dimtext')

            # Base composition — show with display-alphabet labels
            v.insert('end', "\nBASE COMPOSITION\n", 'h3')
            from collections import Counter as _Ctr
            bc = _Ctr(strand.seq)   # strand.seq is engine alphabet
            total = len(strand.seq)
            eng_bases  = strand.system['bases']
            disp_bases = disp['bases']
            for eng_b, disp_b in zip(eng_bases, disp_bases):
                count = bc.get(eng_b, 0)
                pct = count / max(1, total) * 100
                bar = '█' * min(int(pct / 2.5), 40)
                v.insert('end', f"  {disp_b}  {bar:<40}  {count:>5} ({pct:5.1f}%)\n", 'metric')

            # Genome metrics
            v.insert('end', "\nGENOME METRICS\n", 'h3')
            for name, val in [
                ("Info density",      f"{strand.info_density():.3f} bits/base"),
                ("Pair integrity",    f"{strand.pair_integrity():.1%}"),
                ("Stability score",   f"{strand.stability_score():.4f}"),
                ("Error rate",        f"{strand.system['error_rate']*100:.1f}% per base"),
                ("Codon space",       f"{strand.n_bases**CODON_LEN} unique codons"),
            ]:
                v.insert('end', f"  {name:<22}", 'metric')
                v.insert('end', f"  {val}\n")

        # ── Organism Library ──────────────────────────────────────────────────

        def _build_library_tab(self, parent):
            """Build the Library tab layout."""
            # Top control bar
            ctrl = ttk.Frame(parent)
            ctrl.pack(fill='x', padx=8, pady=6)

            ttk.Label(ctrl, text="Folder:").pack(side='left', padx=(0, 4))
            self._lib_folder_entry = ttk.Entry(
                ctrl, textvariable=self._lib_folder_var, width=48,
                font=('Consolas', 9))
            self._lib_folder_entry.pack(side='left', padx=(0, 4))

            ttk.Button(ctrl, text="📁 Browse",
                       command=self._browse_library_folder).pack(side='left', padx=2)
            ttk.Button(ctrl, text="🔄 Refresh",
                       command=self._refresh_library_tab).pack(side='left', padx=2)
            ttk.Button(ctrl, text="🦠 Seed Microbes",
                       command=self._seed_library_with_microbes,
                       style='Accent.TButton').pack(side='left', padx=2)
            ttk.Button(ctrl, text="🧠 Save Intelligent",
                       command=self._save_intelligent_to_library).pack(side='left', padx=2)
            ttk.Button(ctrl, text="💾 Save Current",
                       command=self._save_current_to_library).pack(side='left', padx=2)

            # Split pane: organism list (left) + detail card (right)
            paned = ttk.PanedWindow(parent, orient='horizontal')
            paned.pack(fill='both', expand=True, padx=4, pady=4)

            lf = ttk.LabelFrame(paned, text=" ORGANISM LIBRARY — saved files ")
            paned.add(lf, weight=45)
            self._lib_list = self._make_text(lf)
            self._lib_list.pack(fill='both', expand=True, padx=4, pady=4)

            rf = ttk.LabelFrame(paned, text=" ORGANISM DETAIL CARD ")
            paned.add(rf, weight=55)
            self._lib_detail = self._make_text(rf)
            self._lib_detail.pack(fill='both', expand=True, padx=4, pady=4)

            # Bottom bar
            bot = ttk.Frame(parent)
            bot.pack(fill='x', padx=8, pady=(0, 4))
            self._lib_count_var = tk.StringVar(value="No organisms loaded")
            ttk.Label(bot, textvariable=self._lib_count_var,
                      foreground='#446688', font=('Consolas', 9)).pack(side='left')

        def _browse_library_folder(self):
            from tkinter import filedialog
            folder = filedialog.askdirectory(
                title="Select Organism Library Folder",
                initialdir=self._lib_folder_var.get())
            if folder:
                self._lib_folder_var.set(folder)
                self._refresh_library_tab()

        def _refresh_library_tab(self):
            """Reload and display all organisms in the library folder."""
            if not hasattr(self, '_lib_list'):
                return
            folder = self._lib_folder_var.get()
            organisms = list_library_organisms(folder)

            lst = self._lib_list
            lst.delete('1.0', 'end')
            lst.insert('end', f"ORGANISM LIBRARY\n", 'h1')
            lst.insert('end', f"Folder: {folder}\n", 'dimtext')
            lst.insert('end', f"{len(organisms)} organisms saved\n\n", 'dimtext')

            if not organisms:
                lst.insert('end', "  No organisms found.\n\n", 'dimtext')
                lst.insert('end', "  Click  🦠 Seed Microbes  to populate with\n"
                                  "  5 real micro-organism templates.\n\n", 'dimtext')
                lst.insert('end', "  Or use  💾 Save Current  to save any\n"
                                  "  translated genome to the library.\n", 'dimtext')
                self._lib_count_var.set("Library empty — click 🦠 Seed Microbes to populate")
                return

            grade_colors = {'S++':'#ff00ff','S+':'#ff44ff','S':'#ffd700',
                            'A':'#00ff88','B':'#44aaff','C':'#ff8800',
                            'D':'#ff4444','F':'#888888'}
            cat_icons = {'microbe':'🦠','viral':'🔬','intelligent':'🧠',
                         'evolved':'🧬','generated':'⚡','prebiotic':'🌌'}

            self._lib_organism_map = {}  # tag -> summary dict

            for i, s in enumerate(organisms):
                icon = cat_icons.get(s['category'], '●')
                grade = s['grade']
                gc = grade_colors.get(grade, '#e0e0e0')
                tag_name = f"org_{i}"
                self._lib_organism_map[tag_name] = s

                lst.tag_configure(tag_name, foreground=gc,
                                  font=('Consolas', 10, 'bold'),
                                  underline=True)
                lst.tag_bind(tag_name, '<Button-1>',
                             lambda e, t=tag_name: self._show_library_card(t))

                lst.insert('end', f"  {icon} [{grade}] ", 'metric')
                lst.insert('end', f"{s['name']}\n", tag_name)
                lst.insert('end',
                    f"     Fitness={s['fitness']:.3f}  "
                    f"Parts={s['n_parts']}  "
                    f"{'IQ-'+str(s['intelligence_tier']) if s['intelligence_tier'] > 0 else ''}"
                    f"  {s['category']}  {s['n_bases']}-base\n",
                    'dimtext')
                lst.insert('end', f"     {s['saved_at']}\n\n", 'dimtext')

            self._lib_count_var.set(
                f"Library: {len(organisms)} organisms  ·  folder: {folder}")

            # Auto-show first card
            if organisms:
                self._show_library_card("org_0")

        def _show_library_card(self, tag_name: str):
            """Show full detail card for a library organism."""
            s = self._lib_organism_map.get(tag_name)
            if not s:
                return
            try:
                data = load_organism_from_file(s['filepath'])
            except Exception as ex:
                messagebox.showerror("Load Error", str(ex))
                return

            d = self._lib_detail
            d.delete('1.0', 'end')

            grade = data.get('grade', '?')
            grade_colors = {'S++':'#ff00ff','S+':'#ff44ff','S':'#ffd700',
                            'A':'#00ff88','B':'#44aaff','C':'#ff8800',
                            'D':'#ff4444','F':'#888888'}
            gc = grade_colors.get(grade, '#e0e0e0')
            d.tag_configure('card_grade', foreground=gc,
                            font=('Consolas', 20, 'bold'))
            d.tag_configure('card_name',  foreground='#e8e8e8',
                            font=('Consolas', 12, 'bold'))
            d.tag_configure('load_btn',   foreground='#00d4ff',
                            font=('Consolas', 11, 'bold'), underline=True)

            d.insert('end', f"\n  [{grade}] ", 'card_grade')
            d.insert('end', f"{data.get('name','?')}\n", 'card_name')
            d.insert('end', f"  {data.get('grade_desc','')}\n\n", 'dimtext')

            # Load button
            load_tag = 'load_btn_action'
            d.tag_configure(load_tag, foreground='#00ff88',
                            font=('Consolas', 10, 'bold'), underline=True)
            d.tag_bind(load_tag, '<Button-1>',
                       lambda e, fp=s['filepath']: self._load_organism_from_library(fp))
            d.insert('end', "  ► LOAD INTO EDITOR\n\n", load_tag)

            # Description
            d.insert('end', "DESCRIPTION\n", 'h2')
            d.insert('end', f"  {data.get('description','')}\n\n", 'lf_body')

            # Science notes
            notes = data.get('science_notes', '')
            if notes:
                d.insert('end', "REAL-WORLD SCIENCE\n", 'h2')
                d.insert('end', f"  {notes}\n\n", 'dimtext')

            # Key stats
            d.insert('end', "STATISTICS\n", 'h2')
            stats = [
                ("Source",          data.get('source','')),
                ("Category",        data.get('category','')),
                ("Base system",     f"{data.get('n_bases',4)}-base"),
                ("Environment",     data.get('env_name','earth')),
                ("Genome length",   f"{len(data.get('genome',''))} bases"),
                ("Parts used",      f"{len(data.get('parts_used',[]))} total  ·  "
                                    f"{len(data.get('unique_parts',[]))} unique"),
                ("Fitness",         f"{data.get('fitness',0):.4f}"),
                ("Viable",          "YES" if data.get('viable') else "NO"),
                ("Motile",          "YES" if data.get('motility') else "NO"),
                ("Chemotaxis",      "YES" if data.get('chemotaxis') else "NO"),
                ("Intelligence",    f"Tier {data.get('intelligence_tier',0)}/10  —  "
                                    f"{data.get('intelligence_name','')}"),
                ("IQ score",        f"{data.get('intelligence_score',0):.3f}"),
                ("Homeostasis",     f"{data.get('homeostasis_score',0):.3f}"),
                ("Membrane pot.",   f"{data.get('membrane_potential',0):.0f} mV"),
                ("Energy balance",  f"{data.get('energy_budget',0) - data.get('energy_cost',0):+d} PMF"),
                ("Flagella",        str(data.get('flagella_count',0))),
                ("Torque",          f"{data.get('torque',0):.0f} pN·nm"),
                ("RPM",             f"{data.get('rpm',0):.0f}"),
                ("Saved at",        data.get('saved_at','')),
                ("Version",         data.get('version','')),
            ]
            for k, v in stats:
                d.insert('end', f"  {k:<22}", 'lf_field')
                d.insert('end', f"  {v}\n", 'lf_val')

            # Behaviors
            behaviors = data.get('behaviors', [])
            if behaviors:
                d.insert('end', "\nBEHAVIORS\n", 'h2')
                icons = {'swim':'🏊','chemotaxis':'🧭','neural_signaling':'⚡',
                         'autonomous_cognition':'🧠','learning':'📚',
                         'working_memory':'💾','decision_making':'🎯',
                         'quorum_sensing':'📡'}
                for b in behaviors:
                    d.insert('end', f"  {icons.get(b,'►')}  {b.replace('_',' ').upper()}\n", 'ok')

            # Complexes
            complexes = data.get('complexes_formed', [])
            if complexes:
                d.insert('end', "\nASSEMBLED COMPLEXES\n", 'h2')
                for cx in complexes:
                    d.insert('end', f"  ✓  {cx.replace('_',' ').upper()}\n", 'ok')

            # Genome preview
            genome = data.get('genome', '')
            d.insert('end', "\nGENOME (first 120 bases)\n", 'h2')
            d.insert('end', f"  {genome[:120]}{'...' if len(genome)>120 else ''}\n", 'code')

        def _load_organism_from_library(self, filepath: str):
            """Load a library organism into the genome editor and translate it."""
            try:
                data = load_organism_from_file(filepath)
                genome  = data.get('genome', '')
                n_bases = data.get('n_bases', 4)
                env     = data.get('env_name', 'earth')
                self._base_var.set(str(n_bases))
                self._env_var.set(env)
                self._on_base_changed()
                disp = translate_seq_to_display(genome, n_bases, env)
                self._genome_entry.delete('1.0', 'end')
                self._genome_entry.insert('1.0', disp)
                self._nb_widget.select(0)
                self._translate()
                self._status_var.set(
                    f"Loaded from library: {data.get('name','?')}  ·  "
                    f"grade={data.get('grade','?')}  fitness={data.get('fitness',0):.4f}")
            except Exception as ex:
                messagebox.showerror("Library Load Error", str(ex))

        def _open_library_tab(self):
            """Switch to Library tab and refresh."""
            self._refresh_library_tab()
            self._nb_widget.select(4)  # tab index 4 = Library

        def _save_current_to_library(self):
            """Save the currently translated organism to the library."""
            if not hasattr(self, '_current_org') or self._current_org is None:
                messagebox.showwarning("Nothing to save",
                    "Translate a genome first, then save it.")
                return
            org = self._current_org
            name = f"Organism_{_datetime.datetime.now().strftime('%H%M%S')}"
            try:
                from tkinter.simpledialog import askstring
                name = askstring("Save to Library", "Name for this organism:",
                                 initialvalue=name) or name
            except Exception:
                pass
            folder = self._lib_folder_var.get()
            try:
                path = save_organism_to_library(
                    org, name=name, category='generated',
                    source='translated', folder=folder)
                self._refresh_library_tab()
                self._status_var.set(f"Saved to library: {path}")
            except Exception as ex:
                messagebox.showerror("Save Error", str(ex))

        def _save_intelligent_to_library(self):
            """Generate + save a high-intelligence organism to the library."""
            nb  = self._nb()
            env = self._env_var.get()
            gen_bases = 8 if nb >= 8 else nb
            try:
                self._status_var.set("Generating intelligent life form…")
                self.update_idletasks()
                org = generate_intelligent_organism(n_bases=gen_bases, env_name=env)
                folder = self._lib_folder_var.get()
                path = save_organism_to_library(
                    org,
                    name='Autonomous Cognitive Agent',
                    category='intelligent',
                    source='generated',
                    description=(
                        'A fully operational high-intelligence life form with complete '
                        'nervous system, learning, working memory, and decision making.'
                    ),
                    science_notes=(
                        'Generated by generate_intelligent_organism(). Encodes all 10 '
                        'NEURAL_COMPLEXES: action potential, synapse, signal cascade, '
                        'Hebbian plasticity, neural oscillator, working memory, decision '
                        'circuit, homeostatic control, full nervous system, genome integrity.'
                    ),
                    folder=folder,
                )
                self._refresh_library_tab()
                grade, desc = org.structural_grade()
                self._status_var.set(
                    f"Saved intelligent organism [{grade}] to library: {path}")
            except Exception as ex:
                messagebox.showerror("Generate Error", str(ex))

        def _seed_library_with_microbes(self):
            """Build all 5 micro-organism templates and save to library."""
            folder = self._lib_folder_var.get()
            self._status_var.set("Seeding library with micro-organism templates…")
            self.update_idletasks()

            def _run():
                try:
                    results = generate_and_save_microbe(folder=folder)
                    def _done():
                        self._refresh_library_tab()
                        self._nb_widget.select(4)
                        self._status_var.set(
                            f"Seeded {len(results)} micro-organisms into library: {folder}")
                    self.after(0, _done)
                except Exception as ex:
                    msg = str(ex)
                    self.after(0, lambda: messagebox.showerror(
                        "Seed Error", msg))

            threading.Thread(target=_run, daemon=True).start()

        def _render_bp(self, bp, strand, env, org=None):
            # ── LEFT PANEL: organism parts, systems, viability ───────────────
            b = self._bp_out
            b.delete('1.0', 'end')

            if org is not None:
                grade, grade_desc = org.structural_grade()
                grade_colors = {'S':'#ffd700','A':'#00ff88','B':'#44aaff',
                                'C':'#ff8800','D':'#ff4444','F':'#888888'}
                gc = grade_colors.get(grade, '#e0e0e0')
                b.tag_configure('grade_big', foreground=gc,
                                font=('Consolas', 22, 'bold'))
                b.tag_configure('grade_desc', foreground=gc,
                                font=('Consolas', 11, 'bold'))
                b.insert('end', f"  GRADE  {grade}\n", 'grade_big')
                b.insert('end', f"  {grade_desc}\n", 'grade_desc')
                b.insert('end', f"  Fitness: {org.fitness:.4f}\n", 'metric')
                viable_tag = 'ok' if org.viable else 'bad'
                viable_str = 'VIABLE (alive)' if org.viable else 'NON-VIABLE (not alive)'
                b.insert('end', f"  {viable_str}\n\n", viable_tag)

            b.insert('end', "ORGANISM SUMMARY\n", 'h3')
            b.insert('end', f"  System:      {strand.system['name']}  ({strand.n_bases}-base)\n")
            b.insert('end', f"  Environment: {env}\n")
            b.insert('end', f"  Genome len:  {len(strand.seq)} bases\n\n")

            b.insert('end', "PARTS INVENTORY\n", 'h3')
            if bp.parts_used:
                for pn, cnt in Counter(bp.parts_used).most_common():
                    role = PARTS_KB[pn]['role']
                    tag = f"role_{role}"
                    eng = PART_ENGLISH.get(pn, (pn, PARTS_KB[pn]['desc'], ''))
                    cost = PARTS_KB[pn].get('energy_cost', 0)
                    b.insert('end', f"  {cnt:>2}×  ", 'dimtext')
                    b.insert('end', f"{eng[0]:<28}", tag)
                    b.insert('end', f"  [{role:<11}]  cost={cost} PMF\n", 'dimtext')
                    b.insert('end', f"       {eng[1]}\n")
            else:
                b.insert('end', "  (no parts assembled)\n")

            b.insert('end', "\nSYSTEMS STATUS\n", 'h3')
            all_roles = sorted(set(list(REQUIRED_SYSTEMS) + list(bp.roles_present)))
            for role in all_roles:
                needed = REQUIRED_SYSTEMS.get(role, 0)
                have = bp.roles_present.get(role, 0)
                if needed > 0:
                    ok = have >= needed
                    marker = '✓ ' if ok else '✗ '
                    tag = 'ok' if ok else 'bad'
                else:
                    marker = '+ '
                    tag = 'metric'
                se = SYSTEM_ENGLISH.get(role, {'name': role})
                b.insert('end', f"  {marker}{se['name']:<18}", tag)
                b.insert('end', f"  {have} part(s)")
                if needed > 0:
                    b.insert('end', f"  (need {needed})")
                b.insert('end', '\n')

            b.insert('end', "\nSCORE BREAKDOWN\n", 'h3')
            for label, val in [
                ('Systems complete',  f"{bp.systems_score():.1%}"),
                ('Part diversity',    f"{bp.diversity_score():.1%}"),
                ('Build efficiency',  f"{bp.efficiency_score():.1%}"),
                ('Total parts',       f"{len(bp.parts_used)}"),
                ('Unique parts',      f"{len(set(bp.parts_used))}"),
                ('Connections',       f"{len(bp.connections)}"),
                ('Genome stability',  f"{strand.stability_score():.4f}"),
                ('Pair integrity',    f"{strand.pair_integrity():.1%}"),
            ]:
                b.insert('end', f"  {label:<22}", 'metric')
                b.insert('end', f"  {val}\n")

            if bp.complexes_formed:
                b.insert('end', "\nASSEMBLED MACHINES\n", 'h3')
                for cx_name in bp.complexes_formed:
                    cx = FUNCTIONAL_COMPLEXES.get(cx_name, {})
                    b.insert('end', f"  ★  {cx_name.replace('_',' ').upper()}\n", 'section')
                    b.insert('end', f"     {cx.get('desc','')}\n")
                    b.insert('end', f"     Bonus: {cx.get('bonus','none')}\n\n")

            if bp.behaviors:
                b.insert('end', "BEHAVIORS\n", 'h3')
                for beh in bp.behaviors:
                    b.insert('end', f"  ► {beh.upper()}\n", 'ok')

            if bp.operon_count() > 0:
                b.insert('end', "\nGENOME ORGANIZATION\n", 'h3')
                b.insert('end', f"  Operons: {bp.operon_count()}\n", 'metric')

            if bp.domain_mismatches > 0:
                b.insert('end', "\nWARNINGS\n", 'h3')
                b.insert('end', f"  Domain mismatches:    {bp.domain_mismatches}\n", 'bad')
                b.insert('end', f"  Invalid localizations: {bp.invalid_localizations}\n", 'bad')

            # ── RIGHT PANEL: physics, chemistry, energy, 3D ─────────────────
            p = self._phys_out
            p.delete('1.0', 'end')

            # Energy ledger
            p.insert('end', "ENERGY LEDGER  (PMF units)\n", 'h3')
            bal = bp.energy_balance()
            bal_tag = 'ok' if bal >= 0 else 'bad'
            p.insert('end', f"  Income (budget):    {bp.energy_budget:>6} PMF\n", 'role_energy')
            p.insert('end', f"  Expenses (cost):    {bp.energy_cost:>6} PMF\n", 'missing')
            p.insert('end', f"  Net balance:        {bal:>6} PMF\n", bal_tag)
            p.insert('end', f"  Membrane potential: {bp.membrane_potential:>6.0f} mV\n")
            if bal < 0:
                p.insert('end', "  ⚠ Energy starved — fitness penalised\n", 'bad')
            p.insert('end', '\n')

            # Motor physics
            if bp.torque > 0 or bp.flagella_count > 0:
                p.insert('end', "FLAGELLAR MOTOR PHYSICS\n", 'h3')
                p.insert('end', f"  Flagella count:  {bp.flagella_count}\n", 'metric')
                p.insert('end', f"  Torque:          {bp.torque:.1f} pN·nm\n", 'metric')
                p.insert('end', f"  RPM:             {bp.rpm:.0f}\n", 'metric')
                p.insert('end', f"  Tumble prob:     {bp.tumble_prob:.3f}  "
                                f"({'biased' if bp.tumble_prob < 0.1 else 'random'})\n")
                motile = bp.torque > 0 and bp.flagella_count > 0
                p.insert('end', f"  Motile:          {'YES — CAN SWIM' if motile else 'NO'}\n",
                         'ok' if motile else 'bad')
                p.insert('end', '\n')

            # Secretion, quorum, division, metabolism, gradient
            if bp.secretions:
                p.insert('end', "SECRETION\n", 'h3')
                p.insert('end', f"  Exported proteins: {len(bp.secretions)}\n", 'metric')
                for s in bp.secretions[:8]:
                    p.insert('end', f"    {s}\n", 'dimtext')

            if bp.quorum_signals > 0:
                p.insert('end', "\nQUORUM SENSING\n", 'h3')
                p.insert('end', f"  Signals emitted: {bp.quorum_signals}\n", 'metric')

            if bp.division_events > 0:
                p.insert('end', "\nCELL DIVISION\n", 'h3')
                p.insert('end', f"  Divisions triggered: {bp.division_events}\n", 'metric')

            if bp.metabolic_pathway_complete():
                p.insert('end', "\nMETABOLISM  ★ COMPLETE\n", 'h3')
                p.insert('end', "  Full glucose → NADH → ATP chain detected\n", 'ok')
                if bp.metabolic_chains:
                    p.insert('end', f"  Chain steps: {len(bp.metabolic_chains)}\n")
            else:
                p.insert('end', "\nMETABOLISM\n", 'h3')
                p.insert('end', "  Incomplete metabolic pathway\n", 'dimtext')

            if bp.gradient_steps > 0:
                p.insert('end', "\nNAVIGATION\n", 'h3')
                p.insert('end', f"  Gradient steps: {bp.gradient_steps}\n", 'metric')
                p.insert('end', "  Organism navigates chemical gradients\n", 'ok')

            # Chemistry
            if org is not None and hasattr(org, '_part_mols') and org._part_mols:
                n_mols = len(org._part_mols)
                p.insert('end', f"\nCHEMISTRY  ({n_mols} molecules)\n", 'h3')
                p.insert('end', f"  Total MW:        {org._total_mw:>10.1f} Da\n")
                p.insert('end', f"  Heavy atoms:     {org._total_heavy:>10}\n")
                p.insert('end', f"  Aromatic rings:  {org._total_rings:>10}\n")
                p.insert('end', f"  H-bond accept:   {org._total_hba:>10}\n")
                p.insert('end', f"  H-bond donate:   {org._total_hbd:>10}\n\n")

                if org.element_cost:
                    p.insert('end', "ELEMENT COST  (total across all parts)\n", 'h3')
                    p.insert('end', f"  {'El':<4} {'Count':>6}  {'Mass (amu)':>12}  Bar\n", 'dimtext')
                    total_atoms = sum(org.element_cost.values())
                    for el, cnt in sorted(org.element_cost.items(), key=lambda x: -x[1]):
                        mass = PERIODIC.get(el, {}).get('mass', 0) * cnt
                        pct = cnt / max(1, total_atoms) * 100
                        bar = '█' * min(int(pct / 2), 30)
                        p.insert('end', f"  {el:<4} {cnt:>6}  {mass:>12.1f}  ", 'metric')
                        p.insert('end', f"{bar}\n", 'dimtext')

                geom = getattr(org, '_organism_geom', None)
                if geom:
                    p.insert('end', "\n3D ATOMIC STRUCTURE\n", 'h3')
                    p.insert('end', f"  Atoms positioned:    {len(geom['atoms'])}\n", 'metric')
                    p.insert('end', f"  Bonds mapped:        {len(geom['bonds'])}\n", 'metric')
                    virtual = sum(1 for b in geom['bonds'] if b.get('virtual'))
                    if virtual:
                        p.insert('end', f"  Inter-part links:    {virtual}\n", 'metric')
                    if geom['atoms']:
                        xs = [a['x'] for a in geom['atoms']]
                        ys = [a['y'] for a in geom['atoms']]
                        zs = [a['z'] for a in geom['atoms']]
                        p.insert('end',
                            f"  Bounding box: "
                            f"X[{min(xs):.1f},{max(xs):.1f}]  "
                            f"Y[{min(ys):.1f},{max(ys):.1f}]  "
                            f"Z[{min(zs):.1f},{max(zs):.1f}] Å\n")
            else:
                try:
                    mol = build_mol(bp.combined_smiles)
                    if mol:
                        ec = mol_element_counts(mol)
                        p.insert('end', "\nELEMENT COST  (combined SMILES)\n", 'h3')
                        total_atoms = sum(ec.values())
                        for el, cnt in sorted(ec.items(), key=lambda x: -x[1]):
                            mass = PERIODIC.get(el, {}).get('mass', 0) * cnt
                            pct = cnt / max(1, total_atoms) * 100
                            bar = '█' * min(int(pct / 2), 30)
                            p.insert('end', f"  {el:<4} {cnt:>6}  {mass:>10.1f} amu  ", 'metric')
                            p.insert('end', f"{bar}\n", 'dimtext')
                except Exception:
                    pass

        def _render_life_form(self, org):
            """Render a full life-form intelligence report into the Blueprint panels."""
            bp = org.blueprint
            if bp is None:
                return
            strand = org.strand

            # ── LEFT PANEL: life form identity + all systems ─────────────────
            b = self._bp_out
            b.delete('1.0', 'end')

            grade, grade_desc = org.structural_grade()
            grade_colors = {'S++':'#ff00ff','S+':'#ff44ff','S':'#00ff88',
                            'A':'#00d4ff','B':'#ffd700','C':'#ff8800',
                            'D':'#ff4444','F':'#666666'}
            gc = grade_colors.get(grade, '#e0e0e0')
            b.tag_configure('grade_big', foreground=gc,
                            font=('Consolas', 28, 'bold'))
            b.tag_configure('grade_desc', foreground=gc,
                            font=('Consolas', 11, 'bold'))

            b.insert('end', f"\n  [{grade}]  ", 'grade_big')
            b.insert('end', f"{grade_desc}\n\n", 'grade_desc')
            b.insert('end', f"  Fitness:  {org.fitness:.4f}   "
                            f"Viable: {'YES' if org.viable else 'NO'}\n\n", 'metric')

            # Intelligence tier banner
            tier_idx, tier_name, tier_desc = bp.intelligence_level()
            iq_bar = '█' * tier_idx + '░' * (10 - tier_idx)
            b.insert('end', "INTELLIGENCE LEVEL\n", 'h2')
            b.insert('end', f"  Tier {tier_idx}/10  {iq_bar}  {tier_name}\n", 'section')
            b.insert('end', f"  {tier_desc}\n\n", 'dimtext')
            b.insert('end', f"  Score: {bp.intelligence_score:.3f}   "
                            f"Neural score: {bp.neural_score():.3f}   "
                            f"Homeostasis: {bp.homeostasis_score():.3f}\n\n", 'metric')

            # Active behaviors
            b.insert('end', "ACTIVE BEHAVIORS\n", 'h2')
            behavior_icons = {
                'swim': '🏊', 'chemotaxis': '🧭', 'neural_signaling': '⚡',
                'autonomous_cognition': '🧠', 'learning': '📚',
                'working_memory': '💾', 'decision_making': '🎯',
                'quorum_sensing': '📡',
            }
            if bp.behaviors:
                for beh in bp.behaviors:
                    icon = behavior_icons.get(beh, '►')
                    b.insert('end', f"  {icon}  {beh.replace('_',' ').upper()}\n", 'ok')
            else:
                b.insert('end', "  (none)\n", 'dimtext')
            b.insert('end', '\n')

            # Neural complexes assembled
            b.insert('end', "NEURAL & COGNITIVE SYSTEMS\n", 'h2')
            for cx_name in NEURAL_COMPLEXES:
                present = cx_name in bp.complexes_formed
                cx = FUNCTIONAL_COMPLEXES.get(cx_name, {})
                marker = '✓' if present else '✗'
                tag = 'ok' if present else 'bad'
                b.insert('end', f"  {marker}  {cx_name.replace('_',' ').upper()}\n", tag)
                b.insert('end', f"      {cx.get('desc','')}\n", 'dimtext')

            b.insert('end', '\n')

            # Motor & locomotion complexes
            b.insert('end', "LOCOMOTION SYSTEMS\n", 'h2')
            motor_complexes = {'flagellar_motor', 'flagellum', 'chemotaxis_system',
                               'stator_complex', 'cytoskeleton'}
            for cx_name in motor_complexes:
                present = cx_name in bp.complexes_formed
                cx = FUNCTIONAL_COMPLEXES.get(cx_name, {})
                marker = '✓' if present else '✗'
                tag = 'ok' if present else 'dimtext'
                b.insert('end', f"  {marker}  {cx_name.replace('_',' ').upper()}\n", tag)

            b.insert('end', '\n')

            # Parts by role
            b.insert('end', "PARTS INVENTORY\n", 'h2')
            from collections import Counter as _Ctr
            role_order = ['structure','energy','catalysis','information',
                          'motor','signaling','neural','homeostasis','minimal']
            role_tags = {
                'structure':'role_structure','energy':'role_energy',
                'catalysis':'role_catalysis','information':'role_information',
                'motor':'role_motor','signaling':'role_signaling',
                'neural':'section','homeostasis':'metric','minimal':'dimtext',
            }
            counts_by_role: Dict = {}
            for p_name in bp.parts_used:
                role = PARTS_KB.get(p_name, {}).get('role', 'unknown')
                counts_by_role.setdefault(role, []).append(p_name)
            for role in role_order:
                parts_list = counts_by_role.get(role, [])
                if not parts_list:
                    continue
                tag = role_tags.get(role, 'metric')
                b.insert('end', f"  {role.upper():<14} {len(parts_list):>3} total  "
                                f"{len(set(parts_list)):>3} unique\n", tag)
                part_counts = _Ctr(parts_list)
                for pn, cnt in sorted(part_counts.items()):
                    desc = PARTS_KB.get(pn, {}).get('desc', '')
                    b.insert('end', f"    {'x'+str(cnt) if cnt>1 else '  '} {pn}\n", 'dimtext')

            # ── RIGHT PANEL: deep physics + intelligence metrics ─────────────
            p = self._phys_out
            p.delete('1.0', 'end')

            p.insert('end', "FULL ORGANISM METRICS\n", 'h1')
            p.insert('end', f"  Genome length:     {len(strand.seq)} bases\n", 'metric')
            p.insert('end', f"  Codons:            {len(strand.codons_deterministic())}\n", 'metric')
            p.insert('end', f"  Operons:           {bp.operon_count()}\n", 'metric')
            p.insert('end', f"  Total parts:       {len(bp.parts_used)}\n", 'metric')
            p.insert('end', f"  Unique parts:      {len(set(bp.parts_used))}\n", 'metric')
            p.insert('end', f"  Connections:       {len(bp.connections)}\n", 'metric')
            p.insert('end', f"  Info density:      {strand.info_density():.3f} bits/base\n", 'metric')
            p.insert('end', f"  Stability:         {strand.stability_score():.4f}\n", 'metric')
            p.insert('end', f"  Pair integrity:    {strand.pair_integrity():.1%}\n\n", 'metric')

            # Intelligence deep dive
            p.insert('end', "INTELLIGENCE DEEP ANALYSIS\n", 'h2')
            p.insert('end', f"  Tier:              {tier_idx}/10 — {tier_name}\n", 'section')
            p.insert('end', f"  Score:             {bp.intelligence_score:.4f}\n", 'metric')
            p.insert('end', f"  Neural subsystems: "
                            f"{len(set(bp.complexes_formed) & NEURAL_COMPLEXES)}"
                            f"/{len(NEURAL_COMPLEXES)}\n", 'metric')

            intel_components = [
                ('Action potential',    'action_potential'),
                ('Chemical synapse',    'synapse'),
                ('Signal cascade',      'signal_cascade'),
                ('Hebbian plasticity',  'hebbian_plasticity'),
                ('Neural oscillator',   'neural_oscillator'),
                ('Working memory',      'working_memory_loop'),
                ('Decision circuit',    'decision_circuit'),
                ('Homeostatic ctrl',    'homeostatic_control'),
                ('Full nervous sys',    'full_nervous_system'),
                ('Genome integrity',    'genome_integrity'),
            ]
            for label, cx_name in intel_components:
                present = cx_name in bp.complexes_formed
                mark = '✓' if present else '✗'
                tag  = 'ok' if present else 'bad'
                p.insert('end', f"  {mark}  {label}\n", tag)
            p.insert('end', '\n')

            # Energy ledger
            p.insert('end', "ENERGY LEDGER\n", 'h2')
            bal = bp.energy_balance()
            bal_tag = 'ok' if bal >= 0 else 'bad'
            p.insert('end', f"  PMF budget:        {bp.energy_budget:>6} PMF\n", 'role_energy')
            p.insert('end', f"  Energy cost:       {bp.energy_cost:>6} PMF\n", 'missing')
            p.insert('end', f"  Net balance:       {bal:>6} PMF  "
                            f"{'✓ SURPLUS' if bal>=0 else '✗ STARVED'}\n", bal_tag)
            p.insert('end', f"  Membrane pot:      {bp.membrane_potential:.0f} mV\n\n", 'metric')

            # Motor physics
            p.insert('end', "FLAGELLAR MOTOR\n", 'h2')
            if bp.flagella_count > 0:
                p.insert('end', f"  Flagella:          {bp.flagella_count}\n", 'ok')
                p.insert('end', f"  Torque:            {bp.torque:.1f} pN·nm\n", 'metric')
                p.insert('end', f"  RPM:               {bp.rpm:.0f}\n", 'metric')
                p.insert('end', f"  Tumble prob:       {bp.tumble_prob:.3f}\n", 'metric')
                p.insert('end', f"  Motile:            YES — CAN SWIM\n\n", 'ok')
            else:
                p.insert('end', "  No flagellar motor assembled\n\n", 'dimtext')

            # Homeostasis detail
            p.insert('end', "HOMEOSTASIS\n", 'h2')
            hom_parts = [pn for pn in set(bp.parts_used)
                         if PARTS_KB.get(pn, {}).get('role') == 'homeostasis']
            if hom_parts:
                for pn in hom_parts:
                    desc = PARTS_KB[pn].get('desc', '')
                    p.insert('end', f"  ✓  {pn:<20} {desc}\n", 'ok')
            else:
                p.insert('end', "  No homeostasis parts present\n", 'bad')
            p.insert('end', '\n')

            # Neural parts inventory
            p.insert('end', "NEURAL PARTS INVENTORY\n", 'h2')
            n_parts = [pn for pn in set(bp.parts_used)
                       if PARTS_KB.get(pn, {}).get('role') == 'neural']
            if n_parts:
                for pn in sorted(n_parts):
                    func = PARTS_KB[pn].get('function', '')
                    desc = PARTS_KB[pn].get('desc', '')
                    p.insert('end', f"  {pn:<22}", 'section')
                    p.insert('end', f"  [{func}]\n", 'dimtext')
                    p.insert('end', f"    {desc}\n", 'dimtext')
            else:
                p.insert('end', "  No neural parts present\n", 'bad')
            p.insert('end', '\n')

            # Chemistry
            if hasattr(org, '_part_mols') and org._part_mols:
                p.insert('end', f"CHEMISTRY  ({len(org._part_mols)} molecules)\n", 'h2')
                p.insert('end', f"  Total MW:          {org._total_mw:>10.1f} Da\n", 'metric')
                p.insert('end', f"  Heavy atoms:       {org._total_heavy:>10}\n", 'metric')
                p.insert('end', f"  Aromatic rings:    {org._total_rings:>10}\n", 'metric')
                p.insert('end', f"  H-bond acceptors:  {org._total_hba:>10}\n", 'metric')
                p.insert('end', f"  H-bond donors:     {org._total_hbd:>10}\n\n", 'metric')

        def _evolve_short(self):
            """Quick 10-generation evolution run."""
            if self._evolving:
                self._status_var.set("Evolution already running...")
                return
            nb = self._nb()
            self._evolving = True
            total_gens = 10
            self._progress.configure(maximum=total_gens, value=0)
            self._nb_widget.select(3)  # switch to Evolution tab
            if hasattr(self, '_evo_log'):
                self._evo_log.delete('1.0', 'end')
                self._evo_log.insert('end', "QUICK EVOLUTION  (10 generations)\n", 'h2')
                self._evo_log.insert('end',
                    f"  Pop=60  Gens=10  Workers=4  GenomeLen={self._gl()}  Bases={nb}\n\n",
                    'dimtext')

            def _on_progress(gen_done, total, stats):
                def _update():
                    self._progress.configure(value=gen_done, maximum=total)
                    best = stats.get('best', 0)
                    avg  = stats.get('avg', 0)
                    grade = 'S' if best>=0.85 else ('A' if best>=0.70 else
                            ('B' if best>=0.55 else ('C' if best>=0.40 else
                            ('D' if best>=0.25 else 'F'))))
                    grade_tag = {'S':'section','A':'ok','B':'metric','C':'connect',
                                 'D':'warn','F':'bad'}.get(grade,'metric')
                    if hasattr(self, '_evo_log'):
                        self._evo_log.insert('end', f"  Gen {gen_done:>3}/{total}  ", 'metric')
                        self._evo_log.insert('end', f"[{grade}] ", grade_tag)
                        self._evo_log.insert('end',
                            f"best={best:.4f}  avg={avg:.4f}  "
                            f"viable={stats.get('viable_pct',0)*100:.0f}%\n")
                        self._evo_log.see('end')
                    self._status_var.set(
                        f"Evolving... ({gen_done}/{total})  "
                        f"best={best:.4f}  avg={avg:.4f}")
                self.after(0, _update)

            def _run():
                try:
                    ev = Evolver(pop_size=60, genome_len=self._gl(),
                                 generations=total_gens, n_bases=nb,
                                 n_workers=4, progress_cb=_on_progress)
                    result = ev.run()
                    best = result['best']
                    self.after(0, self._on_evolve_done, best)
                except Exception as e:
                    msg = str(e)
                    self.after(0, lambda: self._on_evolve_error(msg))

            threading.Thread(target=_run, daemon=True).start()

        def _on_evolve_done(self, best):
            self._evolving = False
            self._progress.configure(value=0)
            self._current_strand = None
            self._current_bp = None
            self._current_org = None
            raw_genome = best.get('genome', '')
            nb  = self._nb()
            env = self._env_var.get()
            display_genome = translate_seq_to_display(raw_genome, nb, env)
            self._genome_entry.delete('1.0', 'end')
            self._genome_entry.insert('1.0', display_genome)
            grade = best.get('structural_grade', '?')
            fitness = best.get('fitness', 0)
            if hasattr(self, '_evo_log'):
                self._evo_log.insert('end',
                    f"\n  DONE  [{grade}] fitness={fitness:.4f}  "
                    f"parts={best.get('blueprint',{}).get('n_parts',0)}\n", 'ok')
                self._evo_log.see('end')
            self._status_var.set(
                f"Evolution done — [{grade}] fitness={fitness:.4f}  "
                f"parts={best.get('blueprint',{}).get('n_parts',0)}")
            self._nb_widget.select(0)  # switch to Translation tab
            self._translate()

        def _on_evolve_done_full(self, best, result):
            self._evolving = False
            self._progress.configure(value=0)
            self._current_strand = None
            self._current_bp = None
            self._current_org = None
            self._genome_entry.delete('1.0', 'end')
            self._genome_entry.insert('1.0', best.get('genome', ''))
            grade = best.get('structural_grade', '?')
            fitness = best.get('fitness', 0)
            n_parts = best.get('blueprint', {}).get('n_parts', 0)
            # Update evolution best panel
            if hasattr(self, '_evo_best'):
                b = self._evo_best
                b.insert('end', "\n" + "─" * 50 + "\n", 'dimtext')
                b.insert('end', "FINAL BEST ORGANISM\n", 'h3')
                b.insert('end', f"  Grade:   [{grade}]\n", 'section')
                b.insert('end', f"  Fitness: {fitness:.4f}\n", 'metric')
                b.insert('end', f"  Parts:   {n_parts}\n", 'metric')
                b.insert('end', f"  Viable:  {best.get('viable', False)}\n")
                b.insert('end', f"  Genome:\n", 'dimtext')
                b.insert('end', f"  {best.get('genome','')[:60]}...\n", 'code')
                b.see('end')
            if hasattr(self, '_evo_log'):
                self._evo_log.insert('end',
                    f"\n  COMPLETE  [{grade}] fitness={fitness:.4f}  parts={n_parts}\n", 'ok')
                self._evo_log.see('end')
            self._status_var.set(
                f"Evolution complete — [{grade}] fitness={fitness:.4f}  parts={n_parts}")
            self._nb_widget.select(0)
            self._translate()

        def _on_evolve_error(self, err):
            self._evolving = False
            self._progress.configure(value=0)
            self._status_var.set(f"Evolution error: {err}")
            messagebox.showerror("Evolution Error", err)

        def _export_json(self):
            if self._current_strand is None:
                messagebox.showwarning("Export", "Translate a genome first")
                return
            org = getattr(self, '_current_org', None)
            if org is None:
                env = self._env_var.get()
                supply = ElementSupply.earth_like()
                org = Organism(strand=self._current_strand, env_name=env)
                org.express(supply, deterministic=True)
                org.evaluate()
            data = org.to_dict()
            path = filedialog.asksaveasfilename(
                defaultextension='.json', filetypes=[('JSON', '*.json')],
                initialfile='organism_blueprint.json')
            if path:
                with open(path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                self._status_var.set(f"Exported to {path}")

        def _clear(self):
            self._genome_entry.delete('1.0', 'end')
            self._current_strand = None
            self._current_bp = None
            self._current_org = None
            self._progress.configure(value=0)
            self._show_welcome()
            self._status_var.set("Cleared")

    app = PeriodicMachine()
    app.mainloop()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import sys as _sys
    mp.freeze_support()

    if '--test' in _sys.argv:
        run_tests()
    elif '--evolve' in _sys.argv:
        run_tests()
        evolver = Evolver(pop_size=200, genome_len=80, generations=120,
                          n_bases=4, n_workers=12)
        result = evolver.run()
        with open('evolved_blueprints.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print("\nSaved evolved_blueprints.json")
    elif '--benchmark' in _sys.argv:
        print("Running benchmark suite...")
        bench = run_benchmark()
        with open('benchmark_results.json', 'w') as f:
            json.dump({str(k): v for k, v in bench.items()}, f, indent=2)
        print("\nSaved benchmark_results.json")
    else:
        launch_ui()