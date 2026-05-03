import numpy as np
import random
import json
import time
import math
import multiprocessing as mp
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
from rdkit import RDLogger
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
    },
    4: {  # Earth DNA: 2 complementary pairs
        'pairs': [('A', 'T'), ('C', 'G')],
        'bases': ['A', 'T', 'C', 'G'],
        'error_rate': 0.008,
        'name': 'quaternary',
    },
    6: {  # 3 pairs: more info, less stable, mutation-prone
        'pairs': [('A', 'T'), ('C', 'G'), ('X', 'Y')],
        'bases': ['A', 'T', 'C', 'G', 'X', 'Y'],
        'error_rate': 0.045,
        'name': 'hexary',
    },
    8: {  # 4 pairs: highly unstable
        'pairs': [('A', 'T'), ('C', 'G'), ('X', 'Y'), ('W', 'Z')],
        'bases': ['A', 'T', 'C', 'G', 'X', 'Y', 'W', 'Z'],
        'error_rate': 0.11,
        'name': 'octary',
    },
}


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


# ---------------------------------------------------------------------------
# Molecular fragment library: what codons encode
# ---------------------------------------------------------------------------
# Each codon maps to a real SMILES fragment. Fragments are keyed by the
# first base of the codon, giving each base a chemical "role":
#   A → carbon backbone     T → sulfur/thiol
#   C → nitrogen/amide      G → oxygen/hydroxyl
#   X → silicon             Y → phosphorus
#   W → halogen             Z → ring systems
#   0,1 → minimal (binary)

FRAG_LIB = {
    'A': ['C', 'CC', 'C=C', 'CCC', 'C(C)C'],
    'T': ['S', 'CS', 'C(=O)S', 'SC', 'CSS'],
    'C': ['N', 'CN', 'C(=O)N', 'NC', 'NCC'],
    'G': ['O', 'CO', 'C(=O)O', 'OC', 'OCC'],
    'X': ['[Si]', '[Si](C)C', 'C[Si]C', '[Si]([Si])C'],
    'Y': ['P', 'CP', 'OP(=O)(O)O', 'CP(C)C'],
    'W': ['F', 'Cl', 'Br', 'CF', 'CCl'],
    'Z': ['c1ccccc1', 'C1CC1', 'C1CCC1', 'C1CCCC1', 'c1ccncc1'],
    '0': ['C', 'N'],
    '1': ['O', 'S'],
}

CODON_LEN = 5  # 5-base codons: 4^5=1024 possible (4-base), higher info density than bio's 3


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

    def consume(self, element_counts: Dict[str, int]) -> bool:
        if not self.can_afford(element_counts):
            return False
        for el, need in element_counts.items():
            self.stock[el] -= need
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

    def codons(self) -> List[str]:
        """Transcript -> list of 5-base codons."""
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
        """Inverse of error rate, normalized. 4-base is near-optimal."""
        # Sweet spot: 4 bases. 2 is too simple (low info), 6+ is unstable.
        err = self.system['error_rate']
        info = self.info_density()
        # Balance: high info * low error = good
        return min(1.0, (info / 3.0) * (1.0 - err))

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
                    'desc': 'phosphate — nucleic acid backbone'},
}

# Systems a viable organism needs — like a car needs engine + frame + wheels
REQUIRED_SYSTEMS = {
    'structure':   1,  # at least 1 structural part
    'energy':      1,  # at least 1 energy part
    'catalysis':   1,  # at least 1 catalytic part
}

PART_NAMES = list(PARTS_KB.keys())


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


def codon_to_instruction(codon: str) -> Tuple[int, int]:
    """Decode a 5-base codon into (opcode, operand).
    First base determines opcode, remaining 4 determine operand."""
    base_to_op = {
        'A': Opcode.BUILD, 'T': Opcode.BUILD,     # A/T: build parts
        'C': Opcode.CONNECT, 'G': Opcode.CONNECT,  # C/G: connect
        'X': Opcode.REPEAT,                         # X: repeat/copy
        'Y': Opcode.REGULATE,                       # Y: conditional
        'W': Opcode.BUILD, 'Z': Opcode.BUILD,      # W/Z: build (alt)
        '0': Opcode.BUILD, '1': Opcode.NOOP,       # binary system
    }
    opcode = base_to_op.get(codon[0], Opcode.NOOP)
    operand = sum(ord(c) for c in codon[1:]) if len(codon) > 1 else 0
    return opcode, operand


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
        }


def execute_genome(strand: Strand, env_name: str = 'earth') -> Blueprint:
    """Run the genome as a program. Each codon is an instruction that
    builds, connects, repeats, or regulates parts from the knowledge base."""
    bp = Blueprint()
    codons = strand.codons()
    skip_next = False

    for codon in codons:
        opcode, operand = codon_to_instruction(codon)

        if skip_next:
            skip_next = False
            bp.instruction_log.append('SKIPPED')
            continue

        if opcode == Opcode.BUILD:
            part_idx = operand % len(PART_NAMES)
            part_name = PART_NAMES[part_idx]
            part = PARTS_KB[part_name]
            bp.parts_used.append(part_name)
            bp.smiles_components.append(part['smiles'])
            role = part['role']
            bp.roles_present[role] = bp.roles_present.get(role, 0) + 1
            bp.instruction_log.append(f'BUILD:{part_name}')

        elif opcode == Opcode.CONNECT:
            n = len(bp.parts_used)
            if n >= 2:
                a = (operand) % n
                b = (operand + 1) % n
                if a != b and (a, b) not in bp.connections:
                    bp.connections.append((a, b))
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
            env_hash = sum(ord(c) for c in env_name) % 3
            if operand % 3 != env_hash:
                skip_next = True
            bp.instruction_log.append('REGULATE')

        else:
            bp.instruction_log.append('NOOP')

    return bp


# ---------------------------------------------------------------------------
# Translation: codon -> molecular fragment (low-level, still used for SMILES)
# ---------------------------------------------------------------------------

def codon_to_fragment(codon: str) -> str:
    key = codon[0] if codon[0] in FRAG_LIB else 'A'
    idx = sum(ord(c) for c in codon) % len(FRAG_LIB[key])
    return FRAG_LIB[key][idx]


def translate_strand(strand: Strand) -> str:
    """Strand -> SMILES via blueprint execution."""
    bp = execute_genome(strand)
    return bp.combined_smiles


# ---------------------------------------------------------------------------
# Molecule building + 3D coordinate extraction
# ---------------------------------------------------------------------------

def build_mol(smiles: str) -> Optional[Chem.Mol]:
    """Build an RDKit molecule from SMILES with 3D coordinates.
    For multi-component SMILES (dot-separated), tries whole molecule first,
    then falls back to building the largest parseable fragment."""
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        if AllChem.EmbedMolecule(mol, params) == -1:
            params.useRandomCoords = True
            AllChem.EmbedMolecule(mol, params)
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
    This avoids the multi-component SMILES failure by building each part
    as a separate molecule, then aggregating the results."""
    mols = []
    all_elements: Dict[str, int] = {}
    all_atoms = []
    all_bonds = []
    atom_offset = 0
    spacing = 8.0  # angstroms between parts in 3D

    for i, smiles in enumerate(bp.smiles_components):
        mol = build_mol(smiles)
        if mol is None:
            # Try without charges
            clean = smiles.replace('[N+]', 'N').replace('[Fe+2]', '[Fe]')
            mol = build_mol(clean)
        if mol is not None:
            mols.append(mol)
            # Element counts
            for atom in mol.GetAtoms():
                sym = atom.GetSymbol()
                all_elements[sym] = all_elements.get(sym, 0) + 1
            # 3D geometry with offset per part
            if mol.GetNumConformers() > 0:
                conf = mol.GetConformer()
                x_off = (i % 4) * spacing
                y_off = (i // 4) * spacing
                for j, atom in enumerate(mol.GetAtoms()):
                    pos = conf.GetAtomPosition(j)
                    all_atoms.append({
                        'symbol': atom.GetSymbol(),
                        'x': round(pos.x + x_off, 4),
                        'y': round(pos.y + y_off, 4),
                        'z': round(pos.z, 4),
                        'part_idx': i,
                        'part_name': bp.parts_used[i] if i < len(bp.parts_used) else 'unknown',
                    })
                for b in mol.GetBonds():
                    all_bonds.append({
                        'i': b.GetBeginAtomIdx() + atom_offset,
                        'j': b.GetEndAtomIdx() + atom_offset,
                        'order': b.GetBondTypeAsDouble(),
                    })
                atom_offset += mol.GetNumAtoms()

    # Add inter-part connection bonds (represented as virtual bonds)
    for conn_a, conn_b in bp.connections:
        if conn_a < len(mols) and conn_b < len(mols):
            all_bonds.append({
                'i': sum(m.GetNumAtoms() for m in mols[:conn_a]),
                'j': sum(m.GetNumAtoms() for m in mols[:conn_b]),
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


def extract_3d(mol: Chem.Mol) -> Optional[Dict]:
    if mol.GetNumConformers() == 0:
        return None
    conf = mol.GetConformer()
    atoms = []
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        atoms.append({
            'symbol': atom.GetSymbol(),
            'x': round(pos.x, 4),
            'y': round(pos.y, 4),
            'z': round(pos.z, 4),
        })
    bonds = [{
        'i': b.GetBeginAtomIdx(),
        'j': b.GetEndAtomIdx(),
        'order': b.GetBondTypeAsDouble(),
    } for b in mol.GetBonds()]
    return {'atoms': atoms, 'bonds': bonds}


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

    def express(self, supply: ElementSupply, deterministic: bool = False) -> None:
        """Execute genome program -> assemble blueprint -> build molecules.
        If deterministic=True, uses codons_deterministic (no random splicing).
        Uses build_organism_mol to build each part separately, avoiding
        multi-component SMILES failures."""
        if deterministic:
            self.blueprint = execute_genome_deterministic(self.strand, self.env_name)
        else:
            self.blueprint = execute_genome(self.strand, self.env_name)
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

        # Structural quality score (used for grading)
        quality = (
            diversity * 0.25
            + connectivity * 0.15
            + complexity * 0.20
            + efficiency * 0.15
            + has_info * 0.10
            + viability * 0.15
        )
        self._structural_quality = quality

        # Fitness: alive organisms get 0.50-1.00 range, dead get 0.01-0.49
        if alive:
            self.fitness = 0.50 + quality * 0.40 + rep_stability * 0.10
        else:
            # Partial credit proportional to systems present
            self.fitness = systems * 0.25 + quality * 0.15 + rep_stability * 0.05
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
    seq, n_bases, env_name, gen = args
    supply_map = {
        'earth': ElementSupply.earth_like,
        'silicon': ElementSupply.silicon_world,
        'exotic': ElementSupply.exotic,
    }
    supply = supply_map.get(env_name, ElementSupply.earth_like)()
    strand = Strand(seq, n_bases)
    org = Organism(strand=strand, env_name=env_name, gen_born=gen)
    org.express(supply)
    org.evaluate()
    return org.to_dict()


# ---------------------------------------------------------------------------
# Genetic operators — work on raw sequences
# ---------------------------------------------------------------------------

def crossover(s1: str, s2: str) -> str:
    L = min(len(s1), len(s2))
    if L < 6:
        return s1
    a, b = sorted(random.sample(range(1, L - 1), 2))
    return s1[:a] + s2[a:b] + s1[b:]


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


def tournament(seqs: List[str], fits: List[float], k: int = 4) -> str:
    idxs = random.sample(range(len(seqs)), min(k, len(seqs)))
    return seqs[max(idxs, key=lambda i: fits[i])]


# ---------------------------------------------------------------------------
# Evolution engine: runs multiple base systems in parallel
# ---------------------------------------------------------------------------

ENVS = ['earth', 'silicon', 'exotic']


class Evolver:
    def __init__(self, pop_size: int = 200, genome_len: int = 80,
                 generations: int = 120, n_bases: int = 4,
                 n_workers: int = None):
        self.pop_size = pop_size
        self.genome_len = genome_len
        self.generations = generations
        self.n_bases = n_bases
        self.system = BASE_SYSTEMS[n_bases]
        self.bases = self.system['bases']
        self.n_workers = n_workers or min(mp.cpu_count(), 12)
        self.population = [
            ''.join(random.choice(self.bases) for _ in range(genome_len))
            for _ in range(pop_size)
        ]
        self.history: List[Dict] = []
        self.best_ever: Dict = {'fitness': 0.0}

    def run(self) -> Dict[str, Any]:
        t0 = time.time()
        print(f"[init] system={self.system['name']} n_bases={self.n_bases} "
              f"error_rate={self.system['error_rate']} pop={self.pop_size} "
              f"gens={self.generations} workers={self.n_workers}")

        for gen in range(self.generations):
            gt = time.time()

            tasks = [
                (seq, self.n_bases, ENVS[i % len(ENVS)], gen)
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

            self.history.append({
                'gen': gen, 'best': best_r['fitness'], 'avg': avg,
                'diversity': diversity, 'valid_pct': valid_pct,
            })

            if gen % 10 == 0 or gen == self.generations - 1:
                print(f"  gen {gen:>3d} | best {best_r['fitness']:.4f} | "
                      f"avg {avg:.4f} | viable {valid_pct:.0%} | "
                      f"div {diversity:.2f} | {time.time()-gt:.1f}s")

            # Replication-aware reproduction: use strand's error rate for mutation
            elite_n = max(4, self.pop_size // 10)
            ranked = sorted(range(self.pop_size), key=lambda i: fits[i], reverse=True)
            elites = [self.population[i] for i in ranked[:elite_n]]

            children = list(elites)
            mut_rate = self.system['error_rate'] + 0.08  # base mutation + replication error
            while len(children) < self.pop_size:
                p1 = tournament(self.population, fits)
                p2 = tournament(self.population, fits)
                child = crossover(p1, p2)
                child = mutate_seq(child, self.bases, rate=mut_rate)
                if len(child) > self.genome_len + 20:
                    child = child[:self.genome_len]
                children.append(child)

            self.population = children[:self.pop_size]

        elapsed = time.time() - t0
        print(f"\n[done] {self.system['name']} {self.generations} gens in {elapsed:.1f}s")
        print(f"[best] fitness={self.best_ever['fitness']:.4f} "
              f"atoms={self.best_ever.get('n_heavy_atoms',0)} "
              f"viable={self.best_ever.get('viable', False)}")

        # Final evaluation of top population
        final_tasks = [
            (seq, self.n_bases, ENVS[i % len(ENVS)], self.generations)
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
            'config': {
                'base_system': self.system['name'],
                'n_bases': self.n_bases,
                'error_rate': self.system['error_rate'],
                'pop_size': self.pop_size,
                'genome_len': self.genome_len,
                'generations': self.generations,
                'codon_len': CODON_LEN,
                'environments': ENVS,
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
    for name, part in PARTS_KB.items():
        mol = Chem.MolFromSmiles(part['smiles'])
        assert mol is not None, f"invalid SMILES in KB part '{name}': {part['smiles']}"
        assert part['role'] in ('structure', 'energy', 'catalysis', 'information', 'minimal')
    print(f"  [5] knowledge base OK ({len(PARTS_KB)} parts, all valid SMILES)")

    # 6. Instruction set: codons decode to valid opcodes
    test_codons = ['ATCGA', 'CGTAC', 'XATCG', 'YGTCA', 'TTTTT']
    for tc in test_codons:
        op, operand = codon_to_instruction(tc)
        assert 0 <= op <= 4, f"invalid opcode {op} for codon {tc}"
        assert isinstance(operand, int)
    print("  [6] instruction set OK")

    # 7. Blueprint execution: genome produces parts
    s = Strand.random(80, 4)
    bp = execute_genome(s, 'earth')
    assert len(bp.parts_used) > 0, "genome produced no parts"
    assert len(bp.instruction_log) > 0
    assert all(p in PARTS_KB for p in bp.parts_used)
    print(f"  [7] blueprint: {len(bp.parts_used)} parts, "
          f"systems={bp.systems_score():.2f}, div={bp.diversity_score():.2f}")

    # 8. Blueprint functional completeness is achievable
    complete_count = 0
    for _ in range(50):
        s = Strand.random(80, 4)
        bp = execute_genome(s, 'earth')
        if bp.systems_score() == 1.0:
            complete_count += 1
    print(f"  [8] functional completeness: {complete_count}/50 genomes have all systems")

    # 9. Element supply
    sup = ElementSupply.earth_like()
    assert sup.can_afford({'C': 10, 'N': 5})
    assert sup.richness() > 0.5
    print("  [9] element supply OK")

    # 10. Organism full pipeline with blueprint
    s = Strand.random(80, 4)
    sup = ElementSupply.earth_like()
    org = Organism(strand=s, env_name='earth')
    org.express(sup)
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

    # 15. Fragment library still valid
    for base in FRAG_LIB:
        for frag in FRAG_LIB[base]:
            mol = Chem.MolFromSmiles(frag)
            assert mol is not None, f"invalid fragment: {frag} for base {base}"
    print("  [15] fragment library valid")

    # 16. Reference library: all hardcoded genomes produce expected organisms
    ref_errors = _validate_reference_library()
    assert not ref_errors, f"reference library errors: {ref_errors}"
    print(f"  [16] reference library OK ({len(REFERENCE_LIBRARY)} organisms validated)")

    print("--- all tests passed ---\n")


# ---------------------------------------------------------------------------
# Deterministic genome execution (for UI display — no random splicing)
# ---------------------------------------------------------------------------

def execute_genome_deterministic(strand: Strand, env_name: str = 'earth') -> Blueprint:
    """Like execute_genome but uses deterministic codons for repeatable display."""
    bp = Blueprint()
    codons = strand.codons_deterministic()
    skip_next = False
    env_hash = sum(ord(c) for c in env_name) % 3

    for codon in codons:
        opcode, operand = codon_to_instruction(codon)

        if skip_next:
            skip_next = False
            bp.instruction_log.append('SKIPPED')
            continue

        if opcode == Opcode.BUILD:
            part_idx = operand % len(PART_NAMES)
            part_name = PART_NAMES[part_idx]
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
                    bp.connections.append((a, b))
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
            if operand % 3 != env_hash:
                skip_next = True
            bp.instruction_log.append('REGULATE')
        else:
            bp.instruction_log.append('NOOP')

    return bp

# ---------------------------------------------------------------------------
# DNA-to-English translator
# ---------------------------------------------------------------------------

OPCODE_NAMES = {
    Opcode.BUILD: 'BUILD', Opcode.CONNECT: 'CONNECT',
    Opcode.REPEAT: 'REPEAT', Opcode.REGULATE: 'REGULATE',
    Opcode.NOOP: 'NOOP',
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


def translate_to_english(strand: Strand, env_name: str = 'earth') -> str:
    """Translate a genome into plain English a human can understand."""
    from collections import Counter
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
        opcode, operand = codon_to_instruction(codon)
        step += 1
        if opcode == Opcode.BUILD:
            part_idx = operand % len(PART_NAMES)
            pn = PART_NAMES[part_idx]
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

    # --- Verdict ---
    score = bp.systems_score()
    lines.append("--- VERDICT ---")
    lines.append("")
    if score == 1.0:
        lines.append("ALIVE: This genome encodes a viable organism.")
        lines.append("It has a body, energy, and tools — the minimum for independent life.")
        if bp.roles_present.get('information', 0) > 0:
            lines.append("It also has information storage, so it could reproduce.")
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
    from collections import Counter
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
    para += f"When copied, this DNA {fidelity}."
    return para


def launch_ui():
    """Launch the Periodic Machine graphical interface."""
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox, filedialog
    import threading

    class PeriodicMachine(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("Periodic Machine — Chemistry-DNA Language Interpreter")
            self.geometry("1200x900")
            self.configure(bg='#1a1a2e')
            self.minsize(900, 700)
            self._current_strand = None
            self._current_bp = None
            self._evolving = False
            self._build_styles()
            self._build_ui()

        def _build_styles(self):
            style = ttk.Style(self)
            style.theme_use('clam')
            style.configure('TFrame', background='#1a1a2e')
            style.configure('TLabel', background='#1a1a2e', foreground='#e0e0e0',
                             font=('Consolas', 10))
            style.configure('Header.TLabel', background='#1a1a2e', foreground='#00d4ff',
                             font=('Consolas', 14, 'bold'))
            style.configure('TButton', font=('Consolas', 10), padding=6)
            style.configure('Accent.TButton', font=('Consolas', 10, 'bold'), padding=6)
            style.configure('TLabelframe', background='#1a1a2e', foreground='#00d4ff',
                             font=('Consolas', 10, 'bold'))
            style.configure('TLabelframe.Label', background='#1a1a2e', foreground='#00d4ff',
                             font=('Consolas', 10, 'bold'))
            style.configure('TCombobox', font=('Consolas', 10))

        def _build_ui(self):
            title_frame = ttk.Frame(self)
            title_frame.pack(fill='x', padx=10, pady=(10, 5))
            ttk.Label(title_frame, text="PERIODIC MACHINE",
                      style='Header.TLabel').pack(side='left')
            ttk.Label(title_frame, text="Chemistry-DNA Language Interpreter",
                      foreground='#888').pack(side='left', padx=(15, 0))

            ctrl_frame = ttk.Frame(self)
            ctrl_frame.pack(fill='x', padx=10, pady=5)

            ttk.Label(ctrl_frame, text="Base System:").pack(side='left', padx=(0, 5))
            self._base_var = tk.StringVar(value='4')
            ttk.Combobox(ctrl_frame, textvariable=self._base_var,
                         values=['2', '4', '6', '8'], width=5,
                         state='readonly').pack(side='left', padx=(0, 15))

            ttk.Label(ctrl_frame, text="Environment:").pack(side='left', padx=(0, 5))
            self._env_var = tk.StringVar(value='earth')
            ttk.Combobox(ctrl_frame, textvariable=self._env_var,
                         values=ENVS, width=10,
                         state='readonly').pack(side='left', padx=(0, 15))
            ttk.Label(ctrl_frame, text="Length:").pack(side='left', padx=(0, 5))
            self._len_var = tk.StringVar(value='80')
            ttk.Entry(ctrl_frame, textvariable=self._len_var, width=6).pack(side='left', padx=(0, 15))

            for label, cmd, sty in [
                ("Random Genome", self._gen_random, 'TButton'),
                ("Translate", self._translate, 'Accent.TButton'),
                ("Life Forms", self._show_reference_library, 'TButton'),
                ("Evolve (10 gen)", self._evolve_short, 'TButton'),
                ("Export JSON", self._export_json, 'TButton'),
                ("Clear", self._clear, 'TButton'),
            ]:
                ttk.Button(ctrl_frame, text=label, command=cmd,
                           style=sty).pack(side='left', padx=3)

            input_frame = ttk.LabelFrame(self, text="GENOME INPUT — enter DNA sequence or generate one")
            input_frame.pack(fill='x', padx=10, pady=5)
            self._genome_entry = scrolledtext.ScrolledText(
                input_frame, height=3, font=('Consolas', 11),
                bg='#0f0f23', fg='#00ff88', insertbackground='#00ff88',
                wrap='char', borderwidth=0, highlightthickness=1,
                highlightcolor='#00d4ff', highlightbackground='#333')
            self._genome_entry.pack(fill='x', padx=5, pady=5)

            paned = ttk.PanedWindow(self, orient='horizontal')
            paned.pack(fill='both', expand=True, padx=10, pady=5)

            left_frame = ttk.LabelFrame(paned, text="TRANSLATION — Human-Readable Output")
            paned.add(left_frame, weight=3)
            self._trans = scrolledtext.ScrolledText(
                left_frame, font=('Consolas', 10),
                bg='#0f0f23', fg='#e0e0e0', wrap='word',
                borderwidth=0, highlightthickness=0)
            self._trans.pack(fill='both', expand=True, padx=5, pady=5)
            for tag, cfg in [
                ('header', {'foreground': '#00d4ff', 'font': ('Consolas', 11, 'bold')}),
                ('section', {'foreground': '#ffd700', 'font': ('Consolas', 10, 'bold')}),
                ('build', {'foreground': '#00ff88'}),
                ('connect', {'foreground': '#ff8800'}),
                ('repeat', {'foreground': '#ff44ff'}),
                ('regulate', {'foreground': '#44aaff'}),
                ('noop', {'foreground': '#555555'}),
                ('present', {'foreground': '#00ff88', 'font': ('Consolas', 10, 'bold')}),
                ('missing', {'foreground': '#ff4444', 'font': ('Consolas', 10, 'bold')}),
                ('viable', {'foreground': '#00ff88', 'font': ('Consolas', 12, 'bold')}),
                ('incomplete', {'foreground': '#ff4444', 'font': ('Consolas', 12, 'bold')}),
            ]:
                self._trans.tag_configure(tag, **cfg)

            right_frame = ttk.LabelFrame(paned, text="BLUEPRINT — Parts & Systems")
            paned.add(right_frame, weight=2)
            self._bp_out = scrolledtext.ScrolledText(
                right_frame, font=('Consolas', 10),
                bg='#0f0f23', fg='#e0e0e0', wrap='word',
                borderwidth=0, highlightthickness=0)
            self._bp_out.pack(fill='both', expand=True, padx=5, pady=5)
            for tag, cfg in [
                ('role_structure', {'foreground': '#44aaff'}),
                ('role_energy', {'foreground': '#ffaa00'}),
                ('role_catalysis', {'foreground': '#00ff88'}),
                ('role_information', {'foreground': '#ff44ff'}),
                ('role_minimal', {'foreground': '#888888'}),
                ('metric', {'foreground': '#00d4ff'}),
                ('title', {'foreground': '#ffd700', 'font': ('Consolas', 10, 'bold')}),
            ]:
                self._bp_out.tag_configure(tag, **cfg)

            self._status_var = tk.StringVar(value="Ready — enter a genome or click 'Random Genome'")
            ttk.Label(self, textvariable=self._status_var,
                      foreground='#888', font=('Consolas', 9)).pack(fill='x', padx=10, pady=(0, 5))
            self._show_welcome()

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
            seq = self._genome_entry.get('1.0', 'end').strip()
            nb = self._nb()
            if not seq:
                raise ValueError("No genome entered")
            valid = set(BASE_SYSTEMS[nb]['bases'])
            cleaned = ''.join(c for c in seq.upper() if c in valid)
            if len(cleaned) < CODON_LEN:
                raise ValueError(f"Need at least {CODON_LEN} valid bases")
            return Strand(cleaned, nb)

        # --- welcome ---

        def _show_welcome(self):
            self._trans.delete('1.0', 'end')
            self._trans.insert('end', "PERIODIC MACHINE\n", 'header')
            self._trans.insert('end', "Chemistry-DNA Language Interpreter\n\n", 'section')
            self._trans.insert('end',
                "DNA is a program. Each codon is an instruction.\n"
                "The genome encodes which parts to BUILD,\n"
                "how to CONNECT them, when to REPEAT,\n"
                "and how to REGULATE based on environment.\n\n")
            self._trans.insert('end', "Actions:\n", 'section')
            self._trans.insert('end',
                "  1. Type a genome (e.g. ATCGATCG...) or click Random Genome\n"
                "  2. Click Translate to see what it builds\n"
                "  3. Click Life Forms to browse known organisms\n"
                "  4. Click Evolve to optimize over generations\n"
                "  5. Click Export JSON to save for particle sim\n\n")
            self._trans.insert('end', "Base systems:\n", 'section')
            for nb, si in sorted(BASE_SYSTEMS.items()):
                self._trans.insert('end',
                    f"  {nb}-base ({si['name']}): err={si['error_rate']*100:.1f}%\n")

            self._bp_out.delete('1.0', 'end')
            self._bp_out.insert('end', "BIOLOGICAL PARTS CATALOG\n", 'title')
            self._bp_out.insert('end', f"{len(PARTS_KB)} parts the genome can build:\n\n")
            for name, part in PARTS_KB.items():
                tag = f"role_{part['role']}"
                eng = PART_ENGLISH.get(name, (name, part['desc'], ''))
                self._bp_out.insert('end', f"  {eng[0]}\n", tag)
                self._bp_out.insert('end', f"    {eng[1]}\n")
                self._bp_out.insert('end', f"    {eng[2]}\n\n")

        # --- reference library ---

        def _show_reference_library(self):
            """Display all reference life forms in both panels."""
            self._trans.delete('1.0', 'end')
            self._trans.insert('end', "REFERENCE LIBRARY — Known Life Forms\n", 'header')
            self._trans.insert('end', "Click any organism name below to load its genome.\n\n", 'section')

            categories = [
                ('prebiotic', 'Before Life'),
                ('minimal', 'Minimal Life'),
                ('simple', 'Simple Cells'),
                ('complex', 'Complex Cells'),
                ('specialized', 'Specialized Cells'),
                ('alien', 'Alien Base Systems'),
                ('theoretical', 'Theoretical / Extreme'),
            ]
            cat_refs = {}
            for ref in REFERENCE_LIBRARY:
                cat_refs.setdefault(ref['category'], []).append(ref)

            for cat_key, cat_label in categories:
                refs = cat_refs.get(cat_key, [])
                if not refs:
                    continue
                self._trans.insert('end', f"--- {cat_label.upper()} ---\n", 'section')
                self._trans.insert('end', "\n")
                for ref in refs:
                    sys_name = BASE_SYSTEMS[ref['n_bases']]['name']
                    viable_tag = 'present' if ref['expected_viable'] else 'missing'
                    viable_label = 'ALIVE' if ref['expected_viable'] else 'NOT ALIVE'
                    self._trans.insert('end', f"  {ref['name']}", 'header')
                    self._trans.insert('end', f"  [{sys_name}, {ref['n_bases']}-base]  ")
                    self._trans.insert('end', f"[{viable_label}]\n", viable_tag)
                    self._trans.insert('end', f"    {ref['description']}\n\n")
                    if ref['expected_parts']:
                        part_names = [PART_ENGLISH.get(p, (p,))[0] for p in ref['expected_parts']]
                        self._trans.insert('end', f"    Parts: {', '.join(part_names)}\n", 'build')
                    self._trans.insert('end', f"    Genome: ", 'section')
                    self._trans.insert('end', f"{ref['genome']}\n\n")

            # Right panel: summary table
            self._bp_out.delete('1.0', 'end')
            self._bp_out.insert('end', "LIFE FORMS SUMMARY\n", 'title')
            self._bp_out.insert('end', f"{len(REFERENCE_LIBRARY)} reference organisms\n\n")

            self._bp_out.insert('end', "LOAD AN ORGANISM\n", 'title')
            self._bp_out.insert('end', "Select from the list below:\n\n")

            for i, ref in enumerate(REFERENCE_LIBRARY):
                btn_tag = f"ref_btn_{i}"
                sys_name = BASE_SYSTEMS[ref['n_bases']]['name']
                viable = 'viable' if ref['expected_viable'] else 'non-viable'
                n_parts = len(ref['expected_parts'])

                self._bp_out.tag_configure(btn_tag, foreground='#00d4ff',
                                            underline=True, font=('Consolas', 10, 'bold'))
                self._bp_out.tag_bind(btn_tag, '<Button-1>',
                                       lambda e, r=ref: self._load_reference(r))
                self._bp_out.tag_bind(btn_tag, '<Enter>',
                                       lambda e, t=btn_tag: self._bp_out.tag_configure(
                                           t, foreground='#ffd700'))
                self._bp_out.tag_bind(btn_tag, '<Leave>',
                                       lambda e, t=btn_tag: self._bp_out.tag_configure(
                                           t, foreground='#00d4ff'))

                self._bp_out.insert('end', f"  > {ref['name']}\n", btn_tag)
                self._bp_out.insert('end', f"    {sys_name} | {n_parts} parts | {viable}\n\n")

            self._bp_out.insert('end', "\nBASE SYSTEM COMPARISON\n", 'title')
            for nb in [2, 4, 6, 8]:
                si = BASE_SYSTEMS[nb]
                refs_for_nb = [r for r in REFERENCE_LIBRARY if r['n_bases'] == nb]
                self._bp_out.insert('end', f"\n  {si['name'].upper()} ({nb}-base)\n", 'metric')
                self._bp_out.insert('end', f"    Error rate: {si['error_rate']*100:.1f}%\n")
                self._bp_out.insert('end', f"    Letters: {', '.join(si['bases'])}\n")
                self._bp_out.insert('end', f"    Reference organisms: {len(refs_for_nb)}\n")

            self._status_var.set(
                f"Reference Library — {len(REFERENCE_LIBRARY)} known life forms. "
                f"Click a name to load it.")

        def _load_reference(self, ref):
            """Load a reference organism's genome and translate it."""
            self._base_var.set(str(ref['n_bases']))
            self._genome_entry.delete('1.0', 'end')
            self._genome_entry.insert('1.0', ref['genome'])
            self._status_var.set(f"Loaded: {ref['name']} — click Translate")
            self._translate()

        # --- actions ---

        def _gen_random(self):
            nb, gl = self._nb(), self._gl()
            s = Strand.random(gl, nb)
            self._genome_entry.delete('1.0', 'end')
            self._genome_entry.insert('1.0', s.seq)
            self._status_var.set(f"Generated random {nb}-base genome, {gl} bases")

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
            for line in translation.split('\n'):
                if line.startswith('READING THIS GENOME'):
                    self._trans.insert('end', line + '\n', 'header')
                elif line.startswith('ALIVE:'):
                    self._trans.insert('end', line + '\n', 'viable')
                elif line.startswith('INCOMPLETE:') or line.startswith('NON-VIABLE:'):
                    self._trans.insert('end', line + '\n', 'incomplete')
                elif line.startswith('---'):
                    self._trans.insert('end', line + '\n', 'section')
                elif line.strip().startswith('Step') and 'Make ' in line:
                    self._trans.insert('end', line + '\n', 'build')
                elif line.strip().startswith('Step') and 'Connect' in line:
                    self._trans.insert('end', line + '\n', 'connect')
                elif line.strip().startswith('Step') and ('copy of' in line or 'Copy' in line):
                    self._trans.insert('end', line + '\n', 'repeat')
                elif line.strip().startswith('Step') and 'Check' in line:
                    self._trans.insert('end', line + '\n', 'regulate')
                elif line.strip().startswith('Step') and 'non-coding' in line:
                    self._trans.insert('end', line + '\n', 'noop')
                elif ': YES' in line:
                    self._trans.insert('end', line + '\n', 'present')
                elif ': NO' in line:
                    self._trans.insert('end', line + '\n', 'missing')
                elif line.strip().startswith('(Like ') or line.strip().startswith('('):
                    self._trans.insert('end', line + '\n', 'regulate')
                else:
                    self._trans.insert('end', line + '\n')

            bp = execute_genome_deterministic(strand, env)
            self._current_bp = bp

            # Build full organism for grading and chemistry
            supply = ElementSupply.earth_like()
            org = Organism(strand=strand, env_name=env)
            org.express(supply, deterministic=True)
            org.evaluate()
            self._current_org = org

            self._render_bp(bp, strand, env, org)
            grade, grade_desc = org.structural_grade()
            self._status_var.set(
                f"[{grade}] {grade_desc}  |  "
                f"Fitness: {org.fitness:.2f}  |  "
                f"{len(bp.parts_used)} parts, {len(set(bp.parts_used))} unique")

        def _render_bp(self, bp, strand, env, org=None):
            from collections import Counter
            self._bp_out.delete('1.0', 'end')

            # Structural grade header
            if org is not None:
                grade, grade_desc = org.structural_grade()
                grade_colors = {'S': '#ffd700', 'A': '#00ff88', 'B': '#44aaff',
                                'C': '#ff8800', 'D': '#ff4444', 'F': '#888888'}
                grade_tag = 'grade_display'
                self._bp_out.tag_configure(grade_tag,
                    foreground=grade_colors.get(grade, '#e0e0e0'),
                    font=('Consolas', 16, 'bold'))
                self._bp_out.insert('end', f"  GRADE: {grade}\n", grade_tag)
                self._bp_out.insert('end', f"  {grade_desc}\n", 'title')
                self._bp_out.insert('end', f"  Fitness: {org.fitness:.4f}\n\n")

            self._bp_out.insert('end', "ASSEMBLED ORGANISM\n", 'title')
            self._bp_out.insert('end', f"System: {strand.system['name']}  |  Env: {env}\n\n")

            self._bp_out.insert('end', "PARTS INVENTORY\n", 'title')
            if bp.parts_used:
                for pn, cnt in Counter(bp.parts_used).most_common():
                    tag = f"role_{PARTS_KB[pn]['role']}"
                    eng = PART_ENGLISH.get(pn, (pn, '', ''))
                    self._bp_out.insert('end', f"  {cnt}x ", 'metric')
                    self._bp_out.insert('end', f"{eng[0]}\n", tag)
                    self._bp_out.insert('end', f"     {eng[2]}\n\n")
            else:
                self._bp_out.insert('end', "  (none)\n\n")

            self._bp_out.insert('end', "SYSTEMS STATUS\n", 'title')
            all_roles = set(list(REQUIRED_SYSTEMS) + list(bp.roles_present))
            for role in sorted(all_roles):
                needed = REQUIRED_SYSTEMS.get(role, 0)
                have = bp.roles_present.get(role, 0)
                marker = "[OK]" if needed > 0 and have >= needed else (
                    "[!!]" if needed > 0 else "[++]")
                se = SYSTEM_ENGLISH.get(role, {'name': role})
                tag = f"role_{role}" if f"role_{role}" in (
                    'role_structure', 'role_energy', 'role_catalysis',
                    'role_information', 'role_minimal') else 'metric'
                self._bp_out.insert('end', f"  {marker} {se['name']}: {have}\n", tag)

            self._bp_out.insert('end', "\nMETRICS\n", 'title')
            for label, val in [
                ('Systems', f"{bp.systems_score():.0%}"),
                ('Diversity', f"{bp.diversity_score():.0%}"),
                ('Efficiency', f"{bp.efficiency_score():.0%}"),
                ('Parts', f"{len(bp.parts_used)}"),
                ('Unique', f"{len(set(bp.parts_used))}"),
                ('Connections', f"{len(bp.connections)}"),
                ('Stability', f"{strand.stability_score():.4f}"),
                ('Integrity', f"{strand.pair_integrity():.0%}"),
            ]:
                self._bp_out.insert('end', f"  {label}: ", 'metric')
                self._bp_out.insert('end', f"{val}\n")

            # Chemistry from organism (built per-part, reliable)
            if org is not None and hasattr(org, '_part_mols') and org._part_mols:
                n_mols = len(org._part_mols)
                self._bp_out.insert('end', f"\nCHEMISTRY ({n_mols} molecules built)\n", 'title')
                self._bp_out.insert('end', f"  Total MW: {org._total_mw:.1f} Da\n")
                self._bp_out.insert('end', f"  Heavy atoms: {org._total_heavy}\n")
                self._bp_out.insert('end', f"  Rings: {org._total_rings}\n")
                self._bp_out.insert('end', f"  H-bond acceptors: {org._total_hba}\n")
                self._bp_out.insert('end', f"  H-bond donors: {org._total_hbd}\n")

                if org.element_cost:
                    self._bp_out.insert('end', "\nELEMENT COST\n", 'title')
                    for el, cnt in sorted(org.element_cost.items(), key=lambda x: -x[1]):
                        mass = PERIODIC.get(el, {}).get('mass', 0) * cnt
                        self._bp_out.insert('end', f"  {el}: {cnt}", 'metric')
                        self._bp_out.insert('end', f"  ({mass:.1f} amu)\n")

                geom = getattr(org, '_organism_geom', None)
                if geom:
                    self._bp_out.insert('end', f"\n3D STRUCTURE\n", 'title')
                    self._bp_out.insert('end', f"  {len(geom['atoms'])} atoms positioned\n")
                    self._bp_out.insert('end', f"  {len(geom['bonds'])} bonds mapped\n")
                    virtual = sum(1 for b in geom['bonds'] if b.get('virtual'))
                    if virtual:
                        self._bp_out.insert('end', f"  {virtual} inter-part connections\n")
            else:
                try:
                    mol = build_mol(bp.combined_smiles)
                    if mol:
                        ec = mol_element_counts(mol)
                        self._bp_out.insert('end', "\nELEMENT COST\n", 'title')
                        for el, cnt in sorted(ec.items(), key=lambda x: -x[1]):
                            mass = PERIODIC.get(el, {}).get('mass', 0) * cnt
                            self._bp_out.insert('end', f"  {el}: {cnt}", 'metric')
                            self._bp_out.insert('end', f"  ({mass:.1f} amu)\n")
                except Exception:
                    pass

        def _evolve_short(self):
            if self._evolving:
                self._status_var.set("Evolution already running...")
                return
            nb = self._nb()
            self._evolving = True
            self._status_var.set("Evolving... (10 generations)")

            def _run():
                try:
                    ev = Evolver(pop_size=60, genome_len=self._gl(),
                                 generations=10, n_bases=nb, n_workers=4)
                    result = ev.run()
                    best = result['best']
                    self.after(0, self._on_evolve_done, best)
                except Exception as e:
                    self.after(0, lambda: self._on_evolve_error(str(e)))

            threading.Thread(target=_run, daemon=True).start()

        def _on_evolve_done(self, best):
            self._evolving = False
            self._genome_entry.delete('1.0', 'end')
            self._genome_entry.insert('1.0', best.get('genome', ''))
            grade = best.get('structural_grade', '?')
            self._status_var.set(
                f"Evolution done — [{grade}] fitness={best.get('fitness', 0):.4f}, "
                f"parts={best.get('blueprint', {}).get('n_parts', 0)}")
            self._translate()

        def _on_evolve_error(self, err):
            self._evolving = False
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
    else:
        launch_ui()