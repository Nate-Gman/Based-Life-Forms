# Periodic Machine — a chemistry-DNA language engine

**Periodic Machine v2.5.0** is a single-file artificial-life engine
(`Language.py`, ~21,600 lines) that treats a DNA sequence as an
executable program.  Genomes are read as 5-base codons, decoded into
opcodes, and interpreted into molecular **blueprints** — part lists of
real molecules (SMILES-validated via RDKit), connections, operons,
metabolic chains, motors, neural circuits and behaviors.  Organisms are
then evaluated for viability against environmental element supplies,
scored for fitness, and evolved with mutation/crossover/selection.

The guiding constraint (see `Goal.md`): *be based in reality* — real
DNA semantics and real chemical/physical processes that produce the
same results DNA does when it builds biomachines.  Every headline
number in the engine is **derived, not asserted**: the `--verify`
audit recomputes each value from first principles and compares it
against the literature.

## Requirements

- Python 3.10+ (developed on 3.13, Windows)
- `numpy` — required for the physics sims
- `rdkit` — optional but recommended; the engine degrades gracefully
  without it (SMILES parsing/3D geometry disabled, element accounting
  falls back to estimation).  Auto-installs on first run when possible.
- `pygame` — optional, only for the `--viewer` 3-D molecule viewer
- `tkinter` — ships with CPython; only needed for the GUI

## Quickstart

```bash
python Language.py            # launch the UI (Tkinter)
python Language.py --help     # all modes
```

```bash
python Language.py --test     # 113-test validation suite
python Language.py --verify   # 29-check derived-numbers audit
python Language.py --report   # intelligent organism + chemistry report
```

## CLI modes

| Flag | What it does |
|---|---|
| *(none)* | Launch the Periodic Machine GUI |
| `--test`, `--selftest` | Run the 113-test validation suite |
| `--verify` | Derived-numbers audit — 29 checks, no asserted values |
| `--report` | Build a high-intelligence organism + print a chemistry report |
| `--evolve` | Evolve a population (200 pop × 120 gens × 12 workers) → `evolved_blueprints.json` + `experiment_log.jsonl` |
| `--replicates [N]` | N independent runs → mean best fitness + 95% CI |
| `--compare` | 4-base vs 6-base vs 8-base evolution comparison |
| `--benchmark` | Benchmark suite → `benchmark_results.json` |
| `--human-body` | Human body architecture report → `human_body_constructed.json` |
| `--viewer [geom.json]` | Shaded-sphere 3-D molecule viewer (pygame) |
| `--genbank FILE.gb` | Import a GenBank flat file (real CDS annotation) |
| `--sbml [out.xml]` | Export the FBA network as SBML Level 3 XML |
| `--no-gui` | Initialize caches then exit (CI smoke check) |

Options: `--fast` (skip 3-D embedding, ~5–10× faster), `--seed N`
(reproducible runs), `--resume` (continue `--evolve` from
`evolution_checkpoint.json`).

## What the language looks like

A genome is a string over one of four alphabets — 2, 4, 6 or 8 bases
(`A T C G X Y W Z` at 8-base) — read as 5-base codons.
The first base dispatches the opcode; the remaining four are the operand
(`operand = Σ ord(c)`):

| Opcode | Effect |
|---|---|
| `BUILD` / `CONNECT` / `REPEAT` | add parts, bond them, copy the last part |
| `REGULATE` | environment-conditional skip (promoter logic) |
| `PROMOTER` / `TERMINATOR` | operon boundaries (`CC`/`TT` prefixes, every alphabet ≥4) |
| `MOTOR` / `LOCALIZE` / `POLYMERIZE` / `PHASE` / `CHEMOTAXIS` / `MEMBRANE_INS` | `W`+modifier family (8-base) |
| `SECRETE` / `QUORUM` / `DIVIDE` / `METABOLIZE` / `GRADIENT` | `Z`+modifier family (8-base) |
| `NOOP` | inert spacer (`TG` prefix, or invalid codon) |

Smaller alphabets genuinely cannot encode the full opcode set — a
4-base genome has no W/Z bases — so the template builder emits honest
NOOPs and injects the unencodable parts/effects rather than silently
substituting the wrong instruction.

`translate_to_english()` renders any genome as a step-by-step English
build narrative that mirrors the interpreter's actual control flow
(including `REGULATE` skips).

## Verification status

- **113/113** built-in tests (`--test`)
- **29/29** derived-numbers audit checks (`--verify`)
- pyflakes clean
- Deterministic evolution: seeded runs and checkpoint/resume are
  bit-identical (population + per-generation trajectory)
- All 11 organism templates build and encode 100% of specified parts;
  library round-trips are lossless — and in small alphabets the
  unencodable **opcode effects** (division, quorum, secretion,
  chemotaxis, gradients, phase) are injected alongside the parts
- Standard formats round-trip: FASTA, GenBank (join/complement/remote
  refs), SBML L3, PDB/mmCIF, SDF, organism-library JSON, genome
  compression
- **Input-honest API surface**: invalid physical/math inputs raise
  `ValueError` with the constraint named rather than being silently
  clamped — covers stochastic kinetics (SSA/tau-leap/reaction rates),
  population genetics (Wright–Fisher, Muller's ratchet, SIR, logistic
  growth, mutation/replication-error rates), the motor/diffusion/
  quorum/Turing sims, `Strand`, `ElementSupply`, `Evolver`,
  blood-chemistry and library-deserialization paths

See `updateprogression.md` for the running log of every defect found
and fixed across eight adversarial review passes, and `overview.md`
for the architecture tour + honest limitations.
