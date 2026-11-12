# Periodic Machine — architecture & reality audit overview

Companion to `README.md` (usage) and `updateprogression.md` (change
log).  This file describes how `Language.py` is organized, what each
subsystem models, and — most importantly — **where the model is real
and where it is a documented approximation**.

## Pipeline

```
genome string ──► Strand (replication, transcription, splicing,
│                        methylation, pair integrity)
│                5-base codons
▼
codon ──► codon_to_instruction ──► (Opcode, operand)
│       dispatch table keyed by the alphabet tuple
▼
execute_genome ──► Blueprint
│   parts_used, connections, spatial_graph, localizations,
│   operons, metabolic_chains, energy budget/cost, motors,
│   chemotaxis/quorum/division events, neural+homeostasis parts,
│   intelligence score, replication probability
▼
Organism.evaluate ──► chemistry (RDKit mols / SMILES fallback),
    element cost vs ElementSupply, viability, fitness
▼
Evolver ──► mutation (realistic Ts/Tv/CpG/indels), crossover,
    tournament selection, niche cycling, species clustering,
    deterministic checkpoints + experiment provenance log
```

## Module map (single file, section-delimited)

| Region | Contents |
|---|---|
| Constants / base systems | `CODON_LEN`, alphabets (2/4/6/8-base), per-system error rates, environments (`earth`, `silicon`, `exotic`) |
| `Strand`, `ElementSupply` | double-stranded genome, complement, replication w/ error model, CpG methylation + DNMT1-style inheritance, transcription/splicing |
| Real-biology numerics | standard genetic code, transcription/translation, protein properties (MW, hydropathy, net charge, pI via Henderson–Hasselbalch), Hodgkin–Huxley ODE, metabolism stoichiometry, chemiosmotic thermodynamics, kinetics (MM, Q10, Arrhenius, diffusion), duplex nearest-neighbor thermodynamics, codon usage/CAI |
| Cell & organism physiology | cell cycle, replication fork, membrane potential (GHK/Nernst), ion channels, Ca²⁺ signaling, sarcomere/kinesin/ATP-synthase mechanics, turgor, oxygen transport (MWC hemoglobin, Bohr), buffers, Krogh tissue oxygenation, cardiac AP |
| Genetics & evolution | realistic mutation (Ts/Tv ratio, methylated-CpG hypermutability + deamination, indels), proofreading cascades, N-end rule, telomeres, DNA linking number/compaction, Hardy–Weinberg, drift/Wright–Fisher, selection, mutation load, Muller's ratchet, coalescent, Tajima's D, dN/dS, Luria–Delbrück |
| Stochastic & systems bio | Gillespie SSA, tau-leaping, telegraph bursting, quantal release, gene-expression noise, FBA (32 ATP aerobic / 2 anaerobic, SBML export), sensitivity analysis, enzyme inhibition, lac operon, stringent response, SOS, p53/DSB repair choice |
| Development & ecology | morphogen diffusion + French-flag fates, Delta-Notch lateral inhibition, multicellular development grid, Turing patterns, Hopfield memory, LIF/STDP neurons, repressilator/KaiC clocks, Lotka–Volterra, SIR, plasmid persistence, transposon load |
| Sequence & structure | NW/SW alignment, Karlin–Altschul E-values, ORF finding (6 frames), FASTA/GenBank import, Nussinov RNA fold, restriction sites, sequence logos, CRISPR/PAM, NMD |
| Interpreter | `Opcode`, `LanguageSpec` dispatch, `execute_genome`, `Blueprint` scoring (torque/RPM, membrane potential, operons, metabolic completeness, intelligence tiers), `translate_to_english` parity |
| Evolution | `Evolver` (parallel eval, seeded determinism, checkpoints, landscape/species tracking), `compare_base_systems`, `evolution_replicates`, `record_experiment` |
| Organism factories | `MICROBE_TEMPLATES` (11 real templates), `build_microbe_organism` (genome execution + honest injection of unencodable parts/opcode-effects), `generate_intelligent_organism`, human genome/body caches |
| Export / UI | PDB/mmCIF/SDF/WebGL JSON, genome compression, organism library JSON, pygame 3-D viewer, Tkinter `PeriodicMachine` UI, `--test`/`--verify`/`--report`/`--evolve`/etc. CLI |

## Reality policy

The project's correctness bar, enforced in review:

- **Derived, not asserted.**  `--verify` recomputes every headline
  number (ATP yield, HH spike shape, motor RPM, p50, Tm, FBA fluxes,
  kinesin speed, diffusion times, population-genetics quantities) from
  the same equations and compares it to the literature value.
- **Honest failure.**  Physically or mathematically invalid input
  raises `ValueError`/`FileNotFoundError` rather than being silently
  clamped or substituted.  The silent-clamp sweep removed masked
  inputs across: `blood_ph` (hco3 floor), `logistic_growth` (K/n0
  clamps + spontaneous-generation bug at n0=0), `mullers_ratchet`,
  `consensus_replicate` (error-rate clamp), `mutate_seq` (rate
  bounds), `sir_epidemic` (i0 > population), `splicing_isoforms`,
  `gillespie_ssa`/`tau_leap`/`Reaction` (negative counts/rates),
  `MotorSimulator` (dt/steps), `ElementSupply` (non-dict stock),
  `SignalField`, `simulate_quorum_field` (off-grid cells),
  `multicellular_development`, `turing_pattern`, `Strand`,
  `methylate_cpg`, `Evolver` (genome_len/generations) and
  `organism_from_library_dict` (schema types).  Unknown environment
  names also raise instead of silently assuming earth chemistry.
- **No fabricated biology.**  Unencodable genome instructions produce
  NOOPs + explicit spec-deficit injection (parts *and* opcode effects —
  division/quorum/secretion/chemotaxis/gradient/phase — marked as
  injection), never a random wrong part; remote GenBank coordinates
  are skipped, not invented; degenerate grids and flat Turing fields
  report `nan`/`inf` instead of fake fits; turgor clamps at zero with
  an explicit `plasmolyzed` flag rather than reporting impossible
  negative pressure.
- **Determinism where promised.**  Seeded evolution, `deterministic`
  genome execution, and checkpoint/resume are bit-identical.
- **Serializable results.**  Simulation outputs are plain Python
  types — numpy scalars are cast out (`hopfield_memory`,
  `notch_lateral_inhibition` fixed) so every result survives
  `json.dumps`.

## Known approximations (by design, documented in-code)

- **Alphabet capacity.**  4-base genomes can only encode
  BUILD/CONNECT/PROMOTER/TERMINATOR/NOOP; the W/Z opcode families need
  the 8-base alphabet.  This is a real information-capacity constraint,
  surfaced honestly rather than hidden.
- **Human genome module.**  Ten short real anchor-gene fragments +
  continuance multipliers approximate 3.2 Gbp; it is a coverage model,
  not the human genome.
- **Human body construction.**  A literature-census catalog (36.6 T
  cells, ~7.25 kg proteome vs ICRP ~11 kg) — a scaled inventory, not a
  spatial/developmental simulation of a body.
- **Rate magnitudes.**  Mutation/evolution rates are elevated for
  tractability; fitness is a weighted heuristic, not a measured growth
  rate.
- **SMILES corpus.**  57 parts parse in RDKit but large biomolecules are
  simplified fragments; 3-D geometry is a coarse spatial-graph layout.
- **Neural layers.**  Real HH/LIF/Hopfield/STDP dynamics exist, but the
  "intelligence tier" is a complex-presence score on top of them.
- **Environments.**  `silicon`/`exotic` chemistries are parameterized
  what-ifs (element supply + fidelity modifier), not exobiology.

## Test & audit harness

- `run_tests()` — 113 checks covering interpreters, physics, genetics,
  development, ecology, formats, determinism, checkpoint/resume.
- `run_audit()` — 29 derived-number checks; fails if any recomputed
  value leaves its literature band.
- `_probe.py`, `_probe2.py`, `_probe3.py` — adversarial edge-case
  harnesses; their remaining "failures" are intentionally wrong
  signatures or correct-by-design errors, kept as regression tripwires.

## Repo layout

```
Language.py            the entire engine (~21.6k lines, v2.5.0)
README.md              this project's usage doc
overview.md            architecture + reality audit (this file)
Goal.md                original project goals
Infornmational.md      the original critique/defect list that drives review
updateprogression.md   running log of every fix, pass by pass
ReferenceCode/         UI/functionality references (Leviathan.py,
                       ChiefJuiceV1.17.py, OilFabricationTower.py, ...)
_probe*.py             adversarial test harnesses
*.json / *.cif / *.pdb / *.sdf   generated artifacts
```
