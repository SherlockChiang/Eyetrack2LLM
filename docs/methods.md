# Methods

The analysis models destination allocation conditional on a retained fixation transition being forward, within sentence, and within line. Fixations are converted conservatively, candidate destinations are defined before fitting, and nuisance expectations are estimated with text-level cross-fitting. Raw Pearson residuals are then evaluated for split-half reliability, recovery in simulation, auxiliary learnability, and cross-corpus transfer.

Reliability uses non-overlapping reader halves and repeated partitions. Specification analyses vary risk sets and nuisance features while preserving the inferential text unit. Auxiliary experiments compare gaze supervision with language-model, shuffled, and positional controls. Transfer uses fixed Provo-trained checkpoints on ZuCo without target-driven checkpoint selection.

Reusable implementations are in `src/eyetrack2llm/`; exact public command lines are in [reproducibility.md](reproducibility.md).
