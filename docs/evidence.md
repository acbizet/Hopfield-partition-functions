# Evidence, provenance and limitations

## What is recorded

The archive preserves the exact source JSON files for the headline measurements. `results/provenance.json` maps copied files to their original session paths, with SHA-256 hashes. Different packaged hashes identify portability edits or Markdown math-delimiter normalization. It is a record of original ingestion, not a security attestation or a checksum of later generated documents.

The high-precision experiment fixes the original binary64 bias values as exact rationals. Its independent enumeration reference was evaluated at 100 and 140 digits. Timed kernels do not receive the reference and do not use spin expansion. Both methods use the same precision at a given tolerance.

Parameter selection used an untimed, factorized finite-spin diagnostic. That is feasible for this tiny model; a scalable parameter-selection procedure for general large models is a separate requirement. Analytic cutoffs in the Python positive evaluator address truncation but do not certify every numerical operation.

## Packaging changes

- Session-specific temporary mpmath import fallbacks were removed. Dependencies now come from the Python environment.
- GitHub-compatible display-math delimiters were applied to copied Markdown notes; mathematical content was preserved.
- Original result snapshots were copied without changing numerical values.
- A quick example, native smoke check, figure generator, PDF brief and reproduction guide were added.
- Temporary files, compiled executables and Python caches were excluded.
- New local checks are described separately in `validation.md`; historical full timings are not presented as freshly rerun.

## Claims this repository does not establish

- A general speed advantage over exact enumeration, adaptive quadrature, low-rank samplers or other structure-aware solvers.
- Industrial-scale performance, identified customer demand or a commercial valuation.
- Efficient high-rank scaling, arbitrary real-weight learning, or a formal floating-point certificate.
- A general link between descendant suppression and retrieval reliability.
- Publication priority for the lattice identity, critical coefficients or the Bethe/descent specialization.

At N=8, enumeration visits 256 states while the tightest lattice benchmark visits more than 25 million points. The high-precision comparison is a controlled comparison of two tensor evaluators, not evidence that either is the best algorithm for that instance.

## Research ownership and citation

This package was assembled from an AI-assisted research session. No personal name, affiliation, publication, funding award, or external collaborator has been inferred. Add verified authors and a preferred citation when the account owner is identified. No open-source license has been chosen by the owner yet.
