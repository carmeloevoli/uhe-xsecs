#!/bin/bash
set -euo pipefail

NEVT=${NEVT:-10000}
PYTHON=${PYTHON:-python3}

OUTPUT_DIR=${OUTPUT_DIR:-runs}
PGAMMA_EMIN=${PGAMMA_EMIN:-1e-1}
PGAMMA_EMAX=${PGAMMA_EMAX:-1e2}

for model in Sibyll23e QGSJetII04; do # EposLHC
    for pid in neutron proton; do
        echo "Running p+p model $model for pid $pid"
        "$PYTHON" compute_inelasticity_pp.py --model "$model" --pid "$pid" --nevt "$NEVT" --output-dir "$OUTPUT_DIR"
    done
#    echo "Running p+p energy budget model $model"
#    "$PYTHON" compute_energy_budget_pp.py --model "$model" --nevt "$NEVT" --output-dir "$OUTPUT_DIR"
done

for model in Sophia20; do
    for pid in neutron proton; do
        echo "Running p+gamma model $model for pid $pid"
        "$PYTHON" compute_inelasticity_pgamma.py --model "$model" --pid "$pid" --emin "$PGAMMA_EMIN" --emax "$PGAMMA_EMAX" --nevt "$NEVT" --output-dir "$OUTPUT_DIR"
    done
done

# for model in Sibyll23e EposLHC; do
#     for primary in N14 Fe56; do
#         echo "Running fragment model $model for primary $primary at 1e17 eV"
#         $PYTHON compute_model_A_fragments.py \
#             --model $model \
#             --projectile $primary \
#             --target p \
#             --energy 1e17 \
#             --energy-unit eV \
#             --nevt $NEVT \
#             --output-dir $OUTPUT_DIR
#     done
# done
