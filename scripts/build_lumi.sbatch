#!/bin/bash
# Batch build of gnina for LUMI-G. Submit and walk away -- a failure leaves a
# log rather than killing an interactive session you then have to re-queue.
#
#   export PROJECT=project_465XXXXXX
#   export PREFIX=/project/$PROJECT/gnina-rocm     # optional; this is the default
#   FETCH_ONLY=1 ./scripts/build_lumi.sh          # once, on a login node
#   sbatch --account=$PROJECT scripts/build_lumi.sbatch
#
# Watch it with:  tail -f gnina-build-<jobid>.log
#
# The build is CPU-only (compiling for gfx90a never touches a device), so this
# runs on LUMI-C and spends no GPU-hours. Stages are stamped, so if the job hits
# the wall clock you just resubmit and it resumes at the stage that died.
#SBATCH --job-name=gnina-build
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --output=gnina-build-%j.log

set -euo pipefail

: "${PROJECT:?export PROJECT before submitting, or pass --account=...}"

SRC_ROOT=${SLURM_SUBMIT_DIR:-$PWD}
export NPROC=${SLURM_CPUS_PER_TASK:-32}

echo "job $SLURM_JOB_ID on $(hostname), $NPROC cores"
echo "source:  $SRC_ROOT"
echo "prefix:  ${PREFIX:-<default>}"
echo

srun --ntasks=1 --cpus-per-task="$NPROC" "$SRC_ROOT/scripts/build_lumi.sh"

echo
echo "build finished; verify the device architecture before trusting it:"
echo "  strings ${PREFIX:-\$PREFIX}/lib/libmolgrid.a | grep -o 'gfx[0-9a-z]*' | sort -u"
