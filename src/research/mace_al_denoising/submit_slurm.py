"""Submit the recorded Al preparation as an independent batch job."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex

from src.experiment_runner.slurm import submit_sbatch
from src.simulation.relaxation import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--job', required=True, choices=['prepare_full'])
    args = parser.parse_args()
    job = json.loads(args.plan.read_text())['jobs'][args.job]
    root = Path.cwd()
    out = (root/job['submission_directory']).resolve()
    out.mkdir(parents=True, exist_ok=True)
    record = out/'submission.json'
    if record.exists():
        raise FileExistsError(f'Job already submitted: {record}; inspect Slurm before creating a new attempt.')
    template = out/'config_template.json'
    template.write_bytes(Path(job['config']).read_bytes())
    seconds = job['time_hours']*3600
    runtime_setup = f'''from datetime import datetime, timedelta, timezone
import json, os
from pathlib import Path
cfg = json.loads(Path({str(template)!r}).read_text())
cfg['deadline'] = (datetime.now(timezone.utc) + timedelta(seconds={seconds-900})).isoformat()
Path(os.environ['DENOISING_RUNTIME_CONFIG']).write_text(json.dumps(cfg, indent=2) + '\\n')
'''
    script = f'''#!/bin/bash
#SBATCH --job-name={job['name']}
#SBATCH --partition={job['partition']}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={job['cpus']}
#SBATCH --mem={job['memory']}
#SBATCH --time={job['time_hours']}:00:00
#SBATCH --output={out}/%x_%j.out
#SBATCH --error={out}/%x_%j.err
#SBATCH --chdir={root}
#SBATCH --no-requeue
set -eo pipefail
source /home/infres/vmorozov/miniconda3/etc/profile.d/conda.sh
conda activate pointnet
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH={shlex.quote(str(root))}
export DENOISING_RUNTIME_CONFIG={shlex.quote(str(out))}/runtime_${{SLURM_JOB_ID}}.json
python - <<'PY'
{runtime_setup}PY
nvidia-smi -L
exec python -m src.training_methods.pretrained_mace --config "$DENOISING_RUNTIME_CONFIG" --stage {job['stage']}
'''
    script_path = out/'job.sbatch'
    identifier = submit_sbatch(script, script_path)
    value = dict(job_id=identifier, job=args.job, submitted_at=datetime.now(timezone.utc).isoformat(),
                 script=str(script_path), script_sha256=sha256(script_path), config_snapshot=str(template),
                 config_sha256=sha256(template), plan=job)
    record.write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
