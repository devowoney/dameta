"""
Split a command list over several parallel sbatch jobs (each job = src/scripts/sb_parallel.sh on one node).

Cluster rules (pilot, 2026-11-19 / 20): CPU-only work on sl-mee-br-202..216, at most 12 of my batch jobs at a time (running +
pending; the interactive session is not counted; pilot 2026-11-20, was 8). NO GPU request (--gres): the runs use CPUs only and a
GPU reservation blocks other users (a job without --gres is accepted on these nodes, test job 79166).
Each job goes to a DIFFERENT node, chosen by idle CPUs (sinfo); lines are dealt round-robin so long and short runs
mix in each job.

Usage (project root, login node: only sinfo / squeue / sbatch here):
    python src/scripts/sb_submit.py <cmd_list> <job_name> [--jobs N] [--cpus C] [--after JOBID[:JOBID...]]
Prints the job ids (colon-separated on the last line, for --dependency=afterok:...).
"""

import argparse
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAX_JOBS = 12                  # pilot 2026-11-20 (was 8); the interactive session is not counted


def sh(cmd):
    return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout


def idle_cpus():
    """{node number: idle CPUs} for 202..216 (Odyssey_GPU partition), most idle first."""
    out = sh("sinfo -h -N -p Odyssey_GPU -n sl-mee-br-[202-216] -o '%N %C %t'")
    idle = {}
    for row in out.split("\n"):
        if not row.strip():
            continue
        node, cpus, state = row.split()
        if state in ("idle", "mix"):
            idle[int(node.rsplit("-", 1)[1])] = int(cpus.split("/")[1])
    return dict(sorted(idle.items(), key=lambda kv: -kv[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd_list")
    ap.add_argument("job_name")
    ap.add_argument("--jobs", type=int, default=MAX_JOBS)
    ap.add_argument("--cpus", type=int, default=32, help="CPUs (= parallel lines) per job")
    ap.add_argument("--after", default="", help="afterok dependency job ids, colon-separated")
    a = ap.parse_args()

    lines = [ln for ln in Path(a.cmd_list).read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    mine = len([r for r in sh("squeue -h -u $USER -o %j").split() if r and r != "interactive"])
    n_jobs = min(a.jobs, MAX_JOBS - mine, math.ceil(len(lines) / 1))
    if n_jobs < 1:
        raise SystemExit(f"already {mine} jobs queued (max {MAX_JOBS}): wait for some to finish")
    nodes = [n for n, c in idle_cpus().items() if c >= 4][:n_jobs]
    n_jobs = len(nodes)
    chunks = [lines[i::n_jobs] for i in range(n_jobs)]               # round-robin deal

    list_dir = ROOT / "tmp/logs/sb_lists"
    list_dir.mkdir(parents=True, exist_ok=True)
    dep = f"--dependency=afterok:{a.after}" if a.after else ""
    ids = []
    for i, (node, chunk) in enumerate(zip(nodes, chunks)):
        part = list_dir / f"{a.job_name}_{i}.txt"
        part.write_text("\n".join(chunk) + "\n")
        cpus = min(a.cpus, len(chunk))
        out = sh(f"sbatch --parsable --job-name={a.job_name}_{i} --nodelist=sl-mee-br-{node} "
                 f"--cpus-per-task={cpus} --mem={12 * cpus}G {dep} "
                 f"{ROOT}/src/scripts/sb_parallel.sh {part} {ROOT} {cpus}")
        jid = out.strip().split(";")[0]
        ids.append(jid)
        print(f"job {jid}: {len(chunk)} lines on sl-mee-br-{node} ({cpus} cpus, no GPU)")
    print(":".join(ids))


if __name__ == "__main__":
    main()
