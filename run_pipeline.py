#!/usr/bin/env python3
"""run_pipeline.py
Wrapper to run the full pipeline end-to-end and produce a submission ZIP and report.
Usage: python3 run_pipeline.py --workdir /path/to/workdir
"""
import argparse
import subprocess
import os
import shutil
import sys

def run(cmd, cwd=None):
    print('> ', cmd)
    r = subprocess.run(cmd, shell=True, cwd=cwd)
    if r.returncode != 0:
        raise SystemExit(f'Command failed: {cmd}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', default='.', help='Workspace directory containing scripts and fixed_urls.json')
    parser.add_argument('--zip_name', default='Group_149_Hybrid_RAG.zip')
    parser.add_argument('--max_chunks', type=int, default=None, help='Optional: embed only first N chunks (for smoke tests)')
    args = parser.parse_args()

    wd = os.path.abspath(args.workdir)
    print('Running pipeline in', wd)
    if not os.path.exists(wd):
        raise SystemExit('Workdir does not exist')

    # Step 1: data collection
    fixed = os.path.join(wd, 'fixed_urls.json')
    if not os.path.exists(fixed):
        raise SystemExit('fixed_urls.json not found in workdir; please add it.')

    try:
        run(f'python3 scripts/data_collection.py --fixed fixed_urls.json --out corpus.json --random 300', cwd=wd)
        run(f'python3 scripts/preprocess.py --in corpus.json --out chunks.json', cwd=wd)
        build_idx_cmd = f'python3 scripts/build_index.py --chunks chunks.json --out_dir indices'
        if args.max_chunks:
            build_idx_cmd += f' --max_chunks {args.max_chunks}'
        run(build_idx_cmd, cwd=wd)
        run('python3 scripts/generate_questions.py --chunks chunks.json --out questions.json --num_questions 100', cwd=wd)
        run('python3 scripts/evaluate.py --indices indices --chunks chunks.json --questions_in questions.json --report_out report.json', cwd=wd)
    except SystemExit as e:
        print('Pipeline failed:', e)
        sys.exit(1)

    # Create ZIP
    zip_base = os.path.join(wd, args.zip_name.replace('.zip',''))
    print('Creating ZIP...')
    shutil.make_archive(zip_base, 'zip', wd)
    print('ZIP created at', zip_base + '.zip')

if __name__ == '__main__':
    main()
