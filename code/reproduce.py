"""Reconstruct final paper metrics and verify the final serial timing data."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))

def audit_efficiency(root):
    timing=root/'results/efficiency'
    rows=read_csv(timing/'efficiency_by_dataset.csv')
    summaries=read_csv(timing/'efficiency_manuscript_summary.csv')
    report=[]
    for summary in summaries:
        method=summary['method']
        part=[x for x in rows if x['method']==method]
        assert len(part)==5
        for field,source in [('online_latency_ms_per_sample','end_to_end_ms_per_sample_mean'),
                              ('mean_peak_gpu_memory_mib','peak_gpu_mib_mean'),
                              ('mean_offline_cache_mib','offline_cache_mib_mean'),
                              ('mean_scoring_ranking_ms_per_sample','knowledge_stage_ms_per_sample_mean')]:
            assert np.isclose(float(summary[field]),np.mean([float(x[source]) for x in part]),rtol=1e-10)
        latency=float(summary['online_latency_ms_per_sample'])
        assert np.isclose(float(summary['throughput_samples_per_s']),1000/latency)
        for row in part:
            dataset=row['dataset']
            repeats=read_csv(timing/'runs/attempt_1'/dataset/'timing_repeats.csv')
            run=[x for x in repeats if x['method']==method]
            assert len(run)==5, (dataset,method,len(run))
            for field,source in [('end_to_end_ms_per_sample_mean','end_to_end_ms_per_sample'),
                                  ('knowledge_stage_ms_per_sample_mean','knowledge_stage_ms_per_sample'),
                                  ('peak_gpu_mib_mean','peak_gpu_mib'),('offline_cache_mib_mean','offline_cache_mib')]:
                assert np.isclose(float(row[field]),np.mean([float(x[source]) for x in run]),rtol=1e-10)
        report.append({'method':method,'latency_ms':latency,'repeats_per_dataset':5})
    expected={'CLAP':'21.42','iKnow_dagger':'21.95','SAKI':'55.80'}
    assert {x['method']:format(x['latency_ms'],'.2f') for x in report}==expected
    return report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--bootstrap',action='store_true')
    args=parser.parse_args();root=args.artifacts.resolve()
    command=[sys.executable,str(Path(__file__).with_name('verify_release.py')),'--package',str(root)]
    if args.bootstrap:command.append('--bootstrap')
    subprocess.run(command,check=True)
    timing=audit_efficiency(root)
    out=root/'rebuilt';out.mkdir(exist_ok=True)
    (out/'efficiency_verification.json').write_text(json.dumps(timing,indent=2)+'\n')
    print('PASS: final corrected-cache serial efficiency (not legacy timings).')

if __name__=='__main__':main()
