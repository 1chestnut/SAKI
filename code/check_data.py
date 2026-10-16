"""Check frozen sample identities and, optionally, local audio availability."""
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifests',type=Path,default=Path(__file__).resolve().parents[1]/'manifests')
    p.add_argument('--audio-root',type=Path);a=p.parse_args()
    expected={'01_ESC50':2000,'02_UrbanSound8K':8732,'03_FSD50K':10231,'04_DCASE17_T4':419,'05_AudioSet':17233,'06_TUT2017':6300}
    missing=[]
    for ds,n in expected.items():
        rows=json.loads((a.manifests/ds/'samples.json').read_text(encoding='utf-8'))
        classes=json.loads((a.manifests/ds/'classes.json').read_text(encoding='utf-8'))
        assert len(rows)==n and len({x['audio_path'] for x in rows})==n
        for row in rows:
            assert row['true_indices'] and all(0<=i<len(classes) for i in row['true_indices'])
            assert isinstance(row['seed'],int)
            if a.audio_root:
                path=str(row['audio_path']);prefix='${DATA_ROOT}/';assert path.startswith(prefix)
                if not (a.audio_root/path[len(prefix):]).is_file():missing.append((ds,row['sample_index']))
        print(ds,len(rows),'samples;',len(classes),'candidate classes')
    if missing:raise SystemExit(f'{len(missing)} audio files are unavailable; do not silently change the evaluation subset. First IDs: {missing[:10]}')
    print('PASS: frozen manifest identities, labels and seeds'+(' and audio availability' if a.audio_root else ''))

if __name__=='__main__':main()
