import hashlib
import json
from pathlib import Path

SITE=Path(__file__).resolve().parents[1]/'docs'


def test_common_reference_cases_and_stage_sources():
    folder=SITE/'assets/media/cases'
    records=json.loads((folder/'manifest.json').read_text())
    assert len(records)==20
    for record in records:
        path=folder/record['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
        assert not Path(record['source']).is_absolute()
    data=json.loads((folder/'comparison.json').read_text())
    for task in ['tennis','football']:
        assert set(data[task])=={'fixed','beyond','sonic','ours'}
        assert len({row['reference_sha256'] for row in data[task].values()})==1
        assert len({row['frames'] for row in data[task].values()})==1
        assert {row['display_seed'] for row in data[task].values()}=={202}
