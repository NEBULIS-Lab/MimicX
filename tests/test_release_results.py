import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from scripts.export_release_results import FILES


def test_export_keeps_historical_comparison_in_archive(tmp_path):
    source = tmp_path / 'input'
    source.mkdir()
    for name in FILES:
        (source / name).write_text('metric,value\nerror,0.123\n')
    output = tmp_path / 'published'
    script = Path(__file__).resolve().parents[1] / 'scripts/export_release_results.py'
    subprocess.run([sys.executable, str(script), '--source-dir', str(source),
                    '--output-dir', str(output)], check=True, capture_output=True)
    assert not (output / 'sonic_comparison.csv').exists()
    assert (output / 'archive/sonic_comparison.csv').is_file()
    for record in json.loads((output / 'provenance.json').read_text()):
        path = output / record['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['published_sha256']
        with path.open() as handle:
            assert list(csv.DictReader(handle)) == [{'metric': 'error', 'value': '0.123'}]
