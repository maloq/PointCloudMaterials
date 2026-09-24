"""Evidence fidelity and failure behavior for the cross-study research catalogue."""
import json
from pathlib import Path
import sqlite3

import pytest

from src.experiment_runner import encoder_catalogue as catalogue


def test_csv_preserves_missing_values_duplicate_headers_and_multiline_records():
    header, rows = catalogue.csv_records(b'metric,value,value\n"a\nb",,NaN\nx,0,1\n', 'fixture')
    assert header == ['metric', 'value', 'value']
    assert rows == [(1, ['a\nb', '', 'NaN']), (2, ['x', '0', '1'])]


def test_malformed_csv_fails_with_producer_context():
    with pytest.raises(ValueError, match='run/tables.csv: CSV record 1'):
        catalogue.csv_records(b'a,b\n1,2,3\n', 'run/tables.csv')


def test_markdown_retains_exact_lines_and_source_locations():
    text = '# Result\n\n| Model | AP |\n| --- | ---: |\n| A | **0.12** |\n'
    assert list(catalogue.markdown_records(text)) == [(5, {
        'header_line': '| Model | AP |', 'row_line': '| A | **0.12** |'})]


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    source = tmp_path / 'source'; root = source / 'run'; root.mkdir(parents=True)
    (root / 'README.md').write_text('# Fixture\n\n| Model | AP |\n| --- | ---: |\n| A | 0.12 |\n')
    content = 'metric,value\nAP,0.12\nundefined,\n'
    (root / 'scores.csv').write_text(content)
    (root / 'copy.csv').write_text(content)
    (root / 'metrics.json').write_text('{"bad": NaN, "values": [1, 2]}')
    (root / 'index.html').write_text('<html>gallery</html>')
    code = root / 'code'; code.mkdir(); (code / 'ignore.csv').write_text('x\n1\n')
    (root / 'cycle').symlink_to(root, target_is_directory=True)
    manifest = {'schema_version': 1, 'sources': {'fixture': str(source)},
                'families': [{'id':'f','title':'Family','description':'Fixture'}],
                'studies': [{'id':'s','family':'f','title':'Study','source':'fixture',
                             'path':'run/README.md','evidence':'protocol','status':'consult_record'}],
                'collections':[{'source':'fixture','path':'run','family':'f'}],
                'skip_directories':['code'], 'index_only_csv':[], 'max_import_bytes':10000}
    highlights = [{'id':'h','comparison_group':'g','family':'f','model':'A','metric':'AP',
                   'value':.12,'unit':'fraction','direction':'higher','population':'fixture',
                   'split':'development','horizon_ps':'12','seeds':'1','evidence':'local_report',
                   'source':'fixture','path':'run/README.md','evidence_text':'| A | 0.12 |',
                   'caveat':'Not a scientific result'}]
    manifest_path = tmp_path / 'manifest.json'; manifest_path.write_text(json.dumps(manifest))
    highlights_path = tmp_path / 'highlights.json'; highlights_path.write_text(json.dumps(highlights))
    monkeypatch.setattr(catalogue, 'resolve_path', Path)
    kwargs = dict(manifest_path=manifest_path, highlights_path=highlights_path, output=tmp_path/'out')
    return source, kwargs


def test_build_preserves_provenance_deduplicates_bytes_and_queries_aliases(evidence):
    source, kwargs = evidence
    stats = catalogue.build(**kwargs)
    assert stats['artifacts'] == 5  # Code tree and symlink cycle excluded.
    assert stats['records'] == 6 and stats['stored_records'] == 4
    with sqlite3.connect(kwargs['output']/'technical/results.sqlite') as con:
        assert con.execute('PRAGMA integrity_check').fetchone() == ('ok',)
        assert con.execute('SELECT sha256 FROM headlines').fetchone()[0] == catalogue.digest((source/'run/README.md').read_bytes())
        assert con.execute("SELECT count(*) FROM records WHERE kind='csv'").fetchone()[0] == 4
        assert con.execute("SELECT value FROM csv_cells WHERE artifact_id='fixture:run/copy.csv' AND ordinal=2 AND column_name='value'").fetchone() == ('',)
        assert json.loads(con.execute("SELECT payload_json FROM records WHERE kind='json_result'").fetchone()[0])['bad'] == 'NaN'
    html = (kwargs['output']/'index.html').read_text()
    assert 'storage/fixture/run/scores.csv' in html
    assert (kwargs['output']/'tables/METRICS.md').is_file()
    assert (kwargs['output']/'technical/metric-contract.json').is_file()


def test_changed_quotation_fails_instead_of_publishing_unchecked_score(evidence):
    source, kwargs = evidence
    (source/'run/README.md').write_text('# Corrected result: 0.11\n')
    with pytest.raises(ValueError, match='evidence text changed'):
        catalogue.build(**kwargs)


def test_failed_refresh_preserves_previous_database(evidence):
    source, kwargs = evidence
    catalogue.build(**kwargs)
    database = kwargs['output']/'technical/results.sqlite'
    before = database.read_bytes()
    (source/'run/invalid.csv').write_text('a,b\n1,2,3\n')
    with pytest.raises(ValueError, match='invalid.csv'):
        catalogue.build(**kwargs)
    assert database.read_bytes() == before


def test_missing_registered_source_fails(evidence):
    _, kwargs = evidence
    config = json.loads(kwargs['manifest_path'].read_text())
    config['sources']['missing'] = str(kwargs['output']/'not-present')
    kwargs['manifest_path'].write_text(json.dumps(config))
    with pytest.raises(FileNotFoundError, match='source missing unavailable'):
        catalogue.build(**kwargs)


def test_index_only_data_and_frozen_definitions_are_explicit(evidence):
    source, kwargs = evidence
    tables = source/'run/tables'; tables.mkdir()
    (tables/'METRICS.md').write_text('Historical definition; do not substitute current units.\n')
    (tables/'raw.csv').write_text('value\n1\n2\n')
    config = json.loads(kwargs['manifest_path'].read_text())
    config['index_only_csv'] = ['raw.csv']
    kwargs['manifest_path'].write_text(json.dumps(config))
    catalogue.build(**kwargs)
    with sqlite3.connect(kwargs['output']/'technical/results.sqlite') as con:
        row = con.execute("SELECT row_count,import_status,sha256,definitions FROM artifacts WHERE path='run/tables/raw.csv'").fetchone()
    assert row == (0, 'index_only:per_observation_coordinates', None, 'run/tables/METRICS.md')
