import os
import csv
import pandas as pd
from io import BytesIO
import zipfile
from unittest.mock import patch, MagicMock
from sdevpy.utilities import filemanager
from sdevpy.utilities import xmlmanager, jsonmanager
from sdevpy.projects.datafiles import merge_tsv


def test_check_directory(tmp_path):
    new_dir = tmp_path / "subdir"
    assert not new_dir.exists()
    filemanager.check_directory(str(new_dir))
    assert new_dir.exists()


def test_list_files(tmp_path):
    (tmp_path / "a.tsv").write_text("x")
    (tmp_path / "b.csv").write_text("x")
    result = filemanager.list_files(str(tmp_path))
    assert set(result) == {"a.tsv", "b.csv"}
    # With extension
    (tmp_path / "a.tsv").write_text("x")
    (tmp_path / "b.csv").write_text("x")
    result = filemanager.list_files(str(tmp_path), [".tsv"])
    assert result == ["a.tsv"]
    # No match
    (tmp_path / "a.tsv").write_text("x")
    result = filemanager.list_files(str(tmp_path), [".json"])
    assert result == []


def test_write_csv_creates_file(tmp_path):
    f = str(tmp_path / "out.csv")
    filemanager.write_csv(f)
    assert os.path.exists(f)
    with open(f, newline='', encoding='utf8') as fh:
        rows = list(csv.reader(fh))
    assert rows == [['alpha', 'beta']]


def test_download_unzip(tmp_path):
    # Build an in-memory zip with one file
    buf = BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        zf.writestr("hello.txt", "world")
    buf.seek(0)

    mock_response = MagicMock()
    mock_response.content = buf.read()

    with patch('requests.get', return_value=mock_response):
        filemanager.download_unzip("http://fake/test.zip", str(tmp_path))

    assert (tmp_path / "hello.txt").read_text() == "world"


def test_xmlmanager_roundtrip(tmp_path):
    data = {'alpha': '0.5', 'beta': '1.0', 'label': 'test'}
    xml_file = str(tmp_path / "data.xml")
    xmlmanager.serialize(data, xml_file)
    result = xmlmanager.deserialize(xml_file)
    # xmltodict wraps in a root key; check values are preserved
    root = list(result.values())[0]
    assert root['alpha'] == '0.5'
    assert root['beta'] == '1.0'
    assert root['label'] == 'test'


def test_xmlmanager_custom_root(tmp_path):
    data = {'x': '1'}
    xml_file = str(tmp_path / "data.xml")
    xmlmanager.serialize(data, xml_file, custom_root='MyRoot')
    result = xmlmanager.deserialize(xml_file)
    assert 'MyRoot' in result


def test_xml_to_json(tmp_path):
    data = {'nu': '0.66', 'rho': '0.48'}
    xml_file = str(tmp_path / "data.xml")
    json_file = str(tmp_path / "data.json")
    xmlmanager.serialize(data, xml_file)
    xmlmanager.xml_to_json(xml_file, json_file)
    loaded = jsonmanager.deserialize(json_file)
    root = list(loaded.values())[0]
    assert root['nu'] == '0.66'
    assert root['rho'] == '0.48'


def _write_tsv(path, rows):
    df = pd.DataFrame(rows, columns=['a', 'b'])
    df.to_csv(path, sep='\t', index=False)


def test_merge_tsv(tmp_path):
    _write_tsv(tmp_path / "f1.tsv", [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}])
    _write_tsv(tmp_path / "f2.tsv", [{'a': 5, 'b': 6}])
    merge_tsv(str(tmp_path))
    merged = pd.read_csv(tmp_path / "merged.tsv", sep='\t')
    assert len(merged) == 3
