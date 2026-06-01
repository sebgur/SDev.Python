import pytest
from unittest.mock import patch, MagicMock
from sdevpy.utilities.restapi import fetch_response


def _mock_response(status_code, text="", headers=None):
    r = MagicMock()
    r.status_code = status_code
    r.text = text
    r.headers = headers or {}
    return r


def test_get_200_returns_response():
    mock_r = _mock_response(200)
    with patch('requests.get', return_value=mock_r) as m:
        result = fetch_response("http://x", {}, method='GET')
    assert result is mock_r
    m.assert_called_once()


def test_post_200_returns_response():
    mock_r = _mock_response(200)
    with patch('requests.post', return_value=mock_r) as m:
        result = fetch_response("http://x", {}, payload={'a': 1}, method='POST')
    assert result is mock_r
    m.assert_called_once()


def test_400_raises_value_error():
    with patch('requests.post', return_value=_mock_response(400, "bad")):
        with pytest.raises(ValueError, match="400"):
            fetch_response("http://x", {})


def test_401_raises_permission_error():
    with patch('requests.post', return_value=_mock_response(401)):
        with pytest.raises(PermissionError, match="401"):
            fetch_response("http://x", {})


def test_403_raises_permission_error():
    with patch('requests.post', return_value=_mock_response(403)):
        with pytest.raises(PermissionError, match="403"):
            fetch_response("http://x", {})


def test_404_raises_lookup_error():
    with patch('requests.post', return_value=_mock_response(404)):
        with pytest.raises(LookupError, match="404"):
            fetch_response("http://x", {})


def test_429_raises_runtime_error():
    with patch('requests.post', return_value=_mock_response(429, headers={"Retry-After": "5"})):
        with pytest.raises(RuntimeError, match="429"):
            fetch_response("http://x", {})


def test_500_raises_runtime_error():
    with patch('requests.post', return_value=_mock_response(500, "err")):
        with pytest.raises(RuntimeError, match="500"):
            fetch_response("http://x", {})


def test_connection_error_raises_runtime():
    import requests
    with patch('requests.post', side_effect=requests.exceptions.ConnectionError("fail")):
        with pytest.raises(RuntimeError, match="server"):
            fetch_response("http://x", {})


def test_timeout_raises_runtime():
    import requests
    with patch('requests.post', side_effect=requests.exceptions.Timeout()):
        with pytest.raises(RuntimeError, match="timed out"):
            fetch_response("http://x", {})
