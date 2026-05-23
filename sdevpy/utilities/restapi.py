import requests


def fetch_response(url: str, headers: dict, payload: dict=None, method: str='POST',
                   timeout: tuple[int, int]=(5,30), **kwargs) -> dict: # pragma: no cov
    """ Fetch response given url and headers (mandatory) """
    # Send the request
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, timeout=timeout, **kwargs)
        elif method.upper() == "POST":
            response = requests.post(url, headers=headers, json=payload, timeout=timeout, **kwargs)
        else:
            raise ValueError(f"Unknown method type: {method}")
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"Could not reach the server: {e}") from e
    except requests.exceptions.Timeout as e:
        raise RuntimeError("The request timed out. The server is too slow or unreachable") from e
    except requests.exceptions.TooManyRedirects as e:
        raise RuntimeError("Too many redirects: check URL") from e
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Unexpected network error: {e}") from e

    # Server response: inspect the status code
    status = response.status_code
    if status == 200: # Success
        return response
    elif status == 400:
        raise ValueError(f"Bad request (400): {response.text}")
    elif status == 401:
        raise PermissionError("Unauthorized (401): check your API key")
    elif status == 403:
        raise PermissionError(f"Forbidden (403): you do not have access to {url}")
    elif status == 404:
        raise LookupError(f"Not found (404): URL {url} does not exist")
    elif status == 429: # The server may tell us how long to wait
        retry_after = int(response.headers.get("Retry-After", 10))
        raise RuntimeError(f"Rate limited (429): retry after {retry_after} seconds")
    elif status >= 500:
        raise RuntimeError(f"Server error ({status}): {response.text}")
    else:
        raise RuntimeError(f"Unexpected status code {status}: {response.text}")
