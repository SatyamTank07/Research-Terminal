"""Pytest configuration and zero-network guard fixture.

Ensures that the entire test suite runs offline without initiating any outbound
network HTTP/HTTPS connections (ports 80 and 443) to OpenAI, Yahoo Finance, or the internet.
"""

import socket
import pytest

_original_connect = socket.socket.connect


def _guarded_connect(self, address):
    # address is usually a tuple: (host, port) or (host, port, flowinfo, scopeid)
    if isinstance(address, tuple) and len(address) >= 2:
        host, port = address[0], address[1]
        if port in (80, 443):
            raise RuntimeError(
                f"ZERO-NETWORK VIOLATION: Test attempted external network connection to {host}:{port}!"
            )
    return _original_connect(self, address)


@pytest.fixture(autouse=True, scope="session")
def enforce_zero_external_network():
    """Autouse session fixture installing a strict socket firewall against outbound web traffic."""
    socket.socket.connect = _guarded_connect
    yield
    socket.socket.connect = _original_connect
