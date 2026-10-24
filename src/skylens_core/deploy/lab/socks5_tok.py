#!/usr/bin/env python3
"""A minimal client for getting through the KOREN access proxy (SOCKS5 with an account).

Why it exists
  - `hirun` is x86-64 only, so it does not run on the Raspberry Pi (ARM).
  - `proxychains4` does not mesh with the `nc` in this environment (connection refused).
  - So per-leg measurement needed one client with no dependencies.

Two uses
  1) as SSH's ProxyCommand:
         ssh -o ProxyCommand="python3 socks5.py %h %p" -p 11000 user@10.246.246.9
  2) for a plain unencrypted TCP measurement (stdin is passed straight through):
         dd if=/dev/zero bs=65536 count=1600 | python3 socks5.py 10.246.246.9 15201

The account comes from `~/.koren_portal` (mode 600). The password never appears on
the command line, so it stays out of `ps` and the shell history.
"""

from __future__ import annotations

import os
import select
import socket
import sys

PROXY_HOST = os.environ.get("KOREN_PROXY_HOST", "116.89.174.24")
PROXY_PORT = int(os.environ.get("KOREN_PROXY_PORT", "1080"))
CRED_PATH = os.environ.get("KOREN_CRED", os.path.expanduser("~/.koren_portal"))


def read_credentials(path: str) -> tuple[str, str]:
    env_u = os.environ.get("KOREN_SOCKS_USER")
    env_p = os.environ.get("KOREN_SOCKS_PASS")
    if env_u and env_p:
        return env_u, env_p
    user = password = ""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("ID="):
                user = line[3:]
            elif line.startswith("PW="):
                password = line[3:]
    if not user or not password:
        raise SystemExit(f"{path} 에 ID= 와 PW= 가 모두 있어야 한다")
    return user, password


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise SystemExit("프록시가 연결을 끊었다")
        buf += chunk
    return buf


def socks5_connect(host: str, port: int) -> socket.socket:
    user, password = read_credentials(CRED_PATH)
    sock = socket.create_connection((PROXY_HOST, PROXY_PORT), timeout=20)
    sock.settimeout(30)

    # 1) offer the auth methods we support: 0x02 = username/password
    sock.sendall(b"\x05\x01\x02")
    ver, method = recv_exact(sock, 2)
    if ver != 5 or method != 2:
        raise SystemExit(f"프록시가 계정 인증을 받지 않는다 (ver={ver}, method={method})")

    # 2) account authentication (RFC 1929)
    u = user.encode()
    p = password.encode()
    sock.sendall(bytes([1, len(u)]) + u + bytes([len(p)]) + p)
    _, status = recv_exact(sock, 2)
    if status != 0:
        raise SystemExit("프록시 계정 인증 실패")

    # 3) ask for the destination. Send the address as a domain and let the proxy resolve it.
    target = host.encode()
    sock.sendall(b"\x05\x01\x00\x03" + bytes([len(target)]) + target
                 + port.to_bytes(2, "big"))
    reply = recv_exact(sock, 4)
    if reply[1] != 0:
        raise SystemExit(f"프록시가 목적지 연결을 거부했다 (코드 {reply[1]})")
    atyp = reply[3]
    if atyp == 1:
        recv_exact(sock, 4)
    elif atyp == 3:
        recv_exact(sock, recv_exact(sock, 1)[0])
    elif atyp == 4:
        recv_exact(sock, 16)
    recv_exact(sock, 2)  # port
    sock.settimeout(None)
    if os.environ.get("KOREN_PROBE_ONLY"):
        print("연결 성공", file=sys.stderr)
        sys.exit(0)
    return sock


def pump(sock: socket.socket) -> None:
    """Pump both directions between stdio and the socket."""
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    fds = [stdin.fileno(), sock.fileno()]
    while True:
        ready, _, _ = select.select(fds, [], [])
        if stdin.fileno() in ready:
            data = os.read(stdin.fileno(), 1 << 16)
            if not data:
                try:
                    sock.shutdown(socket.SHUT_WR)
                except OSError:
                    pass
                fds = [sock.fileno()]
            else:
                sock.sendall(data)
        if sock.fileno() in ready:
            data = sock.recv(1 << 16)
            if not data:
                return
            stdout.write(data)
            stdout.flush()


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("사용법: socks5.py <host> <port>")
    sock = socks5_connect(sys.argv[1], int(sys.argv[2]))
    try:
        pump(sock)
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        sock.close()


if __name__ == "__main__":
    main()
