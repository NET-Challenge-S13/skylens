#!/usr/bin/env python3
"""KOREN 접속 프록시(SOCKS5, 계정 인증)를 통과시키는 최소 클라이언트.

왜 만들었나
  - `hirun` 은 x86-64 전용이라 라즈베리파이(ARM)에서 안 돈다.
  - `proxychains4` 는 이 환경의 `nc` 와 맞물리지 않는다(연결이 거부된다).
  - 그래서 구간별 측정을 하려면 의존성 없는 클라이언트가 하나 필요했다.

쓰임 둘
  1) SSH 의 ProxyCommand:
         ssh -o ProxyCommand="python3 socks5.py %h %p" -p 11000 user@10.246.246.9
  2) 암호화 없는 순수 TCP 측정(표준입력을 그대로 흘려보낸다):
         dd if=/dev/zero bs=65536 count=1600 | python3 socks5.py 10.246.246.9 15201

계정은 `~/.koren_portal`(권한 600)에서 읽는다. 명령줄에 비밀번호를 쓰지 않는다.
그래야 `ps` 와 셸 기록에 남지 않는다.
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

    # 1) 사용 가능한 인증 방식 제시: 0x02 = 아이디/비밀번호
    sock.sendall(b"\x05\x01\x02")
    ver, method = recv_exact(sock, 2)
    if ver != 5 or method != 2:
        raise SystemExit(f"프록시가 계정 인증을 받지 않는다 (ver={ver}, method={method})")

    # 2) 계정 인증 (RFC 1929)
    u = user.encode()
    p = password.encode()
    sock.sendall(bytes([1, len(u)]) + u + bytes([len(p)]) + p)
    _, status = recv_exact(sock, 2)
    if status != 0:
        raise SystemExit("프록시 계정 인증 실패")

    # 3) 목적지 연결 요청. 주소는 도메인 그대로 보내 프록시가 풀게 한다.
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
    recv_exact(sock, 2)  # 포트
    sock.settimeout(None)
    return sock


def pump(sock: socket.socket) -> None:
    """표준입출력과 소켓 사이를 양방향으로 흘린다."""
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
