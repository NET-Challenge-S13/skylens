"""WHEP 수신 프로브 — 브라우저 없이 MediaMTX WHEP 종단을 검증한다.

offer 를 만들어 <url>/whep 에 POST 하고, answer 로 연결한 뒤
몇 초간 비디오 프레임이 실제로 도착하는지 센다.

사용: python whep_probe.py http://127.0.0.1:10889/drone [seconds]
"""

import asyncio
import sys
import time
import urllib.request

from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription


async def main(base: str, seconds: float) -> int:
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    pc.addTransceiver("video", direction="recvonly")
    pc.addTransceiver("audio", direction="recvonly")

    frames = {"video": 0, "audio": 0}
    first_at: dict[str, float] = {}

    @pc.on("track")
    def on_track(track):
        async def pump():
            while True:
                try:
                    await track.recv()
                except Exception:
                    return
                frames[track.kind] += 1
                first_at.setdefault(track.kind, time.monotonic())

        asyncio.ensure_future(pump())

    @pc.on("connectionstatechange")
    async def on_state():
        print(f"[상태] {pc.connectionState}", flush=True)

    await pc.setLocalDescription(await pc.createOffer())

    req = urllib.request.Request(
        base.rstrip("/") + "/whep",
        data=pc.localDescription.sdp.encode(),
        headers={"Content-Type": "application/sdp"},
        method="POST",
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=10) as resp:
        answer = resp.read().decode()
        print(f"[WHEP] HTTP {resp.status}  Location: {resp.headers.get('Location')}", flush=True)

    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer, type="answer"))

    await asyncio.sleep(seconds)
    ok = frames["video"] > 0
    ttff = (first_at.get("video", t0) - t0) if ok else None
    print(f"[결과] video {frames['video']} frames, audio {frames['audio']} frames in {seconds:.0f}s", flush=True)
    if ok:
        print(f"[결과] 첫 프레임까지 {ttff*1000:.0f} ms", flush=True)
    await pc.close()
    return 0 if ok else 1


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:10889/drone"
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 5
    sys.exit(asyncio.run(main(url, secs)))
