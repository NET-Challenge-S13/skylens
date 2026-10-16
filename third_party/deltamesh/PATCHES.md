# deltamesh 패치

crates.io `deltamesh` 0.1.0 을 그대로 가져와 SkyLens에 필요한 것만 고쳤다. 루트 `Cargo.toml`의 `[patch.crates-io]`가 이 사본으로 바꿔 끼운다. 원본: https://github.com/me0w2en/deltamesh (MIT 또는 Apache-2.0).

| 변경 | 이유 |
|---|---|
| `src/gpu_bins.rs` `uniform_stride`: 유니폼 슬롯 간격을 `max(정렬값, 구조체 크기)` 에서 `구조체 크기를 정렬값의 배수로 올림` 으로 바꿨다 | 유니폼 구조체 `U` 가 96 바이트인데 RTX 4050(Vulkan)은 동적 오프셋을 64 바이트 배수로 요구한다. 원본은 간격을 96으로 잡아 두 번째 슬롯 오프셋 96이 wgpu 검증에 걸리고(`min_uniform_buffer_offset_alignment`), `gpu-mesh` 로 빌드한 모델 서버의 메시 생성이 첫 구간에서 멈췄다. 올림하면 128 이 된다 |

deltamesh를 새 버전으로 올릴 때는 원본에서 이 문제가 고쳐졌는지 확인하고, 고쳐졌으면 패치를 지운다.
