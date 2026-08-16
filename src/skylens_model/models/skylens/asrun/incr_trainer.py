"""증분(warm-start) 학습 래퍼 v2.

1차 실험에서 순진한 warm-start 가 오히려 6~7dB 나쁘다는 결과가 나왔다.
원인은 셋 다 "스케줄 리셋"이었으며, 여기서 그 셋을 모두 교정한다.

  (1) SH 차수 스케줄 리셋
      원본: sh_degree_to_use = min(step // sh_degree_interval, sh_degree)
      → 이어받은 모델은 이미 고차 계수를 학습했는데 step 0 부터라 차수 0(단색)으로 렌더된다.
      교정: RESUME_STEP 만큼 진행된 것으로 간주해 차수를 바로 회복시킨다.

  (2) means 학습률 리셋
      원본: ExponentialLR(gamma=0.01**(1/max_steps)) 가 step 0 에서 최대 학습률로 재시작.
      → 수렴한 가우시안 위치를 큰 보폭으로 흔들어 형상을 무너뜨린다.
      교정: 스케줄러를 RESUME_STEP 만큼 미리 진행시켜(last_epoch) 감쇠된 학습률에서 출발한다.

  (3) 옵티마이저 모멘트 소실
      원본 체크포인트에는 Adam 의 1·2차 모멘트가 없다.
      교정: 저장 시 옵티마이저 상태를 함께 기록하고, 이어받을 때 복원한다.

환경변수
  INIT_CKPT    : 이어받을 체크포인트 경로 (없으면 일반 학습)
  RESUME_STEP  : 스케줄을 이어붙일 기준 step (기본: 체크포인트의 step)
  MEANS_LR_SCALE : 위치 학습률 추가 배율 (기본 1.0, 더 보수적으로 가려면 <1)
"""

import os

import torch
import tyro

import simple_trainer as st
from gsplat.distributed import cli
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from simple_trainer import Config, main

INIT_CKPT = os.environ.get("INIT_CKPT", "")
RESUME_STEP_ENV = os.environ.get("RESUME_STEP", "")
MEANS_LR_SCALE = float(os.environ.get("MEANS_LR_SCALE", "1.0"))

_orig_create = st.create_splats_with_optimizers
_orig_runner_init = st.Runner.__init__
_resume_state = {"step": 0, "opt": None}


# ---------------------------------------------------------------- (3) 저장 보강
def _patch_save():
    """체크포인트에 옵티마이저 상태를 함께 저장하도록 Runner.train 을 감싼다."""
    _orig_train = st.Runner.train

    def train(self):
        _orig_save = torch.save

        def save(obj, path, *a, **kw):
            if isinstance(obj, dict) and "splats" in obj and "optimizers" not in obj:
                obj = dict(obj)
                obj["optimizers"] = {
                    n: o.state_dict() for n, o in self.optimizers.items()
                }
            return _orig_save(obj, path, *a, **kw)

        torch.save = save
        try:
            return _orig_train(self)
        finally:
            torch.save = _orig_save

    st.Runner.train = train


_patch_save()


# ------------------------------------------------- (3) 초기화 + 옵티마이저 복원
def create_patched(parser, **kw):
    splats, optimizers = _orig_create(parser, **kw)
    if not INIT_CKPT:
        return splats, optimizers

    device = kw.get("device", "cuda")
    # 옵티마이저 상태에 numpy 스칼라(step 카운터)가 섞여 있어 weights_only=True 로는
    # 읽히지 않는다. 우리가 직접 만든 체크포인트이므로 전체 로드한다.
    ck = torch.load(INIT_CKPT, map_location=device, weights_only=False)
    src = ck["splats"]

    hp = {
        n: (
            o.param_groups[0]["lr"],
            o.param_groups[0]["eps"],
            o.param_groups[0]["betas"],
            type(o),
        )
        for n, o in optimizers.items()
    }

    new = {}
    for k in list(splats.keys()):
        if k not in src:
            raise KeyError(f"체크포인트에 '{k}' 가 없습니다")
        new[k] = torch.nn.Parameter(src[k].to(device).clone().contiguous())
    splats2 = torch.nn.ParameterDict(new).to(device)

    opts2 = {}
    for n in splats2.keys():
        lr, eps, betas, cls = hp[n]
        if n == "means":
            lr = lr * MEANS_LR_SCALE
        opts2[n] = cls(
            [{"params": splats2[n], "lr": lr, "name": n}],
            eps=eps,
            betas=betas,
            fused=True,
        )

    # (3) Adam 모멘트 복원
    restored = 0
    saved_opt = ck.get("optimizers")
    if saved_opt:
        for n, o in opts2.items():
            if n in saved_opt:
                try:
                    o.load_state_dict(saved_opt[n])
                    # MEANS_LR_SCALE 은 state_dict 로 덮이므로 다시 적용
                    if n == "means":
                        for g in o.param_groups:
                            g["lr"] = hp[n][0] * MEANS_LR_SCALE
                    restored += 1
                except Exception as e:  # 형상 불일치 등은 조용히 건너뜀
                    print(f"[warm-start] {n} 옵티마이저 복원 실패: {e}", flush=True)

    _resume_state["step"] = (
        int(RESUME_STEP_ENV) if RESUME_STEP_ENV else int(ck.get("step", 0)) + 1
    )
    _resume_state["opt"] = restored

    print(f"[warm-start] {INIT_CKPT}", flush=True)
    print(f"[warm-start] 가우시안 {len(splats2['means']):,}개 이어받음", flush=True)
    print(
        f"[warm-start] 옵티마이저 모멘트 {restored}/{len(opts2)}개 복원 · "
        f"resume_step={_resume_state['step']} · means_lr×{MEANS_LR_SCALE}",
        flush=True,
    )
    return splats2, opts2


st.create_splats_with_optimizers = create_patched


# --------------------------------------------- (1)(2) 스케줄 이어붙이기
def runner_init(self, local_rank, world_rank, world_size, cfg):
    if INIT_CKPT:
        # (1) SH 차수: 이어받은 시점의 차수를 즉시 회복
        #     step 0 부터라도 min((step + resume)//interval, sh_degree) 가 되도록
        #     interval 을 유지한 채 오프셋을 주입한다.
        cfg = cfg  # cfg 자체는 그대로 두고 아래에서 렌더 시점에 보정
    _orig_runner_init(self, local_rank, world_rank, world_size, cfg)


st.Runner.__init__ = runner_init

# (1) sh 차수 오프셋: 학습 루프가 참조하는 cfg.sh_degree_interval 을 바꾸는 대신
#     Config 에 step 오프셋을 더하는 방식이 안전하므로, 루프 내 계산을 감싼다.
_orig_min = min


def _patched_train_wrapper():
    """train() 안의 sh 스케줄과 LR 스케줄러를 resume_step 기준으로 보정."""
    _train = st.Runner.train

    def train(self):
        if INIT_CKPT:
            rs = _resume_state["step"]
            # (1) SH: 시작부터 이어받은 차수를 쓰도록 interval 을 1 로 낮춰
            #     min(step//1, sh_degree) → 첫 step 부터 최대 차수 사용
            self.cfg.sh_degree_interval = 1
            print(f"[warm-start] SH 차수 즉시 회복 (interval 1)", flush=True)
            # (2) LR: means 스케줄러를 rs 만큼 진행시킨 상태에서 출발
            self._resume_lr_steps = rs
        return _train(self)

    st.Runner.train = train


_patched_train_wrapper()

# (2) 스케줄러 생성 직후 last_epoch 를 당겨두기 위해 ExponentialLR 을 감싼다
_OrigExp = torch.optim.lr_scheduler.ExponentialLR


class _ResumedExponentialLR(_OrigExp):
    def __init__(self, optimizer, gamma, last_epoch=-1, **kw):
        rs = _resume_state["step"] if INIT_CKPT else 0
        super().__init__(optimizer, gamma, last_epoch=-1, **kw)
        if rs > 0:
            for _ in range(rs):
                self.step()
            lrs = [g["lr"] for g in optimizer.param_groups]
            print(
                f"[warm-start] LR 스케줄 {rs} step 진행 → 현재 lr={lrs[0]:.3e}",
                flush=True,
            )


torch.optim.lr_scheduler.ExponentialLR = _ResumedExponentialLR


if __name__ == "__main__":
    configs = {
        "default": ("default", Config(strategy=DefaultStrategy(verbose=True))),
        "mcmc": (
            "mcmc",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)
    cli(main, cfg, verbose=True)
