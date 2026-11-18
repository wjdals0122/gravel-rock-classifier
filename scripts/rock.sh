#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
venv=${VENV_DIR:-"$root/.venv"}
python="$venv/bin/python"
data_dir=/home/kangjungmin1/open

usage() {
    cat <<'HELP'
Ubuntu 암석 분류 프로젝트
  rock.sh setup
  rock.sh train
  rock.sh predict IMAGE --checkpoint MODEL [--device cpu|cuda|auto]
  rock.sh evaluate CSV --checkpoint MODEL [--batch-size N] [--device cpu|cuda|auto]
  rock.sh evaluate-folds MANIFEST_DIR CHECKPOINT_DIR [--batch-size N] [--device cpu|cuda|auto]
HELP
}

fail() { echo "$*" >&2; exit 1; }
new_run() {
    mkdir -p "$root/runs"
    mktemp -d "$root/runs/$1-$(date +%Y%m%d-%H%M%S)-XXXXXX"
}

command=${1:-help}
if (( $# )); then shift; fi
case "$command" in
    help|-h|--help) usage; exit 0 ;;
    setup)
        (( $# == 0 )) || fail "사용법: rock.sh setup"
        if [[ ! -x "$python" ]]; then
            python3 -m venv "$venv"
        fi
        exec "$python" -m pip install -r "$root/requirements.txt"
        ;;
    train|predict|evaluate|evaluate-folds) ;;
    *) usage >&2; exit 2 ;;
esac
[[ -x "$python" ]] || fail "가상환경이 필요합니다. 먼저 rock.sh setup을 실행하세요."

case "$command" in
    train)
        (( $# == 0 )) || fail "사용법: rock.sh train"
        [[ -d "$data_dir/train" ]] || fail "학습 폴더가 없습니다: $data_dir/train"
        for name in test_modified.csv sample_submission.csv; do
            [[ -r "$data_dir/$name" ]] || fail "입력 파일을 읽을 수 없습니다: $data_dir/$name"
        done
        run=$(new_run train)
        echo "학습 로그: $run/train.log" >&2
        "$python" -u "$root/model.py" 2>&1 | tee "$run/train.log"
        ;;
    predict|evaluate)
        exec "$python" "$root/$command.py" "$@"
        ;;
    evaluate-folds)
        (( $# >= 2 )) || fail "검증 CSV 디렉터리와 체크포인트 디렉터리를 지정하세요."
        manifests=$1
        checkpoints=$2
        shift 2
        # 모든 입력을 먼저 확인해 누락된 폴드를 실행 전에 찾습니다.
        for fold in {1..5}; do
            for path in "$manifests/fold$fold.csv" "$checkpoints/best_model_fold$fold.pth"; do
                [[ -f "$path" && -r "$path" ]] || fail "입력 파일을 읽을 수 없습니다: $path"
            done
        done
        run=$(new_run evaluate)
        echo "폴드별 평가 결과: $run" >&2
        for fold in {1..5}; do
            echo "Fold $fold/5 평가" >&2
            "$python" "$root/evaluate.py" "$manifests/fold$fold.csv" "$@" \
                --checkpoint "$checkpoints/best_model_fold$fold.pth" > "$run/fold$fold.tmp"
            mv "$run/fold$fold.tmp" "$run/fold$fold.json"
        done
        ;;
esac
