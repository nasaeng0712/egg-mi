"""PhysioNet EEGBCI 피험자 데이터 점검 및 baseline 파이프라인.

여러 Motor Imagery run을 내려받아 8~30 Hz로 필터링한 뒤 데이터 요약,
이벤트/신호 품질 확인, PSD 저장, Epoch 생성, baseline feature 추출 및
train/test 분리를 수행한다.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.datasets import eegbci
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# 실행 설정
# ---------------------------------------------------------

# 대화형 Epoch 창은 프로그램 실행을 기다리게 할 수 있으므로 기본값은 False다.
SHOW_EPOCH_PLOT = False

SUBJECT = 1
RUNS = [4, 8, 12]
LOW_FREQ = 8.0
HIGH_FREQ = 30.0
TEST_SIZE = 0.2
RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
SECTION_WIDTH = 60


def print_section(title: str, step: str | None = None) -> None:
    """모든 실행 단계의 제목을 같은 형식으로 출력한다."""
    heading = f"[{step}] {title}" if step else title
    print(f"\n{'=' * SECTION_WIDTH}")
    print(heading)
    print("=" * SECTION_WIDTH)


def print_item(label: str, value: object) -> None:
    """요약 항목의 들여쓰기와 간격을 통일한다."""
    print(f"{label:<20}: {value}")


def load_subject_data(
    subject: int,
    runs: list[int],
) -> mne.io.BaseRaw:
    """
    PhysioNet EEGBCI 데이터셋에서 한 피험자의 여러 run을 다운로드하고,
    각 EDF 파일을 읽어 하나의 연속 Raw 객체로 결합한다.

    Parameters
    ----------
    subject:
        피험자 번호. EEGBCI 데이터셋에서는 1~109를 사용할 수 있다.
    runs:
        다운로드할 실험 run 번호 목록. 이 프로젝트에서는 [4, 8, 12]를
        사용한다.

    Returns
    -------
    combined_raw:
        필터링한 여러 run을 시간 방향으로 연결한 MNE Raw 객체.
    """
    print_section("EEG 데이터 로딩 및 필터링", "1/7")
    print_item("피험자", subject)
    print_item("Run", runs)

    # data/raw 폴더가 없으면 생성한다.
    # parents=True는 상위 폴더도 만들고, exist_ok=True는 기존 폴더를 허용한다.
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 반환값은 EEG 데이터 자체가 아니라 다운로드된 EDF 파일의 경로 목록이다.
    file_paths = eegbci.load_data(
        subjects=subject,
        runs=runs,
        path=DATA_DIR,
        update_path=False,
    )
    print_item("다운로드 파일 수", len(file_paths))

    # 각 run의 필터링된 Raw 객체를 임시로 저장한다.
    raw_list: list[mne.io.BaseRaw] = []

    for index, file_path in enumerate(file_paths, start=1):
        # preload=True는 신호 전체를 메모리에 올려 필터링과 배열 접근을 허용한다.
        raw = mne.io.read_raw_edf(
            file_path,
            preload=True,
            verbose=False,
        )

        # "C3.." 같은 EEGBCI 채널명을 MNE 표준 형식으로 정리한다.
        eegbci.standardize(raw)

        # C3, Cz, C4 같은 채널에 표준 두피 좌표를 연결한다.
        raw.set_montage("standard_1005", on_missing="warn")

        # 원본을 직접 수정하지 않고 복사본에 Motor Imagery의 주요 대역인
        # 8~30 Hz Band-pass Filter를 한 번만 적용한다.
        raw_filtered = raw.copy()
        raw_filtered.filter(
            l_freq=LOW_FREQ,
            h_freq=HIGH_FREQ,
            verbose=False,
        )

        # 중요: 원본 raw가 아니라 실제 필터링된 객체를 추가한다.
        raw_list.append(raw_filtered)
        print(f"  ✓ Run {index}/{len(file_paths)} 로딩 및 필터 완료")

    if not raw_list:
        raise RuntimeError("로드된 EEG run이 없습니다.")

    # run 4, 8, 12를 시간 방향으로 이어 붙인다.
    # 머신러닝 평가 단계에서는 run 정보를 별도로 보존하는 것이 좋다.
    combined_raw = mne.concatenate_raws(raw_list)
    print(f"  ✓ Band-pass Filter ({LOW_FREQ:g}~{HIGH_FREQ:g} Hz)")
    return combined_raw


def print_data_summary(raw: mne.io.BaseRaw) -> None:
    """결합된 EEG 데이터의 핵심 정보를 출력한다."""
    print_section("EEG 데이터 요약", "2/7")
    duration = raw.n_times / raw.info["sfreq"]
    print_item("채널 수", raw.info["nchan"])
    print_item("샘플링 주파수", f"{raw.info['sfreq']:.1f} Hz")
    print_item("전체 샘플 수", raw.n_times)
    print_item("측정 시간", f"{duration:.1f}초")
    print_item("데이터 Shape", raw.get_data().shape)


def print_event_counts(raw: mne.io.BaseRaw) -> None:
    """Annotation을 Event로 변환하고 종류별 개수를 출력한다."""
    print_section("이벤트 개수", "3/7")
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    reverse_event_id = {code: name for name, code in event_id.items()}

    print_item("전체 이벤트", len(events))
    for code, count in zip(*np.unique(events[:, 2], return_counts=True)):
        print_item(reverse_event_id.get(int(code), str(code)), int(count))


def inspect_signal_quality(raw: mne.io.BaseRaw) -> None:
    """NaN, Inf, 채널 표준편차 및 flat channel을 점검한다."""
    print_section("EEG 신호 품질 검사", "4/7")
    data = raw.get_data(picks="eeg")
    channel_std = np.std(data, axis=1)
    flat_mask = np.isclose(channel_std, 0.0)
    flat_channels = np.asarray(raw.ch_names)[flat_mask].tolist()

    print_item("EEG Shape", data.shape)
    print_item("NaN", int(np.isnan(data).sum()))
    print_item("Inf", int(np.isinf(data).sum()))
    print_item("채널 표준편차 평균", f"{channel_std.mean():.3e}")
    print_item("채널 표준편차 최소", f"{channel_std.min():.3e}")
    print_item("채널 표준편차 최대", f"{channel_std.max():.3e}")
    print_item("Flat Channels", flat_channels or "없음")


def save_psd_figure(raw: mne.io.BaseRaw, subject: int) -> Path:
    """필터링된 EEG의 평균 PSD 그래프를 파일로 저장한다."""
    print_section("PSD 분석 및 저장", "5/7")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    spectrum = raw.compute_psd(
        method="welch",
        fmin=1.0,
        fmax=40.0,
        picks="eeg",
        verbose=False,
    )
    spectrum.plot(average=True, show=False)

    output_path = FIGURE_DIR / f"subject_{subject:03d}_psd.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print_item("저장 경로", output_path)
    return output_path


def create_motor_imagery_epochs(raw: mne.io.BaseRaw) -> mne.Epochs:
    """Raw EEG를 왼손/오른손 운동상상 Epoch로 변환한다."""
    print_section("Motor Imagery Epoch 생성", "6/7")
    events, annotation_event_id = mne.events_from_annotations(
        raw,
        verbose=False,
    )

    missing_events = {"T1", "T2"} - annotation_event_id.keys()
    if missing_events:
        raise ValueError(f"필수 Annotation이 없습니다: {sorted(missing_events)}")

    # EEGBCI Motor Imagery: T1=왼손, T2=오른손.
    motor_imagery_event_id = {
        "left_hand": annotation_event_id["T1"],
        "right_hand": annotation_event_id["T2"],
    }

    # Cue 발생 0.5초 후부터 4.0초까지 EEG 채널만 잘라 Epoch를 만든다.
    epochs = mne.Epochs(
        raw=raw,
        events=events,
        event_id=motor_imagery_event_id,
        tmin=0.5,
        tmax=4.0,
        baseline=None,
        picks="eeg",
        preload=True,
        verbose=False,
    )

    print_item("Epoch Shape", epochs.get_data(copy=False).shape)
    print_item("왼손 Epoch", len(epochs["left_hand"]))
    print_item("오른손 Epoch", len(epochs["right_hand"]))
    return epochs


def extract_baseline_features(
    epochs: mne.Epochs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Epoch에서 간단한 baseline Feature와 Label을 만든다.

    시간축 평균 Feature는 전체 파이프라인 검증용이며, 이후 CSP 같은
    EEG 전용 Feature와 비교하기 위한 baseline으로 사용한다.
    """
    epoch_data = epochs.get_data(copy=False)

    # (Epoch 수, 채널 수, 시간 샘플 수)에서 시간축 평균을 계산한다.
    # 예: (45, 64, 561) -> (45, 64)
    features = epoch_data.mean(axis=2)

    # Annotation의 실제 event code에 의존하지 않고 0=왼손, 1=오른손으로
    # 안전하게 변환한다.
    left_code = epochs.event_id["left_hand"]
    labels = (epochs.events[:, 2] != left_code).astype(np.int64)

    print_item("X Shape", features.shape)
    print_item("y Shape", labels.shape)
    return features, labels


def split_dataset(
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """데이터를 계층화하여 학습용 80%, 시험용 20%로 나눈다."""
    split_data = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    x_train, x_test, y_train, y_test = split_data

    print_item("X_train", x_train.shape)
    print_item("X_test", x_test.shape)
    print_item("y_train", y_train.shape)
    print_item("y_test", y_test.shape)
    return x_train, x_test, y_train, y_test
def prepare_baseline_features(
    subject: int = SUBJECT,
    runs: list[int] = RUNS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cross Validation에 사용할
    분할 전 전체 Feature와 Label을 반환한다.

    Returns
    -------
    features:
        전체 Epoch에서 추출한 baseline Feature.

    labels:
        각 Epoch의 정답 Label.
        0 = 왼손
        1 = 오른손
    """

    # EEG 데이터를 로드하고 8~30Hz 필터를 적용한다.
    raw = load_subject_data(
        subject=subject,
        runs=runs,
    )

    # 연속 EEG를 운동상상 Epoch로 변환한다.
    epochs = create_motor_imagery_epochs(
        raw,
    )

    # 전체 Epoch에서 Feature와 Label을 추출한다.
    features, labels = extract_baseline_features(
        epochs,
    )

    return features, labels
def prepare_baseline_dataset(
    subject: int = SUBJECT,
    runs: list[int] = RUNS,
):
    """
    Baseline SVM에서 사용할 Train/Test 데이터를 준비한다.
    """

    raw = load_subject_data(
        subject=subject,
        runs=runs,
    )

    epochs = create_motor_imagery_epochs(raw)

    features, labels = extract_baseline_features(
        epochs,
    )

    return split_dataset(
        features,
        labels,
    )


def prepare_csp_dataset(
    subject: int = SUBJECT,
    runs: list[int] = RUNS,
):
    """
    CSP 학습을 위한 Epoch 데이터를 준비한다.

    Returns
    -------
    X_train
    X_test
    y_train
    y_test
    """

    raw = load_subject_data(
        subject=subject,
        runs=runs,
    )

    epochs = create_motor_imagery_epochs(raw)

    # (Epoch 수, 채널 수, 시간)
    X = epochs.get_data()

    # 0 = 왼손, 1 = 오른손
    y = epochs.events[:, 2] - 2

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print_section("CSP Dataset")

    print_item("X_train", X_train.shape)
    print_item("X_test", X_test.shape)
    print_item("y_train", y_train.shape)
    print_item("y_test", y_test.shape)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )

    """
    머신러닝에 사용할 Train/Test 데이터를 준비한다.
    """

    raw = load_subject_data(
        subject=subject,
        runs=runs,
    )

    epochs = create_motor_imagery_epochs(raw)

    features, labels = extract_baseline_features(
        epochs,
    )

    return split_dataset(
        features,
        labels,
    )

def print_pipeline_summary(
    raw: mne.io.BaseRaw,
    epochs: mne.Epochs,
    psd_path: Path,
    train_size: int,
    test_size: int,
) -> None:
    """파이프라인 완료 결과를 한눈에 볼 수 있도록 출력한다."""
    print_section("Pipeline Finished Successfully")
    print_item("처리 채널", raw.info["nchan"])
    print_item("전체 Epoch", len(epochs))
    print_item("Train / Test", f"{train_size} / {test_size}")
    print_item("PSD 파일", psd_path)
    print("\n모든 파이프라인 단계가 정상적으로 완료되었습니다.")


def main() -> None:
    """EEG 데이터 점검 및 baseline 파이프라인을 순서대로 실행한다."""
    print_section("EEG Motor Imagery Inspection Pipeline")

    raw = load_subject_data(subject=SUBJECT, runs=RUNS)
    print_data_summary(raw)
    print_event_counts(raw)
    inspect_signal_quality(raw)
    psd_path = save_psd_figure(raw, subject=SUBJECT)

    # Epoch는 한 번만 생성하고 이후 모든 단계에서 같은 객체를 재사용한다.
    epochs = create_motor_imagery_epochs(raw)

    # 기본값 False이므로 일반 실행에서 대화형 창 때문에 멈추지 않는다.
    if SHOW_EPOCH_PLOT:
        epochs.plot(
            n_epochs=min(5, len(epochs)),
            n_channels=min(20, len(epochs.ch_names)),
            scalings="auto",
            block=True,
        )

    print_section("Baseline Feature 및 Train / Test 분리", "7/7")
    features, labels = extract_baseline_features(epochs)
    x_train, x_test, _, _ = split_dataset(features, labels)

    print_pipeline_summary(
        raw=raw,
        epochs=epochs,
        psd_path=psd_path,
        train_size=len(x_train),
        test_size=len(x_test),
    )


if __name__ == "__main__":
    main()
