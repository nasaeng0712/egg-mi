# 파일과 폴더 경로를 운영체제에 독립적으로 다루기 위한 표준 라이브러리
from pathlib import Path

# 그래프 저장과 표시를 위한 라이브러리
import matplotlib.pyplot as plt

# 수치 배열 계산을 위한 라이브러리
import numpy as np

# EEG·MEG 등 뇌신호 분석을 위한 라이브러리
import mne

# PhysioNet EEGBCI 데이터셋을 다운로드하고 정리하는 MNE 모듈
from mne.datasets import eegbci


# ---------------------------------------------------------
# 프로젝트 경로 설정
# ---------------------------------------------------------

# 현재 파일:
# eeg-mi-generalization/scripts/inspect_subject.py
#
# parents[0] = scripts
# parents[1] = eeg-mi-generalization
#
# 따라서 PROJECT_ROOT는 프로젝트 최상위 폴더가 된다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 다운로드한 원본 EEG 데이터를 저장할 폴더
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# PSD 등의 결과 이미지를 저장할 폴더
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"


# ---------------------------------------------------------
# EEG 데이터 다운로드 및 로딩
# ---------------------------------------------------------

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
        다운로드할 실험 run 번호 목록.
        이 프로젝트에서는 [4, 8, 12]를 사용한다.

    Returns
    -------
    combined_raw:
        여러 run을 시간 방향으로 연결한 MNE Raw 객체.
    """

    # data/raw 폴더가 없으면 생성한다.
    #
    # parents=True:
    # data 폴더까지 없으면 상위 폴더도 함께 생성한다.
    #
    # exist_ok=True:
    # 폴더가 이미 존재해도 오류를 발생시키지 않는다.
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"피험자 {subject}의 "
        f"run {runs} 다운로드를 시작합니다."
    )

    # PhysioNet EEGBCI 데이터셋에서 지정한 파일을 다운로드한다.
    #
    # 반환값 file_paths는 EEG 데이터 자체가 아니라,
    # 다운로드된 EDF 파일의 경로 목록이다.
    file_paths = eegbci.load_data(
        subjects=subject,
        runs=runs,
        path=DATA_DIR,
        update_path=False,
    )

    print("다운로드된 파일:")

    # 다운로드된 각 파일의 위치를 출력한다.
    for file_path in file_paths:
        print(f"- {file_path}")

    # 각 run의 Raw 객체를 임시로 저장할 리스트
    raw_list: list[mne.io.BaseRaw] = []

    # 다운로드된 EDF 파일을 하나씩 읽는다.
    for file_path in file_paths:

        # EDF 파일을 MNE Raw 객체로 변환한다.
        #
        # preload=True:
        # EEG 신호 전체를 메모리에 로드한다.
        # 이후 필터링이나 배열 접근이 가능해진다.
        raw = mne.io.read_raw_edf(
            file_path,
            preload=True,
            verbose=False,
        )

        # EEGBCI 채널명을 MNE 표준 형식으로 정리한다.
        #
        # 예:
        # "C3.." → "C3"
        eegbci.standardize(raw)

        # 각 EEG 채널의 두피 위치 좌표를 연결한다.
        #
        # montage는 C3, Cz, C4 같은 채널이
        # 머리의 어느 위치에 있는지를 나타낸다.
        raw.set_montage(
            "standard_1005",
            on_missing="warn",
        )
        # copy()는 Raw 객체를 그대로 복제한다.
        # 앞으로 Filter는 복사본에만 적용한다.
        raw = raw.copy()
        raw.filter(
            l_freq=8.0,      # 8Hz 이하 제거
            h_freq=30.0,     # 30Hz 이상 제거
            verbose=False,
        )

        print("8~30Hz Band-pass Filter 적용 완료")

        # 현재 run의 Raw 객체를 리스트에 추가한다.
        raw_list.append(raw)

    # run 4, 8, 12를 시간 방향으로 이어 붙인다.
    #
    # 주의:
    # 현재는 데이터 탐색용으로 합친다.
    # 머신러닝 평가에서는 run 정보를 별도로 보존해야 한다.
    combined_raw = mne.concatenate_raws(raw_list)

    return combined_raw


# ---------------------------------------------------------
# 기본 데이터 정보 출력
# ---------------------------------------------------------

def print_data_summary(
    raw: mne.io.BaseRaw,
    subject: int,
    runs: list[int],
) -> None:
    """
    EEG 데이터의 채널 수, 샘플링 주파수,
    기록 시간, 이벤트 구조 등을 출력한다.
    """

    # annotation의 T0, T1, T2 정보를
    # 숫자 기반 events 배열로 변환한다.
    events, event_id = mne.events_from_annotations(
        raw,
        verbose=False,
    )

    print()
    print("=" * 60)
    print("EEG 데이터 요약")
    print("=" * 60)

    print(f"피험자 번호: {subject}")
    print(f"사용한 run: {runs}")

    # sfreq = sampling frequency
    # 160Hz라면 채널별로 1초에 160개 값을 측정했다는 의미
    print(f"샘플링 주파수: {raw.info['sfreq']} Hz")

    # EEG 채널 이름 목록의 길이
    print(f"채널 수: {len(raw.ch_names)}")

    # 전체 시간축 샘플 개수
    print(f"전체 샘플 수: {raw.n_times}")

    # raw.times[-1]은 기록의 마지막 시간값
    print(f"전체 기록 시간: {raw.times[-1]:.2f}초")

    # 추출된 전체 이벤트 개수
    print(f"이벤트 수: {len(events)}")

    # 예:
    # {'T0': 1, 'T1': 2, 'T2': 3}
    print(f"이벤트 매핑: {event_id}")

    # 채널 이름 목록 중 앞의 10개만 출력
    print(f"첫 10개 채널: {raw.ch_names[:10]}")

    print("=" * 60)


# ---------------------------------------------------------
# 이벤트 개수 출력
# ---------------------------------------------------------

def print_event_counts(
    raw: mne.io.BaseRaw,
) -> None:
    """
    T0, T1, T2 이벤트가 각각 몇 번 발생했는지 출력한다.
    """

    # 문자열 annotation을 숫자 events로 변환한다.
    events, event_id = mne.events_from_annotations(
        raw,
        verbose=False,
    )

    # 원래 구조:
    # {'T0': 1, 'T1': 2, 'T2': 3}
    #
    # 뒤집은 구조:
    # {1: 'T0', 2: 'T1', 3: 'T2'}
    reverse_event_id = {
        event_number: event_name
        for event_name, event_number in event_id.items()
    }

    # events[:, 2]는 각 이벤트의 숫자 코드만 선택한다.
    #
    # np.unique(..., return_counts=True)는
    # 고유한 이벤트 코드와 발생 횟수를 함께 반환한다.
    unique_codes, counts = np.unique(
        events[:, 2],
        return_counts=True,
    )

    print()
    print("=" * 60)
    print("이벤트별 개수")
    print("=" * 60)

    # 이벤트 코드와 발생 횟수를 하나씩 함께 순회한다.
    for event_code, count in zip(
        unique_codes,
        counts,
    ):
        # 숫자 이벤트 코드를 T0, T1, T2 이름으로 변환한다.
        event_name = reverse_event_id.get(
            int(event_code),
            "unknown",
        )

        print(
            f"{event_name} "
            f"(code={event_code}): "
            f"{count}회"
        )

    print("=" * 60)


# ---------------------------------------------------------
# EEG 기본 품질 검사
# ---------------------------------------------------------

def inspect_signal_quality(
    raw: mne.io.BaseRaw,
) -> None:
    """
    EEG 배열 구조와 NaN, 무한대,
    flat channel 존재 여부를 검사한다.
    """

    # MNE Raw 객체에서 EEG 숫자 배열만 가져온다.
    #
    # 배열 shape:
    # (채널 수, 시간 샘플 수)
    eeg_data = raw.get_data(
        picks="eeg",
    )

    # 각 채널의 표준편차를 계산한다.
    #
    # axis=1:
    # 각 채널의 시간축 방향으로 계산한다.
    channel_stds = np.std(
        eeg_data,
        axis=1,
    )

    # 각 채널의 최대값 - 최소값을 계산한다.
    channel_ranges = np.ptp(
        eeg_data,
        axis=1,
    )

    # NaN은 숫자가 아닌 결측값이다.
    nan_count = int(
        np.isnan(eeg_data).sum()
    )

    # inf는 무한대 값이다.
    inf_count = int(
        np.isinf(eeg_data).sum()
    )

    # 표준편차가 거의 0인 채널을 찾는다.
    #
    # 신호가 시간에 따라 거의 변하지 않으면
    # flat channel일 가능성이 있다.
    flat_channel_indices = np.where(
        channel_stds < 1e-12
    )[0]

    # 숫자 인덱스를 실제 채널 이름으로 변환한다.
    flat_channels = [
        raw.ch_names[index]
        for index in flat_channel_indices
    ]

    print()
    print("=" * 60)
    print("EEG 신호 품질 검사")
    print("=" * 60)

    print(f"데이터 배열 모양: {eeg_data.shape}")
    print(f"NaN 개수: {nan_count}")
    print(f"무한대 값 개수: {inf_count}")

    # MNE는 EEG 데이터를 기본적으로 Volt 단위로 반환한다.
    print(
        "평균 채널 표준편차: "
        f"{channel_stds.mean():.8f} V"
    )

    print(
        "최소 채널 표준편차: "
        f"{channel_stds.min():.8f} V"
    )

    print(
        "최대 채널 표준편차: "
        f"{channel_stds.max():.8f} V"
    )

    print(
        "평균 채널 범위: "
        f"{channel_ranges.mean():.8f} V"
    )

    print(f"Flat channel: {flat_channels}")
    print("=" * 60)


# ---------------------------------------------------------
# PSD 그래프 저장
# ---------------------------------------------------------

def save_psd_figure(
    raw: mne.io.BaseRaw,
    subject: int,
) -> None:
    """
    1~40Hz 구간의 평균 Power Spectral Density 그래프를 저장한다.
    """

    # 결과 이미지 폴더가 없으면 생성한다.
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Welch 방법으로 EEG의 주파수별 파워를 계산한다.
    #
    # fmin=1:
    # 1Hz보다 낮은 매우 느린 변화는 제외한다.
    #
    # fmax=40:
    # 현재 운동상상 분석의 주요 관심 대역보다
    # 조금 넓은 40Hz까지 확인한다.
    spectrum = raw.compute_psd(
        method="welch",
        fmin=1.0,
        fmax=40.0,
        picks="eeg",
        verbose=False,
    )

    # 모든 EEG 채널의 PSD를 평균해 표시한다.
    spectrum.plot(
        average=True,
        show=False,
    )

    # 피험자 번호를 세 자리 숫자로 파일명에 넣는다.
    #
    # subject=1 → subject_001_psd.png
    output_path = (
        FIGURE_DIR
        / f"subject_{subject:03d}_psd.png"
    )

    # PSD 그래프를 PNG 이미지로 저장한다.
    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    # 현재 matplotlib figure를 메모리에서 닫는다.
    plt.close()

    print(
        f"PSD 그래프 저장 완료: {output_path}"
    )


# ---------------------------------------------------------
# Epoch 생성
# ---------------------------------------------------------

def create_motor_imagery_epochs(
    raw: mne.io.BaseRaw,
    
) -> mne.Epochs:
    """
    연속 EEG Raw 데이터를 이벤트 기준으로 잘라
    왼손·오른손 운동상상 Epoch를 생성한다.
    """

    # Raw 객체의 annotation에서 이벤트를 추출한다.
    #
    # events:
    # [이벤트 발생 샘플, 이전 값, 이벤트 코드]
    #
    # annotation_event_id 예:
    # {'T0': 1, 'T1': 2, 'T2': 3}
    events, annotation_event_id = (
        mne.events_from_annotations(
            raw,
            verbose=False,
        )
    )

    # 현재 사용 중인 run 4, 8, 12에서는:
    #
    # T1 = 왼손 운동상상
    # T2 = 오른손 운동상상
    #
    # T0 휴식은 이번 이진 분류에서 제외한다.
    motor_imagery_event_id = {
        "left_hand": annotation_event_id["T1"],
        "right_hand": annotation_event_id["T2"],
    }

    # 각 T1/T2 이벤트를 기준으로 EEG를 자른다.
    # Epoch 객체를 생성한다.
    epochs = mne.Epochs(
    raw=raw,

    # 각 이벤트가 발생한 시간 위치와 코드
    events=events,

    # 왼손과 오른손 이벤트만 선택
    event_id=motor_imagery_event_id,

    # 이벤트 발생 0.5초 후부터 사용
    tmin=0.5,

    # 이벤트 발생 후 4초까지 사용
    tmax=4.0,

    # 초기 버전에서는 baseline 보정을 하지 않는다.
    baseline=None,

    # EEG 채널만 포함한다.
    picks="eeg",

    # Epoch 데이터를 메모리에 미리 로드한다.
    preload=True,

    # MNE의 상세 로그를 줄인다.
    verbose=False,
    )

    # Epoch 객체를 NumPy 배열로 변환한다.
    # shape 구조:
    # (Epoch 개수, 채널 개수, 시간 샘플 개수)
    # Epoch 객체를 NumPy 배열로 변환한다.
    epoch_data = epochs.get_data()

    # 배열의 크기를 출력한다.
    
    # events 배열의 세 번째 열(인덱스 2)은
    # 각 Epoch의 이벤트 코드(T1, T2)를 저장하고 있다.
    labels = epochs.events[:, 2]
    # ---------------------------------------------------------
    # MNE 이벤트 코드 2, 3을 머신러닝용 라벨 0, 1로 변환한다.
    # ---------------------------------------------------------

    # 2(T1, 왼손)를 0으로,
    # 3(T2, 오른손)를 1로 바꾼다.
    binary_labels = labels - 2

    # 변환 결과를 확인한다.
    print(f"이진 라벨 : {binary_labels}")
    print(f"고유 이진 라벨 : {np.unique(binary_labels)}")
    print()
    print("=" * 60)
    print("라벨(Label) 확인")
    print("=" * 60)

    # 모든 Epoch의 라벨 출력
    print(f"전체 라벨 : {labels}")

    # 라벨 종류 출력
    print(f"고유 라벨 : {np.unique(labels)}")

    print("=" * 60)
    print(f"Epoch 배열 Shape: {epoch_data.shape}")
    print()
    print("=" * 60)
    print("Epoch 생성 결과")
    print("=" * 60)

    # 전체 Epoch 개수
    print(f"전체 Epoch 수: {len(epochs)}")

    # NumPy 배열의 shape 출력
    print(f"Epoch 배열 Shape: {epoch_data.shape}")

    # 왼손 운동상상 Epoch 개수
    print(f"왼손 Epoch 수: {len(epochs['left_hand'])}")

    # 오른손 운동상상 Epoch 개수
    print(f"오른손 Epoch 수: {len(epochs['right_hand'])}")

    # 한 Epoch의 시작과 종료 시간
    print(
    f"Epoch 시간 범위: "
    f"{epochs.tmin:.1f}초 ~ "
    f"{epochs.tmax:.1f}초"
    )

    print("=" * 60)
# Epoch 데이터를 NumPy 배열로 가져온다.
    epoch_data = epochs.get_data()

    # 시간축(axis=2)의 평균을 계산한다.
    #
    # 원래 shape
    # (45, 64, 561)
    #
    # 평균 계산 후
    # (45, 64)
    #
    # 즉 Epoch 하나당
    # 채널 64개의 평균값만 남는다.
    features = epoch_data.mean(axis=2)

    print()
    print("=" * 60)
    print("Feature 확인")
    print("=" * 60)

    print(f"Feature Shape : {features.shape}")

    print("=" * 60)
    # 모든 출력과 검사가 끝난 뒤 마지막에 반환한다.
    return epochs


# ---------------------------------------------------------
# 프로그램 실행 순서
# ---------------------------------------------------------

def main() -> None:
    """
    전체 데이터 탐색 과정을 순서대로 실행한다.
    """

    # EEGBCI 참가자 1번을 사용한다.
    subject = 1

    # Run 4, 8, 12는
    # 왼손 대 오른손 운동상상 과제이다.
    runs = [4, 8, 12]

    # EEG 데이터를 다운로드하고 Raw 객체로 로딩한다.
    raw = load_subject_data(
        subject=subject,
        runs=runs,
    )

    # EEG 데이터의 기본 정보를 출력한다.
    print_data_summary(
        raw=raw,
        subject=subject,
        runs=runs,
    )

    # NaN, 무한대, flat channel 등을 확인한다.
    inspect_signal_quality(
        raw=raw,
    )

    # T0, T1, T2 이벤트 발생 횟수를 확인한다.
    print_event_counts(
        raw=raw,
    )

    # 평균 PSD 그래프를 이미지로 저장한다.
    save_psd_figure(
        raw=raw,
        subject=subject,
    )

    # 연속 Raw 데이터를
    # 왼손·오른손 운동상상 Epoch로 변환한다.
    epochs = create_motor_imagery_epochs(
        raw=raw,
    )

    # 생성된 Epoch를 시각적으로 확인한다.
    #
    # 주의:
    # plot 창을 닫아야 Python 프로그램이 완전히 종료될 수 있다.
    epochs.plot(
        n_epochs=5,
        n_channels=20,
        scalings="auto",
    )


# 이 파일이 직접 실행됐을 때만 main()을 호출한다.
#
# 다른 Python 파일에서 import했을 때는
# main()이 자동 실행되지 않는다.
if __name__ == "__main__":
    main()