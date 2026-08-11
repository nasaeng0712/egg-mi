"""
CSP(Common Spatial Pattern) 기반 EEG Feature 추출.

Motor Imagery의 왼손/오른손 EEG에서
두 클래스의 공간적 분산 차이를 최대화하는 Feature를 만든다.
"""

import numpy as np
from mne.decoding import CSP


def create_csp(
    n_components: int = 4,
) -> CSP:
    """
    CSP Feature Extractor를 생성한다.

    Parameters
    ----------
    n_components:
        사용할 CSP component 개수.

    Returns
    -------
    csp:
        MNE CSP 객체.
    """

    return CSP(
        n_components=n_components,

        # 각 CSP component의 평균 log-power를 Feature로 사용한다.
        log=True,

        # covariance는 Epoch 단위로 계산한다.
        cov_est="epoch",

        # 불필요한 상세 로그는 출력하지 않는다.
        verbose=False,
    )


def fit_transform_csp(
    csp: CSP,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> np.ndarray:
    """
    Train EEG로 CSP를 학습하고
    Train CSP Feature를 반환한다.

    Parameters
    ----------
    x_train:
        Shape = (Epoch 수, 채널 수, 시간 샘플 수)

    y_train:
        각 Epoch의 정답 Label.

    Returns
    -------
    x_train_csp:
        CSP로 변환된 Train Feature.
    """

    return csp.fit_transform(
        x_train,
        y_train,
    )


def transform_csp(
    csp: CSP,
    x_test: np.ndarray,
) -> np.ndarray:
    """
    이미 Train 데이터로 학습된 CSP를 이용해
    Test EEG를 변환한다.

    중요:
    Test 데이터에는 fit()을 하지 않는다.
    """

    return csp.transform(
        x_test,
    )