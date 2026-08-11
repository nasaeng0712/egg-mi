"""
CSP(Common Spatial Pattern)

EEG 신호를
CSP Feature로 변환하는 모듈
"""

import numpy as np

from mne.decoding import CSP


def create_csp(
    n_components: int = 4,
) -> CSP:
    """
    CSP 객체 생성
    """

    return CSP(
        n_components=n_components,
        log=True,
        cov_est="epoch",
    )


def fit_transform_csp(
    csp: CSP,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> np.ndarray:
    """
    Train 데이터로 CSP 학습
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
    Test 데이터를
    이미 학습된 CSP로 변환
    """

    return csp.transform(
        x_test,
    )
