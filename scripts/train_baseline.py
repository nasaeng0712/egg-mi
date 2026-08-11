"""Linear SVM 기반 BCI 베이스라인 모델 학습 및 평가 스크립트.

기존의 홀드아웃(학습/테스트) 평가 기능을 유지하면서, 모델 생성·학습·평가와
교차 검증을 각각 독립된 함수로 분리한다.
"""

from typing import Any, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from inspect_subject import prepare_baseline_dataset


# 일부 버전의 inspect_subject.py에는 이 함수가 아직 없을 수 있다.
# 선택적 import로 처리하여 기존 홀드아웃 학습·평가는 그대로 실행되게 한다.
try:
    from inspect_subject import prepare_baseline_features
except ImportError:
    prepare_baseline_features = None


SEPARATOR = "=" * 72
SUB_SEPARATOR = "-" * 72
CV_FOLDS = 5
RANDOM_STATE = 42


def print_section(title: str) -> None:
    """콘솔 결과를 일관된 형식으로 구분하여 출력한다."""
    print(f"\n{SEPARATOR}")
    print(title)
    print(SEPARATOR)


def create_model() -> Pipeline:
    """표준화와 Linear SVM으로 구성된 모델 파이프라인을 생성한다.

    StandardScaler를 파이프라인 안에 포함하면 교차 검증 시 각 fold의 학습
    데이터만으로 스케일링 통계가 계산되어 데이터 누수를 방지할 수 있다.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="linear")),
        ]
    )


def train_model(
    model: BaseEstimator,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> BaseEstimator:
    """전달받은 학습 데이터로 모델을 학습하고 반환한다."""
    model.fit(x_train, y_train)
    return model


def evaluate_model(
    model: BaseEstimator,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[float, np.ndarray, str]:
    """테스트 데이터로 정확도, 혼동 행렬, 분류 보고서를 계산·출력한다."""
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print_section("[1] Hold-out Test 평가 결과")
    print(f"Accuracy              : {accuracy:.4f}")
    print(f"\n{SUB_SEPARATOR}")
    print("Confusion Matrix")
    print(SUB_SEPARATOR)
    print(matrix)
    print(f"\n{SUB_SEPARATOR}")
    print("Classification Report")
    print(SUB_SEPARATOR)
    print(report)

    return accuracy, matrix, report


def _load_full_features() -> Tuple[np.ndarray, np.ndarray]:
    """교차 검증에 필요한 전체 feature와 label을 안전하게 불러온다.

    prepare_baseline_dataset()은 이미 분리된 네 개의 배열만 반환하므로,
    해당 결과를 다시 합치면 최초 데이터 분리 방식에 의존할 수 있다. 따라서
    교차 검증에는 분리 전 전체 데이터를 제공하는 전용 함수를 사용한다.
    """
    if prepare_baseline_features is None:
        raise RuntimeError(
            "5-Fold 교차 검증을 실행할 수 없습니다. "
            "inspect_subject.py에 전체 데이터의 (features, labels)를 반환하는 "
            "prepare_baseline_features() 함수를 추가해 주세요. "
            "기존 prepare_baseline_dataset()만으로는 분리 전 전체 데이터를 "
            "안전하게 얻을 수 없습니다."
        )

    result: Any = prepare_baseline_features()
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise RuntimeError(
            "prepare_baseline_features()는 반드시 "
            "(features, labels) 두 값을 반환해야 합니다."
        )

    features, labels = result
    features = np.asarray(features)
    labels = np.asarray(labels)

    if features.ndim != 2:
        raise RuntimeError(
            "features는 (샘플 수, 특성 수)의 2차원 배열이어야 합니다."
        )
    if labels.ndim != 1:
        labels = labels.ravel()
    if features.shape[0] != labels.shape[0]:
        raise RuntimeError("features와 labels의 샘플 수가 서로 다릅니다.")

    return features, labels


def run_cross_validation(
    model: BaseEstimator,
    features: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """5-Fold Stratified Cross Validation을 실행하고 fold별 점수를 반환한다.

    features와 labels를 생략하면 inspect_subject.py의
    prepare_baseline_features()에서 전체 데이터를 가져온다.
    """
    if features is None or labels is None:
        if features is not None or labels is not None:
            raise ValueError(
                "features와 labels는 함께 전달하거나 모두 생략해야 합니다."
            )
        features, labels = _load_full_features()

    splitter = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    scores = cross_val_score(
        model,
        features,
        labels,
        cv=splitter,
        scoring="accuracy",
    )

    print_section("[2] 5-Fold Stratified Cross Validation 결과")
    for fold_number, score in enumerate(scores, start=1):
        print(f"Fold {fold_number:<2} Accuracy     : {score:.4f}")
    print(SUB_SEPARATOR)
    print(f"Mean Accuracy          : {scores.mean():.4f}")
    print(f"Standard Deviation     : {scores.std():.4f}")

    return scores


def main() -> None:
    """데이터 준비부터 홀드아웃 평가와 교차 검증까지 전체 과정을 실행한다."""
    print_section("Linear SVM Baseline 학습 시작")

    # 기존 함수의 반환 형식은 아래 네 값으로 가정한다.
    x_train, x_test, y_train, y_test = prepare_baseline_dataset()
    print(f"Train samples          : {len(y_train)}")
    print(f"Test samples           : {len(y_test)}")

    model = create_model()
    trained_model = train_model(model, x_train, y_train)
    evaluate_model(trained_model, x_test, y_test)

    # 전체 데이터 제공 함수가 없더라도 기존 학습과 평가는 완료한다. CV만
    # 건너뛰고, 사용자가 무엇을 추가해야 하는지 명확한 안내를 출력한다.
    try:
        run_cross_validation(create_model())
    except RuntimeError as error:
        print_section(
            "[2] 5-Fold Stratified Cross Validation 안내"
        )
        print(f"RuntimeError: {error}")

    print_section("모든 실행 과정 완료")


if __name__ == "__main__":
    main()
