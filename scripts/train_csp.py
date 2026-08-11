"""
CSP + Linear SVM 기반 Motor Imagery 학습 및 평가.

Raw EEG에서 생성한 Epoch를 CSP(Common Spatial Pattern)로 변환한 뒤
Linear SVM을 이용해 왼손/오른손 Motor Imagery를 분류한다.

중요:
CSP는 Train 데이터로만 학습하고,
Test 데이터에는 학습된 CSP를 transform만 적용한다.
"""

from typing import Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from scripts.inspect_subject import prepare_csp_dataset
from src.features.csp import (
    create_csp,
    fit_transform_csp,
    transform_csp,
)


# ---------------------------------------------------------
# 실행 설정
# ---------------------------------------------------------

SECTION_WIDTH = 72


def print_section(title: str) -> None:
    """실행 결과의 출력 형식을 통일한다."""

    print()
    print("=" * SECTION_WIDTH)
    print(title)
    print("=" * SECTION_WIDTH)


def create_model() -> Pipeline:
    """
    CSP Feature를 분류할 Linear SVM 모델을 생성한다.

    StandardScaler:
        CSP Feature의 크기를 정규화한다.

    SVC:
        왼손/오른손을 구분하는 Linear SVM 분류기다.
    """

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "svm",
                SVC(
                    kernel="linear",
                    C=1.0,
                ),
            ),
        ]
    )


def train_model(
    model: Pipeline,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> Pipeline:
    """
    CSP Feature를 이용하여 SVM을 학습한다.
    """

    print_section("[3] Linear SVM 학습")

    print(f"Train Feature Shape : {x_train.shape}")
    print(f"Train Label Shape   : {y_train.shape}")

    model.fit(
        x_train,
        y_train,
    )

    print("[OK] SVM 학습 완료")

    return model


def evaluate_model(
    model: Pipeline,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[float, np.ndarray, str]:
    """
    Test Feature를 이용하여 모델의 분류 성능을 평가한다.
    """

    print_section("[4] CSP + SVM Test 평가")

    # -----------------------------------------------------
    # Test 데이터 예측
    # -----------------------------------------------------

    y_pred = model.predict(
        x_test,
    )

    # -----------------------------------------------------
    # Accuracy
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    # -----------------------------------------------------
    # Confusion Matrix
    # -----------------------------------------------------

    matrix = confusion_matrix(
        y_test,
        y_pred,
    )

    # -----------------------------------------------------
    # Classification Report
    # -----------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        target_names=[
            "left_hand",
            "right_hand",
        ],
        digits=4,
        zero_division=0,
    )

    print(f"Accuracy : {accuracy:.4f}")

    print()
    print("-" * SECTION_WIDTH)
    print("Confusion Matrix")
    print("-" * SECTION_WIDTH)

    print("                 예측")
    print("              왼손  오른손")

    print(
        f"실제 왼손     "
        f"{matrix[0, 0]:>3}   "
        f"{matrix[0, 1]:>3}"
    )

    print(
        f"실제 오른손   "
        f"{matrix[1, 0]:>3}   "
        f"{matrix[1, 1]:>3}"
    )

    print()
    print("-" * SECTION_WIDTH)
    print("Classification Report")
    print("-" * SECTION_WIDTH)

    print(report)

    return (
        accuracy,
        matrix,
        report,
    )


def main() -> None:
    """
    CSP + SVM 전체 학습 파이프라인을 실행한다.
    """

    print_section(
        "CSP + Linear SVM Motor Imagery Pipeline"
    )

    # =====================================================
    # STEP 1
    # CSP용 Epoch 데이터를 준비한다.
    #
    # Baseline과 달리 평균 Feature로 바꾸지 않는다.
    #
    # 예상 Shape
    #
    # X_train
    # (36, 64, 561)
    #
    # X_test
    # (9, 64, 561)
    # =====================================================

    print_section("[1] CSP Dataset 준비")

    (
        x_train_epoch,
        x_test_epoch,
        y_train,
        y_test,
    ) = prepare_csp_dataset()

    print(
        f"X_train Epoch : "
        f"{x_train_epoch.shape}"
    )

    print(
        f"X_test Epoch  : "
        f"{x_test_epoch.shape}"
    )

    # =====================================================
    # STEP 2
    # CSP 생성
    # =====================================================

    print_section("[2] CSP Feature 추출")

    csp = create_csp(
        n_components=4,
    )

    # -----------------------------------------------------
    # Train 데이터
    #
    # CSP가 왼손/오른손을 가장 잘 구분하는
    # 공간 패턴을 Train 데이터에서 학습한다.
    #
    # fit + transform
    # -----------------------------------------------------

    x_train_csp = fit_transform_csp(
        csp=csp,
        x_train=x_train_epoch,
        y_train=y_train,
    )

    # -----------------------------------------------------
    # Test 데이터
    #
    # Test 데이터로 CSP를 다시 학습하면
    # Data Leakage가 발생한다.
    #
    # 따라서 이미 학습된 CSP를 이용하여
    # transform만 한다.
    # -----------------------------------------------------

    x_test_csp = transform_csp(
        csp=csp,
        x_test=x_test_epoch,
    )

    print(
        f"CSP Train Feature : "
        f"{x_train_csp.shape}"
    )

    print(
        f"CSP Test Feature  : "
        f"{x_test_csp.shape}"
    )

    # 예상:
    #
    # (36, 4), (9, 4)

    model = create_model()
    train_model(model, x_train_csp, y_train)
    evaluate_model(model, x_test_csp, y_test)

    print_section("CSP Pipeline Finished Successfully")


if __name__ == "__main__":
    main()
