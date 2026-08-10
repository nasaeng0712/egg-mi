"""
Baseline SVM 학습

평균 Feature를 이용하여
Motor Imagery를 분류한다.
"""

from inspect_subject import prepare_baseline_dataset
from sklearn.svm import SVC
# 분류 성능 평가를 위한 함수들
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
x_train, x_test, y_train, y_test = prepare_baseline_dataset()
# ---------------------------------------------------------
# SVM 모델 생성
# ---------------------------------------------------------

# 가장 기본적인 SVM 분류기를 생성한다.
model = SVC(
    kernel="linear",   # 선형 SVM
    C=1.0,             # 규제 강도
    random_state=42,
)

print("✓ SVM 모델 생성 완료")
# ---------------------------------------------------------
# 모델 학습
# ---------------------------------------------------------

print("모델 학습 시작...")

model.fit(
    x_train,
    y_train,
)

print("✓ 모델 학습 완료")
# ---------------------------------------------------------
# Test 데이터 예측
# ---------------------------------------------------------

y_pred = model.predict(x_test)

print("✓ Test 데이터 예측 완료")
# ---------------------------------------------------------
# Accuracy 계산
# ---------------------------------------------------------

# ---------------------------------------------------------
# Confusion Matrix 계산
# ---------------------------------------------------------

# 행 = 실제 정답
# 열 = 모델의 예측
#
# 현재 라벨:
# 0 = 왼손 운동상상
# 1 = 오른손 운동상상
# ---------------------------------------------------------
# Accuracy 계산
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred,
)
confusion = confusion_matrix(
    y_test,
    y_pred,
)
# ---------------------------------------------------------
# 클래스별 상세 성능 출력
# ---------------------------------------------------------

report = classification_report(
    y_test,
    y_pred,

    # 숫자 0, 1 대신 사람이 읽기 쉬운 이름을 표시한다.
    target_names=[
        "left_hand",
        "right_hand",
    ],

    # 소수점 4자리까지 출력한다.
    digits=4,

    # 특정 클래스 예측이 하나도 없는 경우 경고 대신 0으로 처리한다.
    zero_division=0,
)

print()
print("=" * 60)
print("Classification Report")
print("=" * 60)
print(report)
print()
print("=" * 60)
print("Confusion Matrix")
print("=" * 60)

print("                 예측")
print("              왼손  오른손")
print(f"실제 왼손     {confusion[0, 0]:>3}   {confusion[0, 1]:>3}")
print(f"실제 오른손   {confusion[1, 0]:>3}   {confusion[1, 1]:>3}")

print("=" * 60)

print()
print("=" * 60)
print("Baseline SVM 결과")
print("=" * 60)
print(f"Accuracy : {accuracy:.4f}")
print("=" * 60)