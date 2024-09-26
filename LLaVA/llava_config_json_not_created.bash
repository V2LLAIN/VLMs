#!/bin/bash
# config.json and non_lora_trainables.bin are not generated for few LLaVA versions During LoRA Fine-Tuning

# 결과를 저장할 배열
declare -a results

# 패키지 설치와 버전 확인 함수
install_and_verify() {
    local package_name=$1
    local package_version=$2

    # 패키지 설치
    pip install "${package_name}==${package_version}"

    # 설치된 버전 확인
    installed_version=$(pip show "$package_name" | grep -i version | awk '{print $2}')

    # 버전 일치 여부 확인 및 결과 저장
    if [ "$installed_version" == "$package_version" ]; then
        results+=("$package_name 버전 $package_version 설치 완료.")
    else
        results+=("$package_name 설치 실패: 설치된 버전은 $installed_version 입니다.")
    fi
}

# 패키지 버전 일치시키기
install_and_verify "altair" "5.3.0"
install_and_verify "attrs" "23.2.0"
install_and_verify "bitsandbytes" "0.43.1"
install_and_verify "certifi" "2024.6.2"
install_and_verify "flash-attn" "2.5.9.post1"
install_and_verify "huggingface-hub" "0.23.4"
install_and_verify "pillow" "10.1.0"
install_and_verify "scipy" "1.14.0"
install_and_verify "transformers" "4.36.0"
install_and_verify "uvicorn" "0.30.1"
install_and_verify "peft" "0.11.1"

# 결과 출력
echo "설치 및 버전 확인 결과:"
for result in "${results[@]}"; do
    echo "$result"
done

echo "모든 패키지 업데이트 및 버전 확인이 완료되었습니다."
