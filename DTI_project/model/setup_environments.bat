@echo off
REM ============================================================================
REM DTI_project 필요 패키지 한 번에 설치: p_proj 환경 + DTI_txgnn 환경
REM
REM 새 컴퓨터로 DTI_project 폴더를 옮기셨을 때, 이 스크립트 하나로 두 conda
REM 환경을 만들고 필요한 패키지를 전부 설치합니다. 이미 환경이 있다면 그
REM 환경에 패키지만 (다시) 설치합니다 (덮어써도 안전합니다).
REM
REM 사용법: 아무 conda 환경(또는 base)에서
REM   setup_environments.bat
REM
REM 전제 조건: "conda" 명령이 이 창(cmd.exe)에서 되어야 합니다.
REM           안 되면 한 번만: conda init cmd.exe   실행 후 새 창에서 다시 시도.
REM ============================================================================

setlocal enabledelayedexpansion

echo ================================================================================
echo [1/2] p_proj 환경 설정
echo ================================================================================

call conda create -n p_proj python=3.11 -y
if errorlevel 1 goto :error

call conda activate p_proj
if errorlevel 1 goto :error

REM torch: CPU 전용. GPU(CUDA)가 있는 컴퓨터라면 이 줄 대신
REM   pip install torch --index-url https://download.pytorch.org/whl/cu121
REM 처럼 맞는 CUDA 버전으로 설치하세요.
pip install torch --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :error

pip install ^
    transformers ^
    pandas ^
    numpy ^
    scipy ^
    h5py ^
    duckdb ^
    requests ^
    tqdm ^
    scikit-learn
if errorlevel 1 goto :error

echo.
echo [1/2] p_proj 환경 설정 완료
echo.

echo ================================================================================
echo [2/2] DTI_txgnn 환경 설정
echo ================================================================================

call conda create -n DTI_txgnn python=3.11 -y
if errorlevel 1 goto :error

call conda activate DTI_txgnn
if errorlevel 1 goto :error

REM torch: TxGNN 학습/실행 당시 검증된 버전(2.2.2, CPU 전용)으로 고정.
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :error

REM dgl: TxGNN 이 요구하는 그래프 신경망 라이브러리. 공식 wheel 저장소에서 받습니다.
REM     이 줄이 실패하면 DGL 홈페이지(https://www.dgl.ai/pages/start.html)에서
REM     본인 OS/CPU 에 맞는 설치 명령을 확인해서 대신 실행하세요.
pip install dgl==1.1.2 -f https://data.dgl.ai/wheels/repo.html
if errorlevel 1 goto :error

pip install ^
    pandas==1.5.3 ^
    numpy==1.26.4 ^
    matplotlib ^
    goatools ^
    ftpretty ^
    scipy ^
    scikit-learn ^
    ogb ^
    h5py ^
    requests
if errorlevel 1 goto :error

REM TxGNN 본체: 의존성은 위에서 이미 맞는 버전으로 깔았으므로 --no-deps 로 설치
REM (그대로 설치하면 TxGNN 이 원하는 dgl/torch 버전으로 덮어쓰려 해서 충돌납니다).
pip install txgnn==0.0.3 --no-deps
if errorlevel 1 goto :error

echo.
echo [2/2] DTI_txgnn 환경 설정 완료
echo.

echo ================================================================================
echo 전체 완료: p_proj, DTI_txgnn 두 환경 모두 준비되었습니다.
echo ================================================================================
exit /b 0

:error
echo.
echo [오류] 설치 중 문제가 발생했습니다. 위쪽 오류 메시지를 그대로 알려주시면
echo        원인을 확인해드리겠습니다.
exit /b 1
