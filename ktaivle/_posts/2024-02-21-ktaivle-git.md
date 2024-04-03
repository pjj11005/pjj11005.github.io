---
layout: post
title: 1주차 | GIT
description: KT AIVLE SCHOOL 5기 1주차에 진행한 GIT 강의 내용 정리 글입니다.
sitemap: false
---

GIT은 분산 버전 관리 시스템으로 개발자에게 매우 중요하다  

* this unordered seed list will be replaced by the toc
{:toc}

## GIT 사용

>- 버전 관리 시스템, 가성비 좋다
>- 버전 관리 시스템의 목적 이유
>    - **버전** 관리에 좋다
>        - 버그, 에러 등을 고치기 쉽다
>    - 소스 코드 안전하게 보관 (**백업**)
>    - 서로 다른 사람들끼리 **협업** 가능
>- **GIT 사용 이유 → 디버깅에 사용**

- `GUI에서 GIT` ⇒ `CLI에서 GIT`
    - 문제 발생 시 `CLI에서` 해결 → 따라서 커맨드 라인 알아야한다

- `하나의 버전(단위 작업)은 작을 수록, 하나의 주제만 있으면 좋다`
    - 언제나 실행 가능한 상태 유지하면 좋음
    
### 초기 설정

- user name, email 설정

    ![Untitled](/assets/img/blog/KT_AIVLE/week1/GIT/001.png)

### 기능 및 구조
- 하나의 버전에 복수의 파일 적용, 복수의 버전을 적용
    - `commit`할 파일들만 `staged changes`에 올리고 `add` (장바구니에 물건 담기)
        - `stage area`에 올려놓음 (`commit` 대기)
    - 적용할 버전 별로 나눠서 `commit` 하여 복수의 버전 적용

- `git graph` 확장 프로그램 사용 (GUI 방식)
    - 만든 사람, `commit`한 사람, 수정된 내용 등을 한눈에 알기 쉽다
    
    ![Untitled](/assets/img/blog/KT_AIVLE/week1/GIT/002.png)

- 프로젝트 파일 구조
    
    ![Untitled](/assets/img/blog/KT_AIVLE/week1/GIT/003.png)
    
    > `project folder`
    >    - `working dir`
    >    - `staged area`
    >    - `.git 폴더(repository)` → `저장소` (initialize 시 생성)
        
- 동작의 의미
    - **`add`**: `working dir` 의 파일을 `staged area`로 옮기기
    - **`commit`**: `staged area`의 파일을 `.git 폴더(repository)`에 올리기
        - `commit`으로 생긴 버전 정보를 40자의 해시 값으로 표현 (버전 각각의 `commit` 아이디) → 내용을 기반으로 아이디 생성 (`분산 버전 관리 시스템`이기 때문)

- git은 이전에 `main`에 저장한 `최신 commit id`를 바탕으로 다음 작업의 `commit` 아이디 생성 → `main`의 `last commit` 갱신 (`main`은 최신 `commit`, 가장 마지막의 작업을 가리킨다)
    - 다른 파일의 변경된 점이 없거나, 새롭게 생성된 파일과 기존의 내용이 같으면 이전의 저장된 내용 사용 → `git은 파일이름과 내용을 따로 저장하기 때문`
- `staged area`는 `commit` 작업을 수행해도 사라지지 않고 계속 존재
    - 따라서 이 공간에는 많은 파일들이 존재
    - 각 버전의 스냅샷을 가지고 있어 변경 사항을 알 수 있다
- 각각의 버전은 `그 버전이 만들어진 시점의 스냅샷`이다

### head
- 버전을 생성하면 `head` 생성
    - 버전 생성 시 `commit id` 생성
    - `head`는 `main`을 가리킴 → 따라서 `main`이 새롭게 생성된 `commit id` 가리킴
- 새로운 버전 생성 → `head`가 `parent` 이지만 `main`이 `head`의 대리자이므로 `main`이 가리키는 `commit id`가 `parent`가 됨
    - 그 후, 생성된 `commit id`를 `main`이 가리킴
- `head: 현재 시간, main: 마지막 시간 (시간 여행)`

- **이전 버전으로 디버깅**
    - **`check out`**: `head`를 이동시킨다
        - `working dir`를 예전 상태로 바꿈
        - 다시 원래 상태로 돌아올때는 `main branch`로 `checkout branch` 수행(아니면 더블 클릭)
        - **과거 버전을 확인하여 문제점을 발견한 후, 현재로 돌아와 수정 (과거는 불변, 현재는 가변)**
        - `head`가 버전을 직접 가리키게 한 후 , 새롭게 버전을 만들면 `head`는 대리자가 없기 때문에 직접 새로운 버전을 가리킨다
            - 이때 다시 `git checkout main`을 하면 다시 `head`가 `main`을 가리키고 새롭게 만든 버전은 존재하지 않았던 버전이 되어버림 (새로 만든 버전이 사라지지는 않는다)
    - **`reset`**: `head`가 가리키는 브랜치를 바꾼다
    - git은 안정적이기 때문에 모든 작업은 `recovery` 가능하다

    - `detached head state` 막지 않은 이유: 실험적인 작업을 위해서

    - `commit message`를 변경하면 `commit id`도 변경된다
        - 되돌리려면 기존의 `commit id`로 `reset`하면 된다
        - git은 새로운 `commit id`를 만들 때 복제를 한 후 생성
            - 기존의 `commit id`는 안사라짐

    - reset & checkout 예시

        ![Untitled](/assets/img/blog/KT_AIVLE/week1/GIT/004.png)

        >- `main`이 `v3` 버전을 가리키도록 함
        >- 그 후, `head`가 `exp1` 버전을 가리키도록 함

### branch
- 실험적인 작업을 위해 `branch` 생성
    - 이제 `commit id`를 기억하지 않고도 해당 작업으로 이동 가능

- **branch 병합(merge)** (기존 branch: `main`, 실험적인 branch: `exp`)
    - `merge`는 두 개씩 가능
    - 실험적인 기능이 성공 했을 때 (`main`이 병합)
        - `head`가 `main`을 가리키는 상태에서 `exp branch 병합`
        - 자동으로 ‘Merge …’ 메시지를 가진 `commit` 생성
            - `main과 exp` 두 개의 `parent` 가진다
    - `exp branch`가 `main`을 병합 (`exp`가 병합)
        - 자주할수록 좋음 → 충돌 발생을 막기 위해 필요
    - **결국 합쳐지는 방향이 중요**
    
- **conflict**: 같은 파일을 다른 `branch`에서 수정 후 `merge`할 때 발생
    
    ![Untitled](/assets/img/blog/KT_AIVLE/week1/GIT/005.png)
    
    - git은 `3 way merge` 방식
        - 각 `exp, main branch`의 공통 조상을 `base`로 지정
        - `merge` 수행 시 3개를 비교하며 뒤쪽에서 수정된 부분을 따라간다
        - 양쪽다 바뀌면 `merge` 멈춤 → `conflict` 발생
            
            ![Untitled](/assets/img/blog/KT_AIVLE/week1/GIT/006.png)
            
            >이런 기호가 포함된 상태로 `commit`하면 절대 안된다…

        - **conflict 해결방법**: VScode에 `conflict` 해결을 위한 기능을 지원(`Merge Editor`) → 결과를 작업자가 수정하고 `merge`한 후 `commit`하면 된다.
    
- 작업한 내용들 백업 방법
    - 원격 저장소(`GitHub`)를 이용하여 저장
        - `GitHub`의 `repository` 생성
        - VScode에서 `add remote`로 원격 저장소와 연결(별명 `origin`으로 설정)
        - 그 후, 원격 저장소에 `push`로 업로드
        - `origin/main` (remote tracking branch): 원격 저장소의 branch
        - `push` 후에는 `main/origin`으로 업로드 된부분을 나타낸다

- 원격 저장소의 내용 로컬에 저장 (`clone`)
    - 원격 저장소의 주소 확인 → vscode에서 `clone` → 저장할 로컬 폴더 지정
- `fetch`: `다운로드`
- `pull`: `fetch + merge`

- **협업 시뮬레이션**
    - 작업자 1: 작업 후 `commit + push`
    - 작업자 2: 작업자 1의 작업을 반영한 결과를 원격 저장소에서 `pull`
    - 다음 반대로 진행
    - 위의 작업들 반복
    - **conflict 발생하면 나중에 일하는 작업자가 merge editor로 처리**