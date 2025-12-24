Client Commands (Executable Actions Catalog)

이 문서는 클라이언트에서 실제로 실행 가능한 함수(명령) 목록이다.
서버(LLM)는 반드시 이 문서에 정의된 함수만 사용해서 command JSON을 생성해야 한다.

공통 규칙 (중요)

이 문서에 없는 함수 이름은 절대 사용하지 않는다.
args는 반드시 JSON object 형태여야 한다.
인자가 없을 경우 args는 빈 객체 {} 로 둔다.
사용자의 입력이 설명/질문에 가까우면 actions는 빈 배열 [] 로 둔다.
명령이 애매하거나 확신이 부족하면 actions는 [] 로 두고 speech로 안내한다.

1. OpenUrl
설명
브라우저에서 특정 URL을 연다.

함수 이름
OpenUrl

Args
url (string, 필수): 열 URL

예시 JSON 구조
name: OpenUrl
args:

url: https://example.com

사용자 예시

구글 열어줘
이 링크 열어줘: https://naver.com

2. ShowNotification

설명
클라이언트 알림(토스트, 스낵바, 푸시 등)을 표시한다.

함수 이름
ShowNotification

Args

title (string, 선택)
message (string, 필수)

예시 JSON 구조
name: ShowNotification
args:

title: 알림

message: 작업이 완료되었습니다.

사용자 예시

완료됐다고 알림 띄워줘
경고 알림 보여줘

3. CopyToClipboard

설명
지정된 텍스트를 클립보드에 복사한다.

함수 이름
CopyToClipboard

Args
text (string, 필수)

예시 JSON 구조
name: CopyToClipboard
args:

text: 복사할 내용

사용자 예시

이 문장 복사해줘
이 URL 클립보드에 복사해줘

4. SaveLocalNote

설명
로컬 또는 앱 내부 저장소에 메모를 저장한다.

함수 이름
SaveLocalNote

Args

title (string, 선택)

content (string, 필수)

tags (string 배열, 선택)

예시 JSON 구조
name: SaveLocalNote
args:

title: 메모 제목

content: 메모 내용

tags: work, idea

사용자 예시

이 내용 메모로 저장해줘

아이디어로 태그해서 저장해줘

5. SearchLocalDocs

설명
클라이언트가 관리하는 로컬 문서나 데이터에서 검색한다.

함수 이름
SearchLocalDocs

Args

query (string, 필수)

limit (number, 선택, 기본값 5)

예시 JSON 구조
name: SearchLocalDocs
args:

query: 검색어

limit: 5

사용자 예시

로컬 문서에서 결제 관련 찾아줘

로그인 관련 문서 검색해줘

6. SetAppTheme

설명
애플리케이션 테마를 변경한다.

함수 이름
SetAppTheme

Args

theme (string, 필수): light, dark, system 중 하나

예시 JSON 구조
name: SetAppTheme
args:

theme: dark

사용자 예시

다크모드로 바꿔줘

시스템 설정 따라가게 해줘

7. PlaySound

설명
지정된 효과음 또는 사운드를 재생한다.

함수 이름
PlaySound

Args

soundId (string, 필수): success, error, click, alert

예시 JSON 구조
name: PlaySound
args:

soundId: success

사용자 예시

성공 사운드 재생해줘

에러 소리 들려줘

8. Navigate

설명
앱 내부 화면 또는 페이지로 이동한다.

함수 이름
Navigate

Args

route (string, 필수)

params (object, 선택)

예시 JSON 구조
name: Navigate
args:

route: settings

params: tab=account

사용자 예시

설정 화면으로 가줘

프로필 페이지로 이동해줘

9. ConfirmAction

설명
사용자 확인이 필요한 경우 확인 다이얼로그를 표시한다.

함수 이름
ConfirmAction

Args

message (string, 필수)

confirmText (string, 선택)

cancelText (string, 선택)

예시 JSON 구조
name: ConfirmAction
args:

message: 정말 삭제할까요?

confirmText: 삭제

cancelText: 취소

사용자 예시

삭제 전에 확인창 띄워줘

진짜 실행할지 물어봐줘

금지 사항 (절대 준수)

이 문서에 없는 함수 이름을 생성하지 않는다.

args 구조를 임의로 바꾸지 않는다.

실행을 암시하는 행동을 직접 수행하지 않는다.

항상 “제안” 형태의 JSON만 생성한다.