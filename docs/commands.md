# Client Commands (Executable Actions Catalog)

이 문서는 클라이언트에서 실제로 실행 가능한 **명령(JSON) 생성 전용 함수 카탈로그**입니다.  
서버(LLM)는 반드시 이 문서에 정의된 함수만 사용해서 command JSON을 생성해야 합니다.

---

## 사용 범위 안내 (중요)

- 이 문서는 **명령(JSON) 생성**이 목적입니다.
- 프로그램 개요, 구조, 기술 설명, 동작 원리 같은 질문에는  
  intro.txt, project_overview.txt, tech_stack.md, rag_concept.txt 를 우선 참고해야 합니다.
- 사용자가 **“열어줘 / 복사해줘 / 바꿔줘 / 실행해줘”** 처럼
  실제 행동을 요구할 때만 이 문서를 핵심 근거로 사용합니다.

---

## 공통 규칙 (절대 준수)

- 이 문서에 없는 함수 이름은 절대 생성하지 않습니다.
- args는 반드시 JSON object 형태여야 합니다. (인자가 없으면 {})
- 설명/질문에 가까운 입력이면 actions는 빈 배열 [] 로 둡니다.
- 명령이 애매하거나 확신이 부족하면 actions는 [] 로 두고 speech로 안내합니다.
- 실행을 직접 수행하지 않습니다. 항상 **“제안” 형태의 JSON**만 생성합니다.

---

## 출력 JSON 기본 구조

{
  "type": "command",
  "speech": "사용자에게 보여줄 안내 문구",
  "actions": [
    {
      "name": "함수이름",
      "args": { "param": "value" }
    }
  ]
}

---

## 1. OpenUrl

설명  
브라우저에서 특정 URL을 엽니다.

name  
OpenUrl

args  
- url (string, required): 열 URL

예시  
{
  "name": "OpenUrl",
  "args": { "url": "https://example.com" }
}

---

## 2. ShowNotification

설명  
클라이언트 알림(토스트/스낵바/푸시 등)을 표시합니다.

name  
ShowNotification

args  
- message (string, required)  
- title (string, optional)

예시  
{
  "name": "ShowNotification",
  "args": { "title": "알림", "message": "작업이 완료되었습니다." }
}

---

## 3. CopyToClipboard

설명  
지정된 텍스트를 클립보드에 복사합니다.

name  
CopyToClipboard

args  
- text (string, required)

예시  
{
  "name": "CopyToClipboard",
  "args": { "text": "복사할 내용" }
}

---

## 4. SaveLocalNote

설명  
로컬 또는 앱 내부 저장소에 메모를 저장합니다.

name  
SaveLocalNote

args  
- content (string, required)  
- title (string, optional)  
- tags (string[], optional)

예시  
{
  "name": "SaveLocalNote",
  "args": {
    "title": "메모",
    "content": "내용",
    "tags": ["work", "idea"]
  }
}

---

## 5. SearchLocalDocs

설명  
클라이언트가 관리하는 로컬 문서나 데이터에서 검색합니다.

name  
SearchLocalDocs

args  
- query (string, required)  
- limit (number, optional, default=5)

예시  
{
  "name": "SearchLocalDocs",
  "args": { "query": "결제", "limit": 5 }
}

---

## 6. SetAppTheme

설명  
애플리케이션 테마를 변경합니다.

name  
SetAppTheme

args  
- theme (string, required): light | dark | system

예시  
{
  "name": "SetAppTheme",
  "args": { "theme": "dark" }
}

---

## 7. PlaySound

설명  
지정된 효과음 또는 사운드를 재생합니다.

name  
PlaySound

args  
- soundId (string, required): success | error | click | alert

예시  
{
  "name": "PlaySound",
  "args": { "soundId": "success" }
}

---

## 8. Navigate

설명  
앱 내부 화면 또는 페이지로 이동합니다.

name  
Navigate

args  
- route (string, required)  
- params (object, optional)

예시  
{
  "name": "Navigate",
  "args": {
    "route": "settings",
    "params": { "tab": "account" }
  }
}

---

## 9. ConfirmAction

설명  
사용자 확인이 필요한 경우 확인 다이얼로그를 표시합니다.

name  
ConfirmAction

args  
- message (string, required)  
- confirmText (string, optional)  
- cancelText (string, optional)

예시  
{
  "name": "ConfirmAction",
  "args": {
    "message": "정말 삭제할까요?",
    "confirmText": "삭제",
    "cancelText": "취소"
  }
}

---

## 금지 사항 (절대 준수)

- 이 문서에 없는 함수 이름을 생성하지 않습니다.
- args 구조를 임의로 변경하지 않습니다.
- 실행을 암시하는 행동을 직접 수행하지 않습니다.
- 항상 **“제안” 형태의 JSON**만 생성합니다.
