# commands/registry.py

# 서버가 "실행 가능하다고 인정하는" 명령 목록
ALLOWED_COMMANDS = {
    "OpenUrl": {"args": ["url"]},
    "ShowNotification": {"args": ["message"]},  # title은 optional이면 체크 안 해도 됨
    "CopyToClipboard": {"args": ["text"]},
    "SaveLocalNote": {"args": ["content"]},     # title/tags optional
    "SearchLocalDocs": {"args": ["query"]},     # limit optional
    "SetAppTheme": {"args": ["theme"]},
    "PlaySound": {"args": ["soundId"]},         # ✅ 이 줄 추가
    "Navigate": {"args": ["route"]},            # params optional
    "ConfirmAction": {"args": ["message"]},     # confirmText/cancelText optional
}
