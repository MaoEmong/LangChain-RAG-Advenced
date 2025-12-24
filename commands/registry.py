"""
commands/registry.py
============================================================
명령 화이트리스트 레지스트리

이 모듈은 서버가 실행을 허용하는 명령 목록을 정의합니다.
화이트리스트 방식으로 보안을 강화합니다.

새로운 명령 추가 방법:
1. 이 딕셔너리에 명령 이름과 필수 인자 목록 추가
2. services/command_validator.py가 자동으로 검증
"""

# ============================================================
# 허용된 명령 목록 (화이트리스트)
# ============================================================
# 서버가 "실행 가능하다고 인정하는" 명령 목록
# 
# 구조:
#   "명령이름": {
#       "args": ["필수인자1", "필수인자2", ...]
#   }
#
# Note:
#   - args는 필수 인자만 포함 (optional 인자는 제외)
#   - 새로운 명령 추가 시 여기에 등록해야 함
#   - 등록되지 않은 명령은 자동으로 차단됨

ALLOWED_COMMANDS = {
    # URL 열기
    "OpenUrl": {"args": ["url"]},
    
    # 알림 표시 (title은 optional이므로 체크 안 함)
    "ShowNotification": {"args": ["message"]},
    
    # 클립보드에 복사
    "CopyToClipboard": {"args": ["text"]},
    
    # 로컬 노트 저장 (title/tags는 optional)
    "SaveLocalNote": {"args": ["content"]},
    
    # 로컬 문서 검색 (limit는 optional)
    "SearchLocalDocs": {"args": ["query"]},
    
    # 앱 테마 변경
    "SetAppTheme": {"args": ["theme"]},
    
    # 사운드 재생
    "PlaySound": {"args": ["soundId"]},
    
    # 네비게이션 (params는 optional)
    "Navigate": {"args": ["route"]},
    
    # 확인 다이얼로그 (confirmText/cancelText는 optional)
    "ConfirmAction": {"args": ["message"]},
}
