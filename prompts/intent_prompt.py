# prompts/intent_prompt.py
INTENT_PROMPT_TEMPLATE = """
너는 사용자의 입력을 분류하는 분류기다.
사용자 입력이 "설명/질문"이면 explain,
사용자 입력이 "무언가를 실행/조작/행동 요청"이면 command 로 분류해라.

규칙:
- 사용자가 ~해줘/~해봐/~바꿔줘/~열어줘/~재생해줘/~저장해줘 등 "행동"을 요구하면 command
- 사용자가 ~뭐야/~설명해줘/~왜 그래/~원리가 뭐야 등 "정보"를 요구하면 explain
- 애매하면 explain

출력은 JSON 하나만:
{{
  "intent": "command|explain",
  "reason": "짧은 근거"
}}

[USER_INPUT]
{question}
"""
