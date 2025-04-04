#######################################
# streamlit_app.py
#######################################
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict

#######################################
# 1) 모델 로딩 및 텍스트 생성 함수
#######################################
MODEL_PATH = "/workspace/models/Llama-2-7b-hf"  # 실제 모델 경로로 수정하세요

@st.cache_resource
def load_model():
    """
    모델과 토크나이저를 로딩하여 리턴합니다.
    streamlit에서 @st.cache_resource 데코레이터로 
    매번 재로딩되지 않도록 캐싱합니다.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,   # GPU 사용 시 float16 권장
        device_map="auto"           # GPU 여러 장이 있을 경우 자동 분산
    )
    return tokenizer, model

def hf_generate_text(
    tokenizer,
    model,
    prompt: str,
    temperature: float = 0.7,
    max_new_tokens: int = 500
) -> str:
    """
    Llama 2 HF 모델에서 텍스트를 생성하는 헬퍼 함수.
    (pipeline 대신 model.generate()를 직접 사용한 예시)
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        input_ids,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=0.9
    )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # 프롬프트 제거(옵션)
    prompt_len = len(tokenizer.decode(input_ids[0], skip_special_tokens=True))
    return generated_text[prompt_len:].strip()


#######################################
# 2) 에이전트/회의 로직
#######################################
class AIAgent:
    """
    각 전문가(시장분석가, 프로덕트 매니저 등)를 나타내는 클래스.
    HF 모델(generate_text)을 호출해 답변을 생성.
    """
    def __init__(self, name: str, role: str, temperature: float, personality: str):
        self.name = name
        self.role = role
        self.temperature = temperature
        self.personality = personality
        self.conversation_history: List[Dict] = []

    def generate_response(
        self,
        tokenizer,
        model,
        topic: str,
        other_response: str = "",
        context: str = "",
        round_num: int = 1
    ) -> str:
        """
        round_num == 1: (초기 의견)
        round_num > 1 : (짧은 피드백 및 제안)
        """
        if round_num == 1:
            # 1라운드: 상세 의견
            prompt = f"""
당신은 {self.name}이며, {self.role}입니다.
성격과 말투: {self.personality}

토론 주제: {topic}

분석할 콘텐츠:
{context}

다음 형식으로 의견을 제시해주세요:
1. 현재 상황 분석
2. 기회 요소 발견
3. 해결 방안 제시
4. 구체적 실행 계획
5. 예상되는 도전 과제
"""
        else:
            # 2라운드 이상: 짧은 피드백
            prompt = f"""
당신은 {self.name}이며, {self.role}입니다.
성격과 말투: {self.personality}

이전 대화:
{other_response}

위 내용에 대한 짧은 피드백과 제안을 200자 이내로 제시해주세요.
"""

        try:
            generated_response = hf_generate_text(
                tokenizer=tokenizer,
                model=model,
                prompt=prompt,
                temperature=self.temperature,
                max_new_tokens=600
            ).strip()
            self.conversation_history.append({"role": "assistant", "content": generated_response})
            return generated_response

        except Exception as e:
            return f"응답 생성 중 오류 발생: {str(e)}"


def create_context(transcripts: List[str], video_urls: List[str]) -> str:
    """
    여러 영상에서 추출한 스크립트/URL을 하나의 문자열 컨텍스트로 합치는 예시.
    """
    return "".join([
        f"\n[영상 {i+1}] URL: {url}\n영상 내용 요약:\n{transcript}\n{'-'*50}"
        for i, (transcript, url) in enumerate(zip(transcripts, video_urls))
    ])

def display_message(agent_name: str, message: str):
    """
    Streamlit 환경에서 에이전트별 메시지를 예쁘게 보여주기 위한 유틸.
    """
    style = {
        "시장분석가": {"bg_color": "#E8F4F9", "border_color": "#2196F3", "icon": "📊"},
        "프로덕트 매니저": {"bg_color": "#F3E5F5", "border_color": "#9C27B0", "icon": "💡"},
        "테크리드": {"bg_color": "#E8F5E9", "border_color": "#4CAF50", "icon": "⚙️"},
        "사업전략가": {"bg_color": "#FFF3E0", "border_color": "#FF9800", "icon": "📈"}
    }
    agent_style = style.get(agent_name, {
        "bg_color": "#F5F5F5",
        "border_color": "#9E9E9E",
        "icon": "💭"
    })

    st.markdown(f"""
    <div style="
        background-color: {agent_style['bg_color']};
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid {agent_style['border_color']};
    ">
        <strong>{agent_style['icon']} {agent_name}</strong><br>{message}
    </div>
    """, unsafe_allow_html=True)


def generate_final_summary(
    tokenizer,
    model,
    conversation: List[dict],
    user_prompt: str
) -> str:
    """
    모든 라운드에서 나왔던 전문(에이전트) 의견을 요약하여 최종 제안서 생성.
    """
    # 회의 내용을 간단히 합친 텍스트
    conversation_summary = "\n\n".join([
        f"라운드 {msg['round']} - {msg['agent']}: {msg['response']}"
        for msg in conversation
    ])

    prompt = f"""
주제: {user_prompt}

전문가들의 논의 내용을 바탕으로 다음 형식으로 핵심적인 제안 사항만 간단히 정리해주세요:

1. 프로젝트 개요
2. 핵심 기능
3. 기술 구현 방안
4. 비즈니스 전략
5. 실행 계획
6. 주요 고려사항

전문가 논의 내용:
{conversation_summary}
"""
    try:
        summary = hf_generate_text(tokenizer, model, prompt, temperature=0.7, max_new_tokens=1000)
        return summary.strip()
    except Exception as e:
        return f"최종 요약 생성 중 오류 발생: {str(e)}"


def generate_discussion(
    tokenizer,
    model,
    transcripts: List[str],
    video_urls: List[str],
    user_prompt: str,
    num_rounds: int = 3
) -> tuple:
    """
    - 4명(시장분석가·프로덕트 매니저·테크리드·사업전략가)이 다중 라운드 회의를 진행.
    - 각 라운드에서 번갈아 가며 의견(답변)을 생성.
    - 모든 라운드가 끝나면 최종 요약을 생성.
    """
    # 1) 에이전트(전문가) 생성
    analyst = AIAgent(
        name="시장분석가",
        role="시장 트렌드와 사용자 니즈 분석 전문가",
        temperature=0.7,
        personality="데이터 기반의 객관적인 분석을 제공하며, 시장의 기회와 위험 요소를 파악합니다."
    )
    product_manager = AIAgent(
        name="프로덕트 매니저",
        role="제품 기획 및 전략 수립 전문가",
        temperature=0.8,
        personality="사용자 중심적 사고와 비즈니스 가치를 균형있게 고려합니다."
    )
    tech_lead = AIAgent(
        name="테크리드",
        role="기술 구현 및 아키텍처 설계 전문가",
        temperature=0.7,
        personality="최신 기술 트렌드를 이해하고 실제 구현 가능성을 평가합니다."
    )
    business_strategist = AIAgent(
        name="사업전략가",
        role="비즈니스 모델 및 수익화 전략 전문가",
        temperature=0.8,
        personality="시장성과 수익성을 고려한 사업 전략을 수립합니다."
    )

    # 2) 영상 스크립트 + URL 컨텍스트 만들기
    context = create_context(transcripts, video_urls)
    conversation = []
    agents = [analyst, product_manager, tech_lead, business_strategist]

    # 3) 라운드 진행
    for round_num in range(num_rounds):
        st.markdown(f"### 🔄 라운드 {round_num + 1}")
        # 4명의 전문가가 차례로 의견
        for agent in agents:
            with st.spinner(f'{agent.name}의 의견을 분석 중...'):
                # 바로 이전 라운드까지의 타 전문가 발언들(최대 4개)만 참조 예시
                other_responses = "\n\n".join([
                    f"{msg['agent']}: {msg['response']}"
                    for msg in conversation[-4:] if msg['agent'] != agent.name
                ])

                response = agent.generate_response(
                    tokenizer=tokenizer,
                    model=model,
                    topic=user_prompt,
                    other_response=other_responses,
                    context=context,
                    round_num=round_num + 1
                )
                conversation.append({
                    "agent": agent.name,
                    "response": response,
                    "round": round_num + 1
                })
                display_message(agent.name, response)

        st.markdown(f"""
        <div style="padding: 10px; margin: 20px 0; text-align: center; background-color: #f0f2f6; border-radius: 10px;">
            ✨ 라운드 {round_num + 1} 완료
        </div>
        """, unsafe_allow_html=True)

    # 4) 최종 요약
    final_summary = generate_final_summary(tokenizer, model, conversation, user_prompt)
    return final_summary, conversation


#######################################
# 3) Streamlit 메인 실행 함수
#######################################
def main():
    st.title("Llama 2 기반 다중 전문가 토론 데모")
    st.write("4명의 전문가(시장분석가, 프로덕트 매니저, 테크리드, 사업전략가)가 라운드별로 의견을 제시하고, 마지막에 종합 요약을 생성합니다.")

    # (1) 사용자 입력: 주제
    user_prompt = st.text_input("토론하고 싶은 주제를 입력하세요", value="AI 스타트업 아이디어")

    # (2) 예시 영상 스크립트 입력 (실제로는 파일 업로드나 API 결과일 수 있음)
    st.write("영상 스크립트 및 URL을 예시로 입력해 주세요.")
    video1_url = st.text_input("영상 1 URL", "https://youtube.com/example1")
    video1_script = st.text_area("영상 1 스크립트 요약", "AI 시장 현황에 대한 내용...")

    video2_url = st.text_input("영상 2 URL", "https://youtube.com/example2")
    video2_script = st.text_area("영상 2 스크립트 요약", "딥러닝 모델 서비스화 전략...")

    # 필요한 경우 추가 영상도 받을 수 있음
    transcripts = [video1_script, video2_script]
    video_urls = [video1_url, video2_url]

    # (3) 라운드 수 설정
    num_rounds = st.slider("라운드 수", 1, 5, 3)

    # (4) 버튼 클릭 시 토론 실행
    if st.button("토론 시작"):
        # 모델 로드
        with st.spinner("모델 로딩 중... (처음 한 번만 오래 걸립니다)"):
            tokenizer, model = load_model()

        final_summary, conversation = generate_discussion(
            tokenizer,
            model,
            transcripts=transcripts,
            video_urls=video_urls,
            user_prompt=user_prompt,
            num_rounds=num_rounds
        )

        # 최종 요약 결과 표시
        st.subheader("✅ 최종 제안 요약")
        st.write(final_summary)


if __name__ == "__main__":
    main()
