import os
import base64
import mimetypes
from openai import AzureOpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

endpoint = os.getenv("AZURE_OAI_ENDPOINT")
deployment = os.getenv("AZURE_OAI_DEPLOYMENT")
api_key = os.getenv("AZURE_OAI_KEY")

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version="2024-02-15-preview",
)

SYSTEM_MESSAGE = {
    "role": "system",
    "content": "사용자가 이미지를 이해하는 데 도움이 되는 친절한 AI 도우미입니다."
}

def image_to_base64(uploaded_file):
    return base64.b64encode(uploaded_file.getvalue()).decode("utf-8")

def make_history_text(history):
    text = ""
    for i, item in enumerate(history, start=1):
        text += f"[대화 {i}]\n"
        text += f"사용자: {item['user']}\n"
        text += f"AI: {item['assistant']}\n\n"
    return text.strip()

def analyze_image(uploaded_file, user_prompt, temperature, top_p, max_tokens):
    if uploaded_file is None:
        return "이미지를 업로드해 주세요."

    if not user_prompt or user_prompt.strip() == "":
        user_prompt = "이 이미지를 설명해줘."

    encoded_image = image_to_base64(uploaded_file)

    mime_type = uploaded_file.type
    if mime_type is None:
        mime_type, _ = mimetypes.guess_type(uploaded_file.name)

    if mime_type is None:
        mime_type = "image/png"

    messages = [SYSTEM_MESSAGE]

    for item in st.session_state.history[-3:]:
        messages.append({
            "role": "user",
            "content": item["user"]
        })
        messages.append({
            "role": "assistant",
            "content": item["assistant"]
        })

    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": user_prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{encoded_image}"
                }
            }
        ]
    })

    completion = client.chat.completions.create(
        model=deployment,
        messages=messages,
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=float(top_p)
    )

    answer = completion.choices[0].message.content

    st.session_state.history.append({
        "user": user_prompt,
        "assistant": answer
    })

    st.session_state.history = st.session_state.history[-3:]

    return answer


st.set_page_config(
    page_title="Azure OpenAI 이미지 분석 챗봇",
    page_icon="🖼️",
    layout="wide"
)

if "history" not in st.session_state:
    st.session_state.history = []

st.title("Azure OpenAI 이미지 분석 챗봇")
st.write("이미지를 업로드하고 질문하면 최근 대화 3턴을 기억합니다.")

with st.sidebar:
    st.header("매개변수 설정")
    temperature = st.slider("temperature", 0.0, 1.0, 0.7, 0.1)
    top_p = st.slider("top_p", 0.0, 1.0, 0.95, 0.05)
    max_tokens = st.slider("max_tokens", 100, 2000, 1000, 100)

    if st.button("대화 초기화"):
        st.session_state.history = []
        st.rerun()

uploaded_file = st.file_uploader(
    "이미지 업로드",
    type=["png", "jpg", "jpeg", "webp"]
)

user_prompt = st.text_input(
    "질문",
    value="이 이미지를 설명해줘.",
    placeholder="예: 이 이미지 설명해줘 / 텍스트 읽어줘 / 문제점 찾아줘"
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="업로드한 이미지", use_container_width=True)

if st.button("분석하기"):
    with st.spinner("이미지 분석 중..."):
        answer = analyze_image(
            uploaded_file,
            user_prompt,
            temperature,
            top_p,
            max_tokens
        )

    st.subheader("AI 답변")
    st.write(answer)

st.subheader("최근 대화 3턴")
if st.session_state.history:
    st.text_area(
        "대화 기록",
        value=make_history_text(st.session_state.history),
        height=300
    )
else:
    st.info("아직 대화 기록이 없습니다.")
