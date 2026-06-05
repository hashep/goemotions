import streamlit as st
from datasets import load_dataset
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import FAISS
from transformers.pipelines import pipeline

st.set_page_config(
    page_title="감정 일기 챗봇",
    page_icon="📘"
)

st.title("감정 일기 분석 챗봇")
st.write("PDF 일기를 업로드하세요.")


try:
    OPENAI_API_KEY = st.secrets["openai"]["api_key"]
except Exception:
    st.error("Streamlit Secrets에 OpenAI API Key가 없습니다.")
    st.stop()


@st.cache_resource
def load_emotion_classifier():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=1
    )


emotion_classifier = load_emotion_classifier()


@st.cache_data
def load_goemotions_docs(sample_size=200):
    dataset = load_dataset(
        "go_emotions",
        "simplified",
        split="train"
    )

    samples = dataset.shuffle(seed=42).select(range(sample_size))

    docs = []

    for item in samples:
        text = item["text"]
        emotions = item["labels"]

        if emotions:
            docs.append(
                Document(
                    page_content=f"문장: {text}\n감정 라벨 번호: {emotions}"
                )
            )

    return docs


def extract_text_from_pdf(uploaded_file):
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    return " ".join(
        [doc.page_content for doc in documents]
    )


def classify_emotion(text):
    result = emotion_classifier(
        text,
        truncation=True,
        max_length=512
    )

    if isinstance(result, list):
        result = result[0]

    if isinstance(result, list):
        result = result[0]

    label = result["label"]
    score = result["score"]

    return label, score


def create_vectorstore(user_text, emotion_label, goemotion_docs):
    user_doc = Document(
        page_content=(
            f"일기 내용: {user_text}\n"
            f"감정 분석 결과: {emotion_label}"
        )
    )

    all_docs = goemotion_docs + [user_doc]

    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY
    )

    vectorstore = FAISS.from_documents(
        all_docs,
        embeddings
    )

    return vectorstore


def create_chatbot(vectorstore):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    llm = OpenAI(
        temperature=0.7,
        openai_api_key=OPENAI_API_KEY
    )

    chatbot = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return chatbot


uploaded_file = st.file_uploader(
    "PDF 일기 파일 업로드",
    type="pdf"
)


if uploaded_file:
    with st.spinner("PDF에서 텍스트 추출 중..."):
        diary_text = extract_text_from_pdf(uploaded_file)

    st.success("텍스트 추출 완료")

    with st.spinner("감정 분석 중..."):
        label, score = classify_emotion(diary_text)

    st.markdown(
        f"### 감정 결과: **{label}** "
        f"(신뢰도: {score:.2f})"
    )

    with st.expander("추출된 PDF 텍스트 보기"):
        st.write(diary_text)

    with st.spinner("GoEmotions 데이터 로딩 중..."):
        goemotion_docs = load_goemotions_docs()

    with st.spinner("RAG 벡터스토어 생성 중..."):
        vectorstore = create_vectorstore(
            diary_text,
            label,
            goemotion_docs
        )

        chatbot = create_chatbot(vectorstore)

    st.markdown("---")
    st.subheader("질문을 입력해보세요")

    user_input = st.text_input("질문:")

    if user_input:
        with st.spinner("답변 생성 중..."):
            response = chatbot.run(user_input)

        st.write(response)