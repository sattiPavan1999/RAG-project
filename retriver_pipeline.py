from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()


# -------------------------------
# Load Vector Store
# -------------------------------
def load_vector_store(persist_directory="db/chroma_db"):
    """Load a persisted Chroma vector store."""

    # Use same embedding model as ingestion
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
    )

    print("✅ Loaded Vector Store")
    print(f"🔢 Embeddings in DB: {vector_store._collection.count()}")
    return vector_store


# -------------------------------
# Create Retriever
# -------------------------------
def create_retriever(vector_store, k=3):
    """Convert vector store into retriever."""

    retriever = vector_store.as_retriever(
        search_kwargs={"k": k}
    )

    print("🔍 Retriever ready")
    return retriever


# -------------------------------
# Ask Query
# -------------------------------
def ask_question(retriever, question: str):
    """Retrieve and print relevant documents."""

    print(f"\n🧠 User query: {question}\n")

    # Correct method
    relevant_docs = retriever.invoke(question)

    return relevant_docs


def get_good_answer_from_llm(relevant_docs, question):
    model = ChatOpenAI(model="gpt-4o")

    query_template = f"""
        ## 📝 Question and Context Analysis

        ### ❓ User Query:
        {question}

        ### 📄 Relevant Information (Context):
        {relevant_docs}

        ---

        ## 🎯 Instructions for Response Generation

        1.  **Extract the Answer:** You **must** attempt to answer the **User Query** using **ONLY** the information provided in the **Relevant Information (Context)** section.
        2.  **Strict Constraint:** If, after careful review, you cannot find the answer within the provided context, **DO NOT** use external knowledge or your own information.
        3.  **No Answer Found:** In the event that the answer is not present in the context, your entire response must be the exact phrase: **'answer not found'**
    """

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=query_template)
    ]

    result = model.invoke(messages)
    print("-----Generated Message-----")
    print(result.content)


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    vector_store = load_vector_store()
    retriever = create_retriever(vector_store)
    question = "when was thoughtworks founded?"
    relevant_docs = ask_question(retriever, question)
    get_good_answer_from_llm(relevant_docs, question)
