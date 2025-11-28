from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from retriver_pipeline import load_vector_store, create_retriever, get_good_answer_from_llm

load_dotenv()
model = ChatOpenAI(model="gpt-4o")
vector_store = load_vector_store()
retriver = create_retriever(vector_store)
chat_history = []

def get_response_from_llm(question, related_docs):
    result = get_good_answer_from_llm(related_docs, question)
    chat_history.append(HumanMessage(f"question: {question}"))
    chat_history.append(SystemMessage(f"answer: {result}"))

def remember_the_history(question):
    if chat_history:
        generate_question_prompt = f"""
        Instruction: Create a standalone and meaningful question to make conversations by matching the New Question with the Chat History.

        Chat History: {chat_history}
        New Question: {question}
        """

        standalone = model.invoke(generate_question_prompt).content

        if standalone.strip() == "The question is Not related to this chat":
            final_query = question   # direct
        else:
            final_query = standalone # reformulated
    else:
        final_query = question
    print(final_query)

    # now fetch relevant docs
    related_docs = retriver.invoke(final_query)

    # Now generate an answer
    get_response_from_llm(question, related_docs)


def ask_questions():
    while True:
        question = input("Ask Question: ")
        if question.lower() in ["exit", "close", "stop"]:
            print("Ok Bye")
            break
        remember_the_history(question)



if __name__ == "__main__":
    ask_questions()