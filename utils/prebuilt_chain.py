from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def history_aware_retriever(llm, retriever):
    """
    Create a chain that takes conversation history and returns documents.
    If there is no chat_history, then the input is just passed directly to the retriever.
    If there is chat_history, then the prompt and LLM will be used to generate a search query.
    That search query is then passed to the retriever.

    Args:
    llm: The language model.
    retriever: The retriever to use for finding relevant documents.
    """
    contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
       llm, retriever, contextualize_q_prompt
    )

    return history_aware_retriever

def documents_retriever(llm):
    """
    Create a chain for passing a list of Documents to a model.

    Args:
    llm: The language model.
    """
    system_prompt = (
    "You are an helpfull assistant. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer or the context is not retrieved, SAY THAT YOU DON'T KNOW!!. "
    "Always response in Bahasa Indonesia or Indonesian Language. "
    "Context: {context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    return question_answer_chain