from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os

def rag_pipeline(file_path: str):
    load_dotenv()

    # === Init LLM ===
    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

    # === Init Embeddings ===
    def get_embeddings():
        token = os.environ.get("GITHUB_TOKEN")
        endpoint = "https://models.github.ai/inference"
        model_name = "openai/text-embedding-3-large"
        return OpenAIEmbeddings(
            model=model_name,
            api_key=token,
            base_url=endpoint,
        )
    
    embeddings = get_embeddings()

    # === Init Vector Store ===
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )

    # === Load and Split Document ===
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    _ = vector_store.add_documents(documents=all_splits)

    # === Retrieval Tool ===
    @tool(response_format="content_and_artifact")
    def retrieve(query: str):
        """Retrieve information related to a query."""
        retrieved_docs = vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    # === Node 1: Decide whether to retrieve ===
    def query_or_respond(state: MessagesState):
        """Decide whether to call tools or directly respond."""
        llm_with_tools = llm.bind_tools([retrieve])
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    # === Tool Execution Node ===
    tools = ToolNode([retrieve])

    # === Node 2: Generate final response ===
    def generate(state: MessagesState):
        """Generate a response using retrieved content + prior messages."""
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        conversation_messages = [
            message for message in state["messages"]
            if message.type in ("human", "system") or (message.type == "ai" and not message.tool_calls)
        ]

        system_template = (
            "You are an assistant for question-answering tasks. Use the following pieces of retrieved context "
            "to answer the question. If you don't know the answer, say you don't know. Be thorough:\n\n"
            f"{docs_content}"
        )

        prompt = [SystemMessage(system_template)] + conversation_messages
        response = llm.invoke(prompt)
        return {"messages": [response]}

    # === Graph Construction ===
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("query_or_respond", query_or_respond)
    graph_builder.add_node("tools", tools)
    graph_builder.add_node("generate", generate)

    graph_builder.set_entry_point("query_or_respond")

    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    # === Short-Term Memory ===
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    return graph, memory
