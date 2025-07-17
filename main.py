from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, List, Union
from langchain.tools import tool
from langchain.vectorstores import FAISS
import os
from langchain.embeddings import OpenAIEmbeddings
import wikipedia
from tavily import TavilyClient

os.environ["TAVILY_API_KEY"] = "tvly-dev-............"

embeddings = OpenAIEmbeddings(openai_api_key="sk-proj-...........")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "lsv2_............."
os.environ["LANGCHAIN_PROJECT"] = "checking"

import os
os.environ["OPENAI_API_KEY"] = "sk-proj-............."

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage, SystemMessage, ToolMessage]]


def load_retriever():
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        folder_path=".",                   
        embeddings=embeddings,
        index_name="index",               
        allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(search_kwargs={"k": 2})

# ✅ Load the retriever
retriever = load_retriever()


@tool
def retriever_tool(query:str)->str:
  """this tool searches and returns the information from the mamaearth FAQs."""
  docs=retriever.invoke(query)

  results=[]
  for i,doc in enumerate(docs):
    results.append(f"documents {i+1}: \n{doc.page_content}")
  return "\n\n".join(results)



from tavily import TavilyClient

tavily = TavilyClient(api_key="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")

@tool
def tavily_search_tool(query: str) -> str:
    """Use Tavily Web Search to answer general queries when FAQ doesn't help."""
    try:
        results = tavily.search(query=query, max_results=3)
        snippets = [res['content'] for res in results['results']]
        return "\n\n".join(snippets)
    except Exception as e:
        return f"Error during web search: {e}"


tools=[retriever_tool,tavily_search_tool]
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7).bind_tools(tools)

def model_node(state: AgentState) -> AgentState:
    """Call the LLM with tools."""
    response = llm.invoke(state["messages"])
    state["messages"].append(response)
    return state

def router(state: AgentState) -> str:
    """Decide whether to call tools or end."""
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "call_tool"
    return END

def tool_node_wrapper(state: AgentState) -> AgentState:
    """Execute tools and add results to state."""
    last_msg = state["messages"][-1]
    tool_calls = last_msg.tool_calls


    # Execute each tool call and create ToolMessage for each
    for tool_call in tool_calls:
        tool_name = tool_call['name']
        tool_input = tool_call['args']
        print(f"Tool: {tool_name}")
        print(f"Tool_input: {tool_input}")

        # Find the tool function
        tool_func = next(tool for tool in tools if tool.name == tool_name)

        # Execute the tool
        result = tool_func.invoke(tool_input)

        print("\n📄 Retrieved Chunks Sent to LLM:\n")
        print(result)

        # Add ToolMessage to the state
        state["messages"].append(
            ToolMessage(
                content=str(result),
                name=tool_name,
                tool_call_id=tool_call['id']
            )
        )

    return state



workflow = StateGraph(AgentState)
workflow.add_node("model", model_node)
workflow.add_node("call_tool", tool_node_wrapper)
workflow.set_entry_point("model")
workflow.add_conditional_edges("model", router)
workflow.add_edge("call_tool", "model")
app = workflow.compile()


def interactive_conversation():
    messages = [
        SystemMessage(content="""You are a professional customer support assistant for Mamaearth, a personal care and beauty brand.  Use the context below to
        answer the customer's question completely and accurately. Always include all important steps  or instructions from the context. Always use the `retriever_tool`
        first to answer any customer queries if possible If the answer is not found in the context, or if the user asks something general, you may
        also use the Wikipedia tool to fetch relevant information. If no relevant information is found, say: "Sorry, I couldn't find that information." .

.
""")
    ]

    print("Starting conversation. Type 'exit', 'quit', or 'bye' to end.")

    while True:
        try:
            user_input = input("User: ").strip()
            if user_input.lower() in ['exit', 'quit', 'bye']:
                break
            if not user_input:
                continue

            messages.append(SystemMessage(content="""When answering factual questions about Mamaearth, always use retriever_tool first to find answers from the
      Mamaearth FAQ. If retriever_tool does not find any relevant information, then try to answer using Wikipedia. Use Wikipedia only as a fallback option
      when Mamaearth FAQ does not contain the answer.
"""))
            messages.append(HumanMessage(content=user_input))

            state = {"messages": messages}
            final_state = app.invoke(state)
            messages = final_state["messages"]

            # Print only the last AI response
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    print(f"AI: {msg.content}")
                    break

        except Exception as e:
            print(f"Error occurred: {e}")
            messages.append(AIMessage(content="Sorry, I encountered an error. Please try again."))

if __name__ == "__main__":
  interactive_conversation()
