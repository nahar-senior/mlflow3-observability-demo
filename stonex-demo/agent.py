"""
StoneX Portfolio Intelligence Agent
MLflow 3.0 Observability Demo - Agent Implementation
"""

from typing import Any, Generator, Optional, Sequence, Union

import mlflow
from databricks_langchain import (
    ChatDatabricks,
    VectorSearchRetrieverTool,
    UCFunctionToolkit,
)
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)

# Enable MLflow LangChain autologging for automatic trace capture
mlflow.langchain.autolog()

############################################
# Configuration
############################################

# LLM endpoint name
LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"

# System prompt tailored for wealth management portfolio analysis
system_prompt = """You are an expert portfolio analyst at StoneX Wealth Management. Your role is to help financial advisors and clients understand their portfolios through data-driven insights.

**Your capabilities:**
- Analyze portfolio holdings and calculate risk metrics
- Retrieve real-time market data and earnings intelligence
- Provide actionable recommendations based on fundamentals
- Explain complex financial concepts clearly

**Guidelines:**
- Always use tools to retrieve actual data - never make up numbers or holdings
- For portfolio questions, start by getting holdings data, then enrich with market data
- For risk analysis, use the calculate_portfolio_risk function
- For earnings insights, search the earnings reports
- Be precise with numbers (show decimals for percentages)
- Provide context: compare to benchmarks when relevant
- If data is missing, clearly state what's unavailable

**Tone:** Professional, insightful, and concise. Focus on actionable intelligence."""

###############################################################################
# Initialize Components (with fallback for Model Serving)
###############################################################################

# Module-level variables (for notebook imports)
llm = None
tools = None
_agent_initialized = False

def _initialize_components():
    """
    Initialize LLM and tools.
    Returns (llm, tools) tuple.
    """
    # Initialize LLM
    llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
    
    # Initialize tools
    tools = []
    
    # UC Function Tools
    uc_tool_names = [
        "stonex_demo.portfolio.get_portfolio_summary",
        "stonex_demo.portfolio.get_market_data",
        "stonex_demo.portfolio.calculate_portfolio_risk"
    ]
    uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)
    tools.extend(uc_toolkit.tools)
    
    # Vector Search Tool
    earnings_retriever = VectorSearchRetrieverTool(
        index_name="stonex_demo.portfolio.earnings_reports_index",
        tool_name="search_earnings_reports",
        tool_description="Searches recent earnings reports and financial analysis for publicly traded companies. Use this to get latest earnings results, revenue trends, management guidance, and business insights for specific tickers.",
        num_results=2,
        disable_notice=True
    )
    tools.append(earnings_retriever)
    
    return llm, tools

# Try to initialize eagerly (for notebooks with credentials)
# If it fails (Model Serving), we'll initialize lazily later
try:
    llm, tools = _initialize_components()
    _agent_initialized = True
except Exception:
    # Model Serving context - will initialize on first predict()
    llm = None
    tools = None
    _agent_initialized = False

#####################
# Agent Graph Logic
#####################

def create_tool_calling_agent(
    model: LanguageModelLike,
    tools: Union[Sequence[BaseTool], ToolNode],
    system_prompt: Optional[str] = None,
):
    """
    Creates a LangGraph-based tool-calling agent with MLflow tracing.
    
    The agent follows this flow:
    1. Agent node: LLM decides which tools to call (if any)
    2. Tools node: Execute selected tools
    3. Loop back to agent until final answer is ready
    """
    
    model = model.bind_tools(tools)

    def should_continue(state: ChatAgentState):
        """Routing logic: continue to tools or end"""
        messages = state["messages"]
        last_message = messages[-1]
        
        # If LLM made tool calls, execute them
        if last_message.get("tool_calls"):
            return "continue"
        else:
            return "end"

    # Prepend system prompt to conversation
    if system_prompt:
        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": system_prompt}]
            + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])
    
    model_runnable = preprocessor | model

    def call_model(
        state: ChatAgentState,
        config: RunnableConfig,
    ):
        """Agent decision node - calls LLM with conversation history"""
        response = model_runnable.invoke(state, config)
        return {"messages": [response]}

    # Build the state graph
    workflow = StateGraph(ChatAgentState)

    # Add nodes
    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", ChatAgentToolNode(tools))

    # Define edges
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


class LangGraphChatAgent(ChatAgent):
    """
    Wrapper for LangGraph agent compatible with MLflow's ChatAgent interface.
    Enables deployment to Model Serving with full tracing.
    """
    
    def __init__(self, agent):
        self.agent = agent
    
    def _convert_messages_to_dict(self, messages: list[ChatAgentMessage]) -> list[dict]:
        """Convert ChatAgentMessage objects to dictionaries"""
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        """Synchronous prediction - returns full response"""
        request = {"messages": self._convert_messages_to_dict(messages)}

        messages = []
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                messages.extend(
                    ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
                )
        return ChatAgentResponse(messages=messages)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        """Streaming prediction - yields chunks as they're generated"""
        request = {"messages": self._convert_messages_to_dict(messages)}
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                yield from (
                    ChatAgentChunk(**{"delta": msg}) for msg in node_data["messages"]
                )


###############################################################################
# Agent Creation
###############################################################################

# Create agent if components were initialized successfully
if _agent_initialized and llm is not None and tools is not None:
    agent = create_tool_calling_agent(llm, tools, system_prompt)
    AGENT = LangGraphChatAgent(agent)
    mlflow.models.set_model(AGENT)
    
    print(f"✅ Agent initialized successfully")
    print(f"   - LLM: {LLM_ENDPOINT_NAME}")
    print(f"   - Tools: {len(tools)}")
else:
    AGENT = None
    print("⚠️  Agent initialization deferred (no credentials available yet)")

