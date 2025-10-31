# Databricks notebook source
# MAGIC %md
# MAGIC # StoneX Portfolio Agent - MLflow Tracing
# MAGIC
# MAGIC **What This Agent Does:**
# MAGIC
# MAGIC A financial portfolio intelligence agent that helps wealth managers analyze client portfolios, assess risk, retrieve market data, and search earnings reports. Powered by Claude Sonnet 4 with 4 specialized tools (3 UC Functions + 1 Vector Search).
# MAGIC
# MAGIC This notebook demonstrates MLflow 3.0 automatic tracing - capturing agent reasoning, tool calls, and LLM interactions with zero instrumentation code.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Architecture Overview
# MAGIC
# MAGIC ### Supervisor Agent Pattern
# MAGIC
# MAGIC ```
# MAGIC                              ┌─────────────────────────────────┐
# MAGIC                              │   User Query                    │
# MAGIC                              │   "Analyze C001's portfolio"    │
# MAGIC                              └──────────────┬──────────────────┘
# MAGIC                                             │
# MAGIC                                             ▼
# MAGIC                  ┌───────────────────────────────────────────────────────┐
# MAGIC                  │         LangGraph Supervisor Agent                    │
# MAGIC                  │         LLM: Claude Sonnet 4                          │
# MAGIC                  │         (databricks-claude-3-7-sonnet)                │
# MAGIC                  └───────────────────────┬───────────────────────────────┘
# MAGIC                                          │
# MAGIC                        Orchestrates Tool Calls (Independent & Parallel)
# MAGIC                                          │
# MAGIC           ┌──────────────────┬───────────┴───────────┬──────────────────┐
# MAGIC           │                  │                       │                  │
# MAGIC           ▼                  ▼                       ▼                  ▼
# MAGIC ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
# MAGIC │ get_portfolio_   │ │ calculate_       │ │ get_market_data()│ │ search_earnings_ │
# MAGIC │ summary()        │ │ portfolio_risk() │ │                  │ │ reports()        │
# MAGIC │                  │ │                  │ │                  │ │                  │
# MAGIC │ UC Function      │ │ UC Function      │ │ UC Function      │ │ Vector Search    │
# MAGIC │                  │ │                  │ │                  │ │ (RAG)            │
# MAGIC │ Returns client's │ │ Computes         │ │ Retrieves current│ │                  │
# MAGIC │ stock holdings,  │ │ volatility,      │ │ stock prices,    │ │ Semantic search  │
# MAGIC │ shares, cost     │ │ Sharpe ratio,    │ │ P/E ratio,       │ │ over earnings    │
# MAGIC │ basis            │ │ risk metrics     │ │ market cap       │ │ reports using    │
# MAGIC │                  │ │                  │ │                  │ │ databricks-gte-  │
# MAGIC │                  │ │                  │ │                  │ │ large-en         │
# MAGIC │                  │ │                  │ │                  │ │                  │
# MAGIC │ Data Source:     │ │ Data Source:     │ │ Data Source:     │ │ Data Source:     │
# MAGIC │ portfolio_       │ │ portfolio_       │ │ market_data      │ │ earnings_reports │
# MAGIC │ holdings table   │ │ holdings +       │ │ table            │ │ Vector Search    │
# MAGIC │                  │ │ market_data      │ │                  │ │ Index            │
# MAGIC └──────────────────┘ └──────────────────┘ └──────────────────┘ └──────────────────┘
# MAGIC
# MAGIC Note: All 4 tools are INDEPENDENT - the agent can call any combination based on the query.
# MAGIC       Vector Search does NOT depend on UC Functions.
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install --quiet mlflow[databricks]>=2.16.0 langgraph==0.2.34 langchain-community databricks-langchain
# MAGIC %restart_python

# COMMAND ----------

import mlflow
from agent import AGENT, tools, LLM_ENDPOINT_NAME
from mlflow.types.agent import ChatAgentMessage

print(f"✅ Agent loaded with {len(tools)} tools")
print(f"✅ LLM: {LLM_ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agent Tools
# MAGIC
# MAGIC The agent has 4 tools for portfolio intelligence:

# COMMAND ----------

for i, tool in enumerate(tools, 1):
    print(f"{i}. {tool.name}: {tool.description}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scenario 1: Simple Portfolio Query
# MAGIC
# MAGIC Single tool call to retrieve client holdings.

# COMMAND ----------

with mlflow.start_run(run_name="Scenario_1_Simple_Portfolio"):
    query = "What stocks does client C001 own?"
    messages = [ChatAgentMessage(role="user", content=query)]
    response = AGENT.predict(messages=messages)
    
    print("🤖 Agent Response:")
    print(response.messages[-1].content)
    
    mlflow.log_param("query", query)
    mlflow.log_param("scenario", "simple_portfolio_query")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scenario 2: Portfolio Risk Analysis
# MAGIC
# MAGIC Multi-tool orchestration: portfolio summary + risk calculation + market data.

# COMMAND ----------

with mlflow.start_run(run_name="Scenario_2_Risk_Analysis"):
    query = "Analyze the risk profile for client C002's portfolio. Include current market prices for their top 3 holdings."
    messages = [ChatAgentMessage(role="user", content=query)]
    response = AGENT.predict(messages=messages)
    
    print("🤖 Agent Response:")
    print(response.messages[-1].content)
    
    mlflow.log_param("query", query)
    mlflow.log_param("scenario", "multi_tool_risk_analysis")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scenario 3: Earnings Intelligence
# MAGIC
# MAGIC Vector search to retrieve earnings report highlights.

# COMMAND ----------

with mlflow.start_run(run_name="Scenario_3_Earnings_Intelligence"):
    query = "What were NVIDIA's latest earnings highlights? Focus on revenue growth and AI segment."
    messages = [ChatAgentMessage(role="user", content=query)]
    response = AGENT.predict(messages=messages)
    
    print("🤖 Agent Response:")
    print(response.messages[-1].content)
    
    mlflow.log_param("query", query)
    mlflow.log_param("scenario", "earnings_vector_search")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scenario 4: Complex Analysis
# MAGIC
# MAGIC Combines portfolio, risk, and earnings data for comprehensive analysis.

# COMMAND ----------

with mlflow.start_run(run_name="Scenario_4_Complex_Analysis"):
    query = """Should client C001 be concerned about their technology sector concentration? 
    Analyze their portfolio risk and compare it to recent earnings trends for their tech holdings."""
    messages = [ChatAgentMessage(role="user", content=query)]
    response = AGENT.predict(messages=messages)
    
    print("🤖 Agent Response:")
    print(response.messages[-1].content)
    
    mlflow.log_param("query", query)
    mlflow.log_param("scenario", "complex_orchestration")

# COMMAND ----------

# MAGIC %md
# MAGIC ## View Traces in MLflow UI
# MAGIC
# MAGIC Navigate to the MLflow UI to view detailed traces:
# MAGIC - **Trace timeline:** Agent → Tools → LLM flow
# MAGIC - **Span details:** Inputs, outputs, latencies, token counts
# MAGIC - **Tool orchestration:** See which tools were called and in what order
# MAGIC - **Compare runs:** Select multiple scenarios and compare metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ## Human Feedback
# MAGIC
# MAGIC Add feedback to traces via MLflow UI:
# MAGIC 1. Navigate to any run in the Experiments tab
# MAGIC 2. Click "Add Feedback" button
# MAGIC 3. Add thumbs up/down and comments
# MAGIC 4. Feedback is stored with traces for continuous improvement

# COMMAND ----------

# Display summary
runs_df = mlflow.search_runs(
    filter_string="params.scenario != ''",
    order_by=["start_time DESC"],
    max_results=10
)

if len(runs_df) > 0:
    print("=" * 70)
    print("TRACING DEMO SUMMARY")
    print("=" * 70)
    
    summary = runs_df[["run_name", "params.scenario"]].head(4)
    summary.columns = ["Run Name", "Scenario"]
    
    print("\n📊 Traces Captured:")
    for idx, row in summary.iterrows():
        print(f"  • {row['Run Name']}: {row['Scenario']}")
    
    print(f"\n✅ {len(summary)} traces available in MLflow UI")
    print("=" * 70)
else:
    print("No runs found. Execute scenario cells above first.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC **Run `02_assessments.py`** to see:
# MAGIC - Evaluation with 10 test queries
# MAGIC - 8 LLM judges (2 built-in + 6 custom)
# MAGIC - Quality assessments in MLflow UI
