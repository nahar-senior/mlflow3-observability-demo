# Databricks notebook source
# MAGIC %md
# MAGIC # StoneX Portfolio Agent - MLflow 3.0 Evaluations
# MAGIC
# MAGIC **What This Notebook Does:**
# MAGIC
# MAGIC Evaluates the portfolio agent's quality using MLflow 3.0 LLM Judges. We run 10 test queries through the agent and assess responses across 8 quality dimensions using a combination of built-in and custom judges.

# COMMAND ----------

# MAGIC %md
# MAGIC ## LLM Judges: Quality Assessment Framework
# MAGIC
# MAGIC **What are LLM Judges?**
# MAGIC
# MAGIC LLM Judges use Large Language Models to evaluate agent responses against quality criteria. They provide pass/fail assessments with detailed rationale for each test query.
# MAGIC
# MAGIC ### 8 Judges Used in This Demo
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │                    BUILT-IN JUDGES (2)                      │
# MAGIC │        Pre-configured by Databricks - No setup needed       │
# MAGIC ├─────────────────────────────────────────────────────────────┤
# MAGIC │  1. RelevanceToQuery     Does response address the question?│
# MAGIC │  2. Safety               Is content safe and appropriate?   │
# MAGIC └─────────────────────────────────────────────────────────────┘
# MAGIC
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │                 CUSTOM GUIDELINES JUDGES (6)                │
# MAGIC │         Domain-specific criteria in natural language        │
# MAGIC ├─────────────────────────────────────────────────────────────┤
# MAGIC │  3. Tool Usage           Did agent call appropriate tools?  │
# MAGIC │  4. Data Quality         Includes relevant financial data?  │
# MAGIC │  5. Professional Tone    Maintains wealth mgmt standards?   │
# MAGIC │  6. Regulatory Compliance Follows financial regulations?    │
# MAGIC │  7. Accuracy             No hallucinations or fake data?    │
# MAGIC │  8. Completeness         Adequately answers the question?   │
# MAGIC └─────────────────────────────────────────────────────────────┘
# MAGIC
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │         HARD REQUIREMENTS (Strict Binary Checks) (3)        │
# MAGIC │              Production deployment quality gates            │
# MAGIC ├─────────────────────────────────────────────────────────────┤
# MAGIC │  9. Non-Empty Response   Must return substantive content    │
# MAGIC │ 10. Minimum Length       Must provide adequate detail       │
# MAGIC │ 11. No Placeholders      Must not contain template text     │
# MAGIC └─────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ### Why This Combination?
# MAGIC
# MAGIC | Judge Type | Purpose | Strictness |
# MAGIC |------------|---------|------------|
# MAGIC | **Built-in** | Universal quality checks | Standard |
# MAGIC | **Custom Guidelines (6)** | Domain-specific requirements | Flexible |
# MAGIC | **Hard Requirements (3)** | Production quality gates | Strict binary pass/fail |
# MAGIC
# MAGIC **Guidelines** = Quality scoring for improvement (allows nuance)  
# MAGIC **Hard Requirements** = Deployment gates (strict binary, no partial credit)

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade "mlflow[databricks]>=3.4.0" langgraph==0.2.34 databricks-langchain
# MAGIC %restart_python

# COMMAND ----------

import mlflow
import pandas as pd
from agent import AGENT
from mlflow.types.agent import ChatAgentMessage
from mlflow.genai.scorers import (
    RelevanceToQuery,
    Safety,
    Guidelines,
)

mlflow.set_experiment("/Shared/stonex_portfolio_eval")

print("✅ MLflow Experiment: /Shared/stonex_portfolio_eval")
print("✅ Using MLflow 3.0+ evaluation harness")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation Dataset
# MAGIC
# MAGIC Create test cases with inputs and expected guidelines:

# COMMAND ----------

# Define evaluation dataset - 10 realistic portfolio/financial queries
eval_dataset = [
    {
        "inputs": {"query": "What stocks does client C001 own?"},
    },
    {
        "inputs": {"query": "What is the current price of NVDA?"},
    },
    {
        "inputs": {"query": "Show me Apple's latest earnings highlights"},
    },
    {
        "inputs": {"query": "What is the risk level for client C002's portfolio?"},
    },
    {
        "inputs": {"query": "Get me the market data for Tesla stock"},
    },
    {
        "inputs": {"query": "What are the holdings for client C003?"},
    },
    {
        "inputs": {"query": "What was Microsoft's revenue in Q1 2024?"},
    },
    {
        "inputs": {"query": "Should I invest all my money in cryptocurrency right now?"},
    },
    {
        "inputs": {"query": "Compare the portfolio allocation for clients C001 and C002"},
    },
    {
        "inputs": {"query": "What is Amazon's current stock price and P/E ratio?"},
    },
]

eval_df = pd.DataFrame(eval_dataset)
print(f"📊 Evaluation dataset: {len(eval_df)} test cases")
print("\nTest queries:")
for i, row in enumerate(eval_df['inputs'], 1):
    print(f"  {i}. {row['query']}")
print(f"\n✅ Mix of portfolio queries, market data, earnings, and edge cases (risky advice)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Predict Function
# MAGIC
# MAGIC Wrapper to execute the agent for each query:

# COMMAND ----------

def portfolio_agent_predict(query):
    """
    Execute the portfolio agent and return response.
    Each execution creates a trace automatically.
    
    Args:
        query: The user's question (passed directly from inputs dict)
    """
    # Execute agent - creates trace via mlflow.langchain.autolog()
    messages = [ChatAgentMessage(role="user", content=query)]
    response = AGENT.predict(messages=messages)
    
    # Extract final answer
    final_answer = response.messages[-1].content if response.messages else ""
    
    return final_answer

# Test the predict function
test_result = portfolio_agent_predict("What stocks does client C001 own?")
print("🧪 Test query result:")
print(test_result[:200] + "..." if len(test_result) > 200 else test_result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Evaluation with All Judges
# MAGIC
# MAGIC Single evaluation with built-in + custom judges (all pass/fail, no nulls):

# COMMAND ----------

# Run evaluation with ALL judges (8 total)
results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=portfolio_agent_predict,
    scorers=[
        # ===== BUILT-IN JUDGES (2) =====
        RelevanceToQuery(),           # Does response address the query?
        Safety(),                     # Is response safe and appropriate?
        
        # ===== CUSTOM GUIDELINES (6) =====
        
        # 1. Tool Usage - Check if agent used appropriate tools
        Guidelines(
            name="tool_usage",
            guidelines="""The response should demonstrate appropriate tool usage:
            - For portfolio queries: should call portfolio tools (get_portfolio_summary)
            - For market data: should call market data tools (get_market_data)
            - For earnings: should call earnings tools (search_earnings_reports)
            - Should provide data-backed responses when tools are available
            """
        ),
        
        # 2. Data Quality - Check for specific data (not overly strict)
        Guidelines(
            name="data_quality",
            guidelines="""The response should include relevant financial data when available:
            - Should include ticker symbols when discussing stocks (e.g., AAPL, NVDA, TSLA)
            - Should provide specific numbers when data is retrieved (prices, shares, percentages)
            - Should reference actual data points from the portfolio or market
            - Acceptable to say "data not available" if tools don't return results
            """
        ),
        
        # 3. Professional Tone - Appropriate for wealth management
        Guidelines(
            name="professional_tone",
            guidelines="""The response should maintain professional wealth management standards:
            - Use clear, professional language appropriate for financial services
            - Present information objectively without hype or sensationalism
            - Structure responses logically with clear information hierarchy
            - Maintain a helpful, advisory tone
            """
        ),
        
        # 4. Regulatory Compliance - Critical for financial services
        Guidelines(
            name="regulatory_compliance",
            guidelines="""The response must follow financial advisory regulations:
            - Must NOT provide specific buy/sell recommendations without appropriate disclaimers
            - Must NOT guarantee future returns or predict stock performance with certainty
            - Must NOT use phrases like "definitely will go up" or "guaranteed profit"
            - Should provide factual information based on available data
            - For speculative investment questions, should emphasize risk and need for professional advice
            """
        ),
        
        # 5. Accuracy - No hallucinations
        Guidelines(
            name="accuracy",
            guidelines="""The response should be factually accurate:
            - Should not invent data that wasn't retrieved from tools
            - Should not make up client portfolios or holdings
            - Should acknowledge when specific data is not available
            - Should base all factual claims on retrieved information
            """
        ),
        
        # 6. Completeness - Answers the question
        Guidelines(
            name="completeness",
            guidelines="""The response should adequately address the user's question:
            - Should directly answer what was asked
            - Should provide key information relevant to the query
            - For portfolio queries: include main holdings or summary
            - For market queries: include current price or key metrics
            - For earnings queries: provide highlights or key figures
            - Acceptable to say "not available" if data truly isn't accessible
            """
        ),
        
        # ===== HARD REQUIREMENTS (Binary Pass/Fail Expectations) =====
        
        # 7. Non-Empty Response - MUST return content (strict binary check)
        Guidelines(
            name="non_empty_response",
            guidelines="""HARD REQUIREMENT - Response MUST pass this check:
            - Response must NOT be empty or blank
            - Response must contain actual content (not just whitespace)
            - Response must be more than just a greeting or acknowledgment
            - This is a binary pass/fail - no partial credit
            FAIL if response is empty, blank, or only contains minimal acknowledgment like "OK" or "Hello"
            """
        ),
        
        # 8. Minimum Length - MUST provide adequate detail (strict binary check)
        Guidelines(
            name="minimum_length",
            guidelines="""HARD REQUIREMENT - Response MUST pass this check:
            - Response must contain at least 50 characters
            - Response must include complete sentences
            - Single-word or very short responses are NOT acceptable
            - This is a binary pass/fail - no partial credit
            FAIL if response has fewer than 50 characters or lacks complete sentences
            """
        ),
        
        # 9. No Placeholders - MUST be production-ready (strict binary check)
        Guidelines(
            name="no_placeholders",
            guidelines="""HARD REQUIREMENT - Response MUST pass this check:
            - Response must NOT contain placeholder text like [INSERT], [TODO], or <PLACEHOLDER>
            - Response must NOT contain template variables like {variable}, ${var}, or {{var}}
            - Response must NOT contain "XXXX", "____", or similar placeholder patterns
            - All values must be actual data or clear natural language statements
            - This is a binary pass/fail - no partial credit
            FAIL if any placeholder text or template variables are present
            """
        ),
    ],
)

print("✅ Evaluation complete!")
print(f"Run ID: {results.run_id}")
print(f"\n📊 Judges used (11 total):")
print("   • Built-in (2): RelevanceToQuery, Safety")
print("   • Custom Guidelines (6): tool_usage, data_quality, professional_tone, regulatory_compliance, accuracy, completeness")
print("   • Hard Requirements (3): non_empty_response, minimum_length, no_placeholders")
print("\n💡 Hard Requirements are strict binary pass/fail checks - no partial credit")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 👀 Demo: View Results in MLflow UI
# MAGIC
# MAGIC **Navigate to MLflow UI:**
# MAGIC 1. Experiments → `/Shared/stonex_portfolio_eval`
# MAGIC 2. Click the latest run
# MAGIC 3. **Evaluations tab** → **"Assessments" button**
# MAGIC 4. **See all 8 judges:**
# MAGIC    - `relevance_to_query` (built-in) - Addresses the question?
# MAGIC    - `safety` (built-in) - Safe and appropriate?
# MAGIC    - `tool_usage` (custom) - Used correct tools?
# MAGIC    - `data_quality` (custom) - Included relevant data?
# MAGIC    - `professional_tone` (custom) - Professional language?
# MAGIC    - `regulatory_compliance` (custom) - Follows regulations?
# MAGIC    - `accuracy` (custom) - No hallucinations?
# MAGIC    - `completeness` (custom) - Complete answer?
# MAGIC 5. **Click traces** to see detailed pass/fail rationale
# MAGIC
# MAGIC **Key Points:**
# MAGIC - **All 8 judges in one run!**
# MAGIC - **10 realistic test queries**
# MAGIC - **All produce pass/fail - no nulls**
# MAGIC - Built-in + custom guidelines together
# MAGIC - Demonstrates real-world evaluation patterns
