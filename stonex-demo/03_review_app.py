# Databricks notebook source
# MAGIC %md
# MAGIC # StoneX Portfolio Agent - SME Review App
# MAGIC
# MAGIC **What This Notebook Does:**
# MAGIC
# MAGIC Sets up a Review App for Subject Matter Experts (SMEs) to provide feedback on agent responses. Domain experts can review past agent interactions, provide thumbs up/down ratings, and add detailed comments - all without writing code.

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade "mlflow[databricks]>=3.1.0" openai "databricks-connect>=16.1"
# MAGIC %restart_python

# COMMAND ----------

import mlflow
from mlflow.genai.labeling import get_review_app

# Set experiment (same as tracing notebook)
mlflow.set_experiment("/Shared/stonex_portfolio_eval")

print("✅ MLflow Experiment: /Shared/stonex_portfolio_eval")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Select Traces to Review
# MAGIC
# MAGIC First, let's find recent traces from our agent:

# COMMAND ----------

# Search for recent traces from tracing/evaluation notebooks
# Note: Traces are automatically filtered to the current experiment
traces = mlflow.search_traces(
    max_results=10,
    order_by=["timestamp_ms DESC"]
)

print("=" * 70)
print("AVAILABLE TRACES FOR REVIEW")
print("=" * 70)
print(f"\n📊 Found {len(traces)} recent traces")

if len(traces) > 0:
    print("\nRecent queries:")
    for idx in range(min(10, len(traces))):
        trace = traces.iloc[idx]
        trace_id = traces.index[idx]
        
        # Try to extract query from different possible locations
        query = None
        
        # Method 1: Try inputs column
        if 'inputs' in trace:
            inputs = trace['inputs']
            if inputs is not None:
                if isinstance(inputs, dict):
                    query = inputs.get('query') or inputs.get('question') or inputs.get('messages')
                elif isinstance(inputs, str):
                    query = inputs
        
        # Method 2: Try request column
        if not query and 'request' in trace:
            request = trace['request']
            if request is not None:
                if isinstance(request, dict):
                    query = request.get('query') or request.get('question')
                elif isinstance(request, str):
                    query = request
        
        # Method 3: Try request_metadata
        if not query and 'request_metadata' in trace:
            metadata = trace['request_metadata']
            if metadata is not None and isinstance(metadata, list) and len(metadata) > 0:
                query = metadata[0].get('query', metadata[0].get('question'))
        
        # Fallback: Show trace ID
        if not query or query == 'N/A':
            query = f"Trace {str(trace_id)[:8]}"
        else:
            # Truncate if too long
            query = str(query)[:80]
        
        print(f"  {idx+1}. {query}")
    
    print(f"\n✅ Ready to create labeling session with these traces")
else:
    print("\n⚠️  No traces found. Run 01_tracing.py or 02_assessments.py first!")

print("=" * 70)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Create Labeling Session
# MAGIC
# MAGIC Assign specific traces to reviewers with custom questions:

# COMMAND ----------

from mlflow.genai.labeling import create_labeling_session
from mlflow.genai.label_schemas import create_label_schema, InputCategorical, InputText

if len(traces) > 0:
    # Step 1: Create label schemas for SME feedback
    
    # 1. Overall Quality Rating
    quality_rating = create_label_schema(
        name="quality_rating",
        type="feedback",
        title="Overall Response Quality",
        input=InputCategorical(options=["Excellent", "Good", "Fair", "Poor"]),
        instruction="Rate the overall quality of the agent's response",
        enable_comment=True,
        overwrite=True
    )
    
    # 2. Accuracy Check
    accuracy_check = create_label_schema(
        name="accuracy_check",
        type="feedback",
        title="Factual Accuracy",
        input=InputCategorical(options=["Accurate", "Partially Accurate", "Inaccurate", "Cannot Verify"]),
        instruction="Are the facts, numbers, and data in the response correct?",
        enable_comment=True,
        overwrite=True
    )
    
    # 3. Completeness Check
    completeness_check = create_label_schema(
        name="completeness_check",
        type="feedback",
        title="Response Completeness",
        input=InputCategorical(options=["Complete", "Mostly Complete", "Incomplete", "Missing Key Info"]),
        instruction="Does the response fully answer the client's question?",
        enable_comment=True,
        overwrite=True
    )
    
    # 4. Tool Usage Appropriateness
    tool_usage = create_label_schema(
        name="tool_usage",
        type="feedback",
        title="Tool Usage Appropriateness",
        input=InputCategorical(options=["Correct Tools", "Suboptimal Tools", "Wrong Tools", "Missing Tools"]),
        instruction="Did the agent use the right tools for this query?",
        enable_comment=True,
        overwrite=True
    )
    
    # 5. Compliance Check
    compliance_check = create_label_schema(
        name="compliance_check",
        type="feedback",
        title="Regulatory Compliance",
        input=InputCategorical(options=["Compliant", "Non-Compliant", "Needs Disclaimer", "Unsure"]),
        instruction="Does this follow financial advisory regulations?",
        enable_comment=True,
        overwrite=True
    )
    
    # 6. Professional Tone
    tone_check = create_label_schema(
        name="tone_check",
        type="feedback",
        title="Professional Tone",
        input=InputCategorical(options=["Highly Professional", "Professional", "Casual", "Inappropriate"]),
        instruction="Is the tone appropriate for wealth management?",
        enable_comment=True,
        overwrite=True
    )
    
    # Step 2: Create labeling session with all schemas
    session = create_labeling_session(
        name="StoneX Portfolio Agent Review - Batch 1",
        assigned_users=[],
        label_schemas=[
            quality_rating.name,
            accuracy_check.name,
            completeness_check.name,
            tool_usage.name,
            compliance_check.name,
            tone_check.name
        ]
    )
    
    # Step 3: Add traces to the session (first 5 traces)
    traces_to_add = traces.head(5)
    session.add_traces(traces_to_add)
    
    print("=" * 70)
    print("LABELING SESSION CREATED")
    print("=" * 70)
    print(f"\n✅ Session: {session.name}")
    print(f"📊 Traces added: {len(traces_to_add)}")
    print(f"🔗 Review URL: {session.url}")
    print()
    print("Share this URL with SMEs. They will see:")
    print("  • 5 specific agent interactions to review")
    print("  • Query → Agent Response → Tools Used")
    print()
    print("6 Review Questions per trace:")
    print("  1. Overall Quality: Excellent / Good / Fair / Poor")
    print("  2. Accuracy: Accurate / Partially Accurate / Inaccurate / Cannot Verify")
    print("  3. Completeness: Complete / Mostly Complete / Incomplete / Missing Key Info")
    print("  4. Tool Usage: Correct / Suboptimal / Wrong / Missing Tools")
    print("  5. Compliance: Compliant / Non-Compliant / Needs Disclaimer / Unsure")
    print("  6. Professional Tone: Highly Professional / Professional / Casual / Inappropriate")
    print("  + Comment boxes for detailed feedback on each")
    print()
    print("=" * 70)
else:
    print("⚠️  No traces available. Run 01_tracing.py first to generate traces.")

# COMMAND ----------

# MAGIC %md
# MAGIC **Note:** If queries still show as N/A, traces may use different field names. 
# MAGIC Check the trace structure by running: `print(traces.iloc[0].to_dict())`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: View Labeling Results
# MAGIC
# MAGIC After SMEs complete reviews, access the feedback:

# COMMAND ----------

# View labeling session results
if 'session' in locals():
    # Search for labeled traces in the session
    labeled_traces = mlflow.search_traces(run_id=session.mlflow_run_id)
    
    print("=" * 70)
    print("LABELING SESSION RESULTS")
    print("=" * 70)
    print(f"\n📊 Session: {session.name}")
    print(f"🔗 Review URL: {session.url}")
    print(f"📋 Traces in session: {len(labeled_traces)}")
    print()
    print("To view results:")
    print("  1. Go to MLflow UI → Experiments → /Shared/stonex_portfolio_eval")
    print("  2. Find the run named '{}'".format(session.name))
    print("  3. Click 'Traces' tab to see reviewed traces")
    print("  4. Each trace will have Assessment objects with feedback")
    print()
    print("Or access programmatically:")
    print(f"  labeled_traces = mlflow.search_traces(run_id='{session.mlflow_run_id}')")
    print("  # Access assessments from each trace")
    print()
    print("=" * 70)
else:
    print("⚠️  Run Step 2 to create a labeling session first")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SME Review Workflow
# MAGIC
# MAGIC **What SMEs will see when they open the review URL:**
# MAGIC
# MAGIC 1. **Assigned Chats**
# MAGIC    - List of 5 specific agent interactions
# MAGIC    - Each shows: User query → Agent response → Tools used
# MAGIC
# MAGIC 2. **Provide Feedback (6 questions per chat)**
# MAGIC    - **Overall Quality:** Excellent / Good / Fair / Poor
# MAGIC    - **Factual Accuracy:** Accurate / Partially Accurate / Inaccurate / Cannot Verify
# MAGIC    - **Completeness:** Complete / Mostly Complete / Incomplete / Missing Key Info
# MAGIC    - **Tool Usage:** Correct / Suboptimal / Wrong / Missing Tools
# MAGIC    - **Compliance:** Compliant / Non-Compliant / Needs Disclaimer / Unsure
# MAGIC    - **Professional Tone:** Highly Professional / Professional / Casual / Inappropriate
# MAGIC    - **Comment boxes** for detailed feedback on each
# MAGIC
# MAGIC 3. **Submit Reviews**
# MAGIC    - Answer all questions for each chat
# MAGIC    - Submit when complete
# MAGIC    - Feedback automatically saved to MLflow
# MAGIC
# MAGIC 4. **Impact on Agent Quality**
# MAGIC    - Data science team analyzes feedback patterns
# MAGIC    - Identifies issues: hallucinations, missing data, non-compliance
# MAGIC    - Updates prompts, tools, or guidelines
# MAGIC    - Creates evaluation datasets from labeled examples
# MAGIC    - Tunes custom LLM judges based on expert feedback
# MAGIC
# MAGIC **No coding required - just domain expertise!**

