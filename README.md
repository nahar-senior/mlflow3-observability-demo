# MLflow 3.0 Observability for AI Agents
## StoneX Portfolio Intelligence Demo

A production-ready demonstration of MLflow 3.0's observability capabilities for agentic systems in financial services.

## What This Demo Shows

- **MLflow Tracing**: Automatic observability for LangGraph agents (zero instrumentation)
- **LLM Judges**: Automated quality assessment with built-in and custom judges
- **SME Review App**: Human-in-the-loop validation for agent interactions

## Architecture

- **Agent Framework**: LangGraph with Claude Sonnet 4
- **Tools**: 3 Unity Catalog Functions + 1 Vector Search index
- **Use Case**: Wealth management portfolio analysis

## Setup Instructions

### Prerequisites

- Databricks workspace (AWS, Azure, or GCP)
- MLflow 3.0+
- Unity Catalog enabled

### Configuration

1. Update `stonex-demo/databricks.yml` with your workspace URL:
   ```yaml
   host: <YOUR_DATABRICKS_WORKSPACE_URL>
   ```

2. Install Databricks CLI and authenticate:
   ```bash
   databricks configure --profile default
   ```

### Deployment

Deploy using Databricks Asset Bundles:

```bash
cd stonex-demo
databricks bundle deploy --target dev
```

### Running the Demo

Execute notebooks in order:

1. **00_setup.py** - Creates Unity Catalog tables, Vector Search index, and UC functions
2. **01_tracing.py** - Demonstrates MLflow automatic tracing
3. **02_assessments.py** - Runs LLM judges evaluation
4. **03_review_app.py** - Sets up SME review session

## Key Features

### Automatic Tracing
```python
mlflow.langchain.autolog()  # That's it!
```

### Custom Judges (Plain English)
```python
Guidelines(
    name="regulatory_compliance",
    guidelines="Response must follow financial advisory regulations..."
)
```

### 11 Quality Judges
- 2 Built-in (Relevance, Safety)
- 6 Custom Guidelines (Tool usage, Data quality, Professional tone, Compliance, Accuracy, Completeness)
- 3 Hard Requirements (Non-empty, Minimum length, No placeholders)

## Sample Data

All data in `stonex-demo/data/` is synthetic:
- `portfolio_holdings.csv` - Anonymous client portfolios (C001, C002, C003)
- `market_data.csv` - Public stock market data
- `earnings_reports.csv` - Public company earnings data

## Requirements

See `stonex-demo/requirements.txt`:
- `mlflow[databricks]>=3.4.0`
- `langgraph==0.2.34`
- `databricks-langchain`

## License

Apache 2.0

## Support

For questions or issues, please open a GitHub issue or contact your Databricks account team.

