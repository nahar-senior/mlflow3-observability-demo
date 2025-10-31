# Databricks notebook source
# MAGIC %md
# MAGIC # StoneX Portfolio Intelligence - Setup
# MAGIC
# MAGIC This notebook sets up:
# MAGIC - Unity Catalog: `stonex_demo` catalog
# MAGIC - Tables: portfolio_holdings, market_data, earnings_reports
# MAGIC - Vector Search endpoint and index for earnings intelligence
# MAGIC - UC Functions: get_portfolio_summary, get_market_data, calculate_portfolio_risk

# COMMAND ----------

# MAGIC %pip install --quiet databricks-vectorsearch mlflow[databricks]>=2.16.0 langgraph==0.2.34 langchain-community databricks-langchain
# MAGIC %restart_python

# COMMAND ----------

import requests
import pandas as pd
import io
import time
from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient

# Initialize clients
w = WorkspaceClient()
vs_client = VectorSearchClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Create Unity Catalog & Tables

# COMMAND ----------

# Create catalog and schema
spark.sql("CREATE CATALOG IF NOT EXISTS stonex_demo")
spark.sql("CREATE SCHEMA IF NOT EXISTS stonex_demo.portfolio")

print("✅ Catalog and schema created")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.5. Create Volume and Upload Data

# COMMAND ----------

# Create volume for data files
spark.sql("""
    CREATE VOLUME IF NOT EXISTS stonex_demo.portfolio.demo_data
    COMMENT 'Volume for portfolio demo data files'
""")

volume_path = "/Volumes/stonex_demo/portfolio/demo_data"
print(f"✅ Volume created: {volume_path}")

# COMMAND ----------

# Upload CSV files to volume (for manual setup, these would be uploaded via DAB)
# For now, we'll create them programmatically

import os

# Create sample data directly in volume
csv_data = {
    "portfolio_holdings.csv": """client_id,account_number,ticker,asset_name,quantity,avg_cost,sector,asset_class
C001,ACC-9876,AAPL,Apple Inc.,250,145.20,Technology,Equity
C001,ACC-9876,MSFT,Microsoft Corp.,180,280.50,Technology,Equity
C001,ACC-9876,JPM,JPMorgan Chase,120,155.80,Financials,Equity
C001,ACC-9876,JNJ,Johnson & Johnson,100,165.40,Healthcare,Equity
C001,ACC-9876,BND,Vanguard Total Bond,500,75.20,Fixed Income,Bond ETF
C002,ACC-5432,NVDA,NVIDIA Corp.,75,420.30,Technology,Equity
C002,ACC-5432,GOOGL,Alphabet Inc.,90,138.50,Technology,Equity
C002,ACC-5432,V,Visa Inc.,110,245.60,Financials,Equity
C002,ACC-5432,PG,Procter & Gamble,85,155.20,Consumer Staples,Equity
C002,ACC-5432,AGG,iShares Core Bond,400,102.30,Fixed Income,Bond ETF
C003,ACC-7891,TSLA,Tesla Inc.,45,235.80,Consumer Cyclical,Equity
C003,ACC-7891,META,Meta Platforms,60,315.40,Technology,Equity
C003,ACC-7891,XOM,Exxon Mobil,150,110.60,Energy,Equity
C003,ACC-7891,UNH,UnitedHealth,55,485.20,Healthcare,Equity
C003,ACC-7891,TLT,iShares 20+ Yr Bond,200,95.40,Fixed Income,Bond ETF""",
    
    "market_data.csv": """ticker,current_price,prev_close,day_change_pct,week_change_pct,pe_ratio,dividend_yield,beta,market_cap_b,last_earnings_date,next_earnings_date
AAPL,178.45,176.20,1.28,3.45,28.5,0.52,1.15,2780,2024-02-01,2024-05-02
MSFT,380.25,378.90,0.36,2.10,34.2,0.78,0.95,2825,2024-01-30,2024-04-25
JPM,168.90,167.30,0.96,1.85,10.8,2.45,1.18,485,2024-01-12,2024-04-12
JNJ,162.75,163.20,-0.28,0.45,24.3,3.05,0.55,395,2024-01-23,2024-04-16
NVDA,725.50,718.40,0.99,8.75,68.5,0.04,1.75,1785,2024-02-21,2024-05-22
GOOGL,142.85,141.90,0.67,2.90,25.8,0.00,1.08,1795,2024-01-30,2024-04-25
V,272.30,270.80,0.55,1.95,32.4,0.82,0.98,565,2024-01-25,2024-04-23
PG,158.60,159.10,-0.31,-0.85,25.6,2.42,0.42,375,2024-01-19,2024-04-19
TSLA,195.75,192.30,1.79,4.25,42.8,0.00,2.15,620,2024-01-24,2024-04-23
META,395.80,392.50,0.84,5.60,28.9,0.00,1.25,1005,2024-01-31,2024-04-24
XOM,103.25,102.90,0.34,-1.20,11.2,3.45,0.85,425,2024-02-02,2024-04-26
UNH,512.40,508.70,0.73,2.35,26.5,1.35,0.75,480,2024-01-12,2024-04-16
BND,72.85,72.90,-0.07,0.15,0.0,4.25,0.08,0,,,
AGG,98.40,98.50,-0.10,0.25,0.0,3.85,0.06,0,,,
TLT,91.20,91.60,-0.44,1.85,0.0,4.65,0.18,0,,,""",
    
    "earnings_reports.csv": """doc_id,ticker,company,report_date,indexed_doc
ER001,AAPL,Apple Inc.,2024-02-01,"Apple Q1 2024 Earnings: Record revenue of $119.6B, up 2% YoY. iPhone revenue $69.7B, Services $23.1B (up 11%). Gross margin 45.9%. Management highlighted strong Services growth and AI investments in Apple Silicon. Geographic strength in Americas and Europe, offset by China softness. Returned $27B to shareholders via dividends and buybacks."
ER002,MSFT,Microsoft Corp.,2024-01-30,"Microsoft Q2 FY24 Earnings: Revenue $62B, up 18% YoY. Cloud revenue (Azure, O365) $33.7B, up 24%. Intelligent Cloud segment $25.9B. Azure growth 30% driven by AI services. Operating margin 43%. Copilot AI integration driving Commercial bookings. Gaming revenue up 49% post-Activision acquisition. Management emphasized AI monetization and expanding Azure AI infrastructure."
ER003,NVDA,NVIDIA Corp.,2024-02-21,"NVIDIA Q4 FY24 Earnings: Record revenue $22.1B, up 265% YoY. Data Center revenue $18.4B, up 409% driven by AI accelerators (H100, A100). Gaming $2.9B. Gross margin 76%. Management highlighted insatiable demand for AI computing, new Blackwell architecture launch, and expanding partnerships with cloud hyperscalers. Full-year revenue guidance $110B."
ER004,META,Meta Platforms,2024-01-31,"Meta Q4 2023 Earnings: Revenue $40.1B, up 25% YoY. Family of Apps revenue $38.7B. Daily Active People 3.19B. Ad impressions up 21%, price per ad up 2%. Operating margin 41%. Management announced first-ever dividend ($0.50/share) and $50B buyback. Emphasized Reality Labs investments ($4.6B operating loss), AI-powered ad targeting improvements, and efficiency gains from Year of Efficiency."
ER005,GOOGL,Alphabet Inc.,2024-01-30,"Alphabet Q4 2023 Earnings: Revenue $86.3B, up 13% YoY. Google Search $48B, YouTube Ads $9.2B, Cloud $9.2B (up 26%). Operating margin 30%. Management highlighted Gemini AI model launch, Search Generative Experience expansion, and Cloud AI adoption (Vertex AI, Duet AI). Operating expenses down 2% reflecting efficiency initiatives. Capital expenditures $11B for AI infrastructure."
ER006,JPM,JPMorgan Chase,2024-01-12,"JPMorgan Q4 2023 Earnings: Revenue $39.9B, up 12% YoY. Net income $9.3B. Net Interest Income $23.2B benefiting from rate environment. Investment Banking fees $2.0B, up 9%. Credit card spending up 8%. CET1 ratio 15.0%. Management highlighted strong consumer spending, commercial banking strength, and completed First Republic integration. Increased full-year NII guidance to $89B on sustained higher rates."
ER007,TSLA,Tesla Inc.,2024-01-24,"Tesla Q4 2023 Earnings: Revenue $25.2B, up 3% YoY. Automotive revenue $21.6B. Delivered 1.81M vehicles in 2023. Gross margin 17.6% (pressure from price cuts). Energy generation/storage revenue $1.4B, up 10%. Management emphasized next-gen platform development (lower-cost vehicle), Cybertruck production ramp, and FSD (Full Self-Driving) improvements. 2024 delivery growth may be notably lower as focus shifts to next-gen vehicle."
ER008,UNH,UnitedHealth,2024-01-12,"UnitedHealth Q4 2023 Earnings: Revenue $94.4B, up 14% YoY. Full-year revenue $371.6B. Medical care ratio 82.3%. UnitedHealthcare served 29.9M members. Optum revenue $60.8B, up 24%. Operating margin 8.4%. Management provided 2024 EPS guidance $27.50-28.00, highlighted Optum Health growth (value-based care), pharmacy care services expansion, and investments in care delivery infrastructure and digital health."
ER009,V,Visa Inc.,2024-01-25,"Visa Q1 FY24 Earnings: Revenue $8.6B, up 9% YoY. Payments volume $3.3T, up 8%. Cross-border volume up 16% (ex Russia). Processed transactions 53.8B, up 10%. Operating margin 67%. Management emphasized international growth momentum, expansion of value-added services (fraud, data analytics), and growth in new flows (B2B, account-to-account). Consumer spending remains resilient despite macro uncertainty."
ER010,PG,Procter & Gamble,2024-01-19,"P&G Q2 FY24 Earnings: Revenue $21.9B, flat YoY. Organic sales up 5% (volume +2%, pricing +3%). Operating margin 24.8%. Beauty and Grooming categories strong. Management maintained full-year guidance: organic sales +4-5%, EPS $6.35-6.53. Emphasized productivity savings ($2.5B program) funding innovation and marketing, pricing discipline to offset commodity inflation, and market share gains in 8 of 10 categories."""
}

# Write files to volume
for filename, content in csv_data.items():
    file_path = f"{volume_path}/{filename}"
    dbutils.fs.put(file_path, content, overwrite=True)
    print(f"✅ Uploaded: {filename}")

# COMMAND ----------

# Load data from volume into tables
csv_files = ["portfolio_holdings", "market_data", "earnings_reports"]

for table_name in csv_files:
    file_path = f"{volume_path}/{table_name}.csv"
    
    df = spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load(file_path)
    
    df.write.mode("overwrite").saveAsTable(f"stonex_demo.portfolio.{table_name}")
    
    count = df.count()
    print(f"✅ Created table: stonex_demo.portfolio.{table_name} ({count} rows)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create Vector Search Endpoint

# COMMAND ----------

endpoint_name = "stonex_portfolio_endpoint"

try:
    endpoints = vs_client.list_endpoints()
    endpoint_names = [ep['name'] for ep in endpoints.get('endpoints', [])]
    
    if endpoint_name not in endpoint_names:
        vs_client.create_endpoint(name=endpoint_name, endpoint_type="STANDARD")
        print(f"✅ Creating vector search endpoint: {endpoint_name}")
        
        # Wait for endpoint to be ready
        max_wait = 600  # 10 minutes
        wait_interval = 15
        elapsed = 0
        
        while elapsed < max_wait:
            endpoint_info = vs_client.get_endpoint(endpoint_name)
            state = endpoint_info.get('endpoint_status', {}).get('state', 'UNKNOWN')
            print(f"   Endpoint state: {state}")
            
            if state in ["ONLINE", "PROVISIONED"]:
                print("✅ Endpoint is ready")
                break
            elif state in ["FAILED", "OFFLINE"]:
                print(f"❌ Endpoint failed with state: {state}")
                break
            
            time.sleep(wait_interval)
            elapsed += wait_interval
    else:
        print(f"✅ Endpoint already exists: {endpoint_name}")
        
except Exception as e:
    print(f"❌ Error with vector search endpoint: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Enable CDC and Create Vector Search Index

# COMMAND ----------

# Enable Change Data Feed on earnings_reports table
try:
    spark.sql("""
        ALTER TABLE stonex_demo.portfolio.earnings_reports 
        SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
    """)
    print("✅ CDC enabled on earnings_reports table")
except Exception as e:
    print(f"⚠️  CDC setup: {e}")

# COMMAND ----------

# Create vector search index for earnings intelligence
try:
    index = vs_client.create_delta_sync_index(
        endpoint_name=endpoint_name,
        source_table_name="stonex_demo.portfolio.earnings_reports",
        index_name="stonex_demo.portfolio.earnings_reports_index",
        pipeline_type="TRIGGERED",
        primary_key="doc_id",
        embedding_source_column="indexed_doc",
        embedding_model_endpoint_name="databricks-gte-large-en"
    )
    
    print("✅ Vector search index created: stonex_demo.portfolio.earnings_reports_index")
    print("   (Index will sync in background)")
    
except Exception as e:
    error_msg = str(e)
    if "already exists" in error_msg.lower():
        print("✅ Vector search index already exists")
    else:
        print(f"❌ Error creating index: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create UC Functions (Tools)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION stonex_demo.portfolio.get_portfolio_summary(
# MAGIC   client_id STRING COMMENT 'The unique client identifier (e.g., C001, C002, C003). Required to look up the client portfolio.'
# MAGIC )
# MAGIC RETURNS TABLE(
# MAGIC   ticker STRING COMMENT 'Stock ticker symbol',
# MAGIC   asset_name STRING COMMENT 'Full name of the asset',
# MAGIC   quantity DOUBLE COMMENT 'Number of shares held',
# MAGIC   avg_cost DOUBLE COMMENT 'Average cost per share',
# MAGIC   sector STRING COMMENT 'Market sector',
# MAGIC   asset_class STRING COMMENT 'Asset classification'
# MAGIC )
# MAGIC COMMENT 'Returns all portfolio holdings for a specific client by their client_id. Use this to see what stocks and assets a client currently owns, including quantities and cost basis.'
# MAGIC RETURN (
# MAGIC   SELECT ticker, asset_name, quantity, avg_cost, sector, asset_class
# MAGIC   FROM stonex_demo.portfolio.portfolio_holdings
# MAGIC   WHERE client_id = get_portfolio_summary.client_id
# MAGIC );

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION stonex_demo.portfolio.get_market_data(
# MAGIC   ticker STRING COMMENT 'Stock ticker symbol (e.g., AAPL, MSFT, GOOGL). Required to look up current market data and price information.'
# MAGIC )
# MAGIC RETURNS TABLE(
# MAGIC   ticker STRING COMMENT 'Stock ticker symbol',
# MAGIC   current_price DOUBLE COMMENT 'Current stock price in USD',
# MAGIC   day_change_pct DOUBLE COMMENT 'Percentage change today',
# MAGIC   week_change_pct DOUBLE COMMENT 'Percentage change over past week',
# MAGIC   pe_ratio DOUBLE COMMENT 'Price to earnings ratio',
# MAGIC   dividend_yield DOUBLE COMMENT 'Annual dividend yield percentage',
# MAGIC   next_earnings_date STRING COMMENT 'Date of next earnings report'
# MAGIC )
# MAGIC COMMENT 'Returns current market data, pricing, and fundamental metrics for a specific stock ticker. Use this to get real-time price information, performance metrics, and valuation data for any stock.'
# MAGIC RETURN (
# MAGIC   SELECT ticker, current_price, day_change_pct, week_change_pct, 
# MAGIC          pe_ratio, dividend_yield, next_earnings_date
# MAGIC   FROM stonex_demo.portfolio.market_data
# MAGIC   WHERE ticker = get_market_data.ticker
# MAGIC );

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION stonex_demo.portfolio.calculate_portfolio_risk(
# MAGIC   client_id STRING COMMENT 'The unique client identifier (e.g., C001, C002, C003). Required to analyze the specific client portfolio risk.'
# MAGIC )
# MAGIC RETURNS TABLE(
# MAGIC   total_positions INT COMMENT 'Total number of positions held',
# MAGIC   equity_exposure_pct DOUBLE COMMENT 'Percentage of portfolio in equities',
# MAGIC   tech_concentration_pct DOUBLE COMMENT 'Percentage concentrated in tech sector',
# MAGIC   avg_beta DOUBLE COMMENT 'Average portfolio beta (market sensitivity)',
# MAGIC   risk_level STRING COMMENT 'Overall risk assessment: Low, Moderate, or High'
# MAGIC )
# MAGIC COMMENT 'Calculates comprehensive risk metrics for a client portfolio including sector concentration, market exposure, beta analysis, and overall risk level assessment. Use this when analyzing portfolio risk or discussing risk management strategies.'
# MAGIC RETURN (
# MAGIC   WITH holdings AS (
# MAGIC     SELECT h.*, m.beta, m.current_price
# MAGIC     FROM stonex_demo.portfolio.portfolio_holdings h
# MAGIC     LEFT JOIN stonex_demo.portfolio.market_data m ON h.ticker = m.ticker
# MAGIC     WHERE h.client_id = calculate_portfolio_risk.client_id
# MAGIC   ),
# MAGIC   calcs AS (
# MAGIC     SELECT 
# MAGIC       COUNT(*) as total_positions,
# MAGIC       SUM(CASE WHEN asset_class = 'Equity' THEN quantity * current_price ELSE 0 END) / 
# MAGIC         NULLIF(SUM(quantity * current_price), 0) * 100 as equity_pct,
# MAGIC       SUM(CASE WHEN sector = 'Technology' THEN quantity * current_price ELSE 0 END) / 
# MAGIC         NULLIF(SUM(quantity * current_price), 0) * 100 as tech_pct,
# MAGIC       AVG(CASE WHEN beta IS NOT NULL THEN beta ELSE 1.0 END) as avg_beta
# MAGIC     FROM holdings
# MAGIC   )
# MAGIC   SELECT 
# MAGIC     total_positions,
# MAGIC     ROUND(equity_pct, 2) as equity_exposure_pct,
# MAGIC     ROUND(tech_pct, 2) as tech_concentration_pct,
# MAGIC     ROUND(avg_beta, 2) as avg_beta,
# MAGIC     CASE
# MAGIC       WHEN tech_pct > 40 OR avg_beta > 1.3 THEN 'High'
# MAGIC       WHEN tech_pct > 25 OR avg_beta > 1.1 THEN 'Moderate'
# MAGIC       ELSE 'Low'
# MAGIC     END as risk_level
# MAGIC   FROM calcs
# MAGIC );

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Grant Permissions

# COMMAND ----------

# Grant permissions to all users
try:
    spark.sql("GRANT USE CATALOG ON CATALOG stonex_demo TO `account users`")
    spark.sql("GRANT USE SCHEMA ON SCHEMA stonex_demo.portfolio TO `account users`")
    spark.sql("GRANT SELECT ON SCHEMA stonex_demo.portfolio TO `account users`")
    spark.sql("GRANT EXECUTE ON SCHEMA stonex_demo.portfolio TO `account users`")
    print("✅ Permissions granted")
except Exception as e:
    print(f"⚠️  Permissions: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Verify Setup

# COMMAND ----------

print("=" * 60)
print("SETUP VERIFICATION")
print("=" * 60)

# Check tables
print("\n📊 Tables:")
tables = spark.sql("SHOW TABLES IN stonex_demo.portfolio").collect()
for table in tables:
    count = spark.sql(f"SELECT COUNT(*) as cnt FROM stonex_demo.portfolio.{table.tableName}").first()['cnt']
    print(f"   ✓ {table.tableName}: {count} rows")

# Check functions
print("\n🔧 Functions:")
funcs = spark.sql("SHOW USER FUNCTIONS IN stonex_demo.portfolio").collect()
for func in funcs:
    print(f"   ✓ {func.function}")

# Check vector search
print("\n🔍 Vector Search:")
try:
    endpoint_info = vs_client.get_endpoint(endpoint_name)
    state = endpoint_info.get('endpoint_status', {}).get('state', 'UNKNOWN')
    print(f"   ✓ Endpoint: {endpoint_name} ({state})")
    
    try:
        index_info = vs_client.get_index(endpoint_name, "stonex_demo.portfolio.earnings_reports_index")
        print(f"   ✓ Index: earnings_reports_index")
    except:
        print(f"   ⚠️  Index may still be syncing")
except Exception as e:
    print(f"   ❌ {e}")

print("\n✅ Setup complete! Ready for agent demo.")
print("=" * 60)

