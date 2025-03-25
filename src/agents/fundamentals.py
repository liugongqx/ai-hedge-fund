from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from utils.progress import progress
import json

from tools.api import get_financial_metrics, get_company_facts

# Expanded industry-specific threshold ranges
industry_thresholds = {
    "Software - Infrastructure": {
        "return_on_equity": 0.15,  # High ROE for software infrastructure companies
        "net_margin": 0.20,        # High net margin for software infrastructure
        "operating_margin": 0.15,  # High operating margin for software infrastructure
        
        "revenue_growth": 0.10,    # Higher growth expectations
        "earnings_growth": 0.10,   # High earnings growth
        "book_value_growth": 0.10, # Rapid book value increase
        
        "current_ratio": 1.5,      # Higher liquidity needs
        "debt_to_equity": 0.5,    # More conservative debt
        "fcf_earnings_ratio": 0.8, # Strong cash flow conversion
        
        "price_to_earnings": 35,   # Higher P/E for growth potential
        "price_to_book": 5,        # Higher P/B for software infrastructure
        "price_to_sales": 8        # Higher P/S for software infrastructure companies
    },
    "Retail - Consumer Discretionary": {
        "return_on_equity": 0.10,  # Moderate ROE for consumer discretionary retail
        "net_margin": 0.05,        # Lower net margin for retail
        "operating_margin": 0.05,  # Lower operating margin for retail
        
        "revenue_growth": 0.07,    # Moderate growth
        "earnings_growth": 0.08,   # Steady earnings
        "book_value_growth": 0.05, # Slower book value growth
        
        "current_ratio": 1.5,      # Standard liquidity
        "debt_to_equity": 1.0,     # Typical retail sector debt
        "fcf_earnings_ratio": 0.7, # Moderate cash flow
        
        "price_to_earnings": 20,   # Lower P/E 
        "price_to_book": 2,        # Lower P/B
        "price_to_sales": 1        # Lower P/S 
    },
    "Energy - Exploration & Production": {
        "return_on_equity": 0.08,  # Moderate ROE for energy exploration
        "net_margin": 0.10,        # Moderate net margin for energy companies
        "operating_margin": 0.12,  # Moderate operating margin for energy companies
        
        "revenue_growth": 0.05,    # Cyclical growth
        "earnings_growth": 0.06,   # Moderate earnings
        "book_value_growth": 0.04, # Slower growth
        
        "current_ratio": 1.5,      # Steady liquidity
        "debt_to_equity": 1.5,     # Higher capital intensity
        "fcf_earnings_ratio": 0.8, # Solid cash flow
        
        "price_to_earnings": 15,   # Lower P/E
        "price_to_book": 1.5,      # Lower P/B
        "price_to_sales": 1.2      # Moderate P/S
    },
    "Financials - Banks": {
        "return_on_equity": 0.15,  # Strong ROE for banking sector
        "net_margin": 0.20,        # Healthy net margin for financial institutions
        "operating_margin": 0.20,  # Healthy operating margin for financials
        
        "revenue_growth": 0.10,    # Steady growth
        "earnings_growth": 0.12,   # Consistent earnings
        "book_value_growth": 0.10, # Moderate growth
        
        "current_ratio": 1.8,      # Strong liquidity
        "debt_to_equity": 2.0,     # Higher leverage typical in banking
        "fcf_earnings_ratio": 0.75,# Moderate cash flow
        
        "price_to_earnings": 25,   # Moderate P/E
        "price_to_book": 2.5,      # Book value important in financial sector
        "price_to_sales": 3        # Moderate P/S
    },
    "Manufacturing - Industrial Goods": {
        "return_on_equity": 0.10,  # Moderate ROE for industrial manufacturing
        "net_margin": 0.07,        # Lower net margin for manufacturing
        "operating_margin": 0.08,  # Lower operating margin for manufacturing
        
        "revenue_growth": 0.06,    # Slower growth
        "earnings_growth": 0.07,   # Steady earnings
        "book_value_growth": 0.05, # Modest growth
        
        "current_ratio": 1.6,      # Steady liquidity
        "debt_to_equity": 1.2,     # Moderate debt
        "fcf_earnings_ratio": 0.7, # Moderate cash flow
        
        "price_to_earnings": 18,   # Lower P/E
        "price_to_book": 2,        # Lower P/B
        "price_to_sales": 1.5      # Moderate P/S
    },
    "Healthcare - Pharmaceuticals": {
        "return_on_equity": 0.18,  # Strong ROE for pharmaceutical companies
        "net_margin": 0.25,        # High net margin for pharma
        "operating_margin": 0.20,  # High operating margin for pharma
        
        "revenue_growth": 0.12,    # Strong growth potential
        "earnings_growth": 0.15,   # Consistent earnings
        "book_value_growth": 0.12, # Steady growth
        
        "current_ratio": 2.0,      # Strong liquidity
        "debt_to_equity": 0.5,     # Conservative debt
        "fcf_earnings_ratio": 0.85,# Strong cash flow
        
        "price_to_earnings": 30,   # Higher P/E for growth
        "price_to_book": 4,        # Higher P/B 
        "price_to_sales": 5        # Higher P/S
    }
}

def get_thresholds_list(industry: str, thresholds: dict, metric_type: str, metrics: object) -> list[tuple]:
    """
    Generate a list of thresholds for specific metric types based on industry
    
    :param industry: Industry of the company
    :param thresholds: Dictionary of industry-specific thresholds
    :param metric_type: Type of metrics to compare
    :param metrics: Financial metrics object
    :return: List of tuples with (metric_value, threshold)
    """
    # Get the threshold values based on the industry, default to 'tech'
    industry_threshold = thresholds.get(industry, thresholds["Software - Infrastructure"])
    
    if industry_threshold == thresholds["Software - Infrastructure"] and industry not in thresholds:
        print(f"Industry '{industry}' not found. Falling back to default 'Software - Infrastructure' thresholds.")
    
    if metric_type == "profitability":
        return [
            (metrics.return_on_equity, industry_threshold["return_on_equity"]),
            (metrics.net_margin, industry_threshold["net_margin"]),
            (metrics.operating_margin, industry_threshold["operating_margin"])
        ]
    
    elif metric_type == "growth":
        return [
            (metrics.revenue_growth, industry_threshold["revenue_growth"]),
            (metrics.earnings_growth, industry_threshold["earnings_growth"]),
            (metrics.book_value_growth, industry_threshold["book_value_growth"])
        ]
    
    elif metric_type == "financial_health":
        # Calculate free cash flow to earnings ratio
        fcf_earnings_ratio = (metrics.free_cash_flow_per_share / metrics.earnings_per_share) if metrics.earnings_per_share else None
        
        return [
            (metrics.current_ratio, industry_threshold["current_ratio"]),
            (metrics.debt_to_equity, industry_threshold["debt_to_equity"]),
            (fcf_earnings_ratio, industry_threshold["fcf_earnings_ratio"])
        ]
    
    elif metric_type == "valuation":
        return [
            (metrics.price_to_earnings_ratio, industry_threshold["price_to_earnings"]),
            (metrics.price_to_book_ratio, industry_threshold["price_to_book"]),
            (metrics.price_to_sales_ratio, industry_threshold["price_to_sales"])
        ]
    
    return []

def fundamentals_agent(state: AgentState):
    """Analyzes fundamental data and generates trading signals for multiple tickers."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    # Initialize fundamental analysis for each ticker
    fundamental_analysis = {}

    for ticker in tickers:
        progress.update_status("fundamentals_agent", ticker, "Fetching financial metrics")

        # Get the financial metrics
        # ttm (trailing twelve months) is the default period as it can get latest data
        # annual is used for long-term analysis
        financial_metrics = get_financial_metrics(
            ticker=ticker,
            end_date=end_date,
            period="annual",
            limit=10,
        )

        if not financial_metrics:
            progress.update_status("fundamentals_agent", ticker, "Failed: No financial metrics found")
            continue

        # Pull the most recent financial metrics
        metrics = financial_metrics[0]

        # Get the company's industry
        industry = get_company_facts(ticker).industry
        
        # Fallback to a default industry if not found
        if industry not in industry_thresholds:
            industry = "Software - Infrastructure"
            print(f"Industry '{industry}' not found. Falling back to default thresholds.")

        # Get industry-specific thresholds
        thresholds = industry_thresholds[industry]

        # Initialize signals list for different fundamental aspects
        signals = []
        reasoning = {}

        progress.update_status("fundamentals_agent", ticker, "Analyzing profitability")
        # 1. Profitability Analysis
        return_on_equity = metrics.return_on_equity
        net_margin = metrics.net_margin
        operating_margin = metrics.operating_margin

        profitability_score = 0
        if return_on_equity and return_on_equity > thresholds["return_on_equity"]:
            profitability_score += 1
        if net_margin and net_margin > thresholds["net_margin"]:
            profitability_score += 1
        if operating_margin and operating_margin > thresholds["operating_margin"]:
            profitability_score += 1

        signals.append("bullish" if profitability_score >= 2 else "bearish" if profitability_score == 0 else "neutral")
        reasoning["profitability_signal"] = {
            "signal": signals[0],
            "details": (f"ROE: {return_on_equity:.2%}" if return_on_equity else "ROE: N/A") + 
                       ", " + (f"Net Margin: {net_margin:.2%}" if net_margin else "Net Margin: N/A") + 
                       ", " + (f"Op Margin: {operating_margin:.2%}" if operating_margin else "Op Margin: N/A"),
        }

        progress.update_status("fundamentals_agent", ticker, "Analyzing growth")
        # 2. Growth Analysis
        revenue_growth = metrics.revenue_growth
        earnings_growth = metrics.earnings_growth
        book_value_growth = metrics.book_value_growth

        growth_score = 0
        if revenue_growth and revenue_growth > thresholds["revenue_growth"]:
            growth_score += 1
        if earnings_growth and earnings_growth > thresholds["earnings_growth"]:
            growth_score += 1
        if book_value_growth and book_value_growth > thresholds["book_value_growth"]:
            growth_score += 1

        signals.append("bullish" if growth_score >= 2 else "bearish" if growth_score == 0 else "neutral")
        reasoning["growth_signal"] = {
            "signal": signals[1],
            "details": (f"Revenue Growth: {revenue_growth:.2%}" if revenue_growth else "Revenue Growth: N/A") + 
                       ", " + (f"Earnings Growth: {earnings_growth:.2%}" if earnings_growth else "Earnings Growth: N/A"),
        }

        progress.update_status("fundamentals_agent", ticker, "Analyzing financial health")
        # 3. Financial Health
        current_ratio = metrics.current_ratio
        debt_to_equity = metrics.debt_to_equity
        free_cash_flow_per_share = metrics.free_cash_flow_per_share
        earnings_per_share = metrics.earnings_per_share

        health_score = 0
        if current_ratio and current_ratio > thresholds["current_ratio"]:
            health_score += 1
        if debt_to_equity and debt_to_equity < thresholds["debt_to_equity"]:
            health_score += 1
        if free_cash_flow_per_share and earnings_per_share and free_cash_flow_per_share > earnings_per_share * thresholds["fcf_earnings_ratio"]:
            health_score += 1

        signals.append("bullish" if health_score >= 2 else "bearish" if health_score == 0 else "neutral")
        reasoning["financial_health_signal"] = {
            "signal": signals[2],
            "details": (f"Current Ratio: {current_ratio:.2f}" if current_ratio else "Current Ratio: N/A") + 
                       ", " + (f"D/E: {debt_to_equity:.2f}" if debt_to_equity else "D/E: N/A"),
        }

        progress.update_status("fundamentals_agent", ticker, "Analyzing valuation ratios")
        # 4. Price to X ratios
        pe_ratio = metrics.price_to_earnings_ratio
        pb_ratio = metrics.price_to_book_ratio
        ps_ratio = metrics.price_to_sales_ratio

        price_ratio_score = 0
        if pe_ratio and pe_ratio > thresholds["price_to_earnings"]:
            price_ratio_score += 1
        if pb_ratio and pb_ratio > thresholds["price_to_book"]:
            price_ratio_score += 1
        if ps_ratio and ps_ratio > thresholds["price_to_sales"]:
            price_ratio_score += 1

        signals.append("bearish" if price_ratio_score >= 2 else "bullish" if price_ratio_score == 0 else "neutral")
        reasoning["price_ratios_signal"] = {
            "signal": signals[3],
            "details": (f"P/E: {pe_ratio:.2f}" if pe_ratio else "P/E: N/A") + 
                       ", " + (f"P/B: {pb_ratio:.2f}" if pb_ratio else "P/B: N/A") + 
                       ", " + (f"P/S: {ps_ratio:.2f}" if ps_ratio else "P/S: N/A"),
        }

        progress.update_status("fundamentals_agent", ticker, "Calculating final signal")
        # Determine overall signal
        bullish_signals = signals.count("bullish")
        bearish_signals = signals.count("bearish")

        if bullish_signals > bearish_signals:
            overall_signal = "bullish"
        elif bearish_signals > bullish_signals:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        # Calculate confidence level
        total_signals = len(signals)
        confidence = round(max(bullish_signals, bearish_signals) / total_signals, 2) * 100

        fundamental_analysis[ticker] = {
            "signal": overall_signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        progress.update_status("fundamentals_agent", ticker, "Done")

    # Create the fundamental analysis message
    message = HumanMessage(
        content=json.dumps(fundamental_analysis),
        name="fundamentals_agent",
    )

    # Print the reasoning if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(fundamental_analysis, "Fundamental Analysis Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["fundamentals_agent"] = fundamental_analysis

    return {
        "messages": [message],
        "data": data,
    }