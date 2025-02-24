from autogen import AssistantAgent, UserProxyAgent, register_function
import yfinance as yf
import matplotlib.pyplot as plt
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()  
API_KEY = os.getenv("API_KEY")
  
genai.configure(api_key=API_KEY)

def get_stock_price(ticker: str):
    """Get historical stock price data for a given ticker symbol."""
    stock = yf.Ticker(ticker)
    data = stock.history(period="1y")
    if data.empty:
        return f"No data found for ticker {ticker}."
    return data.to_string()

def plot_stock_price(ticker: str):
    """Visualize stock price history for a given ticker symbol."""
    stock = yf.Ticker(ticker)
    data = stock.history(period="1y")
    if data.empty:
        return f"No data available to plot for ticker {ticker}."
    
    plt.figure(figsize=(10, 5))
    plt.plot(data['Close'])
    plt.title(f"{ticker} Stock Price History")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    filename = f"{ticker}_price_chart.png"
    plt.savefig(filename)
    plt.close() 
    return f"Chart saved as {filename}"

def list_available_extensions():
    """List all registered extensions/tools."""
    available_tools = ["get_stock_price", "plot_stock_price", "list_available_extensions", "describe"]
    return "Available extensions: " + ", ".join(available_tools)

def describe():
    """Provide detailed descriptions of all registered tools."""
    tool_descriptions = {
        "get_stock_price": "Get historical stock price data for a given ticker symbol.",
        "plot_stock_price": "Visualize stock price history for a given ticker symbol.",
        "list_available_extensions": "Lists all registered extensions/tools.",
        "describe": "Provides detailed descriptions of all registered tools."
    }
    result = "\n".join(f"{tool}: {desc}" for tool, desc in tool_descriptions.items())
    return result

llm_config = {
    "config_list": [{
        "model": "gemini-pro",
        "api_key": API_KEY,
        "api_type": "google"
    }],
    "temperature": 0.7
}

financial_analyst = AssistantAgent(
    name="Financial_Analyst",
    system_message="Analyze stock data for a ticker. Plot history and mention the chart filename. Manage the resource token carefully",
    llm_config=llm_config
)

user_proxy = UserProxyAgent(
    name="User_Proxy",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "coding", "use_docker": False},
    default_auto_reply="Please continue using available tools",
    max_consecutive_auto_reply=5,
)
\
register_function(
    get_stock_price,
    caller=financial_analyst, 
    executor=user_proxy,       
    name="get_stock_price",
    description="Get historical stock price data for a given ticker symbol"
)

register_function(
    plot_stock_price,
    caller=financial_analyst,
    executor=user_proxy,
    name="plot_stock_price",
    description="Visualize stock price history for a given ticker symbol"
)

register_function(
    list_available_extensions,
    caller=financial_analyst,
    executor=user_proxy,
    name="list_available_extensions",
    description="List all registered extensions/tools"
)

register_function(
    describe,
    caller=financial_analyst,
    executor=user_proxy,
    name="describe",
    description="Provide detailed descriptions of all registered tools"
)

def start_analysis(task):
    user_proxy.initiate_chat(
        financial_analyst,
        message=task
    )

if __name__ == "__main__":
    start_analysis("Analyze Alphabet Inc. stock performance and visualize the price history")
