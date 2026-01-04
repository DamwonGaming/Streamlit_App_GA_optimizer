#strategy.py
import pandas as pd
import numpy as np
import yfinance as yf
import os

def load_data():
    cache_file = "hs300_cache.parquet"
    # 尝试从缓存加载
    if os.path.exists(cache_file):
        return pd.read_parquet(cache_file)
    try:
        # 获取沪深300 ETF (510300.SS)
        ticker = "510300.SS"
        data = yf.download(ticker, start="2010-01-01", progress=False)
        if not data.empty:
            df = data[['Close']].copy()
            df.columns = ['price']
            df.index.name = 'date'
            df['returns'] = df['price'].pct_change()
            df['cumulative'] = (1 + df['returns']).cumprod()
            # 保存缓存
            df.to_parquet(cache_file)
            print(f"✅ 已加载 {ticker} 数据 ({len(df)} 天)")
            return df
    except Exception as e:
        print(f"数据下载失败: {e}")
    # 如果都失败，生成模拟价格序列（用于演示）
    print("使用模拟数据启动（仅用于演示）")
    dates = pd.date_range("2015-01-01", "2025-12-31", freq="B")  # 工作日
    np.random.seed(42)
    price = 3000 * np.cumprod(1 + np.random.normal(0.0008, 0.015, len(dates)))  # 模拟价格路径
    df = pd.DataFrame({"price": price}, index=dates)
    df['returns'] = df['price'].pct_change()
    df['cumulative'] = (1 + df['returns']).cumprod()
    return df


def double_ma_strategy(df, short_period: int, long_period: int, drop_na: bool = True):
    """
    双均线交易策略：
    - 当短期均线上穿长期均线 → 买入（持仓为1）
    - 当短期均线下穿长期均线 → 卖出（持仓为0）
    - 使用前一日信号决定当日持仓（避免未来函数）
    
    Args:
        drop_na: 是否删除滚动窗口导致的NaN值
    """
    df = df.copy()
    df['short_ma'] = df['price'].rolling(short_period).mean()
    df['long_ma'] = df['price'].rolling(long_period).mean()
    df['signal'] = np.where(df['short_ma'] > df['long_ma'], 1, 0)  # 信号：1=多头，0=空仓
    df['position'] = df['signal'].shift(1).fillna(0)  # 延迟一天执行（T-1信号用于T日）
    df['strategy_returns'] = df['position'] * df['returns']  # 策略日收益
    
    # 删除滚动窗口导致的NaN值，确保比较的一致性
    if drop_na:
        df = df.dropna(subset=['short_ma', 'long_ma'])
    
    df['cumulative_strategy'] = (1 + df['strategy_returns']).cumprod()  # 策略累计净值
    return df


def calculate_metrics(df, start_date=None):
    """
    计算策略绩效指标：
    - 夏普比率（年化）
    - 最大回撤
    - 总收益率
    
    Args:
        start_date: 如果提供，只计算从该日期开始的数据
    """
    if start_date is not None:
        df = df[df.index >= start_date]
    
    returns = df['strategy_returns'].dropna()
    if len(returns) == 0 or returns.std() == 0:
        return {'sharpe_ratio': 0.0, 'max_drawdown': -1.0, 'total_return': 0.0}
    
    # 年化夏普比率（假设252个交易日）
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    
    # 计算最大回撤
    cum = (1 + returns).cumprod()
    drawdown = (cum - cum.cummax()) / cum.cummax()  # 回撤 = (当前净值 - 历史最高) / 历史最高
    
    return {
        'sharpe_ratio': float(sharpe),
        'max_drawdown': float(drawdown.min()),  # 负值，如 -0.2 表示 20% 回撤
        'total_return': float(cum.iloc[-1] - 1)  # 总收益，如 0.5 表示 50%
    }
