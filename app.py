#app.py
# 导入必要库
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from strategy import load_data, double_ma_strategy, calculate_metrics
from ga_optimizer import run_ga_optimizer  
import warnings
warnings.filterwarnings('ignore')

# 设置页面
st.set_page_config(page_title="双均线策略智能优化", layout="wide")
st.title("📈 双均线交易策略的多目标智能优化")
st.markdown("### 基于NSGA-II遗传算法 | 最大化夏普比率 & 最小化最大回撤")

# 初始化session_state
if 'first_run' not in st.session_state:
    st.session_state.first_run = True
    st.success("🎯 系统启动成功！已加载沪深300数据")

# 加载数据
@st.cache_data(show_spinner="📊 加载市场数据中...")
def get_data():
    return load_data()

df = get_data()

# 确定有效开始日期（跳过滚动窗口的NaN）
if 'effective_start_date' not in st.session_state or st.session_state.effective_start_date is None:
    st.session_state.effective_start_date = df.index[200] if len(df) > 200 else df.index[0]

# 初始化其他session_state变量
default_params = {
    'short_period': 20,
    'long_period': 120,
    'ga_done': False,
    'F': None,
    'X': None,
    'indices': None,
    'optimization_completed': False,
    'optimization_triggered': False
}

for key, default_value in default_params.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# 创建两列布局
col_left, col_right = st.columns([1, 1])

# ==================== 左侧：手动调参 ====================
with col_left:
    st.markdown("## 🔧 手动参数调优")
    
    with st.container():
        st.markdown("### 参数设置")
        col1, col2 = st.columns(2)
        
        with col1:
            short_val = st.slider(
                "短期均线周期", 5, 50, st.session_state.short_period,
                help="短期移动平均线的计算窗口"
            )
        
        with col2:
            long_val = st.slider(
                "长期均线周期", 20, 200, st.session_state.long_period,
                help="长期移动平均线的计算窗口"
            )
        
        # 应用按钮
        if st.button("🚀 应用当前参数", width='stretch'):
            st.session_state.short_period = short_val
            st.session_state.long_period = long_val
            st.rerun()

# 使用当前或默认参数
current_short = st.session_state.short_period
current_long = st.session_state.long_period

# 计算手动策略表现
df_manual = double_ma_strategy(df, current_short, current_long, drop_na=True)
effective_start_date = st.session_state.effective_start_date
if isinstance(effective_start_date, bool) and not effective_start_date:
    effective_start_date = None
metrics_manual = calculate_metrics(df_manual, start_date=effective_start_date)

with col_left:
    # 绩效指标卡片
    st.markdown("### 策略绩效")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("夏普比率", f"{metrics_manual['sharpe_ratio']:.3f}",
                 delta="风险调整收益", delta_color="off")
    with col_m2:
        st.metric("最大回撤", f"{metrics_manual['max_drawdown']:.2%}",
                 delta="风险指标", delta_color="inverse")
    with col_m3:
        st.metric("总收益", f"{metrics_manual['total_return']:.2%}",
                 delta="绝对收益", delta_color="off")
    
    # 资金曲线图
    st.markdown("### 📈 资金曲线对比")
    
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=df.index, y=df['cumulative'],
        name="买入持有", 
        line=dict(width=1.5, color='#1f77b4', dash='dot'),
        opacity=0.7
    ))
    fig_equity.add_trace(go.Scatter(
        x=df_manual.index, y=df_manual['cumulative_strategy'],
        name=f"双均线策略({current_short},{current_long})",
        line=dict(width=2.5, color='#ff7f0e')
    ))
    
    fig_equity.update_layout(
        height=400,
        margin=dict(l=50, r=20, t=40, b=80),  # 增加底部边距，为图例留出空间
        hovermode="x unified",
        xaxis_title=dict(
            text="日期",
            font=dict(size=12, color='black', family='Arial, sans-serif')
        ),
        yaxis_title=dict(
            text="累计收益",
            font=dict(size=12, color='black', family='Arial, sans-serif')
        ),
        # 移除图表内的图例
        showlegend=False,
        font=dict(family="Arial, sans-serif", size=12, color='black'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        title=dict(
            text="资金曲线对比（双均线 vs 买入持有）",
            font=dict(size=16, color='black', family='Arial, sans-serif'),
            x=0.5
        )
    )
    
    # 设置坐标轴
    fig_equity.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#e0e0e0',
        linecolor='black',
        linewidth=1,
        mirror=True,
        title_font=dict(size=12, color='black', family='Arial, sans-serif'),
        tickfont=dict(size=11, color='black', family='Arial, sans-serif'),
        showline=True
    )
    fig_equity.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#e0e0e0',
        linecolor='black',
        linewidth=1,
        mirror=True,
        title_font=dict(size=12, color='black', family='Arial, sans-serif'),
        tickfont=dict(size=11, color='black', family='Arial, sans-serif'),
        showline=True
    )
    
    st.plotly_chart(fig_equity, use_container_width=True)
    
    # 在图表下方添加自定义图例
    st.markdown(f"""
    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; border: 1px solid #dee2e6; margin-top: 10px;">
        <div style="display: flex; align-items: center; justify-content: center; flex-wrap: wrap; gap: 20px;">
            <div style="display: flex; align-items: center;">
                <div style="width: 20px; height: 3px; background-color: #1f77b4; margin-right: 8px; border-style: dashed;"></div>
                <span style="font-size: 12px; color: black;">买入持有</span>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="width: 20px; height: 3px; background-color: #ff7f0e; margin-right: 8px;"></div>
                <span style="font-size: 12px; color: black;">双均线策略({current_short},{current_long})</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==================== 右侧：智能优化 ====================
with col_right:
    st.markdown("## 🤖 智能参数优化")
    
    # 优化按钮
    if st.button("🚀 启动遗传算法优化 (30-50秒)", 
                 width='stretch',
                 type="primary"):
        st.session_state.optimization_triggered = True
        st.rerun()
    
    # 执行优化
    if st.session_state.optimization_triggered and not st.session_state.optimization_completed:
        with st.spinner("🔬 正在进行多目标优化... 请稍候"):
            F, X, indices = run_ga_optimizer(df, n_gen=100, pop_size=60)
            st.session_state.F = F
            st.session_state.X = X
            st.session_state.indices = indices
            st.session_state.ga_done = True
            st.session_state.optimization_completed = True
        st.success("✅ 优化完成！已找到帕累托最优解")
        st.rerun()

# 显示优化结果
if st.session_state.ga_done and st.session_state.F is not None:
    F = st.session_state.F
    X = st.session_state.X
    indices = st.session_state.indices
    
    with col_right:
        # 获取三个关键解
        comp_idx = indices['compromise_idx']
        sharpe_idx = indices['best_sharpe_idx']
        mdd_idx = indices['best_mdd_idx']
        
        comp_short, comp_long = int(np.round(X[comp_idx][0])), int(np.round(X[comp_idx][1]))
        comp_sharpe, comp_mdd = -F[comp_idx, 0], F[comp_idx, 1]
        
        sharpe_short, sharpe_long = int(np.round(X[sharpe_idx][0])), int(np.round(X[sharpe_idx][1]))
        sharpe_val, sharpe_mdd = -F[sharpe_idx, 0], F[sharpe_idx, 1]
        
        mdd_short, mdd_long = int(np.round(X[mdd_idx][0])), int(np.round(X[mdd_idx][1]))
        mdd_sharpe, mdd_val = -F[mdd_idx, 0], F[mdd_idx, 1]
        
        # 优化结果概览
        st.markdown("### 🎯 优化结果对比")
        
        # 创建对比表格
        comparison_data = {
            "优化类型": ["手动参数", "平衡型(推荐)", "激进型", "保守型"],
            "短期均线": [current_short, comp_short, sharpe_short, mdd_short],
            "长期均线": [current_long, comp_long, sharpe_long, mdd_long],
            "夏普比率": [f"{metrics_manual['sharpe_ratio']:.3f}", 
                       f"{comp_sharpe:.3f}", f"{sharpe_val:.3f}", f"{mdd_sharpe:.3f}"],
            "最大回撤": [f"{metrics_manual['max_drawdown']:.2%}", 
                       f"{comp_mdd:.2%}", f"{sharpe_mdd:.2%}", f"{mdd_val:.2%}"],
            "夏普提升": ["-", 
                       f"{(comp_sharpe - metrics_manual['sharpe_ratio']):+.3f}",
                       f"{(sharpe_val - metrics_manual['sharpe_ratio']):+.3f}",
                       f"{(mdd_sharpe - metrics_manual['sharpe_ratio']):+.3f}"],
            "回撤改善": ["-", 
                       f"{(comp_mdd - metrics_manual['max_drawdown']):+.2%}",
                       f"{(sharpe_mdd - metrics_manual['max_drawdown']):+.2%}",
                       f"{(mdd_val - metrics_manual['max_drawdown']):+.2%}"]
        }
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, hide_index=True)
        
        # 应用优化参数的按钮
        st.markdown("### ⚡ 一键应用优化参数")
        
        cols = st.columns(3)
        with cols[0]:
            if st.button("应用平衡型", width='stretch', 
                        help=f"短期{comp_short}天, 长期{comp_long}天"):
                st.session_state.short_period = comp_short
                st.session_state.long_period = comp_long
                st.rerun()
        
        with cols[1]:
            if st.button("应用激进型", width='stretch',
                        help=f"短期{sharpe_short}天, 长期{sharpe_long}天"):
                st.session_state.short_period = sharpe_short
                st.session_state.long_period = sharpe_long
                st.rerun()
        
        with cols[2]:
            if st.button("应用保守型", width='stretch',
                        help=f"短期{mdd_short}天, 长期{mdd_long}天"):
                st.session_state.short_period = mdd_short
                st.session_state.long_period = mdd_long
                st.rerun()

# 帕累托前沿图（横跨两列）
st.markdown("---")
st.markdown("## 📊 帕累托前沿：多目标权衡分析")

if st.session_state.ga_done and st.session_state.F is not None:
    F = st.session_state.F
    X = st.session_state.X
    indices = st.session_state.indices
    
    # 只选择关键的20个点显示
    n_points = min(20, len(F))
    if len(F) > n_points:
        # 均匀采样
        indices_sampled = np.linspace(0, len(F)-1, n_points, dtype=int)
        F_display = F[indices_sampled]
        X_display = X[indices_sampled]
    else:
        F_display = F
        X_display = X
    
    # 获取关键点索引
    sharpe_idx = np.argmax(-F_display[:, 0])
    mdd_idx = np.argmin(F_display[:, 1])
    
    # 重新计算距离找到折衷点
    F_norm = F_display.copy()
    F_norm[:, 0] = (F_norm[:, 0] - F_norm[:, 0].min()) / (F_norm[:, 0].max() - F_norm[:, 0].min() + 1e-10)
    F_norm[:, 1] = (F_norm[:, 1] - F_norm[:, 1].min()) / (F_norm[:, 1].max() - F_norm[:, 1].min() + 1e-10)
    ideal = np.array([0, 0])
    distances = np.sqrt(np.sum((F_norm - ideal)**2, axis=1))
    comp_idx = np.argmin(distances)
    
    # 创建帕累托前沿图
    fig_pareto = go.Figure()
    
    # 帕累托前沿点
    fig_pareto.add_trace(go.Scatter(
        x=-F_display[:, 0], y=F_display[:, 1],
        mode='markers',
        marker=dict(
            size=8,
            color='#1f77b4',
            opacity=0.6,
            line=dict(width=1, color='white')
        ),
        name='帕累托前沿',
        hovertemplate="短期%{text[0]}天, 长期%{text[1]}天<br>夏普: %{x:.3f}<br>回撤: %{y:.3f}<extra></extra>",
        text=[[f"{int(np.round(x[0]))}", f"{int(np.round(x[1]))}"] for x in X_display]
    ))
    
    # 当前手动参数点
    fig_pareto.add_trace(go.Scatter(
        x=[metrics_manual['sharpe_ratio']],
        y=[metrics_manual['max_drawdown']],
        mode='markers+text',
        marker=dict(size=20, color='#2ca02c', symbol='diamond'),
        text=[f"手动({current_short},{current_long})"],
        textposition="top center",
        name='手动参数',
        hovertemplate=f"手动参数({current_short},{current_long})<br>夏普: {metrics_manual['sharpe_ratio']:.3f}<br>回撤: {metrics_manual['max_drawdown']:.3f}<extra></extra>"
    ))
    
    # 平衡型点
    fig_pareto.add_trace(go.Scatter(
        x=[-F_display[comp_idx, 0]],
        y=[F_display[comp_idx, 1]],
        mode='markers+text',
        marker=dict(size=25, color='#ff7f0e', symbol='star'),
        text=["平衡型(推荐)"],
        textposition="top center",
        name='平衡型(推荐)',
        hovertemplate=f"平衡型({int(np.round(X_display[comp_idx][0]))},{int(np.round(X_display[comp_idx][1]))})<br>夏普: {-F_display[comp_idx, 0]:.3f}<br>回撤: {F_display[comp_idx, 1]:.3f}<extra></extra>"
    ))
    
    # 激进型点
    fig_pareto.add_trace(go.Scatter(
        x=[-F_display[sharpe_idx, 0]],
        y=[F_display[sharpe_idx, 1]],
        mode='markers+text',
        marker=dict(size=20, color='#d62728', symbol='triangle-up'),
        text=["激进型"],
        textposition="top center",
        name='激进型',
        hovertemplate=f"激进型({int(np.round(X_display[sharpe_idx][0]))},{int(np.round(X_display[sharpe_idx][1]))})<br>夏普: {-F_display[sharpe_idx, 0]:.3f}<br>回撤: {F_display[sharpe_idx, 1]:.3f}<extra></extra>"
    ))
    
    # 保守型点
    fig_pareto.add_trace(go.Scatter(
        x=[-F_display[mdd_idx, 0]],
        y=[F_display[mdd_idx, 1]],
        mode='markers+text',
        marker=dict(size=20, color='#9467bd', symbol='triangle-down'),
        text=["保守型"],
        textposition="top center",
        name='保守型',
        hovertemplate=f"保守型({int(np.round(X_display[mdd_idx][0]))},{int(np.round(X_display[mdd_idx][1]))})<br>夏普: {-F_display[mdd_idx, 0]:.3f}<br>回撤: {F_display[mdd_idx, 1]:.3f}<extra></extra>"
    ))
    
    # 更新布局
    fig_pareto.update_layout(
        height=500,
        margin=dict(l=60, r=20, t=40, b=80),  # 增加底部边距，为图例留出空间
        title=dict(
            text="帕累托前沿：夏普比率 vs 最大回撤",
            font=dict(size=18, color='black', family='Arial, sans-serif'),
            x=0.5
        ),
        xaxis_title=dict(
            text="夏普比率 (↑ 收益能力)",
            font=dict(size=13, color='black', family='Arial, sans-serif')
        ),
        yaxis_title=dict(
            text="最大回撤 (↓ 风险控制)",
            font=dict(size=13, color='black', family='Arial, sans-serif')
        ),
        # 移除图表内的图例
        showlegend=False,
        hovermode='closest',
        font=dict(family="Arial, sans-serif", size=12, color='black'),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # 添加网格线和坐标轴
    fig_pareto.update_xaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='#e0e0e0',
        linecolor='black',
        linewidth=1,
        mirror=True,
        showline=True,
        title_font=dict(size=13, color='black', family='Arial, sans-serif'),
        tickfont=dict(size=11, color='black', family='Arial, sans-serif'),
        zeroline=False
    )
    fig_pareto.update_yaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='#e0e0e0',
        linecolor='black',
        linewidth=1,
        mirror=True,
        showline=True,
        title_font=dict(size=13, color='black', family='Arial, sans-serif'),
        tickfont=dict(size=11, color='black', family='Arial, sans-serif'),
        zeroline=False
    )
    
    st.plotly_chart(fig_pareto, use_container_width=True)
    
    # 在图表下方添加自定义图例
    st.markdown(f"""
    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; border: 1px solid #dee2e6; margin-top: 10px;">
        <div style="display: flex; align-items: center; justify-content: center; flex-wrap: wrap; gap: 15px;">
            <div style="display: flex; align-items: center;">
                <div style="width: 12px; height: 12px; background-color: #1f77b4; border-radius: 50%; margin-right: 6px;"></div>
                <span style="font-size: 12px; color: black;">帕累托前沿</span>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="width: 12px; height: 12px; background-color: #2ca02c; margin-right: 6px; clip-path: polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%);"></div>
                <span style="font-size: 12px; color: black;">手动参数({current_short},{current_long})</span>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="color: #ff7f0e; font-size: 16px; margin-right: 4px;">★</div>
                <span style="font-size: 12px; color: black;">平衡型(推荐)</span>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="color: #d62728; font-size: 16px; margin-right: 4px;">▲</div>
                <span style="font-size: 12px; color: black;">激进型</span>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="color: #9467bd; font-size: 16px; margin-right: 4px;">▼</div>
                <span style="font-size: 12px; color: black;">保守型</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 图表解读
    st.markdown("---")
    st.markdown("## 🔍 图表解读")
    
    # 使用两列布局
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 图例说明")
        
        # 使用卡片样式展示图例
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #1f77b4; margin-bottom: 10px;">
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <div style="width: 12px; height: 12px; background-color: #1f77b4; border-radius: 50%; margin-right: 8px;"></div>
                <div>
                    <strong style="color: black;">帕累托前沿</strong><br>
                    <span style="color: #666; font-size: 12px;">遗传算法找到的最优解集合，代表了在不同风险收益权衡下的最优参数组合</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #2ca02c; margin-bottom: 10px;">
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <div style="width: 12px; height: 12px; background-color: #2ca02c; margin-right: 8px; clip-path: polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%);"></div>
                <div>
                    <strong style="color: black;">手动参数({current_short},{current_long})</strong><br>
                    <span style="color: #666; font-size: 12px;">当前手动设置的参数，用于与优化结果对比</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #ff7f0e; margin-bottom: 10px;">
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <div style="color: #ff7f0e; font-size: 18px; margin-right: 8px;">★</div>
                <div>
                    <strong style="color: black;">平衡型(推荐)</strong><br>
                    <span style="color: #666; font-size: 12px;">距离"理想点"(0,0)最近，在收益和风险间取得最佳平衡</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("### 📊 优化解析")
        
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #d62728; margin-bottom: 10px;">
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <div style="color: #d62728; font-size: 18px; margin-right: 8px;">▲</div>
                <div>
                    <strong style="color: black;">激进型</strong><br>
                    <span style="color: #666; font-size: 12px;">夏普比率最高，适合风险承受能力强的投资者</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #9467bd; margin-bottom: 10px;">
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <div style="color: #9467bd; font-size: 18px; margin-right: 8px;">▼</div>
                <div>
                    <strong style="color: black;">保守型</strong><br>
                    <span style="color: #666; font-size: 12px;">最大回撤最小，适合风险厌恶型投资者</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 使用建议 - 使用更好的配色
        st.markdown(f"""
        <div style="background-color: #e8f4f8; padding: 15px; border-radius: 8px; border-left: 4px solid #1f77b4; margin-top: 10px;">
            <h4 style="margin-top: 0; color: #1a5276; font-size: 16px;">💡 使用建议</h4>
            <ul style="color: #2c3e50; font-size: 14px; margin-bottom: 0; padding-left: 20px;">
                <li><strong style="color: #2c3e50;">平衡型</strong>适合大多数投资者</li>
                <li><strong style="color: #2c3e50;">激进型</strong>适合追求高收益的风险承受者</li>
                <li><strong style="color: #2c3e50;">保守型</strong>适合风险厌恶型投资者</li>
                <li>点击"应用"按钮一键使用优化参数</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # 优化目标解析
    st.markdown("---")
    col_goal1, col_goal2 = st.columns(2)
    
    with col_goal1:
        st.markdown("""
        <div style="background-color: #f0f8ff; padding: 15px; border-radius: 8px; border: 1px solid #d6e4ff; margin-top: 10px;">
            <h4 style="margin-top: 0; color: #1a5276; font-size: 16px;">📈 优化目标冲突</h4>
            <p style="color: #2c3e50; font-size: 14px; margin-bottom: 0;">
                无法同时最大化夏普和最小化回撤，需要在两者之间寻找最佳平衡
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_goal2:
        st.markdown("""
        <div style="background-color: #f0f8ff; padding: 15px; border-radius: 8px; border: 1px solid #d6e4ff; margin-top: 10px;">
            <h4 style="margin-top: 0; color: #1a5276; font-size: 16px;">📊 帕累托前沿特征</h4>
            <p style="color: #2c3e50; font-size: 14px; margin-bottom: 0;">
                每个点对应一组(短期,长期)参数，右上方向移动提高夏普比率但增加回撤风险，左下方向移动降低回撤风险但降低夏普比率
            </p>
        </div>
        """, unsafe_allow_html=True)

# 重置按钮
st.markdown("---")
if st.session_state.ga_done:
    if st.button("🔄 重新运行优化", width='stretch'):
        st.session_state.optimization_triggered = False
        st.session_state.optimization_completed = False
        st.session_state.ga_done = False
        st.session_state.F = None
        st.session_state.X = None
        st.session_state.indices = None
        st.rerun()

# 侧边栏信息
st.sidebar.markdown("---")
st.sidebar.markdown("""
## 📋 系统说明

**研究课题**：
双均线交易策略的多目标智能优化

**核心方法**：
NSGA-II遗传算法
- 种群规模：60
- 进化代数：100
- 评估次数：≈6000

**优化目标**：
1. 最大化夏普比率
2. 最小化最大回撤

**参数约束**：
- 短期均线：5-50天
- 长期均线：20-200天
- 最小间隔：20天

**数据范围**：
沪深300指数 (2010-2025)
- 训练期间：完整历史数据
- 回测期间：2010-2025
- 交易日：约252天/年
""")
