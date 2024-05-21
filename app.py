import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 定义 DTW 距离函数
def dtw_distance(series1, series2):
    n, m = len(series1), len(series2)
    dtw_matrix = np.zeros((n + 1, m + 1))

    for i in range(1, n + 1):
        dtw_matrix[i, 0] = float('inf')
    for j in range(1, m + 1):
        dtw_matrix[0, j] = float('inf')
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(series1[i - 1] - series2[j - 1])
            last_min = min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])
            dtw_matrix[i, j] = cost + last_min

    distance = dtw_matrix[n, m]
    path = []

    i, j = n, m
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        i, j = min((i - 1, j), (i, j - 1), (i - 1, j - 1), key=lambda x: dtw_matrix[x[0], x[1]])

    path.reverse()
    return distance, path


# Streamlit 应用程序
st.set_page_config(page_title="外购件与自制件在线分析与预测平台", layout="wide")

# 侧边栏文件上传和参数设置
st.sidebar.title("参数设置")
uploaded_file = st.sidebar.file_uploader("上传测量数据文件", type=["xlsx"])
interval = st.sidebar.selectbox("选择重采样频率", options=["D", "W", "M"], index=0,
                                format_func=lambda x: {"D": "每日", "W": "每周", "M": "每月"}[x])
st.sidebar.markdown("---")
show_trend = st.sidebar.checkbox("显示偏差数据趋势图", value=True)
show_correlation = st.sidebar.checkbox("显示偏差数据相关性图", value=True)
show_dtw = st.sidebar.checkbox("显示DTW对齐路径图", value=True)

# 标签页设置
tabs = st.tabs(["数据预览", "分析结果", "详细分析", "导出结果"])

if uploaded_file is not None:
    # 读取 Excel 文件
    data = pd.read_excel(uploaded_file)

    with tabs[0]:
        st.write("### 数据预览")
        st.dataframe(data.head())

    # 将测量日期转换为 datetime 对象
    data['MEASURE_BEGDATE_a'] = pd.to_datetime(data['MEASURE_BEGDATE_a'])
    data['MEASURE_BEGDATE_b'] = pd.to_datetime(data['MEASURE_BEGDATE_b'])

    # 分别移除每个部件的 NaN 值以及它们的时间戳
    a_clean = data.dropna(subset=['DEVIATION_a'])[['DEVIATION_a', 'MEASURE_BEGDATE_a']].reset_index(drop=True)
    b_clean = data.dropna(subset=['DEVIATION_b'])[['DEVIATION_b', 'MEASURE_BEGDATE_b']].reset_index(drop=True)

    # 提取清洗后的偏差值及其对应的时间戳
    deviation_a_clean = a_clean['DEVIATION_a'].values
    deviation_b_clean = b_clean['DEVIATION_b'].values
    time_a_clean = a_clean['MEASURE_BEGDATE_a'].values
    time_b_clean = b_clean['MEASURE_BEGDATE_b'].values

    # 计算 DTW 距离和路径
    dtw_dist, path = dtw_distance(deviation_a_clean, deviation_b_clean)

    # 重采样数据以进行相关性计算
    data_a_resampled = data.set_index('MEASURE_BEGDATE_a')['DEVIATION_a'].resample(interval).mean().interpolate()
    data_b_resampled = data.set_index('MEASURE_BEGDATE_b')['DEVIATION_b'].resample(interval).mean().interpolate()

    data_merged = pd.merge(data_a_resampled, data_b_resampled, left_index=True, right_index=True, how='inner')

    correlation = data_merged.corr().iloc[0, 1]

    # 线性回归
    X = data_merged[['DEVIATION_a']].values
    y = data_merged['DEVIATION_b'].values
    model = LinearRegression()
    model.fit(X, y)
    regression_coefficient = model.coef_[0]
    intercept = model.intercept_

    # 根据 DTW 距离和相关系数进行分类
    dtw_class = "高度相关" if dtw_dist <= 30 else "中度相关" if dtw_dist <= 60 else "低度相关"

    corr_abs = abs(correlation)
    corr_class = "强相关" if corr_abs >= 0.6 else "中等相关" if corr_abs >= 0.4 else "弱相关" if corr_abs >= 0.2 else "无相关"

    # 分析结果页面
    with tabs[1]:
        st.write("### 分析结果")
        st.write(f"**DTW 距离**: {dtw_dist:.3f}")
        st.write(f"**DTW 分类**: {dtw_class}")
        st.write(f"**相关系数**: {correlation:.3f}")
        st.write(f"**相关系数分类**: {corr_class}")
        st.write(f"**回归系数**: {regression_coefficient:.3f}")
        st.write(f"**截距**: {intercept:.3f}")

    # 详细分析页面
    with tabs[2]:
        st.write("### 详细分析")
        analysis_results = {
            'DTW 距离': round(dtw_dist, 3),
            '相关系数': round(correlation, 3),
            '回归系数': round(regression_coefficient, 3),
            '截距': round(intercept, 3),
            'DTW 分类': dtw_class,
            '相关系数分类': corr_class
        }
        st.dataframe(pd.DataFrame([analysis_results]))

        if show_trend:
            st.write("### 偏差数据趋势图")
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=data_merged.index, y=data_merged['DEVIATION_a'], mode='lines', name='外购件偏差值',
                           line=dict(color="#88C1D0")))
            fig.add_trace(
                go.Scatter(x=data_merged.index, y=data_merged['DEVIATION_b'], mode='lines', name='自制件偏差值',
                           line=dict(color="#FFB27F")))
            fig.update_layout(title='偏差数据趋势图', xaxis_title='时间', yaxis_title='偏差值')
            st.plotly_chart(fig, use_container_width=True)

        if show_correlation:
            st.write("### 偏差数据相关性图")
            fig = px.imshow(data_merged.corr(), text_auto=True, aspect="auto", color_continuous_scale="Blues")
            fig.update_layout(title='偏差数据相关性图', xaxis_title='测量点', yaxis_title='测量点')
            st.plotly_chart(fig, use_container_width=True)

        if show_dtw:
            st.markdown("### DTW 对齐路径图")
            fig, ax = plt.subplots(figsize=(14, 7))
            for (i, j) in path:
                ax.plot([time_a_clean[i], time_b_clean[j]], [deviation_a_clean[i], deviation_b_clean[j]], color='gray',
                        linestyle='dotted')
            ax.plot(time_a_clean, deviation_a_clean, label='外购件偏差值', color="#88C1D0")
            ax.plot(time_b_clean, deviation_b_clean, label='自制件偏差值', color="#FFB27F")
            ax.legend()
            ax.set_xlabel('时间', fontsize=16)
            ax.set_ylabel('偏差值', fontsize=16)
            ax.set_title('DTW 对齐路径图', fontsize=16)
            st.pyplot(fig)

    # 导出结果页面
    with tabs[3]:
        st.write("### 导出分析结果")
        if st.button("导出分析结果"):
            results_df = pd.DataFrame([analysis_results])
            results_df.to_csv("analysis_results.csv", index=False)
            st.success("分析结果已导出: analysis_results.csv")

# 页脚附加信息
st.markdown("---")
st.markdown("© 2024 外购件与自制件在线分析平台. 版权所有.")
