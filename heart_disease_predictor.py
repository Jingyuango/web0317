import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# 配置页面
st.set_page_config(
    page_title="多任务燃料特性预测系统",
    page_icon="🔥",
    layout="wide"
)

# 加载所有模型
MODELS = {
    'C': 'C综合.pkl',
    'H': 'H综合.pkl',
    'O': 'O综合.pkl',
    'N': 'N综合.pkl',
    'FC': 'FC综合.pkl',
    'VM': 'VM综合.pkl',
    'ASH': 'ASH综合.pkl',
    'HHV': 'HHV综合.pkl',
    'EY': 'EY综合.pkl'
}


loaded_models = {}
try:
    for model_name, model_path in MODELS.items():
        loaded_models[model_name] = joblib.load(model_path)
    st.success("所有模型加载成功！")
except Exception as e:
    st.error(f"模型加载失败：{str(e)}")
    st.stop()

    
# 侧边栏输入
st.sidebar.header("⚙️ 输入参数")
with st.sidebar.expander("基础参数", expanded=True):
    C = st.number_input("C (%)", min_value=0.0, max_value=100.0, value=50.0)
    H = st.number_input("H (%)", min_value=0.0, max_value=100.0, value=6.0)
    O = st.number_input("O (%)", min_value=0.0, max_value=100.0, value=30.0)
    N = st.number_input("N (%)", min_value=0.0, max_value=100.0, value=1.0)
    FC = st.number_input("FC (%)", min_value=0.0, max_value=100.0, value=10.0)
    VM = st.number_input("VM (%)", min_value=0.0, max_value=100.0, value=30.0)
    ASH = st.number_input("ASH (%)", min_value=0.0, max_value=100.0, value=5.0)

with st.sidebar.expander("高级参数", expanded=True):
    HT = st.number_input("HT (°C)", min_value=0, max_value=2000, value=800)
    Ht = st.number_input("Ht (s)", min_value=0.0, max_value=100.0, value=10.0)

# 特征工程
@st.cache_data
def calculate_features(inputs):
    df = pd.DataFrame([inputs])
    
    # 计算衍生特征
    df['o_raw/c_raw'] = df['O'] / df['C'] * 12/16
    df['h_raw/c_raw'] = df['H'] / df['C'] * 12
    df['R'] = np.log(df['Ht'] * np.exp((df['HT'] - 100) / 14.75))
    df['HHV_cal'] = 0.4059 * df['C']
    
    return df

# 主界面
st.title("🔥 多任务燃料特性预测系统")
st.markdown("""
本系统基于机器学习模型预测燃料特性参数，提供以下功能：
- **9个关键参数预测**：C, H, O, N, FC, VM, ASH, HHV, EY
- **特征重要性分析**：SHAP解释模型预测
- **实时计算**：自动计算衍生特征
""")

# 模型预测部分
if st.button("开始预测"):
    with st.spinner('正在计算中...'):
        # 准备输入数据
        input_data = {
            'C': C, 'H': H, 'O': O, 'N': N,
            'FC': FC, 'VM': VM, 'ASH': ASH,
            'HT': HT, 'Ht': Ht
        }
        
        features_df = calculate_features(input_data)
        
        # 创建结果容器
        results = {}
        shap_values = {}
        
        # 遍历所有模型进行预测
        for target, model_path in MODELS.items():
            model = joblib.load(model_path)
            
            # O模型的特殊处理
            if target == 'O':
                X = features_df[['C','H','N','FC','VM','ASH','HHV_cal','o_raw/c_raw','h_raw/c_raw','R']]
            else:
                X = features_df.drop(columns=['HHV_cal','o_raw/c_raw','h_raw/c_raw','R'], errors='ignore')
            
            # 预测并保存结果
            results[target] = model.predict(X)[0]
            
            # 计算SHAP值
            explainer = shap.Explainer(model)
            shap_values[target] = explainer(X)
        
        # 显示预测结果
        st.subheader("📊 预测结果")
        cols = st.columns(3)
        for i, (k, v) in enumerate(results.items()):
            with cols[i%3]:
                st.metric(
                    label=f"{k} 预测值",
                    value=f"{v:.2f}",
                    help=f"{k}参数的预测结果"
                )
        
        # SHAP可视化
        st.subheader("🔍 特征影响分析 (SHAP)")
        selected_target = st.selectbox("选择分析目标参数", list(MODELS.keys()))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_values[selected_target],
            features_df,
            plot_type="bar",
            show=False
        )
        plt.title(f"{selected_target} - 特征重要性")
        st.pyplot(fig)
        
        # 特征关系分析
        st.subheader("📈 特征关系可视化")
        selected_feature = st.selectbox("选择分析特征", features_df.columns)
        
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        plt.scatter(features_df[selected_feature], results[selected_target])
        plt.xlabel(selected_feature)
        plt.ylabel(selected_target)
        plt.title(f"{selected_feature} vs {selected_target}")
        st.pyplot(fig2)

# 数据说明
with st.expander("📚 数据说明", expanded=True):
    st.markdown("""
    **输入参数说明**：
    - C/H/O/N: 元素含量百分比
    - FC/VM/ASH: 工业分析参数（固定碳、挥发分、灰分）
    - HT/Ht: 热解温度和时间
    
    **衍生特征公式**：
    - o_raw/c_raw = (O/C) × 12/16
    - h_raw/c_raw = (H/C) × 12
    - R = ln(Ht × exp((HT-100)/14.75))
    - HHV_cal = 0.4059 × C
    """)

st.markdown("---")
st.caption("科研预测系统 | © 2025 燃料特性分析实验室")