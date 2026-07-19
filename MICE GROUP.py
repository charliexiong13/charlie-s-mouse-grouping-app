import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from sklearn.preprocessing import StandardScaler
import io
import warnings

warnings.filterwarnings('ignore')


# ==================== 分组函数（蛇形初始化 + 纯SSB优化） ====================
def balance_groups(df, metric_cols, n_groups, n_restarts=30, n_iter=2000):
    """
    针对"各组均值接近总体均值"优化的分组算法：
    1. 蛇形分配：按第一个指标排序，S形循环分配，确保该指标组间绝对均衡。
    2. 局部微调：在蛇形基础上进行小范围随机重分配和交换优化，仅降低SSB（组间平方和）。
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(df[metric_cols].values)
    n_total = len(df)

    base_size = n_total // n_groups
    remainder = n_total % n_groups
    group_sizes = np.array([base_size + (1 if i < remainder else 0) for i in range(n_groups)])

    total_mean = X.mean(axis=0)

    def compute_ssb(labels):
        """计算组间平方和（各组均值与总体均值的差距）"""
        ssb = 0.0
        for g in range(n_groups):
            mask = labels == g
            if mask.sum() == 0:
                continue
            g_mean = X[mask].mean(axis=0)
            diff = g_mean - total_mean
            ssb += mask.sum() * np.dot(diff, diff)
        return ssb

    # 🔴🔴🔴【核心修改】损失函数只包含SSB，彻底忽略组内差异，严格满足您的需求 🔴🔴🔴
    def total_loss(labels):
        return compute_ssb(labels)

    # ---------- 蛇形初始化 ----------
    first_metric = metric_cols[0]
    sorted_indices = np.argsort(df[first_metric].values)
    labels = np.zeros(n_total, dtype=int)

    for idx_in_sorted, orig_idx in enumerate(sorted_indices):
        group_idx = idx_in_sorted % n_groups
        if (idx_in_sorted // n_groups) % 2 == 1:
            group_idx = n_groups - 1 - group_idx
        labels[orig_idx] = group_idx

    current_loss = total_loss(labels)
    best_labels = labels.copy()
    best_ssb = compute_ssb(labels)

    # ---------- 局部微调（仅优化SSB） ----------
    for restart in range(n_restarts):
        test_labels = best_labels.copy()
        num_to_shuffle = min(10, max(3, n_total // 10))
        shuffle_idx = np.random.choice(n_total, num_to_shuffle, replace=False)
        current_groups = test_labels[shuffle_idx]
        np.random.shuffle(current_groups)
        for i, idx in enumerate(shuffle_idx):
            test_labels[idx] = current_groups[i]

        for g in range(n_groups):
            while np.sum(test_labels == g) > group_sizes[g]:
                overs = np.where(test_labels == g)[0]
                move_idx = np.random.choice(overs)
                available = [h for h in range(n_groups) if np.sum(test_labels == h) < group_sizes[h]]
                if available:
                    test_labels[move_idx] = np.random.choice(available)

        for it in range(n_iter):
            i, j = np.random.choice(n_total, 2, replace=False)
            if test_labels[i] == test_labels[j]:
                continue
            test_labels[i], test_labels[j] = test_labels[j], test_labels[i]
            new_loss = total_loss(test_labels)
            if new_loss < current_loss:
                current_loss = new_loss
            else:
                test_labels[i], test_labels[j] = test_labels[j], test_labels[i]

        improved = True
        while improved:
            improved = False
            for i in range(n_total):
                old_g = test_labels[i]
                best_g = old_g
                best_l = current_loss
                for g in range(n_groups):
                    if g == old_g or np.sum(test_labels == g) >= group_sizes[g]:
                        continue
                    test_labels[i] = g
                    new_l = total_loss(test_labels)
                    if new_l < best_l:
                        best_l = new_l
                        best_g = g
                    test_labels[i] = old_g
                if best_g != old_g:
                    test_labels[i] = best_g
                    current_loss = best_l
                    improved = True

        current_ssb = compute_ssb(test_labels)
        if current_ssb < best_ssb:
            best_ssb = current_ssb
            best_labels = test_labels.copy()

    # ---------- 计算距离指标（仅用于展示，不参与分组决策） ----------
    distances = np.zeros(n_total)  # 到组中心距离
    within_discrepancy = np.zeros(n_total)  # 组内差异

    for g in range(n_groups):
        mask = best_labels == g
        if mask.sum() == 0:
            continue
        group_data = X[mask]
        center = group_data.mean(axis=0)
        distances[mask] = np.linalg.norm(group_data - center, axis=1)

        n_group = mask.sum()
        if n_group > 1:
            indices = np.where(mask)[0]
            for idx, i in enumerate(indices):
                other = indices[np.arange(n_group) != idx]
                within_discrepancy[i] = np.mean([np.linalg.norm(X[i] - X[j]) for j in other])
        else:
            within_discrepancy[mask] = 0.0

    result_df = df.copy()
    result_df['分组'] = [f'G{g + 1}' for g in best_labels]
    result_df['到组中心距离'] = distances.round(4)
    result_df['组内差异'] = within_discrepancy.round(4)

    group_stats = result_df.groupby('分组')[metric_cols].mean().reset_index()

    return result_df, best_ssb, group_stats


# ==================== Streamlit 界面 ====================
st.set_page_config(page_title="小鼠均衡分组工具 - 纯均值匹配版", layout="wide")
st.title("🐭 QINGSHAN'S IMAGINAL DISK💿")

st.markdown("""
**📊 如何判断**：请直接查看下方的 **"各组指标均值对比"** 表格，以及 **"各组与总体均值差异"** 可视化表格（红色偏高，绿色偏低）。
**📌 补充说明**：结果表中的"组内差异"仅用于参考，**不影响**分组决策，您可以忽略它。
""")

uploaded_file = st.file_uploader("📁 上传 Excel 文件", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"读取 Excel 失败: {e}")
        st.stop()

    st.subheader("📊 数据预览")
    st.dataframe(df.head(10), use_container_width=True)

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        st.error("Excel 中没有数值列，无法进行均衡分组！")
        st.stop()

    metric_cols = st.multiselect(
        "🎯 选择用于分组的指标列（**第一个指标将作为蛇形排序依据**，请将体重等最重要指标置于首位）",
        options=numeric_cols,
        default=numeric_cols
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        n_groups = st.number_input("🔢 分组数量", min_value=2, max_value=50, value=2, step=1)
    with col2:
        n_restarts = st.number_input("🔄 局部微调次数", min_value=10, max_value=100, value=30, step=5)
    with col3:
        n_iter = st.number_input("⚡ 交换迭代次数", min_value=1000, max_value=5000, value=2000, step=500)

    if st.button("🚀 开始均衡分组", type="primary", use_container_width=True):
        if not metric_cols:
            st.warning("请至少选择一个指标列！")
        else:
            with st.spinner(f"正在优化分组（全力让各组均值贴近总体均值），请稍候..."):
                result_df, ssb, group_stats = balance_groups(
                    df, metric_cols, n_groups,
                    n_restarts=int(n_restarts),
                    n_iter=int(n_iter)
                )

            st.success("✅ 分组完成！各组均值已最大限度贴近总体均值。")
            st.balloons()

            # ---- 按组别排序 ----
            group_order = [f'G{i + 1}' for i in range(n_groups)]
            result_df['分组'] = pd.Categorical(result_df['分组'], categories=group_order, ordered=True)
            result_df = result_df.sort_values('分组').reset_index(drop=True)
            cols = result_df.columns.tolist()
            cols.remove('分组')
            result_df = result_df[['分组'] + cols]

            # ---- 展示分组结果 ----
            st.subheader("📋 分组结果明细（按组别排列）")
            st.dataframe(result_df, use_container_width=True)

            st.subheader("📊 各组样本数量")
            group_counts = result_df['分组'].value_counts().sort_index()
            count_df = pd.DataFrame({'组别': group_counts.index, '样本数': group_counts.values})
            st.dataframe(count_df, use_container_width=True)

            # ---- ★★★ 核心评判表：各组均值对比 ★★★ ----
            st.subheader("📈 各组指标均值对比（⭐ 核心评判表：数值越接近总体均值越好）")
            numeric_cols_stats = group_stats.select_dtypes(include='number').columns.tolist()
            if numeric_cols_stats:
                st.dataframe(group_stats.style.format({col: "{:.3f}" for col in numeric_cols_stats}),
                             use_container_width=True)
            else:
                st.dataframe(group_stats, use_container_width=True)

            # ---- 显示总体均值参考 ----
            st.subheader("📌 总体均值参考")
            overall_means = df[metric_cols].mean().to_frame().T
            st.dataframe(overall_means.style.format("{:.3f}"), use_container_width=True)

            # ---- 🆕 新增：各组与总体均值差异（可视化） ----
            st.subheader("📊 各组与总体均值差异（组均值 - 总体均值）")
            diff_df = group_stats.set_index('分组')[metric_cols] - overall_means.loc[0, metric_cols]
            # 使用红-绿渐变，红色↑偏高，绿色↓偏低，颜色深浅代表偏离程度
            styled_diff = diff_df.style.background_gradient(
                cmap='RdYlGn',
                axis=None,
                vmin=-diff_df.abs().max().max(),
                vmax=diff_df.abs().max().max()
            ).format("{:.3f}")
            st.dataframe(styled_diff, use_container_width=True)
            st.caption("💡 红色表示该组该指标均值高于总体均值，绿色表示低于总体均值，颜色越深偏离越大。")

            # ---- 显示优化指标 ----
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.metric("📐 组间平方和 (SSB)", f"{ssb:.6f}",
                          help="值越接近0，说明各组均值与总体均值越一致，这是本工具唯一的优化目标")
            with col_metric2:
                mean_values = group_stats[metric_cols].values
                cv = np.std(mean_values, axis=0).mean() / np.mean(mean_values, axis=0).mean()
                st.metric("📊 均值变异系数", f"{cv:.6f}",
                          help="衡量各组均值的变异程度，越小越均衡")

            st.subheader("🔬 单指标方差分析（ANOVA）")
            anova_results = []
            for col in metric_cols:
                groups_data = [result_df[result_df['分组'] == f'G{g + 1}'][col].values
                               for g in range(n_groups)]
                f_stat, p_val = f_oneway(*groups_data)
                anova_results.append({
                    '指标': col,
                    'F 值': f"{f_stat:.4f}",
                    'p 值': f"{p_val:.4f}",
                    '平衡判断': '✅ 平衡 (p>0.05)' if p_val > 0.05 else '⚠️ 有差异 (p≤0.05)'
                })
            anova_df = pd.DataFrame(anova_results)
            st.dataframe(anova_df, use_container_width=True)

            # ---- 导出 ----
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                result_df.to_excel(writer, sheet_name='分组结果', index=False)
                group_stats.to_excel(writer, sheet_name='组间平衡统计', index=False)
                anova_df.to_excel(writer, sheet_name='ANOVA', index=False)
                ssb_df = pd.DataFrame({'指标': ['组间平方和(SSB)'], '数值': [ssb]})
                ssb_df.to_excel(writer, sheet_name='汇总', index=False)

            processed_data = output.getvalue()
            st.download_button(
                label="📥 下载分组结果 Excel",
                data=processed_data,
                file_name="小鼠分组结果_纯均值匹配.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
