#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2021. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2021/5/27 10:30
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: None
# @FileName : main.py
# @Software : Python3.6; PyCharm; Windows10
# @Hardware : Intel Core i7-4712MQ; NVIDIA GeForce 840M
# @Version  : V2.0 - ZL.Z：2022/5/31 - 2022/6/6
#             特征工程修改，添加统计相关：重测信度、ANOVA、PCA；
#             V1.2: 2022/1/21
#             特征选择中的卡方检验换为方差分析
#             V1.1: 2021/8/23 - 2021/8/24
#             仅针对构音任务的特征进行可视化/筛选.
#             V1.0: 2021/5/27 - 2021/5/2
#             First version.
# @License  : None
# @Brief    : 针对快速言语认知评估提取的特征进行数据探索、数据清洗与特征工程等

from config import *
import os
import datetime
import numpy as np
import pandas as pd
from collections import OrderedDict
import seaborn as sns
import pingouin as pg
import scikit_posthocs as scp
from pca import pca
from itertools import combinations
import matplotlib.pyplot as plt
import ptitprince as pt
from statannotations.Annotator import Annotator
from sklearn.preprocessing import PowerTransformer


def norm_test_power_trans(data: pd.DataFrame, save_dir: str = '', trans=True):
    """
    正态性检验和正态分布转换
    :param data: 输入数据，pd.DataFrame类型，列数据为一个样本，包含或不包含预测值，即y
    :param save_dir: 保存结果路径
    :param trans:是否进行正态分布转换
    :return: 检验结果、转换后的数据（若trans=False，则为原始数据）
    """
    # 正态性检验：Shapiro-Wilk Test
    res = pg.normality(data)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    res.to_csv(os.path.join(save_dir, 'norm_test.csv'))
    print(res)
    data_normal = data
    if trans:
        for i_col in res.index:
            if not res.loc[i_col]['normal']:
                print(f"对 {i_col} 进行正态转换")
                # 使用sklearn中方法利用Box-Cox transform转成正态分布，且进行归一化
                power = PowerTransformer(method='box-cox', standardize=True)
                x = power.fit_transform(data[i_col].to_numpy().reshape(-1, 1))
                data_normal[i_col] = pd.Series(x.reshape(1, -1)[0])
            else:
                pass
        print(data_normal)
    return res, data_normal


def test_retest_icc(data: pd.DataFrame, targets: str, raters: str, save_dir: str, fig=True):
    """
    test-retest reliability重测信度ICC指标
    :param data: 输入数据，pd.DataFrame类型，列数据为一个样本,列包含特征+targets+raters
    :param targets: data中包含目标的列的名称，如为id号（同一个id有两次测验）
    :param raters: data中包含打分者的列的名称，如为session测试时间点
    :param save_dir: 文件保存路径
    :param fig: 对否可视化各个特征的ICC
    :return: feat_icc所有特征对应的ICC
    """
    feat_icc_all = pd.DataFrame()
    for feat_name in data.drop(columns=[targets, raters]).columns:  # 遍历每一个特征
        # 一次对每一个特征求ICC：targets为id号（同一个id有两次测验），raters为session测试时间点，ratings为特征值
        icc_all = pg.intraclass_corr(data, targets=targets, raters=raters,
                                     ratings=feat_name, nan_policy='omit')  # 若包含缺失值，则删除该样本再进行ICC
        # 仅保留ICC(2,k):  average measurement, absolute agreement, 2-way mixed effects model
        icc = icc_all[icc_all['Type'] == 'ICC2k']
        icc.insert(0, 'Features', feat_name)
        feat_icc_all = pd.concat([feat_icc_all, icc])
    feat_icc = feat_icc_all.loc[:, ['Features', 'ICC', 'CI95%', 'F', 'df1', 'df2', 'pval']].reset_index(drop=True)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    feat_icc.to_csv(os.path.join(save_dir, 'icc.csv'), index=False)
    print(feat_icc)
    if fig:
        plt.figure(figsize=(9, 6), dpi=300, tight_layout=True)
        # plt.title('ICC with 95% CI of features', fontdict={'family': font_family, 'size': 16})
        # plt.ylabel('Features', fontdict={'family': font_family, 'size': 14})
        plt.xlabel('ICC (95% CI)', fontdict={'family': font_family, 'size': 14})
        plt.axvline(0.50, c='gray', lw=1.2, ls='--', zorder=0)
        plt.axvline(0.75, c='gray', lw=1.2, ls='--', zorder=0)
        plt.axvline(0.90, c='gray', lw=1.2, ls='--', zorder=0)
        for i_feat in feat_icc.index:
            ci = list(feat_icc.loc[i_feat]['CI95%'])
            plt.hlines(i_feat + 1, xmin=ci[0], xmax=ci[1], colors='k', linestyles='solid', lw=2, zorder=1)
            plt.vlines(ci[0], ymin=i_feat + 0.7, ymax=i_feat + 1.3, colors='k', linestyles='solid', lw=2, zorder=2)
            plt.vlines(ci[1], ymin=i_feat + 0.7, ymax=i_feat + 1.3, colors='k', linestyles='solid', lw=2, zorder=2)
            plt.scatter(feat_icc.loc[i_feat]['ICC'], i_feat + 1, s=80, c='blue')
        plt.yticks(range(0, len(feat_icc.index) + 2), [''] + feat_icc['Features'].tolist() + [''],
                   fontproperties=font_family, fontsize=12)
        # plt.text(0.71, len(feat_icc.index) + 1.5, '0.75', fontdict={'family': font_family, 'size': 10})
        plt.xticks([0.0, 0.2, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 1.0], fontproperties=font_family, fontsize=10)
        plt.ylim(len(feat_icc.index) + 1, 0)
        plt.xlim(0.2, 1.0)
        for sp in plt.gca().spines:
            plt.gca().spines[sp].set_color('k')
            plt.gca().spines[sp].set_linewidth(1)
        plt.gca().tick_params(direction='in', color='k', length=5, width=1)
        plt.grid(False)
        fig_file = os.path.join(save_dir, "ICC.png")
        plt.savefig(fig_file, dpi=600, bbox_inches='tight', pad_inches=0.2)
        plt.savefig(fig_file.replace('.png', '.tif'), dpi=600, bbox_inches='tight', pad_inches=0.2,
                    pil_kwargs={"compression": "tiff_lzw"})
        plt.savefig(fig_file.replace('.png', '.svg'), format='svg', bbox_inches='tight', pad_inches=0.2)
        plt.show()
    return feat_icc


def correlation(data: pd.DataFrame, save_dir: str = '', method='spearman', padjust='none'):
    """
    相关性矩阵
    :param data: 数据，包含预测值，即y
    :param save_dir: 保存结果路径
    :param method: 计算相关性的方法
    :param padjust: p值校正方法
    :return: 相关性矩阵df_r_p_value，下三角为r值，上三角为p值（若校正则为校正后的）
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df_r_p_value = data.rcorr(method=method, upper='pval', decimals=6, padjust=padjust, stars=False)
    df_r_p_value.to_csv(os.path.join(save_dir, "r_p_value.csv"))
    res_sign_corr = data.rcorr(method=method, upper='pval', decimals=6, padjust=padjust, stars=True)
    res_sign_corr.to_csv(os.path.join(save_dir, "sign_corr.csv"))
    print(df_r_p_value)
    r_matrix = np.tril(df_r_p_value.to_numpy(), k=-1).astype(np.float) + \
               np.tril(df_r_p_value.to_numpy(), k=-1).T.astype(np.float) + np.eye(df_r_p_value.shape[0])
    p_matrix = np.triu(df_r_p_value.to_numpy(), k=1).astype(np.float) + \
               np.triu(df_r_p_value.to_numpy(), k=1).T.astype(np.float) + np.eye(df_r_p_value.shape[0])
    pval_stars = {0.0001: '****', 0.001: '***', 0.01: '**', 0.05: '*'}

    def replace_pval(x):
        for key, value in pval_stars.items():
            if x < key:
                return value
        return ''
    p_matrix_star = pd.DataFrame(p_matrix).applymap(replace_pval).to_numpy()
    plt.subplots(figsize=(9, 7), dpi=300, tight_layout=True)
    title_t = f"{method.title()} correction corrected for multiple comparisons with {padjust.title()}."
    # plt.title(title_t, fontdict={'family': font_family, 'size': 14})
    mask = np.zeros_like(r_matrix, dtype=np.bool)  # 定义一个大小一致全为零的矩阵  用布尔类型覆盖原来的类型,此时全为False
    mask[np.triu_indices_from(mask, k=1)] = True  # 返回除对角线的矩阵上三角，并将其设置为true，作为热力图掩码进行屏蔽
    ax = sns.heatmap(r_matrix, mask=mask, annot=True, cbar=False,
                     annot_kws={'size': 10, 'color': 'k'}, fmt='.2f', square=True,
                     cmap="coolwarm", xticklabels=False, yticklabels=False)  # 下三角r值
    mask = np.zeros_like(r_matrix, dtype=np.bool)
    mask[np.tril_indices_from(mask)] = True  # 返回包含对角线的矩阵下三角，并将其设置为true，作为热力图掩码进行屏蔽
    sns.heatmap(r_matrix, mask=mask, annot=p_matrix_star, ax=ax,
                annot_kws={'size': 14, 'weight': 'bold', 'color': 'k'}, fmt='', square=True, vmin=-1, vmax=1,
                cmap="coolwarm", xticklabels=df_r_p_value.columns, yticklabels=df_r_p_value.index,
                cbar_kws={"pad": 0.02})  # 上三角p值
    # cbar_kws={'label': 'p-value', 'orientation': 'horizontal', 'shrink': 1.0, "format": "%.2f", "pad": 0.25}
    plt.tick_params(bottom=False, left=False)
    plt.xticks(fontsize=12, rotation=45, ha="right", rotation_mode="anchor")
    plt.yticks(fontsize=12, rotation=0)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, "heatmap.png"), dpi=600, bbox_inches='tight', pad_inches=0.2)
    plt.savefig(os.path.join(save_dir, "heatmap.tif"), dpi=600, bbox_inches='tight', pad_inches=0.2,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.savefig(os.path.join(save_dir, "heatmap.svg"), bbox_inches='tight', pad_inches=0.2)
    plt.show()
    plt.close()
    return df_r_p_value


def anova_oneway(data: pd.DataFrame, save_dir: str = '', dvs=None, between=None, parametric=True,
                 padjust=None, fig=True, grp_name=None):
    """
    单向方差分析
    :param data: 数据，包含标签，即label
    :param save_dir: 保存结果路径
    :param dvs: 包含因变量的列的名称，列表类型
    :param between: 包含组别的列的名称，即标签名，string类型
    :param parametric: 是否使用非参数检验, False时使用非参数检验
    :param padjust: 使用非参数检验时事后检验的p值校正方法
    :param grp_name: list或OrderedDict类型，元素为string类型,可视化中，纵坐标，即组的名称映射，顺序为从上到下的显示顺序
                list: ['0', '1', '2']，按照所列的顺序从上到下依次显示（元素必须存在于data[between]列中）
                OrderedDict: {'0':'first', '1':'second', '2':'third'}，
                将键替换为值，并按照所列的顺序从上到下依次显示（键必须存在于data[between]列中），替换后的值同时在save_dir文件中对应更改
    :param fig: 对否可视化各个特征的ANOVA
    :return: 方差分析结果，包含事后检验（仅针对参数检验中的显著结果事后检验有意义）
    """
    data[between] = data[between].astype(str)
    if isinstance(grp_name, list):
        order = grp_name
    elif isinstance(grp_name, OrderedDict):
        data[between] = data[between].map(grp_name)
        order = list(grp_name.values())
    elif grp_name is None:
        order = None
    else:
        raise TypeError(f'{grp_name} 错误格式，应为list或OrderedDict格式')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    res = pd.DataFrame()
    for dv in dvs:
        if parametric:
            # 步骤一：方差齐性检验
            data_nonan = data.dropna(subset=[dv])  # pg.homoscedasticity要求非nan
            homo = pg.homoscedasticity(data_nonan, dv=dv, group=between)  # Levene方法
            if homo['equal_var'].tolist()[0]:  # 满足方差齐性，使用标准的ANOVA和Tukey-HSD post-hoc test
                # 步骤二：方差分析(ANOVA)
                aov = pg.anova(data, dv=dv, between=between)
                # 步骤三：事后检验post-hoc test：Tukey-HSD post-hoc test
                hoc = pg.pairwise_tukey(data, dv=dv, between=between, effsize='eta-square')
                p_corrected = 'hoc-p-tukey'
                title_t = f"One-way ANOVA and post-hoc test using Tukey-HSD.\n" \
                          f"$F({int(aov['ddof1'][0])}, {int(aov['ddof2'][0])}) = {aov['F'][0]:.2f}, p = " \
                          f"{aov['p-unc'][0]:.3g}, \eta^2 = {aov['np2'][0]:.3f}, N = {data[dv].count()}$"
            else:  # 不满足方差齐性，使用Welch ANOVA和Games-Howell post-hoc test
                # 步骤二：方差分析(ANOVA)
                aov = pg.welch_anova(data, dv=dv, between=between)
                # 步骤三：事后检验post-hoc test：Games-Howell post-hoc test
                hoc = pg.pairwise_gameshowell(data, dv=dv, between=between, effsize='eta-square')
                p_corrected = 'hoc-pval'
                title_t = f"One-way Welch ANOVA and post-hoc test using Games-Howell.\n" \
                          f"$F({int(aov['ddof1'][0])}, {aov['ddof2'][0]:.2f}) = {aov['F'][0]:.2f}, p = " \
                          f"{aov['p-unc'][0]:.3g}, \eta^2 = {aov['np2'][0]:.3f}, N = {data[dv].count()}$"
            ind = {}
            for i_ind in hoc.index:
                if i_ind == 0:
                    ind[i_ind] = dv
                else:
                    ind[i_ind] = ''
            res_dv = pd.concat([homo.reset_index(drop=True).add_prefix('homo-'), aov.add_prefix('aov-'),
                                hoc.add_prefix('hoc-')], axis=1).rename(index=ind)
            p_values = [d[p_corrected] for i, d in res_dv.iterrows()]
        else:  # 非参数检验直接使用Kruskal-Wallis秩和检验
            # 步骤一：方差分析(ANOVA)
            aov = pg.kruskal(data, dv=dv, between=between).rename(index={'Kruskal': dv}, columns={'H': 'chi-squared'})
            # 增加效应量Effect size：Eta-squared η2 = (H-k+1)/(n-k)，其中H为Kruskal统计量,k为组别数，n为总的样本数
            # ref: 1.	Tomczak M, Tomczak E. The need to report effect size estimates revisited.
            # An overview of some recommended measures of effect size. Trends in Sport Sciences 2014;1(21):19-25.
            # J.Cohen 提出的标准:
            # 0.01- < 0.06时为小效应(small effect)，0.06 – < 0.14时为中等效应(moderate effect)，>= 0.14为大效应(large effect).
            n2 = (aov['chi-squared'][0] - len(data[between].unique()) + 1) / \
                 (data[dv].count() - len(data[between].unique()))
            pd_n2 = pd.DataFrame({'aov-eta-squared': [n2], 'aov-es-magnitude':
                ['small' if n2 < 0.06 else 'large' if n2 >= 0.14 else 'moderate']})
            # 步骤二：事后检验post-hoc test：Conover´s post-hoc test
            data.dropna(subset=[dv], inplace=True)  # scikit_posthocs包需提前去除np.nan才能保持结果与R或SPSS一致
            hoc_p = scp.posthoc_conover(data, dv, between, p_adjust=None).to_numpy()
            hoc_p_adj = scp.posthoc_conover(data, dv, between, p_adjust=padjust).to_numpy()
            pval = hoc_p[np.triu_indices_from(hoc_p, k=1)]
            pval_adj = hoc_p_adj[np.triu_indices_from(hoc_p_adj, k=1)]
            grp = data.groupby(between, observed=True)[dv]
            labels = np.array(list(grp.groups.keys()))
            gmeans = grp.mean().to_numpy()
            ng = aov['ddof1'][0] + 1
            g1, g2 = np.array(list(combinations(np.arange(ng), 2))).T
            hoc = pd.DataFrame({'A': labels[g1], 'B': labels[g2], 'mean(A)': gmeans[g1],
                                'mean(B)': gmeans[g2], 'pval': pval, 'pval_adj': pval_adj})
            ind = {}
            for i_ind in hoc.index:
                if i_ind == 0:
                    ind[i_ind] = dv
                else:
                    ind[i_ind] = ''
            res_dv = pd.concat([aov.reset_index(drop=True).add_prefix('aov-'), pd_n2,
                                hoc.add_prefix('hoc-')], axis=1).rename(index=ind)
            p_values = [d['hoc-pval_adj'] for i, d in res_dv.iterrows()]
            title_t = f"Kruskal-Wallis H-test and post-hoc test using Conover's test with {str(padjust).title()} correction.\n" \
                      f"$\chi^2 ({int(res_dv['aov-ddof1'][0])}) = {res_dv['aov-chi-squared'][0]:.2f}, p = " \
                      f"{res_dv['aov-p-unc'][0]:.3g}, \eta^2 = {res_dv['aov-eta-squared'][0]:.3f}, N = {data[dv].count()}$"
        res = pd.concat([res, res_dv])
        if fig:
            plt.subplots(figsize=(7, 6), dpi=300, tight_layout=True)
            # plt.title(title_t, fontdict={'family': font_family, 'size': 12})
            plot_args = {'data': data, 'order': order, 'orient': 'h'}
            ax = pt.RainCloud(x=between, y=dv, cut=2, width_viol=1.0, pointplot=True, point_size=4, **plot_args)
            ax.xaxis.label.set_size(14)
            ax.yaxis.label.set_size(14)
            plt.ylabel('')
            pairs = [(str(d['hoc-A']), str(d['hoc-B'])) for i, d in res_dv.iterrows()]
            annotator = Annotator(ax, pairs, x=dv, y=between, **plot_args)
            annotator.configure(text_format='star', loc='inside', fontsize=16)
            annotator.set_pvalues(p_values).annotate()
            for sp in plt.gca().spines:
                plt.gca().spines[sp].set_color('k')
                plt.gca().spines[sp].set_linewidth(1)
            plt.xticks(fontproperties=font_family)
            plt.yticks(fontproperties=font_family)
            plt.gca().tick_params(labelsize=12, direction='in', color='k', length=5, width=1)
            plt.tick_params('y', labelsize=14)
            # plt.xlim(left=-0.1)
            plt.grid(False)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(os.path.join(save_dir, f"anova_oneway_{dv.split('(')[0]}.png"),
                        dpi=600, bbox_inches='tight', pad_inches=0.2)
            plt.savefig(os.path.join(save_dir, f"anova_oneway_{dv.split('(')[0]}.tif"),
                        dpi=600, bbox_inches='tight', pad_inches=0.2,
                        pil_kwargs={"compression": "tiff_lzw"})
            plt.savefig(os.path.join(save_dir, f"anova_oneway_{dv.split('(')[0]}.svg"),
                        bbox_inches='tight', pad_inches=0.2)
            plt.show()
            plt.close()
    res.to_csv(os.path.join(save_dir, "anova_oneway.csv"))
    print(res)
    return res


def pca_biplot(data: pd.DataFrame, label: str, save_dir: str, feat_map=None):
    """
    绘制PCA降维的biplot图
    :param data: 输入数据，pd.DataFrame类型，列数据为一个样本，包含标签，即label
    :param label: 标签名
    :param save_dir: 文件保存路径
    :param feat_map: None或者dict类型，dict时键为特征类别，值为该类别包含的特征；默认None时为全部的特征对应各自的类别
    :return: None
    """
    # 绘制PCA的biplot
    model_pca = pca(n_components=2, verbose=2, normalize=True)
    model_pca.fit_transform(data.set_index(label))
    # Pre-processing
    plt.subplots(figsize=(8, 6), dpi=300, tight_layout=True)
    plt.xlabel('PC1', fontdict={'family': font_family, 'size': 14})
    plt.ylabel('PC2', fontdict={'family': font_family, 'size': 14})
    plt.tick_params(labelsize=10)
    y, topfeat, n_feat = model_pca._fig_preprocessing(None, None, None)
    PC = [0, 1]
    topfeat = pd.concat([topfeat.iloc[PC, :], topfeat.loc[~topfeat.index.isin(PC), :]])
    topfeat.reset_index(inplace=True)
    # Collect coefficients
    coeff = model_pca.results['loadings'].iloc[0:n_feat, :]
    # Use the PCs only for scaling purposes
    mean_x = np.mean(model_pca.results['PC'].iloc[:, PC[0]].values)
    mean_y = np.mean(model_pca.results['PC'].iloc[:, PC[1]].values)
    # Plot and scale values for arrows and text by taking the absolute minimum range of the x-axis and y-axis.
    max_axis = np.max(np.abs(model_pca.results['PC'].iloc[:, PC]).min(axis=1))
    max_arrow = np.abs(coeff).max().max()
    scale = (np.max([1, np.round(max_axis / max_arrow, 2)])) * 0.93
    # For vizualization purposes we will keep only the unique feature-names
    topfeat = topfeat.drop_duplicates(subset=['feature'])
    if topfeat.shape[0] < n_feat:
        n_feat = topfeat.shape[0]
    # Plot arrows and text
    if feat_map is None:
        color_dict = dict(zip(topfeat['feature'], sns.color_palette('muted', n_feat)))
    else:
        color_list = sns.color_palette('muted', len(feat_map))
        color_dict = {}
        for i_feat in range(len(feat_map)):
            color_dict = dict(color_dict, **dict(zip(list(feat_map.values())[i_feat],
                                                     [color_list[i_feat]] * len(list(feat_map.values())[i_feat]))))
    for i in range(0, n_feat):
        getfeat = topfeat['feature'].iloc[i]
        label = getfeat
        getcoef = coeff[getfeat].values
        # Set first PC vs second PC direction. Note that these are not neccarily the best loading.
        xarrow = getcoef[PC[0]] * scale  # First PC in the x-axis direction
        yarrow = getcoef[PC[1]] * scale  # Second PC in the y-axis direction
        plt.arrow(mean_x, mean_y, xarrow - mean_x, yarrow - mean_y, color=color_dict[label], head_width=0.025 * scale,
                  length_includes_head=True, lw=1.5, overhang=0.5)
        plt.text(xarrow * 1.12, yarrow * 1.16, label + f"\n({topfeat['loading'].iloc[i]:.3f})",
                 color='k', ha='center', va='center', fontdict={'family': font_family, 'size': 12})
        if feat_map is None:
            lg_t = label
        else:
            lg_t = list(feat_map.keys())[list(feat_map.values()).index([i for i in list(feat_map.values()) if label in i][0])]
        plt.scatter(xarrow, yarrow, s=80, color=color_dict[label], label=lg_t)
    handles, labels = plt.gca().get_legend_handles_labels()  # 去除重复图例标签文本
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper left", prop={'family': font_family, 'size': 12},
               frameon=True, facecolor='w', edgecolor='k')
    for sp in plt.gca().spines:
        plt.gca().spines[sp].set_color('k')
        plt.gca().spines[sp].set_linewidth(1)
    plt.gca().tick_params(direction='in', color='k', length=5, width=1)
    plt.grid(True)
    plt.xticks(fontproperties=font_family)
    plt.yticks(fontproperties=font_family)
    plt.xlim(round(-abs(max_axis)*1.2), round(abs(max_axis)*1.2))
    plt.ylim(round(-abs(max_axis)*1.2), round(abs(max_axis)*1.2))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fig_file = os.path.join(save_dir, "pca_biplot_nop.png")
    plt.savefig(fig_file, dpi=600, bbox_inches='tight', pad_inches=0.2)
    plt.savefig(fig_file.replace('.png', '.tif'), dpi=600, bbox_inches='tight', pad_inches=0.2,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.savefig(fig_file.replace('.png', '.svg'), format='svg', bbox_inches='tight', pad_inches=0.2)
    plt.show()
    plt.close()


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    print("---------- Start Time: %s ----------" % start_time.strftime("%Y-%m-%d %H:%M:%S"))
    current_path = os.getcwd()
    data_path = os.path.join(current_path, "data")
    res_path = os.path.join(current_path, r"results/statistical_analysis")
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    feat_f = os.path.join(data_path, 'features_used_model.csv')
    pd_data_icc_all = pd.read_csv(os.path.join(data_path, 'features_used_icc.csv'))
    pd_data_icc_all.columns = pd_data_icc_all.columns.str.replace(r"^.*-|\(.*$", '', regex=True)  # 删除特征名中-之前字符和(之后字符
    pd_data_icc = pd_data_icc_all.drop(columns=['edu', 'age', 'gender'])
    pd_data_all = pd.read_csv(feat_f)
    pd_data_all.columns = pd_data_all.columns.str.replace(r"^.*-|\(.*$", '', regex=True)  # 删除特征名中-之前字符和(之后字符
    pd_data_all.rename(columns={'Frenchay Score': 'm-FDA score'}, inplace=True)
    pd_data_feat = pd_data_all.drop(columns=['SubjectID', 'Age', 'Gender', 'Edu', 'Group',
                                             'm-FDA score', 'Dysarthria Severity'])
    pd_data_feat_fre = pd_data_all.drop(columns=['SubjectID', 'Age', 'Gender', 'Edu', 'Group', 'Dysarthria Severity'])
    pd_data_feat_frerank = pd_data_all.drop(columns=['SubjectID', 'Age', 'Gender', 'Edu', 'Group', 'm-FDA score'])
    # print(pd_data_feat_frerank[pd_data_feat_frerank['Dysarthria Severity'] == 0].describe([.5]).round(2))
    # print(pd_data_feat_frerank[pd_data_feat_frerank['Dysarthria Severity'] == 1].describe([.5]).round(2))
    # print(pd_data_feat_frerank[pd_data_feat_frerank['Dysarthria Severity'] == 2].describe([.5]).round(2))
    # 缺失值分组均值填充
    feat_frerank_nonan = pd_data_feat_frerank.fillna(pd_data_feat_frerank.groupby('Dysarthria Severity').transform('mean'))
    # 重测信度
    test_retest_icc(pd_data_icc, targets='id', raters='session', save_dir=os.path.join(res_path, 'icc'))
    # 正态性检验
    norm_test_power_trans(pd_data_feat_fre, save_dir=os.path.join(res_path, 'norm_test'), trans=False)
    # 相关性
    correlation(pd_data_feat_fre, save_dir=os.path.join(res_path, 'corr'), method='spearman', padjust='holm')
    # one-way ANOVA，非参数Kruskal-Wallis秩和检验
    anova_oneway(pd_data_feat_frerank, save_dir=os.path.join(res_path, 'anova_oneway'), dvs=pd_data_feat.columns,
                 between='Dysarthria Severity', parametric=False, padjust='fdr_bh',
                 grp_name=OrderedDict({'0': 'Healthy', '1': 'Mild', '2': 'Moderate\n-severe'}))
    # PCA: biplot
    pca_biplot(feat_frerank_nonan, 'Dysarthria Severity', save_dir=os.path.join(res_path, 'pca_biplot'),
               feat_map={'SP task': ['MPT'], 'SZR task': ['SZR'],
                         'DDK task': ['DDK rate', 'DDK regularity', 'VOT', 'pause duration']})

    end_time = datetime.datetime.now()
    print("---------- End Time: %s ----------" % end_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("---------- Time Used: %s ----------" % (end_time - start_time))
    with open(os.path.join(current_path, "results/finished.txt"), "w") as ff:
        ff.write("------------------ Started at %s -------------------\r\n" % start_time.strftime("%Y-%m-%d %H:%M:%S"))
        ff.write("------------------ Finished at %s -------------------\r\n" % end_time.strftime("%Y-%m-%d %H:%M:%S"))
        ff.write("------------------ Time Used %s -------------------\r\n" % (end_time - start_time))
