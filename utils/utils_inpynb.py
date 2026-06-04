import math

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import List

def grid_search_matrix(df, fixed_column: str, x_column: str, y_column: str, value_columns: List[str] | None = None, is_fixed_column = True):
    all_fixed_values = [fixed_column]
    if is_fixed_column:
        all_fixed_values = pd.unique(df[fixed_column])

    total_columns = len(all_fixed_values)
    ax_rows = value_columns

    drop_columns = set([fixed_column, x_column, y_column])

    if value_columns == None:
        ax_rows = [c for c in df.columns if c not in drop_columns]

    total_rows = len(ax_rows)

    if is_fixed_column:
        fig, ax = plt.subplots(figsize=(7*total_columns,6*total_rows), ncols=total_columns, nrows= total_rows)
    else:
        fig, ax = plt.subplots(figsize=(7*total_rows,6*total_columns), ncols=total_rows, nrows=total_columns)

    ax_flat = ax.flatten() if total_rows > 1 else ax

    ax_idx = 0
    for value_column in ax_rows:
        for fixed_in in all_fixed_values:
            if is_fixed_column:
                result_heatmap = \
                    df[df[fixed_column] == fixed_in][[x_column, y_column, value_column]]\
                    .pivot(index=y_column, columns=x_column, values=value_column)
            else:
                result_heatmap = \
                    df[[x_column, y_column, value_column]]\
                    .pivot(index=y_column, columns=x_column, values=value_column)
            
            sns.heatmap(result_heatmap, annot=True, cmap=sns.color_palette("magma", as_cmap=True), ax=ax_flat[ax_idx], vmin=0, vmax=1)
            ax_flat[ax_idx].set_xlabel(x_column)
            ax_flat[ax_idx].set_ylabel(y_column)
            if is_fixed_column:
                ax_flat[ax_idx].set_title(f"{value_column} - {fixed_column}: {fixed_in}")
            else:
                ax_flat[ax_idx].set_title(f"{fixed_column}\n{value_column}")

            ax_idx += 1
        
    plt.show()

def mean_std_grid_search_matrix(df, fixed_column: str, x_column: str, y_column: str, value_columns: List[str] | None = None):
    all_fixed_values = pd.unique(df[fixed_column])

    total_columns = len(all_fixed_values)
    
    drop_columns = set([fixed_column, x_column, y_column])
    if value_columns is None:
        ax_rows = [c for c in df.columns if c not in drop_columns]
    else:
        ax_rows = value_columns

    total_rows = len(ax_rows)

    fig, ax = plt.subplots(figsize=(7*total_columns, 6*total_rows), ncols=total_columns, nrows=total_rows)

    # Garante que ax_flat seja sempre iterável, mesmo se for apenas 1 gráfico (1x1)
    if total_rows == 1 and total_columns == 1:
        ax_flat = [ax]
    else:
        ax_flat = ax.flatten()

    ax_idx = 0
    for value_column in ax_rows:
        for fixed_in in all_fixed_values:
            df_filtered = df[df[fixed_column] == fixed_in]
            
            pivot_mean = df_filtered.pivot_table(
                index=y_column, columns=x_column, values=value_column, aggfunc='mean'
            )
            pivot_std = df_filtered.pivot_table(
                index=y_column, columns=x_column, values=value_column, aggfunc='std'
            ).fillna(0)
            
            annot_df = pd.DataFrame(index=pivot_mean.index, columns=pivot_mean.columns)
            for col in pivot_mean.columns:
                annot_df[col] = pivot_mean[col].map(lambda m: f"{m:.2f}") + " ± " + pivot_std[col].map(lambda s: f"{s:.2f}")
                
            sns.heatmap(
                data=pivot_mean, 
                annot=annot_df, 
                fmt="",
                cmap=sns.color_palette("magma", as_cmap=True), 
                ax=ax_flat[ax_idx], 
                vmin=0, 
                vmax=1
            )
            
            ax_flat[ax_idx].set_xlabel(x_column)
            ax_flat[ax_idx].set_ylabel(y_column)
            ax_flat[ax_idx].set_title(f"{value_column} - {fixed_column}: {fixed_in}")

            ax_idx += 1
            
    plt.tight_layout()
    plt.show()