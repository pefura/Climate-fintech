import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch # Ensure this is imported
from matplotlib import ticker as mtick # Ensure this is imported
from datetime import timedelta
import traceback

# --- Configuration ---
APP_DIR = os.path.dirname(os.path.abspath(__file__)) # Should resolve to repo root on Streamlit Cloud

# --- Asset Agnostic Settings ---
HORIZONS_AVAILABLE = [1, 2, 3, 4, 5, 6]

# --- WHEAT Futures Specific Configuration ---
WHEAT_FEATURE_FILE_NAME = "feature_engineered_ZW_F_target_climate_financial.csv"
WHEAT_FULL_FEATURE_PATH = os.path.join(APP_DIR, WHEAT_FEATURE_FILE_NAME)
# Updated model name pattern to match the last WHEAT training script output
WHEAT_MODEL_NAME_PATTERN = "strategy_b_model_h{horizon}_test_eval.joblib"
WHEAT_TARGET_RETURN_COL_TO_DROP_FROM_X = 'Target_Return' # Used if dropping it from X
WHEAT_RAW_PRICE_COL_TO_DROP_FROM_X = 'ZW=F' # Used if dropping it from X
WHEAT_EXISTING_TARGET_COL_IN_FEATURES = 'Target_Direction' # If an old target col is in the feature CSV

# --- CORN ETF Specific Configuration ---
CORN_FEATURE_FILE_NAME = "feature_engineered_CORN_target_climate_financial_trade.csv"
CORN_FULL_FEATURE_PATH = os.path.join(APP_DIR, CORN_FEATURE_FILE_NAME)
# Updated model name pattern to match the CORN training script output
CORN_MODEL_NAME_PATTERN = "CORN_xgb_h{horizon}p_fwd_hpo_test_eval.joblib"
CORN_TARGET_RETURN_COL_TO_DROP_FROM_X = 'Target_Return_CORN'
CORN_RAW_PRICE_COL_TO_DROP_FROM_X = 'CORN'
CORN_EXISTING_TARGET_COL_IN_FEATURES = 'Target_Direction' # If an old target col is in the feature CSV

# Models are in the same directory as the app.py
FULL_MODEL_DIR_PATH_APP = APP_DIR

print(f"--- Path Configuration (app.py) ---")
print(f"APP_DIR / Repo root: {APP_DIR}")
print(f"WHEAT Feature path: {os.path.abspath(WHEAT_FULL_FEATURE_PATH)}")
print(f"CORN Feature path: {os.path.abspath(CORN_FULL_FEATURE_PATH)}")
print(f"Model directory path: {os.path.abspath(FULL_MODEL_DIR_PATH_APP)}")
print(f"--- End Path Configuration ---")

# --- Helper Functions ---
@st.cache_data(ttl=3600)
def load_feature_data(file_path):
    absolute_path = os.path.abspath(file_path)
    if not os.path.exists(absolute_path):
        st.error(f"Data file NOT FOUND at {absolute_path}. Ensure it's in the GitHub repository at the correct location relative to app.py.")
        return None
    try:
        df = pd.read_csv(absolute_path, index_col='Date', parse_dates=['Date'])
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data from {absolute_path}: {e}")
        st.text(traceback.format_exc()); return None

@st.cache_resource(max_entries=15)
def load_model_components(asset_name, horizon_weeks):
    if asset_name == "WHEAT":
        model_filename_pattern = WHEAT_MODEL_NAME_PATTERN
    elif asset_name == "CORN":
        model_filename_pattern = CORN_MODEL_NAME_PATTERN
    else:
        st.error(f"Unknown asset: {asset_name}")
        return None

    model_filename = model_filename_pattern.format(horizon=horizon_weeks)
    model_filepath = os.path.join(FULL_MODEL_DIR_PATH_APP, model_filename)
    
    if os.path.exists(model_filepath):
        try:
            components = joblib.load(model_filepath)
            # Check for essential keys from the updated training script
            # 'test_metrics' itself should be a dictionary
            essential_keys = ['model', 'imputer', 'scaler', 'selected_features', 'imputer_train_columns', 'test_metrics']
            if not all(k in components for k in essential_keys):
                 st.warning(f"Model for {asset_name} H{horizon_weeks} is missing one or more essential components: {essential_keys}. Predictions might fail.")
                 return None # Or handle partially, but better to fail early
            if not isinstance(components['test_metrics'], dict):
                st.warning(f"Model for {asset_name} H{horizon_weeks}: 'test_metrics' is not a dictionary. Cannot retrieve detailed metrics.")
                # Allow to proceed but metrics might be NaN
            return components
        except Exception as e:
            st.error(f"Error loading model for {asset_name} H{horizon_weeks} from {model_filepath}: {e}")
            st.text(traceback.format_exc()); return None
    else:
        st.error(f"Model file NOT FOUND for {asset_name} H{horizon_weeks} at: {model_filepath} (resolved to {os.path.abspath(model_filepath)})")
        return None

def get_latest_features_for_prediction(df_full_data, imputer_train_columns_from_model, asset_config):
    if df_full_data is None or df_full_data.empty: return None, None
    
    cols_to_drop_for_X = []
    if asset_config['existing_target_col_in_features'] and asset_config['existing_target_col_in_features'] in df_full_data.columns:
        cols_to_drop_for_X.append(asset_config['existing_target_col_in_features'])
    if asset_config['raw_price_col_to_drop'] and asset_config['raw_price_col_to_drop'] in df_full_data.columns:
        cols_to_drop_for_X.append(asset_config['raw_price_col_to_drop'])
    if asset_config['target_return_col_to_drop'] and asset_config['target_return_col_to_drop'] in df_full_data.columns:
        cols_to_drop_for_X.append(asset_config['target_return_col_to_drop'])

    # Also remove any multi-horizon target names if they were accidentally included in the feature CSV
    # (The training script should have dropped these before creating X_features_base_full)
    for h in HORIZONS_AVAILABLE:
        cols_to_drop_for_X.append(f"Target_Return_{asset_config['asset_name']}_{h}p_fwd")
        cols_to_drop_for_X.append(f"Target_Direction_{asset_config['asset_name']}_{h}p_fwd")
        # Generic pattern from older scripts if still present by mistake
        cols_to_drop_for_X.append(f"Target_Direction_h{h}")


    X_data = df_full_data.drop(columns=[col for col in cols_to_drop_for_X if col in df_full_data.columns], errors='ignore')
    
    # Ensure all features are numeric
    X_data = X_data.apply(pd.to_numeric, errors='coerce')
    X_data.dropna(axis=1, how='all', inplace=True) # Drop cols that became all NaNs after coercion
        
    if X_data.empty: return None, None
    
    latest_features_raw = X_data.iloc[[-1]] # Get the last row
    last_data_date = X_data.index[-1]
    
    # Reindex to match the columns the IMPUTER was trained on
    latest_features_reindexed_for_imputer = latest_features_raw.reindex(columns=imputer_train_columns_from_model, fill_value=np.nan)
    
    return latest_features_reindexed_for_imputer, last_data_date

def make_prediction(latest_features_df_for_imputer, components):
    if latest_features_df_for_imputer is None or latest_features_df_for_imputer.empty: return None, None, None, None
    try:
        imputer_cols = components['imputer_train_columns'] # Columns imputer was fit on
        selected_feats = components['selected_features']    # Columns model uses after selection
        
        # 1. Impute: Input df should already be aligned with imputer_cols
        features_imputed_df = pd.DataFrame(
            components['imputer'].transform(latest_features_df_for_imputer),
            columns=imputer_cols, # Imputer outputs same columns it was fit on
            index=latest_features_df_for_imputer.index
        )
        
        # 2. Scale: Scaler was fit on the same columns as imputer in the training script
        features_scaled_df = pd.DataFrame(
            components['scaler'].transform(features_imputed_df), # Scale all imputed features
            columns=imputer_cols, # Scaler outputs same columns it was fit on
            index=features_imputed_df.index
        )
        
        # 3. Select Features: Select the subset of features the model was trained on
        features_selected_df = features_scaled_df[selected_feats]
        
        # 4. Predict
        pred_direction = components['model'].predict(features_selected_df)[0]
        pred_proba = components['model'].predict_proba(features_selected_df)[0]

        # Get precision and recall from the 'test_metrics' dictionary
        precision_val = components.get('test_metrics', {}).get('precision_up', np.nan)
        recall_val = components.get('test_metrics', {}).get('recall_up', np.nan)
        
        return int(pred_direction), pred_proba[1], precision_val, recall_val
    except Exception as e:
        st.error(f"Error during prediction pipeline: {e}")
        st.text(traceback.format_exc())
        return None, None, None, None

def get_individual_trading_advice(direction, probability_up, horizon_weeks, prediction_date, precision, recall, asset_name):
    if direction is None or probability_up is None: # Check for np.nan as well
        return f"##### {asset_name} - Week starting {prediction_date.strftime('%Y-%m-%d')} (H{horizon_weeks})\n*Prediction could not be made or metrics are unavailable.*"
    
    advice_lines = []
    advice_lines.append(f"##### {asset_name} - Week starting {prediction_date.strftime('%Y-%m-%d')} (Horizon: {horizon_weeks} Week{'s' if horizon_weeks > 1 else ''} Ahead)")
    
    action_color = "forestgreen" if direction == 1 else "orangered"
    action_text = "UP (Consider LONG)" if direction == 1 else "DOWN (Consider SHORT or AVOID)"
    advice_lines.append(f"- **Prediction:** <span style='color:{action_color}; font-weight:bold;'>{action_text}</span>")
    
    prob_text = "N/A" if pd.isna(probability_up) else f"{probability_up:.2%}"
    advice_lines.append(f"- **Probability of UP:** {prob_text}")

    confidence_level_text = "N/A"
    if pd.notna(probability_up) and pd.notna(direction):
        if direction == 1: # UP
            if probability_up > 0.70: confidence_level_text = "**High**"
            elif probability_up > 0.55: confidence_level_text = "**Moderate**"
            else: confidence_level_text = "Low"
        else: # DOWN
            if probability_up < 0.30: confidence_level_text = "**High** (for DOWN)"
            elif probability_up < 0.45: confidence_level_text = "**Moderate** (for DOWN)"
            else: confidence_level_text = "Low (for DOWN)"
    advice_lines.append(f"- **Confidence in Prediction:** {confidence_level_text}")
    
    metrics_text_parts = []
    if pd.notna(precision): metrics_text_parts.append(f"Precision: {precision:.2%}")
    if pd.notna(recall): metrics_text_parts.append(f"Recall: {recall:.2%}")
    
    if metrics_text_parts:
        advice_lines.append(f"- *Model Test Metrics (for H{horizon_weeks}): {', '.join(metrics_text_parts)}*")
    else:
        advice_lines.append(f"- *Model Test Metrics (for H{horizon_weeks}): Not available*")

    return "\n".join(advice_lines)

def plot_future_predictions_overview(df_forecasts, asset_name, plot_title_suffix=""):
    if df_forecasts.empty:
        st.warning(f"No future forecast data to plot for {asset_name}."); return None
    
    fig, ax_dir = plt.subplots(figsize=(15, 7))
    bar_width = 0.4
    x_positions = np.arange(len(df_forecasts['Prediction_For_Week_Starting']))

    for i, row in df_forecasts.iterrows():
        bar_plot_value = 0 # Default for NaN prediction
        color = 'grey'
        
        if pd.notna(row['Predicted_Direction']):
            bar_plot_value = 1 if row['Predicted_Direction'] == 1 else -1
            color = 'forestgreen' if row['Predicted_Direction'] == 1 else 'orangered'
        ax_dir.bar(x_positions[i], bar_plot_value, width=bar_width, color=color, alpha=0.7)
        
        metric_text_parts = []
        if pd.notna(row['Model_Test_Precision']): metric_text_parts.append(f"P: {row['Model_Test_Precision']:.2f}")
        # else: metric_text_parts.append("P: N/A") # Optional: show N/A explicitly
        if pd.notna(row['Model_Test_Recall']): metric_text_parts.append(f"R: {row['Model_Test_Recall']:.2f}")
        # else: metric_text_parts.append("R: N/A") # Optional

        metric_text = "\n".join(metric_text_parts) if metric_text_parts else "Metrics N/A"

        text_color_metric, fw, va, ty = ('black', 'normal', 'bottom', 0.15)
        if bar_plot_value == 1: ty, va, text_color_metric, fw = (0.85, 'top', 'white', 'bold')
        elif bar_plot_value == -1: ty, va, text_color_metric, fw = (-0.85, 'bottom', 'white', 'bold')
        ax_dir.text(x_positions[i], ty, metric_text, ha='center', va=va, fontsize=7, color=text_color_metric, fontweight=fw)

    ax_dir.set_xticks(x_positions)
    ax_dir.set_xticklabels([d.strftime('%Y-%m-%d') for d in df_forecasts['Prediction_For_Week_Starting']], rotation=45, ha="right")
    ax_dir.set_yticks([-1, 0, 1]); ax_dir.set_yticklabels(['Down', 'N/A', 'Up']) # Simplified for clarity
    ax_dir.axhline(0, color='grey', linewidth=0.8); ax_dir.set_ylabel('Predicted Direction', color='black')
    ax_dir.tick_params(axis='y', labelcolor='black')
    
    ax_prob = ax_dir.twinx()
    # Plot only non-NaN probabilities
    valid_probs = df_forecasts[df_forecasts['Probability_Up'].notna()]
    if not valid_probs.empty:
        valid_x_pos = x_positions[df_forecasts['Probability_Up'].notna().values] # Get corresponding x positions
        ax_prob.plot(valid_x_pos, valid_probs['Probability_Up'], marker='o', linestyle='--', color='dodgerblue', label='Prob(Up) Trend')
        for i, row in valid_probs.iterrows(): # Annotate only valid probabilities
            idx_in_original_x = df_forecasts.index.get_loc(i) # find original index for x_position
            ax_prob.scatter(x_positions[idx_in_original_x], row['Probability_Up'], color='blue', s=50, zorder=5)
            ax_prob.text(x_positions[idx_in_original_x], row['Probability_Up'] + 0.02, f"{row['Probability_Up']:.2f}", ha='center', va='bottom', fontsize=8, color='blue')

    ax_prob.set_ylabel('Predicted Probability of Up Move', color='dodgerblue')
    ax_prob.tick_params(axis='y', labelcolor='dodgerblue'); ax_prob.set_ylim(0, 1)
    ax_prob.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))


    direction_legend_elements = [ Patch(facecolor='forestgreen', alpha=0.7, label='Predicted Up'), Patch(facecolor='orangered', alpha=0.7, label='Predicted Down'), Patch(facecolor='grey', alpha=0.7, label='Prediction N/A')]
    lines_prob, _ = ax_prob.get_legend_handles_labels()
    prob_threshold_line = ax_prob.axhline(0.5, color='dimgray', linestyle=':', linewidth=1, label='Prob Threshold (0.5)')
    all_legend_handles = direction_legend_elements + lines_prob + [prob_threshold_line]
    
    # Filter out None from lines_prob if any plots were skipped due to all NaNs
    all_legend_handles = [h for h in all_legend_handles if h is not None]

    ax_prob.legend(handles=all_legend_handles, loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize='small')
    title = f'{asset_name} - Future Predictions Overview {plot_title_suffix}'
    ax_dir.set_title(title, fontsize=12); ax_dir.set_xlabel('Start Date of Predicted Week')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); return fig

# --- Main Application ---
def main():
    st.set_page_config(layout="wide", page_title="Market Prediction Tool")
    st.title("📈 Commodity Market Predictor")

    selected_asset_display = st.sidebar.selectbox(
        "Select Asset:",
        ("WHEAT Futures (ZW=F)", "CORN ETF (CORN)")
    )

    asset_name_short = ""
    full_feature_path_selected = ""
    asset_specific_config = {}

    if selected_asset_display == "WHEAT Futures (ZW=F)":
        asset_name_short = "WHEAT"
        full_feature_path_selected = WHEAT_FULL_FEATURE_PATH
        asset_specific_config = {
            'asset_name': asset_name_short,
            'existing_target_col_in_features': WHEAT_EXISTING_TARGET_COL_IN_FEATURES,
            'raw_price_col_to_drop': WHEAT_RAW_PRICE_COL_TO_DROP_FROM_X,
            'target_return_col_to_drop': WHEAT_TARGET_RETURN_COL_TO_DROP_FROM_X
        }
    elif selected_asset_display == "CORN ETF (CORN)":
        asset_name_short = "CORN"
        full_feature_path_selected = CORN_FULL_FEATURE_PATH
        asset_specific_config = {
            'asset_name': asset_name_short,
            'existing_target_col_in_features': CORN_EXISTING_TARGET_COL_IN_FEATURES,
            'raw_price_col_to_drop': CORN_RAW_PRICE_COL_TO_DROP_FROM_X,
            'target_return_col_to_drop': CORN_TARGET_RETURN_COL_TO_DROP_FROM_X
        }
    
    st.header(f"Predictions for: {selected_asset_display}")
    st.markdown(f"Predicting future market direction for {asset_name_short} using latest available data.")

    df_full_historical_data = load_feature_data(full_feature_path_selected)
    if df_full_historical_data is None:
        st.error(f"Halting app execution: Feature data for {asset_name_short} could not be loaded."); st.stop()

    st.sidebar.header("Prediction Range Selection")
    min_h_sidebar, max_h_sidebar = min(HORIZONS_AVAILABLE), max(HORIZONS_AVAILABLE)
    
    start_horizon = st.sidebar.number_input(
        f"Start Prediction Horizon (Weeks Ahead, {min_h_sidebar}-{max_h_sidebar}):",
        min_value=min_h_sidebar, max_value=max_h_sidebar, value=min_h_sidebar, step=1, key=f"{asset_name_short}_start_h"
    )
    end_horizon = st.sidebar.number_input(
        f"End Prediction Horizon (Weeks Ahead, {min_h_sidebar}-{max_h_sidebar}):",
        min_value=min_h_sidebar, max_value=max_h_sidebar, value=max_h_sidebar, step=1, key=f"{asset_name_short}_end_h"
    )

    if start_horizon > end_horizon:
        st.sidebar.error("Start horizon cannot be after end horizon. Adjusting end horizon.")
        end_horizon = start_horizon
    
    selected_range_horizons_for_display = [h for h in HORIZONS_AVAILABLE if start_horizon <= h <= end_horizon]

    all_future_predictions_list = []
    latest_features_for_imputer_base, last_data_date_base = None, None
    
    base_model_components = load_model_components(asset_name_short, HORIZONS_AVAILABLE[0])
    if base_model_components and 'imputer_train_columns' in base_model_components:
        imputer_cols_from_base_model = base_model_components['imputer_train_columns']
        latest_features_for_imputer_base, last_data_date_base = get_latest_features_for_prediction(
            df_full_historical_data, imputer_cols_from_base_model, asset_specific_config
        )
    else:
        err_msg = f"Could not load base model (H{HORIZONS_AVAILABLE[0]}) for {asset_name_short} "
        if not base_model_components: err_msg += " (model not found)."
        elif 'imputer_train_columns' not in base_model_components: err_msg += "(missing 'imputer_train_columns' key in payload)."
        else: err_msg += "(unknown reason)."
        st.error(err_msg + " Predictions cannot proceed."); st.stop()

    if latest_features_for_imputer_base is None or last_data_date_base is None:
        st.error(f"Could not retrieve latest features for {asset_name_short}. Predictions cannot proceed."); st.stop()

    st.write(f"*Predictions for {asset_name_short} based on data up to: **{last_data_date_base.strftime('%Y-%m-%d')}***")
    
    any_model_loaded_successfully = False
    for h_offset in HORIZONS_AVAILABLE:
        prediction_date = last_data_date_base + timedelta(weeks=h_offset)
        model_comps_h = load_model_components(asset_name_short, h_offset)
        
        pred_dir, prob_up, prec_val, rec_val = np.nan, np.nan, np.nan, np.nan
        
        if model_comps_h:
            any_model_loaded_successfully = True
            # The latest_features_for_imputer_base is already aligned with the imputer columns
            # as per the base model. We assume all models for an asset use the same imputer_train_columns.
            pred_dir, prob_up, prec_val, rec_val = make_prediction(latest_features_for_imputer_base, model_comps_h)
        
        all_future_predictions_list.append({
            'Horizon': h_offset, 'Prediction_For_Week_Starting': prediction_date,
            'Predicted_Direction': pred_dir if pd.notna(pred_dir) else np.nan, # Ensure NaNs are stored if pred fails
            'Probability_Up': prob_up if pd.notna(prob_up) else np.nan,
            'Model_Test_Precision': prec_val if pd.notna(prec_val) else np.nan,
            'Model_Test_Recall': rec_val if pd.notna(rec_val) else np.nan
        })
    
    if not any_model_loaded_successfully:
        st.error(f"No models could be loaded for {asset_name_short}. Cannot generate predictions or plot."); st.stop()

    df_all_forecasts = pd.DataFrame(all_future_predictions_list)

    if selected_range_horizons_for_display:
        df_plot_forecasts = df_all_forecasts[df_all_forecasts['Horizon'].isin(selected_range_horizons_for_display)]
        plot_header_suffix = ""
        current_min_h_plot = df_plot_forecasts["Horizon"].min() if not df_plot_forecasts.empty else start_horizon
        current_max_h_plot = df_plot_forecasts["Horizon"].max() if not df_plot_forecasts.empty else end_horizon

        if len(selected_range_horizons_for_display) == 1:
            plot_header_suffix = f"for Week {selected_range_horizons_for_display[0]} Ahead"
        elif current_min_h_plot == min(HORIZONS_AVAILABLE) and current_max_h_plot == max(HORIZONS_AVAILABLE):
             plot_header_suffix = f"for Next {len(HORIZONS_AVAILABLE)} Weeks"
        else:
            plot_header_suffix = f"for Weeks {current_min_h_plot} to {current_max_h_plot} Ahead"

        st.header(f"🔮 {asset_name_short} - Future Predictions Overview {plot_header_suffix}")

        if not df_plot_forecasts.empty:
            fig_overview = plot_future_predictions_overview(df_plot_forecasts, asset_name_short)
            if fig_overview: st.pyplot(fig_overview); plt.close(fig_overview)
            else: st.warning(f"Could not generate plot for {asset_name_short} for selected range {start_horizon}-{end_horizon}.")
        else: st.info(f"No prediction data available for {asset_name_short} for selected plot range: Weeks {start_horizon} to {end_horizon}.")
    else: st.warning("No valid horizon range selected for plotting.")

    if selected_range_horizons_for_display:
        st.subheader(f"🎯 {asset_name_short} - Detailed Advice for Selected Range: Weeks {start_horizon} to {end_horizon} Ahead")
        advice_displayed_count = 0
        for h_advice in selected_range_horizons_for_display:
            forecast_row = df_all_forecasts[df_all_forecasts['Horizon'] == h_advice]
            if not forecast_row.empty:
                prediction_data_for_h = forecast_row.iloc[0]
                advice_text = get_individual_trading_advice(
                    prediction_data_for_h['Predicted_Direction'],
                    prediction_data_for_h['Probability_Up'],
                    h_advice,
                    prediction_data_for_h['Prediction_For_Week_Starting'],
                    prediction_data_for_h['Model_Test_Precision'],
                    prediction_data_for_h['Model_Test_Recall'],
                    asset_name_short
                )
                st.markdown(advice_text, unsafe_allow_html=True)
                st.markdown("---")
                advice_displayed_count += 1
        if advice_displayed_count == 0 and not any_model_loaded_successfully:
             pass
        elif advice_displayed_count == 0:
             st.info(f"No prediction data found for {asset_name_short} within selected range to display advice.")
    else:
        st.info("Select a valid range in sidebar to see detailed advice.")

    st.sidebar.markdown("---"); st.sidebar.markdown("Created by Pefura Yone")

if __name__ == "__main__":
    main()
