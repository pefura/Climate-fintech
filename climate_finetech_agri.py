import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from datetime import timedelta
import traceback

# --- Configuration ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Asset Agnostic Settings ---
HORIZONS_AVAILABLE = [1, 2, 3, 4, 5, 6] # Common for both for now

# --- WHEAT Futures Specific Configuration ---
WHEAT_FEATURE_FILE_NAME = "feature_engineered_ZW_F_target_climate_financial.csv"
WHEAT_FULL_FEATURE_PATH = os.path.join(APP_DIR, WHEAT_FEATURE_FILE_NAME)
WHEAT_MODEL_NAME_PATTERN = "strategy_b_model_h{horizon}.joblib" # As seen in your GitHub
WHEAT_TARGET_RETURN_COL = 'Target_Return' # Used if dropping it from X
WHEAT_RAW_PRICE_COL_TO_DROP = 'ZW=F' # Used if dropping it from X

# --- CORN ETF Specific Configuration ---
CORN_FEATURE_FILE_NAME = "feature_engineered_CORN_target_climate_financial_trade.csv"
CORN_FULL_FEATURE_PATH = os.path.join(APP_DIR, CORN_FEATURE_FILE_NAME)
CORN_MODEL_NAME_PATTERN = "CORN_xgb_h{horizon}p_fwd_tuned.joblib" # From your training script output
CORN_TARGET_RETURN_COL = 'Target_Return_CORN' # If this was the base for `Target_Direction` and needs dropping from X
CORN_RAW_PRICE_COL_TO_DROP = 'CORN' # From your CORN training script

# Models are in the same directory as the app.py (root of the repo on Streamlit Cloud)
FULL_MODEL_DIR_PATH_APP = APP_DIR

print(f"--- Path Configuration (app.py) ---")
print(f"APP_DIR / Repo root: {APP_DIR}")
print(f"WHEAT Feature path: {os.path.abspath(WHEAT_FULL_FEATURE_PATH)}")
print(f"CORN Feature path: {os.path.abspath(CORN_FULL_FEATURE_PATH)}")
print(f"Model directory path: {os.path.abspath(FULL_MODEL_DIR_PATH_APP)}")
print(f"--- End Path Configuration ---")

# --- Helper Functions ---
@st.cache_data(ttl=3600) # Cache data for 1 hour
def load_feature_data(file_path):
    absolute_path = os.path.abspath(file_path)
    if not os.path.exists(absolute_path):
        st.error(f"Data file NOT FOUND at {absolute_path}. Please ensure it's in the GitHub repository at the correct location relative to the app script.")
        # Try to load from raw GitHub URL as a fallback if local path fails (for testing/flexibility)
        # This part is tricky because the file_path here is constructed from APP_DIR.
        # For GitHub deployment, APP_DIR should be correct.
        return None
    try:
        df = pd.read_csv(absolute_path, index_col='Date', parse_dates=['Date'])
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data from {absolute_path}: {e}")
        st.text(traceback.format_exc()); return None

@st.cache_resource(max_entries=15) # Cache a few loaded models
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
    
    # st.info(f"Attempting to load model for {asset_name} H{horizon_weeks}: {os.path.abspath(model_filepath)}") # Debug
    if os.path.exists(model_filepath):
        try:
            components = joblib.load(model_filepath)
            # Ensure essential keys exist from the training script
            if not all(k in components for k in ['model', 'selected_features', 'imputer_train_columns']):
                 st.warning(f"Model for {asset_name} H{horizon_weeks} is missing essential components (model, selected_features, imputer_train_columns). Predictions might fail.")
            return components
        except Exception as e:
            st.error(f"Error loading model for {asset_name} H{horizon_weeks} from {model_filepath}: {e}")
            st.text(traceback.format_exc()); return None
    else:
        st.error(f"Model file NOT FOUND for {asset_name} H{horizon_weeks} at: {model_filepath} (resolved to {os.path.abspath(model_filepath)})")
        return None

def get_latest_features_for_prediction(df_full_data, all_train_feature_names_from_model, asset_config):
    if df_full_data is None or df_full_data.empty: return None, None
    
    # Columns to drop: existing target, raw price, and any other specific return cols for that asset
    cols_to_drop_for_X = [asset_config['existing_target_col']] # The pre-engineered target in the feature file
    if asset_config['raw_price_col_to_drop'] and asset_config['raw_price_col_to_drop'] in df_full_data.columns:
        cols_to_drop_for_X.append(asset_config['raw_price_col_to_drop'])
    if asset_config['target_return_col_to_drop'] and asset_config['target_return_col_to_drop'] in df_full_data.columns:
        cols_to_drop_for_X.append(asset_config['target_return_col_to_drop'])

    # Also remove any multi-horizon target names if they were accidentally included
    for h in HORIZONS_AVAILABLE:
        cols_to_drop_for_X.append(f"{asset_config['asset_name']}_Direction_{h}p_fwd") # General pattern

    X_data = df_full_data.drop(columns=[col for col in cols_to_drop_for_X if col in df_full_data.columns], errors='ignore')
    
    # Ensure all features are numeric, handle errors by trying to convert, then drop if still non-numeric
    non_numeric_cols = X_data.select_dtypes(exclude=np.number).columns
    if len(non_numeric_cols) > 0:
        X_data = X_data.copy() # Avoid SettingWithCopyWarning
        for col in non_numeric_cols:
            try:
                X_data[col] = pd.to_numeric(X_data[col], errors='coerce')
            except Exception:
                X_data.drop(col, axis=1, inplace=True, errors='ignore') # Drop if conversion fails
        # Drop columns that became all NaNs after coercion
        X_data.dropna(axis=1, how='all', inplace=True)
        
    if X_data.empty: return None, None
    
    latest_features_raw = X_data.iloc[[-1]] # Get the last row
    last_data_date = X_data.index[-1]
    
    # Reindex to match the columns the model was trained on (handles missing/extra columns)
    latest_features_reindexed = latest_features_raw.reindex(columns=all_train_feature_names_from_model, fill_value=np.nan)
    
    return latest_features_reindexed, last_data_date

def make_prediction(latest_features_df, components):
    if latest_features_df is None or latest_features_df.empty: return None, None, None, None
    try:
        # Assuming 'imputer_train_columns' are the columns imputer was fit on,
        # and 'selected_features' are the subset scaler and model use.
        imputer_cols = components['imputer_train_columns']
        selected_feats = components['selected_features']
        
        # Align latest_features_df to the columns the imputer expects
        latest_features_aligned_for_imputer = latest_features_df.reindex(columns=imputer_cols, fill_value=np.nan)

        features_imputed_df = pd.DataFrame(
            components['imputer'].transform(latest_features_aligned_for_imputer),
            columns=imputer_cols,
            index=latest_features_aligned_for_imputer.index
        )
        
        # Scale only the selected features (or all if selector passed all)
        # The scaler in components was likely fit on selected_features OR all_train_feature_names
        # For simplicity here, assume scaler was fit on the same set of features that are selected
        # This needs to match how it was done in training!
        # A safer approach: save scaler_cols in payload.
        # For now, assume scaler operates on the same columns that are selected.
        
        # If scaler was fit on all_train_feature_names before selection:
        features_to_scale = features_imputed_df # Scale all imputed features
        features_scaled_df = pd.DataFrame(
            components['scaler'].transform(features_to_scale),
            columns=imputer_cols, # Scaler outputs same columns it was fit on
            index=features_to_scale.index
        )
        features_selected_df = features_scaled_df[selected_feats] # Then select
        
        pred_direction = components['model'].predict(features_selected_df)[0]
        pred_proba = components['model'].predict_proba(features_selected_df)[0]

        # Get precision and recall from the model components if they exist
        precision_val = components.get('test_set_precision', np.nan)
        recall_val = components.get('test_set_recall', np.nan)
        
        return int(pred_direction), pred_proba[1], precision_val, recall_val
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.text(traceback.format_exc())
        return None, None, None, None

def get_individual_trading_advice(direction, probability_up, horizon_weeks, prediction_date, precision, recall, asset_name):
    if direction is None or probability_up is None:
        return f"##### {asset_name} - Week starting {prediction_date.strftime('%Y-%m-%d')} (H{horizon_weeks})\n*Prediction could not be made.*"
    
    advice_lines = []
    advice_lines.append(f"##### {asset_name} - Week starting {prediction_date.strftime('%Y-%m-%d')} (Horizon: {horizon_weeks} Week{'s' if horizon_weeks > 1 else ''} Ahead)")
    
    action_color = "forestgreen" if direction == 1 else "orangered"
    action_text = "UP (Consider LONG)" if direction == 1 else "DOWN (Consider SHORT or AVOID)"
    advice_lines.append(f"- **Prediction:** <span style='color:{action_color}; font-weight:bold;'>{action_text}</span>")
    advice_lines.append(f"- **Probability of UP:** {probability_up:.2%}")

    confidence_level_text = ""
    if direction == 1: # UP
        if probability_up > 0.70: confidence_level_text = "**High**"
        elif probability_up > 0.55: confidence_level_text = "**Moderate**"
        else: confidence_level_text = "Low"
    else: # DOWN
        if probability_up < 0.30: confidence_level_text = "**High** (for DOWN)"
        elif probability_up < 0.45: confidence_level_text = "**Moderate** (for DOWN)"
        else: confidence_level_text = "Low (for DOWN)"
    advice_lines.append(f"- **Confidence in Prediction:** {confidence_level_text}")
    
    if pd.notna(precision) and pd.notna(recall):
        advice_lines.append(f"- *Model Test Metrics (for H{horizon_weeks}): Precision: {precision:.2%}, Recall: {recall:.2%}*")
    elif pd.notna(precision):
        advice_lines.append(f"- *Model Test Metric (for H{horizon_weeks}): Precision: {precision:.2%}*")
    elif pd.notna(recall):
         advice_lines.append(f"- *Model Test Metric (for H{horizon_weeks}): Recall: {recall:.2%}*")

    return "\n".join(advice_lines)

def plot_future_predictions_overview(df_forecasts, asset_name, plot_title_suffix=""):
    # ... (This function remains largely the same, just ensure it uses asset_name in title) ...
    if df_forecasts.empty:
        st.warning(f"No future forecast data to plot for {asset_name}."); return None
    fig, ax_dir = plt.subplots(figsize=(15, 7))
    bar_width = 0.4
    x_positions = np.arange(len(df_forecasts['Prediction_For_Week_Starting']))
    for i, row in df_forecasts.iterrows():
        bar_plot_value, color = (0, 'grey')
        if pd.notna(row['Predicted_Direction']):
            bar_plot_value, color = (1, 'forestgreen') if row['Predicted_Direction'] == 1 else (-1, 'orangered')
        ax_dir.bar(x_positions[i], bar_plot_value, width=bar_width, color=color, alpha=0.7)
        
        # Display Precision if available, else 'N/A'
        metric_text = f"P: {row['Model_Test_Precision']:.2f}" if pd.notna(row['Model_Test_Precision']) else "P: N/A"
        metric_text += f"\nR: {row['Model_Test_Recall']:.2f}" if pd.notna(row['Model_Test_Recall']) else "\nR: N/A"

        text_color_metric, fw, va, ty = ('black', 'normal', 'bottom', 0.15)
        if bar_plot_value == 1: ty, va, text_color_metric, fw = (0.85, 'top', 'white', 'bold')
        elif bar_plot_value == -1: ty, va, text_color_metric, fw = (-0.85, 'bottom', 'white', 'bold')
        ax_dir.text(x_positions[i], ty, metric_text, ha='center', va=va, fontsize=7, color=text_color_metric, fontweight=fw)

    ax_dir.set_xticks(x_positions)
    ax_dir.set_xticklabels([d.strftime('%Y-%m-%d') for d in df_forecasts['Prediction_For_Week_Starting']], rotation=45, ha="right")
    ax_dir.set_yticks([-1, 0, 1]); ax_dir.set_yticklabels(['Down/Neutral', 'N/A', 'Up'])
    ax_dir.axhline(0, color='grey', linewidth=0.8); ax_dir.set_ylabel('Predicted Direction', color='black')
    ax_dir.tick_params(axis='y', labelcolor='black')
    ax_prob = ax_dir.twinx()
    ax_prob.plot(x_positions, df_forecasts['Probability_Up'], marker='o', linestyle='--', color='dodgerblue', label='Prob(Up) Trend')
    for i, row in df_forecasts.iterrows():
        if pd.notna(row['Probability_Up']):
            ax_prob.scatter(x_positions[i], row['Probability_Up'], color='blue', s=50, zorder=5)
            ax_prob.text(x_positions[i], row['Probability_Up'] + 0.02, f"{row['Probability_Up']:.2f}", ha='center', va='bottom', fontsize=8, color='blue')
    ax_prob.set_ylabel('Predicted Probability of Up Move', color='dodgerblue')
    ax_prob.tick_params(axis='y', labelcolor='dodgerblue'); ax_prob.set_ylim(0, 1)
    direction_legend_elements = [ Patch(facecolor='forestgreen', alpha=0.7, label='Predicted Up'), Patch(facecolor='orangered', alpha=0.7, label='Predicted Down'), Patch(facecolor='grey', alpha=0.7, label='Prediction N/A')]
    lines_prob, _ = ax_prob.get_legend_handles_labels()
    prob_threshold_line = ax_prob.axhline(0.5, color='dimgray', linestyle=':', linewidth=1, label='Prob Threshold (0.5)')
    all_legend_handles = direction_legend_elements + lines_prob + [prob_threshold_line]
    ax_prob.legend(handles=all_legend_handles, loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize='small')
    title = f'{asset_name} - Future Predictions Overview {plot_title_suffix}' # Added asset_name
    # ... (rest of title logic from your original script) ...
    ax_dir.set_title(title, fontsize=12); ax_dir.set_xlabel('Start Date of Predicted Week')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); return fig

# --- Main Application ---
def main():
    st.set_page_config(layout="wide", page_title="Market Prediction Tool")
    st.title("📈 Commodity Market Predictor") # Generic title

    # --- Asset Selection ---
    selected_asset = st.sidebar.selectbox(
        "Select Asset:",
        ("WHEAT Futures (ZW=F)", "CORN ETF (CORN)") # Assuming CORN is an ETF for this example
    )

    asset_name_short = ""
    full_feature_path_selected = ""
    asset_specific_config = {}

    if selected_asset == "WHEAT Futures (ZW=F)":
        asset_name_short = "WHEAT"
        full_feature_path_selected = WHEAT_FULL_FEATURE_PATH
        asset_specific_config = {
            'asset_name': asset_name_short,
            'existing_target_col': 'Target_Direction', # WHEAT's original target col name
            'raw_price_col_to_drop': WHEAT_RAW_PRICE_COL_TO_DROP,
            'target_return_col_to_drop': WHEAT_TARGET_RETURN_COL
        }
    elif selected_asset == "CORN ETF (CORN)":
        asset_name_short = "CORN"
        full_feature_path_selected = CORN_FULL_FEATURE_PATH
        asset_specific_config = {
            'asset_name': asset_name_short,
            'existing_target_col': 'Target_Direction', # CORN's original target col name in its feature file
            'raw_price_col_to_drop': CORN_RAW_PRICE_COL_TO_DROP,
            'target_return_col_to_drop': CORN_TARGET_RETURN_COL # If it existed in feature file
        }
    
    st.header(f"Predictions for: {selected_asset}")
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
    latest_features_base, last_data_date_base = None, None
    
    # Load a base model (e.g., H1) to get the 'all_train_feature_names'
    # This assumes all models for an asset were trained on a superset of features before selection
    base_model_components = load_model_components(asset_name_short, HORIZONS_AVAILABLE[0])
    if base_model_components and 'imputer_train_columns' in base_model_components : # Use imputer_train_columns as the definitive list
        all_train_cols = base_model_components['imputer_train_columns']
        latest_features_base, last_data_date_base = get_latest_features_for_prediction(
            df_full_historical_data, all_train_cols, asset_specific_config
        )
    elif base_model_components and 'all_train_feature_names' in base_model_components: # Fallback
        all_train_cols = base_model_components['all_train_feature_names']
        latest_features_base, last_data_date_base = get_latest_features_for_prediction(
            df_full_historical_data, all_train_cols, asset_specific_config
        )
    else:
        st.error(f"Could not load base model (H{HORIZONS_AVAILABLE[0]}) for {asset_name_short} or it's missing feature name list. Predictions cannot proceed."); st.stop()

    if latest_features_base is None or last_data_date_base is None:
        st.error(f"Could not retrieve latest features for {asset_name_short}. Predictions cannot proceed."); st.stop()

    st.write(f"*Predictions for {asset_name_short} based on data up to: **{last_data_date_base.strftime('%Y-%m-%d')}***")
    
    any_model_loaded_successfully = False
    for h_offset in HORIZONS_AVAILABLE:
        prediction_date = last_data_date_base + timedelta(weeks=h_offset) # Assuming weekly data
        model_comps = load_model_components(asset_name_short, h_offset)
        pred_dir, prob_up, prec_val, rec_val = np.nan, np.nan, np.nan, np.nan # Initialize all
        
        if model_comps:
            any_model_loaded_successfully = True
            # Align features specifically for this model if its training feature set differs
            # The model_payload should contain 'imputer_train_columns' which are the full set before selection
            # and 'selected_features' which are the ones used by the actual model.
            features_for_this_model = latest_features_base.reindex(columns=model_comps['imputer_train_columns'], fill_value=np.nan)

            pred_dir, prob_up, prec_val, rec_val = make_prediction(features_for_this_model, model_comps)
        
        all_future_predictions_list.append({
            'Horizon': h_offset, 'Prediction_For_Week_Starting': prediction_date,
            'Predicted_Direction': pred_dir, 'Probability_Up': prob_up,
            'Model_Test_Precision': prec_val, # Changed from Accuracy
            'Model_Test_Recall': rec_val      # Added Recall
        })
    
    if not any_model_loaded_successfully:
        st.error(f"No models could be loaded for {asset_name_short}. Cannot generate predictions or plot."); st.stop()

    df_all_forecasts = pd.DataFrame(all_future_predictions_list)

    if selected_range_horizons_for_display:
        df_plot_forecasts = df_all_forecasts[df_all_forecasts['Horizon'].isin(selected_range_horizons_for_display)]
        # ... (plot title logic remains same) ...
        plot_header_suffix = "" # Default suffix
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
            fig_overview = plot_future_predictions_overview(df_plot_forecasts, asset_name_short) # Pass asset_name
            if fig_overview: st.pyplot(fig_overview); plt.close(fig_overview)
            else: st.warning(f"Could not generate plot for {asset_name_short} for selected range {start_horizon}-{end_horizon}.")
        else: st.info(f"No prediction data available for {asset_name_short} for selected plot range: Weeks {start_horizon} to {end_horizon}.")
    else: st.warning("No valid horizon range selected for plotting.")

    if selected_range_horizons_for_display:
        st.subheader(f"🎯 {asset_name_short} - Detailed Advice for Selected Range: Weeks {start_horizon} to {end_horizon} Ahead")
        advice_displayed_count = 0
        for h_advice in selected_range_horizons_for_display:
            prediction_data_for_h = df_all_forecasts[df_all_forecasts['Horizon'] == h_advice].iloc[0] \
                if not df_all_forecasts[df_all_forecasts['Horizon'] == h_advice].empty else None
            if prediction_data_for_h is not None:
                advice_text = get_individual_trading_advice(
                    prediction_data_for_h['Predicted_Direction'],
                    prediction_data_for_h['Probability_Up'],
                    h_advice,
                    prediction_data_for_h['Prediction_For_Week_Starting'],
                    prediction_data_for_h['Model_Test_Precision'], # Pass Precision
                    prediction_data_for_h['Model_Test_Recall'],    # Pass Recall
                    asset_name_short # Pass asset_name
                )
                st.markdown(advice_text, unsafe_allow_html=True)
                # Removed direct printing of accuracy here, now part of advice text
                st.markdown("---")
                advice_displayed_count += 1
        if advice_displayed_count == 0 and not any_model_loaded_successfully :
             pass # Avoid showing "No prediction data" if no models loaded at all
        elif advice_displayed_count == 0 :
             st.info(f"No prediction data found for {asset_name_short} within selected range to display advice.")
    else:
        st.info("Select a valid range in sidebar to see detailed advice.")

    st.sidebar.markdown("---"); st.sidebar.markdown("Created by Pefura Yone")

if __name__ == "__main__":
    main()
