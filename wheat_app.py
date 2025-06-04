import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch # For custom legend in the new plot
from datetime import timedelta
import traceback

# --- Configuration ---
APP_DIR = os.path.dirname(os.path.abspath(__file__)) # This will be the root of the cloned repo on Streamlit Cloud

# 1. Feature Data from Raw GitHub URL
#    Make sure this URL is correct and the file is publicly accessible.
FULL_FEATURE_PATH_APP = "https://raw.githubusercontent.com/pefura/Climate-fintech/main/feature_engineered_ZW_F_target_climate_financial.csv" # Encapsulate in quotes

# 2. Model Files Location

FULL_MODEL_DIR_PATH_APP = APP_DIR # Models are in the same directory as wheat_app.py


# Optional: Print resolved paths to terminal when Streamlit starts, for verification
# These prints are more useful for local debugging. On Streamlit Cloud, check deployment logs.
print(f"--- Path Configuration (app.py) ---")
print(f"app.py directory (APP_DIR): {APP_DIR}") # On Streamlit Cloud, this will be something like /mount/src/climate-fintech
print(f"Full feature path (URL): {FULL_FEATURE_PATH_APP}")
print(f"Full model directory path (resolved): {os.path.abspath(FULL_MODEL_DIR_PATH_APP)}")
print(f"--- End Path Configuration ---")

TARGET_RETURN_COL = 'Target_Return'
RAW_PRICE_COL_NAME_IN_ORIGINAL_FILE = 'ZW=F'
HORIZONS_AVAILABLE = [1, 2, 3, 4, 5, 6]

# --- Helper Functions ---
@st.cache_data
def load_feature_data(file_path_or_url): # Renamed to reflect it can be a URL
    # No need for os.path.abspath or os.path.exists if it's a URL
    try:
        # Pandas can read directly from a URL
        df = pd.read_csv(file_path_or_url, index_col='Date', parse_dates=['Date'])
        df.sort_index(inplace=True)
        # st.success(f"Feature data loaded successfully from {file_path_or_url}") # Optional success message
        return df
    except Exception as e:
        st.error(f"An error occurred during data loading from {file_path_or_url}: {e}")
        st.text(traceback.format_exc()); return None

@st.cache_resource
def load_model_components(horizon_weeks):
    # FULL_MODEL_DIR_PATH_APP is now APP_DIR (root of the repo)
    model_filename = os.path.join(FULL_MODEL_DIR_PATH_APP, f"strategy_b_model_h{horizon_weeks}.joblib")
    # For Streamlit Cloud, os.path.abspath might give paths like /mount/src/climate-fintech/strategy_b_model_h1.joblib
    # absolute_model_path = os.path.abspath(model_filename) # Good for debugging locally

    # On Streamlit Cloud, relative paths from the script's location are usually fine.
    # If APP_DIR is the root, then model_filename is already effectively correct.
    
    # Let's keep abspath for local debugging print, but primarily rely on model_filename
    # st.info(f"Attempting to load model: {os.path.abspath(model_filename)}")

    if os.path.exists(model_filename): # This check will work on Streamlit Cloud too
        try:
            components = joblib.load(model_filename)
            return components
        except Exception as e:
            st.error(f"Error loading model components for H{horizon_weeks} from {model_filename}: {e}")
            st.text(traceback.format_exc()); return None
    else:
        st.error(f"Model file NOT FOUND for Horizon {horizon_weeks} at expected path: {model_filename} (resolved to {os.path.abspath(model_filename)})")
        return None

# ... (rest of your get_latest_features_for_prediction, make_prediction, get_individual_trading_advice, plot_future_predictions_overview functions remain THE SAME as your last working version) ...
def get_latest_features_for_prediction(df_full_data, all_train_feature_names_from_model):
    if df_full_data is None or df_full_data.empty: return None, None
    cols_to_drop_for_X = [TARGET_RETURN_COL] + [f'Target_Direction_h{h}' for h in HORIZONS_AVAILABLE]
    if RAW_PRICE_COL_NAME_IN_ORIGINAL_FILE in df_full_data.columns:
        if RAW_PRICE_COL_NAME_IN_ORIGINAL_FILE not in cols_to_drop_for_X:
            cols_to_drop_for_X.append(RAW_PRICE_COL_NAME_IN_ORIGINAL_FILE)
    X_data = df_full_data.drop(columns=[col for col in cols_to_drop_for_X if col in df_full_data.columns], errors='ignore')
    non_numeric_cols = X_data.select_dtypes(exclude=np.number).columns
    if len(non_numeric_cols) > 0:
        X_data = X_data.copy()
        for col in non_numeric_cols:
            try: X_data.loc[:, col] = pd.to_numeric(X_data[col], errors='coerce')
            except Exception: X_data.drop(col, axis=1, inplace=True, errors='ignore')
        X_data.dropna(axis=1, how='all', inplace=True)
    if X_data.empty: return None, None
    latest_features_raw = X_data.iloc[[-1]]
    last_data_date = X_data.index[-1]
    latest_features_reindexed = latest_features_raw.reindex(columns=all_train_feature_names_from_model)
    return latest_features_reindexed, last_data_date

def make_prediction(latest_features_df, components):
    if latest_features_df is None or latest_features_df.empty: return None, None
    try:
        imputer, scaler, model = components['imputer'], components['scaler'], components['model']
        all_train_feats, selected_feats = components['all_train_feature_names'], components['selected_features_names']
        if not list(latest_features_df.columns) == list(all_train_feats):
            latest_features_df = latest_features_df.reindex(columns=all_train_feats)
        features_imputed_df = pd.DataFrame(imputer.transform(latest_features_df), columns=all_train_feats, index=latest_features_df.index)
        features_scaled_df = pd.DataFrame(scaler.transform(features_imputed_df), columns=all_train_feats, index=latest_features_df.index)
        features_selected_df = features_scaled_df[selected_feats]
        pred_direction = model.predict(features_selected_df)[0]
        pred_proba = model.predict_proba(features_selected_df)[0]
        return int(pred_direction), pred_proba[1]
    except Exception: return None, None

def get_individual_trading_advice(direction, probability_up, horizon_weeks, prediction_date):
    if direction is None or probability_up is None:
        return f"##### Week starting {prediction_date.strftime('%Y-%m-%d')} (H{horizon_weeks})\n*Prediction could not be made.*"
    advice_lines = []
    advice_lines.append(f"##### Week starting {prediction_date.strftime('%Y-%m-%d')} (Horizon: {horizon_weeks} Week{'s' if horizon_weeks > 1 else ''} Ahead)")
    action_color = "green" if direction == 1 else "red"
    action_text = "UP (Consider LONG)" if direction == 1 else "DOWN (Consider SHORT or AVOID)"
    advice_lines.append(f"- **Prediction:** <span style='color:{action_color}; font-weight:bold;'>{action_text}</span>")
    advice_lines.append(f"- **Probability of UP:** {probability_up:.2%}")
    confidence_level_text = ""
    if direction == 1: # UP
        if probability_up > 0.70: confidence_level_text = "**High**"
        elif probability_up > 0.55: confidence_level_text = "**Moderate**"
        else: confidence_level_text = "Low"
    else: # DOWN
        if probability_up < 0.30: confidence_level_text = "**High**"
        elif probability_up < 0.45: confidence_level_text = "**Moderate**"
        else: confidence_level_text = "Low"
    advice_lines.append(f"- **Confidence in Prediction:** {confidence_level_text}")
    signal_summary = ""
    if "High" in confidence_level_text:
        signal_summary = f"Strong signal for {'upward' if direction == 1 else 'downward'} movement."
    elif "Moderate" in confidence_level_text:
        signal_summary = f"Moderate signal for {'upward' if direction == 1 else 'downward'} movement."
    else: signal_summary = f"Weak signal. Exercise caution or await further confirmation."
    advice_lines.append(f"- **Signal Strength:** *{signal_summary}*")
    return "\n".join(advice_lines)

def plot_future_predictions_overview(df_forecasts, plot_title_suffix=""):
    if df_forecasts.empty:
        st.warning("No future forecast data to plot."); return None
    fig, ax_dir = plt.subplots(figsize=(15, 7))
    bar_width = 0.4
    x_positions = np.arange(len(df_forecasts['Prediction_For_Week_Starting']))
    for i, row in df_forecasts.iterrows():
        bar_plot_value, color = (0, 'grey')
        if pd.notna(row['Predicted_Direction']):
            bar_plot_value, color = (1, 'forestgreen') if row['Predicted_Direction'] == 1 else (-1, 'orangered')
        ax_dir.bar(x_positions[i], bar_plot_value, width=bar_width, color=color, alpha=0.7)
        acc_text = f"Acc: {row['Model_Test_Accuracy']:.2f}" if pd.notna(row['Model_Test_Accuracy']) else "Acc: N/A"
        text_color_acc, fw, va, ty = ('black', 'normal', 'bottom', 0.15)
        if bar_plot_value == 1: ty, va, text_color_acc, fw = (0.85, 'top', 'white', 'bold')
        elif bar_plot_value == -1: ty, va, text_color_acc, fw = (-0.85, 'bottom', 'white', 'bold')
        ax_dir.text(x_positions[i], ty, acc_text, ha='center', va=va, fontsize=8, color=text_color_acc, fontweight=fw)
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
    direction_legend_elements = [
        Patch(facecolor='forestgreen', alpha=0.7, label='Predicted Up'),
        Patch(facecolor='orangered', alpha=0.7, label='Predicted Down'),
        Patch(facecolor='grey', alpha=0.7, label='Prediction N/A')
    ]
    lines_prob, _ = ax_prob.get_legend_handles_labels()
    prob_threshold_line = ax_prob.axhline(0.5, color='dimgray', linestyle=':', linewidth=1, label='Prob Threshold (0.5)')
    all_legend_handles = direction_legend_elements + lines_prob + [prob_threshold_line]
    ax_prob.legend(handles=all_legend_handles, loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize='small')
    title = f'Future Predictions Overview {plot_title_suffix}'
    current_min_h_plot = df_forecasts["Horizon"].min()
    current_max_h_plot = df_forecasts["Horizon"].max()
    if len(df_forecasts) == 1:
        title = f'Future Prediction for Week {current_min_h_plot} Ahead {plot_title_suffix}'
    elif current_min_h_plot != min(HORIZONS_AVAILABLE) or current_max_h_plot != max(HORIZONS_AVAILABLE):
         title = f'Future Predictions for Weeks {current_min_h_plot} to {current_max_h_plot} Ahead {plot_title_suffix}'
    else:
         title = f'Future Predictions Overview for Next {len(HORIZONS_AVAILABLE)} Weeks {plot_title_suffix}'
    ax_dir.set_title(title, fontsize=12)
    ax_dir.set_xlabel('Start Date of Predicted Week')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); return fig

# --- Main Application ---
def main():
    st.set_page_config(layout="wide", page_title="Market Prediction Tool")
    st.title("📈 Market Direction Predictor of Prices of Wheat futures")
    st.markdown("Predict future market direction using the latest available data.")

    df_full_historical_data = load_feature_data(FULL_FEATURE_PATH_APP)
    if df_full_historical_data is None:
        st.error("Halting app execution: Feature data could not be loaded."); st.stop()

    st.sidebar.header("Prediction Range Selection")
    min_h_sidebar, max_h_sidebar = min(HORIZONS_AVAILABLE), max(HORIZONS_AVAILABLE)
    
    global start_horizon, end_horizon # Make accessible for plot title

    start_horizon = st.sidebar.number_input(
        f"Start Prediction Horizon (Weeks Ahead, {min_h_sidebar}-{max_h_sidebar}):",
        min_value=min_h_sidebar, max_value=max_h_sidebar, value=min_h_sidebar, step=1
    )
    end_horizon = st.sidebar.number_input(
        f"End Prediction Horizon (Weeks Ahead, {min_h_sidebar}-{max_h_sidebar}):",
        min_value=min_h_sidebar, max_value=max_h_sidebar, value=max_h_sidebar, step=1
    )

    if start_horizon > end_horizon:
        st.sidebar.error("Start horizon cannot be after end horizon. Adjusting end horizon.")
        end_horizon = start_horizon
    
    selected_range_horizons_for_display = [h for h in HORIZONS_AVAILABLE if start_horizon <= h <= end_horizon]

    all_future_predictions_list = []
    latest_features_base, last_data_date_base = None, None
    
    first_model_components = load_model_components(HORIZONS_AVAILABLE[0])
    if first_model_components:
        latest_features_base, last_data_date_base = get_latest_features_for_prediction(
            df_full_historical_data, first_model_components['all_train_feature_names']
        )
    else:
        st.error(f"Could not load base model (H{HORIZONS_AVAILABLE[0]}) to determine feature set. Predictions cannot proceed."); st.stop()

    if latest_features_base is None or last_data_date_base is None:
        st.error("Could not retrieve latest features from data. Predictions cannot proceed."); st.stop()

    st.write(f"*Predictions based on data up to: **{last_data_date_base.strftime('%Y-%m-%d')}***")
    
    any_model_loaded_successfully = False
    for h_offset in HORIZONS_AVAILABLE:
        prediction_date = last_data_date_base + timedelta(weeks=h_offset)
        model_comps = load_model_components(h_offset)
        pred_dir, prob_up, acc = np.nan, np.nan, np.nan
        if model_comps:
            any_model_loaded_successfully = True
            aligned_feats = latest_features_base.reindex(columns=model_comps['all_train_feature_names'])
            pred_dir, prob_up = make_prediction(aligned_feats, model_comps)
            acc = model_comps.get('test_set_accuracy', np.nan)
        all_future_predictions_list.append({
            'Horizon': h_offset, 'Prediction_For_Week_Starting': prediction_date,
            'Predicted_Direction': pred_dir, 'Probability_Up': prob_up,
            'Model_Test_Accuracy': acc
        })
    
    if not any_model_loaded_successfully:
        st.error("No models could be loaded. Cannot generate predictions or plot."); st.stop()

    df_all_forecasts = pd.DataFrame(all_future_predictions_list)

    if selected_range_horizons_for_display:
        df_plot_forecasts = df_all_forecasts[df_all_forecasts['Horizon'].isin(selected_range_horizons_for_display)]
        plot_header_suffix = ""
        if len(selected_range_horizons_for_display) == 1:
            plot_header_suffix = f"for Week {selected_range_horizons_for_display[0]} Ahead"
        elif start_horizon == min_h_sidebar and end_horizon == max_h_sidebar:
             plot_header_suffix = f"for Next {len(HORIZONS_AVAILABLE)} Weeks"
        else:
            plot_header_suffix = f"for Weeks {start_horizon} to {end_horizon} Ahead"
        st.header(f"🔮 Future Predictions Overview {plot_header_suffix}")
        if not df_plot_forecasts.empty:
            fig_overview = plot_future_predictions_overview(df_plot_forecasts)
            if fig_overview: st.pyplot(fig_overview); plt.close(fig_overview)
            else: st.warning(f"Could not generate plot for selected range {start_horizon}-{end_horizon}.")
        else: st.info(f"No prediction data available for selected plot range: Weeks {start_horizon} to {end_horizon}.")
    else: st.warning("No valid horizon range selected for plotting.")

    if selected_range_horizons_for_display:
        st.subheader(f"🎯 Detailed Advice for Selected Range: Weeks {start_horizon} to {end_horizon} Ahead")
        advice_displayed_count = 0
        for h_advice in selected_range_horizons_for_display:
            prediction_data_for_h = df_all_forecasts[df_all_forecasts['Horizon'] == h_advice].iloc[0] \
                if not df_all_forecasts[df_all_forecasts['Horizon'] == h_advice].empty else None
            if prediction_data_for_h is not None:
                advice_text = get_individual_trading_advice(
                    prediction_data_for_h['Predicted_Direction'],
                    prediction_data_for_h['Probability_Up'],
                    h_advice,
                    prediction_data_for_h['Prediction_For_Week_Starting']
                )
                st.markdown(advice_text, unsafe_allow_html=True)
                if pd.notna(prediction_data_for_h['Model_Test_Accuracy']):
                    st.caption(f"H{h_advice} Model Test Accuracy: {prediction_data_for_h['Model_Test_Accuracy']:.2%}")
                st.markdown("---")
                advice_displayed_count += 1
        if advice_displayed_count == 0 and not any_model_loaded_successfully :
             pass
        elif advice_displayed_count == 0 :
             st.info("No prediction data found within selected range to display advice.")
    else:
        st.info("Select a valid range in sidebar to see detailed advice.")

    st.sidebar.markdown("---"); st.sidebar.markdown("Created by Pefura Yone")

if __name__ == "__main__":
    main()
