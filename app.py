import csv
import io
import json
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from services.crowd_counter import analyze_image, count_people


# ============================================================
# INTELLICROWD CONFIGURATION
# ============================================================

APP_NAME = "IntelliCrowd"
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"

OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

HISTORY_FILE = OUTPUT_DIR / "crowd_history.csv"

st.set_page_config(
    page_title="IntelliCrowd | Crowd Intelligence",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================

if "crowd_history" not in st.session_state:
    st.session_state.crowd_history = []

if "last_image_count" not in st.session_state:
    st.session_state.last_image_count = None

if "last_image_name" not in st.session_state:
    st.session_state.last_image_name = None

if "video_results" not in st.session_state:
    st.session_state.video_results = []

if "camera_running" not in st.session_state:
    st.session_state.camera_running = False


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    .stApp {
        background-color: #0b1020;
        color: #e5e7eb;
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stSidebar"] {
        background-color: #11182b;
        border-right: 1px solid #26334d;
    }

    [data-testid="stMetric"] {
        background-color: #141e33;
        border: 1px solid #283750;
        padding: 20px;
        border-radius: 12px;
    }

    [data-testid="stMetricValue"] {
        color: #60a5fa;
    }

    .hero {
        padding: 28px;
        border-radius: 18px;
        background: linear-gradient(
            120deg,
            #172554,
            #123047,
            #164e63
        );
        border: 1px solid #294967;
        margin-bottom: 24px;
    }

    .hero h1 {
        font-size: 36px;
        margin-bottom: 8px;
        color: #ffffff;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 15px;
    }

    .section-card {
        background: #141e33;
        border: 1px solid #283750;
        padding: 18px;
        border-radius: 12px;
        margin-bottom: 12px;
    }

    .status-good {
        color: #4ade80;
        font-weight: 600;
    }

    .muted {
        color: #94a3b8;
    }

    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
        min-height: 42px;
    }

    h1, h2, h3 {
        color: #f8fafc;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def add_history(source, filename, count, duration=None):
    """Save a completed prediction to session history and CSV."""

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "filename": filename,
        "estimated_count": int(count),
        "duration_seconds": duration,
    }

    st.session_state.crowd_history.append(record)

    try:
        row_exists = HISTORY_FILE.exists()

        with open(
            HISTORY_FILE,
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=record.keys(),
            )

            if not row_exists:
                writer.writeheader()

            writer.writerow(record)

    except OSError as error:
        st.warning(f"Could not save history: {error}")


def load_history():
    """Read saved history when available."""

    if HISTORY_FILE.exists():
        try:
            return pd.read_csv(HISTORY_FILE)
        except (OSError, pd.errors.ParserError):
            pass

    return pd.DataFrame(
        columns=[
            "timestamp",
            "source",
            "filename",
            "estimated_count",
            "duration_seconds",
        ]
    )


def count_uploaded_image(uploaded_file):
    """Run the actual CSRNet counting service on an uploaded image."""

    suffix = Path(uploaded_file.name).suffix.lower() or ".jpg"

    temp_path = None
    start_time = time.perf_counter()

    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = Path(temp_file.name)

        predicted_count = count_people(temp_path)

        duration = time.perf_counter() - start_time

        return predicted_count, duration

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def create_line_chart(data, x_column, y_column, title):
    """Create a reusable Plotly line chart."""

    fig = px.line(
        data,
        x=x_column,
        y=y_column,
        title=title,
        markers=True,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1020",
        plot_bgcolor="#11182b",
        font_color="#e5e7eb",
        margin=dict(l=20, r=20, t=55, b=20),
    )

    return fig


def analyze_video(video_path, sample_interval=2.0, max_frames=30):
    """
    Analyze selected frames using the existing CSRNet service.

    Counts are per sampled frame, not a count of unique people.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError("Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 25.0

    video_duration = frame_count / fps if frame_count > 0 else 0

    interval_frames = max(1, int(fps * sample_interval))

    results = []
    frame_index = 0
    processed_frames = 0

    progress = st.progress(0)
    status = st.empty()

    try:
        while processed_frames < max_frames:
            success, frame = cap.read()

            if not success:
                break

            if frame_index % interval_frames == 0:
                # Resize large frames to reduce CPU and memory usage.
                height, width = frame.shape[:2]
                max_dimension = 960

                scale = min(
                    1.0,
                    max_dimension / max(height, width),
                )

                if scale < 1.0:
                    frame = cv2.resize(
                        frame,
                        (
                            int(width * scale),
                            int(height * scale),
                        ),
                        interpolation=cv2.INTER_AREA,
                    )

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                pil_image = Image.fromarray(rgb_frame)

                with tempfile.NamedTemporaryFile(
                    suffix=".jpg",
                    delete=False,
                ) as temp_file:
                    frame_path = Path(temp_file.name)
                    pil_image.save(frame_path, format="JPEG")

                try:
                    predicted_count = count_people(frame_path)
                finally:
                    frame_path.unlink(missing_ok=True)

                timestamp_seconds = frame_index / fps

                results.append(
                    {
                        "timestamp_seconds": timestamp_seconds,
                        "estimated_count": int(predicted_count),
                    }
                )

                processed_frames += 1

                if video_duration > 0:
                    progress.progress(
                        min(
                            1.0,
                            timestamp_seconds / video_duration,
                        )
                    )

                status.write(
                    f"Processed frame {processed_frames} "
                    f"| Time: {timestamp_seconds:.1f}s "
                    f"| Estimated people: {predicted_count}"
                )

            frame_index += 1

    finally:
        cap.release()

    progress.empty()

    return results


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 12px 0 24px 0;">
            <div style="font-size: 34px;">👥</div>
            <div style="
                font-size: 25px;
                font-weight: 800;
                color: #60a5fa;
            ">
                IntelliCrowd
            </div>
            <div style="color: #94a3b8; font-size: 12px;">
                AI Crowd Intelligence Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Image Analysis",
            "Video Analysis",
            "Live Monitoring",
            "Crowd Analytics",
            "Model Evaluation",
            "Results & Reports",
            "Settings",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("### System Status")
    st.markdown(
        '<span class="status-good">● Dashboard online</span>',
        unsafe_allow_html=True,
    )
    st.caption("Inference device: CPU")
    st.caption("Model: CSRNet / SHA weights")


# ============================================================
# PAGE 1: OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        """
        <div class="hero">
            <h1>Intelligent Crowd Monitoring</h1>
            <p>
                Analyze crowd images and video using a pretrained
                deep-learning crowd-counting model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    history = load_history()

    total_analyses = len(history)

    if total_analyses:
        average_count = history["estimated_count"].mean()
        maximum_count = history["estimated_count"].max()
        latest_count = history.iloc[-1]["estimated_count"]
    else:
        average_count = 0
        maximum_count = 0
        latest_count = 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Analyses",
        f"{total_analyses:,}",
    )

    col2.metric(
        "Latest Estimate",
        f"{int(latest_count):,}",
    )

    col3.metric(
        "Average Estimate",
        f"{average_count:.1f}",
    )

    col4.metric(
        "Highest Estimate",
        f"{int(maximum_count):,}",
    )

    st.subheader("Recent Activity")

    if not history.empty:
        st.dataframe(
            history.tail(10).iloc[::-1],
            use_container_width=True,
            hide_index=True,
        )

        chart_data = history.copy()
        chart_data["timestamp"] = pd.to_datetime(
            chart_data["timestamp"],
            errors="coerce",
        )

        st.plotly_chart(
            create_line_chart(
                chart_data,
                "timestamp",
                "estimated_count",
                "Recent Crowd Estimates",
            ),
            use_container_width=True,
        )

    else:
        st.info(
            "No analyses recorded yet. "
            "Start with Image Analysis to generate your first prediction."
        )

    st.subheader("Available Modules")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### 🖼️ Image Analysis")
        st.write("Estimate crowd size in an uploaded image.")

    with c2:
        st.markdown("### 🎞️ Video Analysis")
        st.write("Sample video frames and estimate crowd size over time.")

    with c3:
        st.markdown("### 📊 Crowd Analytics")
        st.write("Review saved estimates and analysis history.")


# ============================================================
# PAGE 2: IMAGE ANALYSIS
# ============================================================

elif page == "Image Analysis":

    st.title("🖼️ Image-Based Crowd Counting")

    st.write(
        "Upload a crowd image to estimate the number of people "
        "and visualize CSRNet's predicted spatial density."
    )

    st.info(
        "The count is a model estimate, not a verified number. "
        "The heatmap visualizes predicted density and does not identify "
        "or track individual people. Validate accuracy against annotated data."
    )

    uploaded_image = st.file_uploader(
        "Choose a crowd image",
        type=["jpg", "jpeg", "png"],
        key="image_upload",
    )

    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image).convert("RGB")

            left, right = st.columns([1.2, 1])

            with left:
                st.image(
                    image,
                    caption=uploaded_image.name,
                    use_container_width=True,
                )
                st.caption(
                    f"Image dimensions: {image.width} × {image.height} pixels"
                )

            with right:
                st.markdown("### Analysis Configuration")
                st.write("Model: CSRNet")
                st.write("Weights: SHA / ShanghaiTech Part A")
                st.write("Inference device: CPU")

                run_analysis = st.button(
                    "🔍 Analyze Image",
                    type="primary",
                    use_container_width=True,
                    key="run_image_analysis",
                )

            if run_analysis:
                suffix = Path(uploaded_image.name).suffix.lower() or ".jpg"
                temp_path = None
                start_time = time.perf_counter()

                try:
                    with tempfile.NamedTemporaryFile(
                        suffix=suffix,
                        delete=False,
                    ) as temp_file:
                        temp_file.write(uploaded_image.getvalue())
                        temp_path = Path(temp_file.name)

                    with st.spinner("Running CSRNet inference and generating density map..."):
                        count, density_map = analyze_image(temp_path)

                    duration = time.perf_counter() - start_time
                    st.session_state.last_image_count = count
                    st.session_state.last_image_name = uploaded_image.name

                    add_history(
                        "Image",
                        uploaded_image.name,
                        count,
                        duration,
                    )

                    st.success("Image analysis completed.")

                    metric1, metric2, metric3 = st.columns(3)
                    metric1.metric("Estimated People", f"{count:,}")
                    metric2.metric("Image Width", f"{image.width:,} px")
                    metric3.metric("Inference Time", f"{duration:.2f} s")

                    st.divider()
                    original_col, density_col = st.columns(2)

                    with original_col:
                        st.subheader("Original Image")
                        st.image(image, use_container_width=True)

                    with density_col:
                        st.subheader("CSRNet Density Heatmap")
                        # Matplotlib is used only to render the heatmap.
                        import matplotlib
                        matplotlib.use("Agg")
                        import matplotlib.pyplot as plt

                        density_for_display = np.maximum(density_map, 0)
                        fig, ax = plt.subplots(figsize=(7, 5))
                        heatmap = ax.imshow(
                            density_for_display,
                            cmap="jet",
                            interpolation="bilinear",
                        )
                        ax.set_title("Predicted Crowd Density")
                        ax.axis("off")
                        fig.colorbar(
                            heatmap,
                            ax=ax,
                            fraction=0.046,
                            pad=0.04,
                            label="Relative predicted density",
                        )
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                    st.caption(
                        "The heatmap is a visualization of CSRNet's spatial output. "
                        "It is not a person-by-person detection map."
                    )

                except Exception as error:
                    st.error(f"Image analysis failed: {error}")

                finally:
                    if temp_path is not None:
                        temp_path.unlink(missing_ok=True)

            elif (
                st.session_state.last_image_count is not None
                and st.session_state.last_image_name == uploaded_image.name
            ):
                st.metric(
                    "Most Recent Estimated Count",
                    f"{st.session_state.last_image_count:,}",
                )
                st.caption(
                    "Click Analyze Image to run inference again and regenerate its heatmap."
                )

        except Exception as error:
            st.error(f"Could not read image: {error}")


# ============================================================
# ============================================================
# PAGE 3: VIDEO ANALYSIS
# ============================================================

elif page == "Video Analysis":

    st.title("🎞️ Recorded Video Analysis")

    st.write(
        "Sample frames from a video and estimate crowd size "
        "at each sampled timestamp."
    )

    st.warning(
        "This is frame-based counting. The same person may appear "
        "in multiple frames, so adding frame counts does not give "
        "the number of unique people in the video."
    )

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload",
    )

    sample_interval = st.slider(
        "Seconds between sampled frames",
        min_value=1,
        max_value=10,
        value=2,
    )

    max_frames = st.slider(
        "Maximum frames to analyze",
        min_value=1,
        max_value=50,
        value=10,
    )

    if uploaded_video is not None:

        st.video(uploaded_video)

        if st.button(
            "Analyze Video",
            type="primary",
        ):

            video_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    suffix=Path(uploaded_video.name).suffix or ".mp4",
                    delete=False,
                ) as temp_file:
                    temp_file.write(uploaded_video.getvalue())
                    video_path = Path(temp_file.name)

                with st.spinner(
                    "Analyzing sampled video frames..."
                ):
                    results = analyze_video(
                        video_path,
                        sample_interval=sample_interval,
                        max_frames=max_frames,
                    )

                if results:
                    result_df = pd.DataFrame(results)

                    st.session_state.video_results = results

                    st.success(
                        f"Processed {len(result_df)} sampled frames."
                    )

                    st.metric(
                        "Average Estimated Crowd",
                        f"{result_df['estimated_count'].mean():.1f}",
                    )

                    st.metric(
                        "Maximum Frame Estimate",
                        f"{result_df['estimated_count'].max():,}",
                    )

                    st.plotly_chart(
                        create_line_chart(
                            result_df,
                            "timestamp_seconds",
                            "estimated_count",
                            "Estimated Crowd Over Video Time",
                        ),
                        use_container_width=True,
                    )

                    st.dataframe(
                        result_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    for index, row in result_df.iterrows():
                        add_history(
                            "Video",
                            uploaded_video.name
                            + f" / frame {index + 1}",
                            int(row["estimated_count"]),
                        )

                    csv_bytes = result_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        "Download Video Analysis CSV",
                        data=csv_bytes,
                        file_name="video_crowd_analysis.csv",
                        mime="text/csv",
                    )

                else:
                    st.warning(
                        "No frames could be analyzed."
                    )

            except Exception as error:
                st.error(f"Video analysis failed: {error}")

            finally:
                if video_path is not None:
                    video_path.unlink(missing_ok=True)

    if st.session_state.video_results:
        st.subheader("Previous Video Results")

        st.dataframe(
            pd.DataFrame(st.session_state.video_results),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# PAGE 4: LIVE MONITORING
# ============================================================

elif page == "Live Monitoring":

    st.title("📹 Live Camera Monitoring")

    st.write(
        "Capture a frame from your local webcam and run crowd counting."
    )

    st.warning(
        "This feature uses a camera connected to the computer "
        "running Streamlit. If deployed remotely, it may access "
        "the server's camera rather than your browser's camera."
    )

    camera_image = st.camera_input(
        "Capture an image from your camera"
    )

    if camera_image is not None:

        camera_pil = Image.open(camera_image).convert("RGB")

        st.image(
            camera_pil,
            caption="Captured camera frame",
            use_container_width=True,
        )

        if st.button(
            "Analyze Captured Frame",
            type="primary",
        ):

            temp_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".jpg",
                    delete=False,
                ) as temp_file:
                    temp_path = Path(temp_file.name)
                    camera_pil.save(temp_path, format="JPEG")

                with st.spinner(
                    "Estimating crowd size..."
                ):
                    count = count_people(temp_path)

                st.metric(
                    "Estimated People in Frame",
                    f"{count:,}",
                )

                add_history(
                    "Camera",
                    "Captured frame",
                    count,
                )

            except Exception as error:
                st.error(
                    f"Camera-frame analysis failed: {error}"
                )

            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)


# ============================================================
# PAGE 5: CROWD ANALYTICS
# ============================================================

elif page == "Crowd Analytics":

    st.title("📊 Crowd Analytics")

    history = load_history()

    if history.empty:

        st.info(
            "No saved analyses are available yet. "
            "Run image or video analysis first."
        )

    else:

        history["timestamp"] = pd.to_datetime(
            history["timestamp"],
            errors="coerce",
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Records",
            f"{len(history):,}",
        )

        col2.metric(
            "Mean Estimate",
            f"{history['estimated_count'].mean():.1f}",
        )

        col3.metric(
            "Maximum Estimate",
            f"{history['estimated_count'].max():,.0f}",
        )

        st.subheader("Crowd Estimates Over Time")

        st.plotly_chart(
            create_line_chart(
                history.dropna(subset=["timestamp"]),
                "timestamp",
                "estimated_count",
                "Crowd Estimate History",
            ),
            use_container_width=True,
        )

        st.subheader("Analysis Source Distribution")

        source_counts = (
            history["source"]
            .value_counts()
            .rename_axis("source")
            .reset_index(name="analyses")
        )

        fig = px.pie(
            source_counts,
            names="source",
            values="analyses",
            title="Analyses by Input Type",
            hole=0.45,
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b1020",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.subheader("Saved Records")

        st.dataframe(
            history.sort_values(
                "timestamp",
                ascending=False,
            ),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# PAGE 6: MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    st.title("🧪 Model Evaluation")

    st.write(
        "Evaluate model predictions against manually annotated "
        "ground-truth crowd counts."
    )

    st.warning(
        "Evaluation metrics are meaningful only when each prediction "
        "is paired with a verified ground-truth count. No accuracy "
        "or benchmark score is fabricated here."
    )

    st.markdown(
        """
        Upload a CSV file with these two columns:

        - `ground_truth`: actual number of people
        - `prediction`: model-estimated number of people

        Each row must refer to the same image or sample.
        """
    )

    sample_template = pd.DataFrame(
        {
            "ground_truth": [50, 100, 150],
            "prediction": [53, 95, 161],
        }
    )

    st.download_button(
        "Download Evaluation CSV Template",
        data=sample_template.to_csv(index=False),
        file_name="evaluation_template.csv",
        mime="text/csv",
    )

    evaluation_file = st.file_uploader(
        "Upload annotated evaluation CSV",
        type=["csv"],
        key="evaluation_upload",
    )

    if evaluation_file is not None:

        try:
            evaluation_df = pd.read_csv(evaluation_file)

            required_columns = {
                "ground_truth",
                "prediction",
            }

            if not required_columns.issubset(
                evaluation_df.columns
            ):
                st.error(
                    "CSV must contain ground_truth and prediction columns."
                )

            else:

                evaluation_df = evaluation_df[
                    ["ground_truth", "prediction"]
                ].apply(pd.to_numeric, errors="coerce")

                evaluation_df = evaluation_df.dropna()

                if evaluation_df.empty:
                    st.error(
                        "No valid numeric evaluation rows were found."
                    )

                else:
                    ground_truth = evaluation_df[
                        "ground_truth"
                    ].to_numpy(dtype=float)

                    prediction = evaluation_df[
                        "prediction"
                    ].to_numpy(dtype=float)

                    errors = prediction - ground_truth

                    mae = float(
                        np.mean(np.abs(errors))
                    )

                    rmse = float(
                        np.sqrt(np.mean(errors ** 2))
                    )

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "MAE",
                        f"{mae:.3f}",
                    )

                    col2.metric(
                        "RMSE",
                        f"{rmse:.3f}",
                    )

                    col3.metric(
                        "Samples Evaluated",
                        f"{len(evaluation_df):,}",
                    )

                    st.caption(
                        "MAE and RMSE are computed only from "
                        "the uploaded annotation/prediction pairs."
                    )

                    comparison = evaluation_df.copy()

                    comparison["error"] = errors
                    comparison["absolute_error"] = np.abs(errors)

                    st.subheader("Prediction Comparison")

                    st.plotly_chart(
                        px.scatter(
                            comparison,
                            x="ground_truth",
                            y="prediction",
                            title="Predicted vs Ground-Truth Counts",
                            trendline=None,
                        ).update_layout(
                            template="plotly_dark",
                            paper_bgcolor="#0b1020",
                            plot_bgcolor="#11182b",
                        ),
                        use_container_width=True,
                    )

                    st.dataframe(
                        comparison,
                        use_container_width=True,
                        hide_index=True,
                    )

        except Exception as error:
            st.error(
                f"Could not evaluate the uploaded CSV: {error}"
            )


# ============================================================
# PAGE 7: RESULTS & REPORTS
# ============================================================

elif page == "Results & Reports":

    st.title("📁 Results & Reports")

    history = load_history()

    if history.empty:

        st.info(
            "No analysis results have been saved yet."
        )

    else:

        st.subheader("Export Results")

        st.dataframe(
            history.sort_values(
                "timestamp",
                ascending=False,
            ),
            use_container_width=True,
            hide_index=True,
        )

        csv_data = history.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download All Results as CSV",
            data=csv_data,
            file_name="intellicrowd_results.csv",
            mime="text/csv",
            type="primary",
        )

        report = {
            "project": APP_NAME,
            "generated_at": datetime.now().isoformat(),
            "total_analyses": int(len(history)),
            "average_estimated_count": float(
                history["estimated_count"].mean()
            ),
            "maximum_estimated_count": int(
                history["estimated_count"].max()
            ),
            "records": history.astype(str).to_dict(
                orient="records"
            ),
        }

        st.download_button(
            "⬇️ Download JSON Report",
            data=json.dumps(
                report,
                indent=2,
            ),
            file_name="intellicrowd_report.json",
            mime="application/json",
        )

        st.caption(
            "Reports contain model estimates, not verified ground-truth counts."
        )


# ============================================================
# PAGE 8: SETTINGS
# ============================================================

elif page == "Settings":

    st.title("⚙️ Settings")

    st.subheader("Inference Settings")

    st.selectbox(
        "Model",
        [
            "CSRNet - SHA pretrained weights",
        ],
        help="The current crowd-counting service uses CSRNet with SHA weights.",
    )

    st.selectbox(
        "Inference device",
        [
            "CPU",
        ],
    )

    st.slider(
        "Maximum video dimension",
        min_value=320,
        max_value=1280,
        value=960,
        step=160,
        help=(
            "The current video-processing helper uses a fixed "
            "960-pixel maximum dimension."
        ),
    )

    st.slider(
        "Default video sample interval (seconds)",
        min_value=1,
        max_value=10,
        value=2,
    )

    st.divider()

    st.subheader("System Information")

    st.write(f"**Application:** {APP_NAME}")
    st.write("**Counting model:** CSRNet")
    st.write("**Weights:** SHA")
    st.write("**Inference device:** CPU")
    st.write(f"**Python project folder:** `{BASE_DIR}`")
    st.write(f"**Results folder:** `{OUTPUT_DIR}`")

    st.divider()

    st.subheader("Data Management")

    st.caption(
        "Clearing the current session does not delete saved CSV results."
    )

    if st.button(
        "Clear Current Session History",
        type="secondary",
    ):

        st.session_state.crowd_history = []
        st.session_state.last_image_count = None
        st.session_state.last_image_name = None
        st.session_state.video_results = []

        st.success(
            "Current session data has been cleared. "
            "Previously saved CSV records remain on disk."
        )

        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align: center;
        color: #64748b;
        font-size: 12px;
        padding: 8px 0 20px 0;
    ">
        IntelliCrowd | AI-Based Crowd Intelligence Dashboard<br>
        Research prototype — model predictions require validation.
    </div>
    """,
    unsafe_allow_html=True,
)