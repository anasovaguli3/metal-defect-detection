"""
DefectVision Streamlit web application.

Launch from project root:
    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

# Project root must be on sys.path before app/ submodules import ``src.*``
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image

from feature_analysis import (  # noqa: E402
    capture_feature_maps,
    extract_filter_weights,
    feature_layer_names,
    fig_feature_map_grid,
    fig_filter_grid,
    fig_input_vs_deep_maps,
    iter_test_images_by_category,
)
from visualizations import (  # noqa: E402
    fig_confusion_matrix,
    fig_f1_vs_threshold,
    fig_models_comparison,
    fig_per_defect_recall,
    fig_roc_curve,
    fig_score_distribution,
)

from utils import (  # noqa: E402
    CATEGORY_ORDER,
    CATEGORY_TITLES,
    MODEL_PRESETS,
    apply_model_preset,
    compare_custom_models,
    evaluate_test_set_by_category,
    get_checkpoint_info,
    get_inference_threshold,
    load_model_for_inference,
    run_inference,
)
from src.model import count_parameters, get_model_architecture, get_model_checkpoint_path  # noqa: E402
from src.utils import device_label, load_config, resolve_path  # noqa: E402


def _render_result(result: dict, threshold: float, title: str | None = None) -> None:
    """Render inference result for a single image."""
    if title:
        st.markdown(f"### {title}")

    if result["is_defect"]:
        st.error(f"❌ DEFECT — confidence {result['confidence']:.1f}%")
    else:
        st.success(f"✅ GOOD — confidence {result['confidence']:.1f}%")

    st.metric(
        label="Reconstruction Error",
        value=f"{result['error']:.6f}",
        delta=f"threshold {threshold:.6f}",
        delta_color="inverse" if result["is_defect"] else "normal",
    )

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Original vs Reconstructed**")
        sub_l, sub_r = st.columns(2)
        with sub_l:
            st.image(result["original"], caption="Original", use_container_width=True)
        with sub_r:
            st.image(result["reconstructed"], caption="Reconstructed", use_container_width=True)

    with col_right:
        st.markdown("**Error Heatmap & Verdict**")
        sub_l, sub_r = st.columns(2)
        with sub_l:
            st.image(result["heatmap"], caption="Error Heatmap", use_container_width=True)
        with sub_r:
            st.image(result["overlay"], caption="Overlay", use_container_width=True)
            verdict_color = "#ff4b4b" if result["is_defect"] else "#21c354"
            st.markdown(
                f"<div style='padding:1rem;border-radius:8px;"
                f"background:{verdict_color}22;border:2px solid {verdict_color};'>"
                f"<h3 style='color:{verdict_color};margin:0;'>{result['label']}</h3>"
                f"<p>Confidence: {result['confidence']:.1f}%</p>"
                f"<p>Error: {result['error']:.6f}</p></div>",
                unsafe_allow_html=True,
            )


def _render_batch_summary(rows: list[dict]) -> None:
    """Show compact summary table for multiple uploads."""
    good = sum(1 for r in rows if not r["is_defect"])
    defect = len(rows) - good

    c1, c2, c3 = st.columns(3)
    c1.metric("Total images", len(rows))
    c2.metric("Good", good)
    c3.metric("Defect", defect)

    df = pd.DataFrame(
        [
            {
                "File": r["filename"],
                "Result": r["label"],
                "Error": round(r["error"], 6),
                "Confidence %": round(r["confidence"], 1),
            }
            for r in rows
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_inspection_tab(
    config: dict,
    model,
    device,
    threshold: float,
) -> None:
    uploaded_files = st.file_uploader(
        "Upload Images",
        type=["png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=True,
        help="Select one or more JPEG/PNG images",
    )

    if not uploaded_files:
        st.info("👆 Upload one or more images to begin inspection.")
        return

    batch_results: list[dict] = []
    for uploaded in uploaded_files:
        image = Image.open(uploaded).convert("RGB")
        result = run_inference(image, model, device, config, threshold)
        result["filename"] = uploaded.name
        batch_results.append(result)

    st.divider()
    st.subheader("Batch Summary")
    _render_batch_summary(batch_results)

    st.divider()

    if len(batch_results) == 1:
        _render_result(batch_results[0], threshold)
    else:
        st.subheader("Detailed Results")
        for item in batch_results:
            icon = "❌" if item["is_defect"] else "✅"
            with st.expander(
                f"{icon} {item['filename']} — {item['label']} (error: {item['error']:.4f})"
            ):
                _render_result(item, threshold)


def _summary_table_rows(eval_result: dict) -> list[dict]:
    """Build flat rows for the per-category summary table."""
    rows: list[dict] = []
    for key in sorted(eval_result["categories"].keys(), key=lambda n: (
        CATEGORY_ORDER.index(n) if n in CATEGORY_ORDER else 99,
        n,
    )):
        cat = eval_result["categories"][key]
        if cat["ground_truth"] == "good":
            status = f"{cat['correct']} correct, {cat['errors']} false alarm"
        else:
            status = f"{cat['correct']} detected, {cat['errors']} missed"
        rows.append(
            {
                "Category": cat["title"],
                "Total": cat["total"],
                "→ GOOD": cat["pred_good"],
                "→ DEFECT": cat["pred_defect"],
                "Correct": cat["correct"],
                "Errors": cat["errors"],
                "Status": status,
                "Mean error": round(cat["mean_error"], 4),
            }
        )
    return rows


def _render_evaluation_tab(
    config: dict,
    model,
    device,
    threshold: float,
    ckpt_info: dict | None = None,
) -> None:
    st.markdown(
        "Полная оценка на **test-наборе MVTec cable**: все изображения загружаются "
        "из `data/processed/cable/test/` по папкам-категориям."
    )

    test_root = resolve_path(config["data"]["processed_dir"]) / config["data"]["category"] / "test"
    st.caption(f"Test data: `{test_root}`")

    run = st.button("▶ Run evaluation", type="primary", use_container_width=False)

    if not run and "eval_result" not in st.session_state:
        st.info("Нажми **Run evaluation** — прогон по всем категориям займёт ~1–2 минуты.")
        return

    cache_key = (
        threshold,
        str(resolve_path(get_model_checkpoint_path(config))),
        get_model_architecture(config),
        ckpt_info.get("epoch", 0),
    )
    if run or st.session_state.get("eval_cache_key") != cache_key:
        progress = st.progress(0.0, text="Loading test images…")
        status = st.empty()
        base_config = load_config(str(PROJECT_ROOT / "config.yaml"))

        def on_progress(fraction: float, category: str) -> None:
            progress.progress(min(fraction, 1.0), text=f"Evaluating: {category}…")

        try:
            result = evaluate_test_set_by_category(
                config, model, device, threshold, progress_callback=on_progress
            )
            status.text("Сравнение Conv AE vs U-Net…")
            result["models_comparison"] = compare_custom_models(
                base_config,
                progress_callback=lambda f, lbl: progress.progress(
                    min(0.5 + f * 0.5, 1.0),
                    text=f"Model comparison: {lbl}…",
                ),
            )
        except FileNotFoundError as exc:
            st.error(str(exc))
            return
        finally:
            progress.empty()
            status.empty()

        st.session_state["eval_result"] = result
        st.session_state["eval_cache_key"] = cache_key

    eval_result = st.session_state["eval_result"]
    overall = eval_result["overall"]

    st.divider()
    st.subheader("Overall (all test images)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total", overall["total"])
    c2.metric("Predicted GOOD", overall["pred_good"])
    c3.metric("Predicted DEFECT", overall["pred_defect"])
    c4.metric("Correct", overall["correct"])
    c5.metric("Accuracy", f"{overall['accuracy']:.1%}")
    st.caption(f"Threshold: **{eval_result['threshold']:.4f}**")

    st.divider()
    st.subheader("By category")

    summary_rows = _summary_table_rows(eval_result)
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # CSV download
    csv_buffer = StringIO()
    summary_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download summary (CSV)",
        data=csv_buffer.getvalue(),
        file_name="defectvision_eval_by_category.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Category details")

    for key in sorted(eval_result["categories"].keys(), key=lambda n: (
        CATEGORY_ORDER.index(n) if n in CATEGORY_ORDER else 99,
        n,
    )):
        cat = eval_result["categories"][key]
        icon = "✅" if cat["errors"] == 0 else "⚠️"
        header = (
            f"{icon} **{cat['title']}** — "
            f"Total **{cat['total']}** · "
            f"GOOD **{cat['pred_good']}** · "
            f"DEFECT **{cat['pred_defect']}**"
        )
        if cat["ground_truth"] == "good":
            header += f" · false alarms: **{cat['errors']}**"
        else:
            header += f" · detected: **{cat['correct']}/{cat['total']}**"

        with st.expander(header):
            detail_df = pd.DataFrame(
                [
                    {
                        "File": r["filename"],
                        "Ground truth": r["ground_truth"],
                        "Prediction": r["prediction"].upper(),
                        "OK": "✓" if r["is_correct"] else "✗",
                        "Error": round(r["error"], 6),
                    }
                    for r in cat["rows"]
                ]
            )
            st.dataframe(detail_df, use_container_width=True, hide_index=True)

            wrong = [r for r in cat["rows"] if not r["is_correct"]]
            if wrong:
                st.markdown("**Ошибки модели:**")
                cols = st.columns(min(len(wrong), 4))
                for i, row in enumerate(wrong[:8]):
                    with cols[i % len(cols)]:
                        img = Image.open(row["path"]).convert("RGB")
                        st.image(img, caption=f"{row['filename']}\n→ {row['prediction']}", use_container_width=True)

    _render_full_metrics_block(eval_result)
    _render_visualizations_block(eval_result)
    _render_error_gallery(eval_result)


def _show_fig(fig: plt.Figure) -> None:
    st.pyplot(fig, clear_figure=True)


def _render_visualizations_block(eval_result: dict) -> None:
    """ROC, confusion matrix, distributions, baseline comparison."""
    y_true = eval_result.get("y_true")
    scores = eval_result.get("scores")
    y_pred = eval_result.get("y_pred")
    if y_true is None or scores is None:
        return

    threshold = eval_result["threshold"]
    st.divider()
    st.subheader("Visualizations")

    row1a, row1b = st.columns(2)
    with row1a:
        st.markdown("**ROC Curve**")
        _show_fig(fig_roc_curve(y_true, scores))
    with row1b:
        st.markdown("**Confusion Matrix**")
        _show_fig(fig_confusion_matrix(y_true, y_pred))

    row2a, row2b = st.columns(2)
    with row2a:
        st.markdown("**Score Distribution**")
        _show_fig(fig_score_distribution(y_true, scores, threshold))
    with row2b:
        st.markdown("**F1 vs Threshold**")
        _show_fig(
            fig_f1_vs_threshold(
                y_true,
                scores,
                threshold,
                eval_result.get("optimal_threshold"),
            )
        )

    row3a, row3b = st.columns(2)
    with row3a:
        st.markdown("**Recall per Defect Type**")
        if eval_result.get("per_defect_recall"):
            _show_fig(fig_per_defect_recall(eval_result["per_defect_recall"]))
    with row3b:
        st.markdown("**Conv AE vs U-Net**")
        if eval_result.get("models_comparison"):
            cmp = eval_result["models_comparison"]
            _show_fig(
                fig_models_comparison(
                    cmp["conv_ae_metrics"],
                    cmp["unet_metrics"],
                )
            )


def _render_error_gallery(eval_result: dict) -> None:
    """False positives and false negatives from test set."""
    all_rows = [
        row
        for cat in eval_result["categories"].values()
        for row in cat["rows"]
        if not row["is_correct"]
    ]
    if not all_rows:
        return

    fp_rows = [r for r in all_rows if r["label"] == 0]
    fn_rows = [r for r in all_rows if r["label"] == 1]

    st.divider()
    st.subheader("Error Analysis")

    if fp_rows:
        st.markdown(f"**False Positives** — good → DEFECT ({len(fp_rows)})")
        cols = st.columns(min(4, len(fp_rows)))
        for i, row in enumerate(fp_rows[:8]):
            with cols[i % len(cols)]:
                st.image(
                    Image.open(row["path"]).convert("RGB"),
                    caption=f"{row['filename']}\nerror {row['error']:.4f}",
                    use_container_width=True,
                )

    if fn_rows:
        st.markdown(f"**False Negatives** — missed defects ({len(fn_rows)})")
        cols = st.columns(min(4, len(fn_rows)))
        for i, row in enumerate(fn_rows[:8]):
            with cols[i % len(cols)]:
                st.image(
                    Image.open(row["path"]).convert("RGB"),
                    caption=f"{row['filename']}\nerror {row['error']:.4f}",
                    use_container_width=True,
                )


def _render_full_metrics_block(eval_result: dict) -> None:
    """Full classification metrics on the entire test set (all images)."""
    metrics = eval_result.get("metrics")
    if not metrics:
        return

    overall = eval_result["overall"]
    conf = eval_result["confusion"]

    st.divider()
    st.subheader("Full evaluation — all test images")
    st.caption(
        f"{overall['total']} images · {overall['n_good']} good · {overall['n_defect']} defect · "
        f"threshold **{eval_result['threshold']:.4f}**"
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    m2.metric("Precision", f"{metrics['precision']:.1%}")
    m3.metric("Recall", f"{metrics['recall']:.1%}")
    m4.metric("F1 Score", f"{metrics['f1']:.3f}")
    auc = metrics.get("auc_roc", float("nan"))
    m5.metric("AUC-ROC", f"{auc:.3f}" if auc == auc else "—")

    st.markdown("**Confusion matrix**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("True Negative (good→GOOD)", conf["tn"])
    c2.metric("False Positive (good→DEFECT)", conf["fp"])
    c3.metric("False Negative (missed defect)", conf["fn"])
    c4.metric("True Positive (defect caught)", conf["tp"])

    st.markdown(
        f"- **Specificity** (good recognized): {conf['tn']}/{overall['n_good']} "
        f"= {conf['tn'] / overall['n_good']:.1%}\n"
        f"- **Defect recall** (defects caught): {conf['tp']}/{overall['n_defect']} "
        f"= {conf['tp'] / overall['n_defect']:.1%}"
    )

    st.markdown("**Recall by defect type**")
    recall_df = pd.DataFrame(
        [
            {
                "Category": row["title"],
                "Images": row["count"],
                "Detected": row["detected"],
                "Missed": row["missed"],
                "Recall": f"{row['recall']:.1%}",
            }
            for row in eval_result.get("per_defect_recall", [])
        ]
    )
    st.dataframe(recall_df, use_container_width=True, hide_index=True)

    if eval_result.get("models_comparison"):
        st.markdown("**Conv AE vs U-Net (обе custom-модели)**")
        cmp = eval_result["models_comparison"]
        conv_m = cmp["conv_ae_metrics"]
        unet_m = cmp["unet_metrics"]
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Conv AE AUC", f"{conv_m.get('auc_roc', 0):.3f}")
        b2.metric("Conv AE F1", f"{conv_m.get('f1', 0):.3f}")
        b3.metric("U-Net AUC", f"{unet_m.get('auc_roc', 0):.3f}")
        b4.metric("U-Net F1", f"{unet_m.get('f1', 0):.3f}")
        b5.metric("Conv AE Recall", f"{conv_m.get('recall', 0):.1%}")

    if eval_result.get("alt_metrics"):
        st.markdown("**Alternative threshold (balanced)**")
        alt = eval_result["alt_metrics"]
        st.caption(
            f"Threshold {eval_result.get('alt_threshold', 0):.4f} — "
            f"{eval_result.get('alt_threshold_source', '')}"
        )
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Alt F1", f"{alt.get('f1', 0):.3f}")
        a2.metric("Alt Recall", f"{alt.get('recall', 0):.1%}")
        a3.metric("Alt Precision", f"{alt.get('precision', 0):.1%}")
        a4.metric("Alt Accuracy", f"{alt.get('accuracy', 0):.1%}")

    metrics_df = pd.DataFrame(
        [
            {"Metric": "Accuracy", "Value": f"{metrics['accuracy']:.4f}"},
            {"Metric": "Precision", "Value": f"{metrics['precision']:.4f}"},
            {"Metric": "Recall", "Value": f"{metrics['recall']:.4f}"},
            {"Metric": "F1 Score", "Value": f"{metrics['f1']:.4f}"},
            {"Metric": "AUC-ROC", "Value": f"{auc:.4f}" if auc == auc else "n/a"},
            {"Metric": "Threshold", "Value": f"{eval_result['threshold']:.6f}"},
            {"Metric": "True Negatives", "Value": str(conf["tn"])},
            {"Metric": "False Positives", "Value": str(conf["fp"])},
            {"Metric": "False Negatives", "Value": str(conf["fn"])},
            {"Metric": "True Positives", "Value": str(conf["tp"])},
        ]
    )
    csv_metrics = StringIO()
    metrics_df.to_csv(csv_metrics, index=False)
    st.download_button(
        label="Download full metrics (CSV)",
        data=csv_metrics.getvalue(),
        file_name="defectvision_full_metrics.csv",
        mime="text/csv",
    )


def _render_feature_analysis_tab(
    config: dict,
    model,
    device,
    arch: str,
) -> None:
    """Feature maps and conv filters — separate from evaluation metrics."""
    st.markdown(
        "Визуализация **фильтров первого conv-слоя** и **feature maps** по слоям encoder. "
        "Работает для загруженных фото и для **всех** изображений test-набора."
    )

    layers = feature_layer_names(arch)
    shallow, deep = layers[0], layers[-1]

    st.subheader("1. Learned filters (первый Conv слой)")
    filters = extract_filter_weights(model, arch)
    if filters is not None:
        _show_fig(
            fig_filter_grid(
                filters,
                title=f"First-layer filters — {arch}",
                max_filters=32,
            )
        )
        st.caption(f"Показано до 32 из {len(filters)} фильтров первого свёрточного слоя.")
    else:
        st.warning("Фильтры недоступны для этой архитектуры.")

    st.divider()
    st.subheader("2. Feature maps")

    source = st.radio(
        "Источник изображений",
        options=["upload", "test_all"],
        format_func=lambda x: "Загрузить фото" if x == "upload" else "Все test-изображения (по категориям)",
        horizontal=True,
    )

    selected_layer = st.selectbox("Слой для детальной сетки карт", layers, index=layers.index(deep))

    if source == "upload":
        uploaded = st.file_uploader(
            "Image for feature maps",
            type=["png", "jpg", "jpeg", "bmp"],
            key="feature_upload",
        )
        if not uploaded:
            st.info("Загрузи изображение или выбери «Все test-изображения».")
            return

        image = Image.open(uploaded).convert("RGB")
        activations = capture_feature_maps(model, image, config, device)

        col_a, col_b = st.columns(2)
        with col_a:
            _show_fig(fig_input_vs_deep_maps(image, activations, shallow_layer=shallow, deep_layer=deep))
        with col_b:
            _show_fig(fig_feature_map_grid(activations[selected_layer], layer_name=selected_layer))
        return

    test_root = resolve_path(config["data"]["processed_dir"]) / config["data"]["category"] / "test"
    if not test_root.exists():
        st.error(f"Test folder not found: `{test_root}`")
        return

    run_all = st.button("▶ Построить feature maps для всех test-изображений", type="primary")
    cache_key = (arch, str(resolve_path(get_model_checkpoint_path(config))), selected_layer)

    if run_all or st.session_state.get("feature_maps_cache_key") != cache_key:
        if run_all:
            items = iter_test_images_by_category(test_root, CATEGORY_ORDER)
            progress = st.progress(0.0, text="Feature maps…")
            grouped: dict[str, list[dict]] = {}

            for idx, (category, path) in enumerate(items):
                progress.progress((idx + 1) / len(items), text=f"{category}: {path.name}")
                image = Image.open(path).convert("RGB")
                activations = capture_feature_maps(model, image, config, device)
                grouped.setdefault(category, []).append(
                    {
                        "path": path,
                        "filename": path.name,
                        "image": image,
                        "activations": activations,
                    }
                )

            progress.empty()
            st.session_state["feature_maps_by_category"] = grouped
            st.session_state["feature_maps_cache_key"] = cache_key

    grouped = st.session_state.get("feature_maps_by_category")
    if not grouped:
        st.info("Нажми кнопку выше — прогон по ~150 test-фото займёт несколько минут.")
        return

    total = sum(len(v) for v in grouped.values())
    st.success(f"Готово: **{total}** изображений в **{len(grouped)}** категориях.")

    for category in sorted(grouped.keys(), key=lambda n: (
        CATEGORY_ORDER.index(n) if n in CATEGORY_ORDER else 99,
        n,
    )):
        rows = grouped[category]
        title = CATEGORY_TITLES.get(category, category)
        with st.expander(f"**{title}** — {len(rows)} images", expanded=(category == "good")):
            for row in rows:
                with st.expander(f"{row['filename']}", expanded=False):
                    c1, c2 = st.columns(2)
                    with c1:
                        _show_fig(
                            fig_input_vs_deep_maps(
                                row["image"],
                                row["activations"],
                                shallow_layer=shallow,
                                deep_layer=deep,
                            )
                        )
                    with c2:
                        _show_fig(
                            fig_feature_map_grid(
                                row["activations"][selected_layer],
                                layer_name=selected_layer,
                            )
                        )


@st.cache_resource
def _load_model_and_config(model_preset: str):
    """Cache model and config for faster repeated inference."""
    base_config = load_config(str(PROJECT_ROOT / "config.yaml"))
    config = apply_model_preset(base_config, model_preset)
    model, device = load_model_for_inference(config)
    ckpt_info = get_checkpoint_info(get_model_checkpoint_path(config))
    return config, model, device, ckpt_info


def main() -> None:
    st.set_page_config(
        page_title="DefectVision",
        page_icon="🔍",
        layout="wide",
    )

    st.title("DefectVision — Cable Inspection")
    st.markdown(
        "Anomaly detection для **MVTec AD cable** через reconstruction error autoencoder."
    )

    with st.sidebar:
        st.header("Model")
        preset_labels = {k: v["label"] for k, v in MODEL_PRESETS.items()}
        model_preset = st.selectbox(
            "Active model",
            options=list(MODEL_PRESETS.keys()),
            format_func=lambda k: preset_labels[k],
            index=0,
        )

    try:
        config, model, device, ckpt_info = _load_model_and_config(model_preset)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Train the model with `python scripts/train.py` after preparing data.")
        return

    default_threshold, threshold_source = get_inference_threshold(config, model, device)
    model_path = resolve_path(get_model_checkpoint_path(config))
    arch = get_model_architecture(config)

    with st.sidebar:
        st.header("Settings")
        threshold = st.slider(
            "Reconstruction error threshold",
            min_value=0.001,
            max_value=0.5,
            value=min(default_threshold, 0.5),
            step=0.001,
            format="%.3f",
        )
        st.divider()
        st.subheader("Model Info")
        st.write("**Product:** Cable (MVTec AD)")
        st.write(f"**Architecture:** {arch} ({config['model']['name']})")
        if ckpt_info.get("epoch"):
            st.write(f"**Checkpoint epoch:** {ckpt_info['epoch']}")
        if ckpt_info.get("val_loss"):
            st.write(f"**Val loss:** {ckpt_info['val_loss']:.6f}")
        if arch == "conv_ae":
            st.write(f"**Latent dim:** {config['model']['latent_dim']}")
        st.write(f"**Parameters:** {count_parameters(model):,}")
        st.write(f"**Device:** {device_label(device)}")
        st.caption(f"File: `{model_path.name}`")
        st.caption(f"Threshold: {default_threshold:.4f} ({threshold_source})")
        if model_preset == "unet":
            st.caption("U-Net ещё учится — порог пересчитывается на val/good.")
        if st.button("Reload model (clear cache)"):
            st.cache_resource.clear()
            st.session_state.pop("eval_result", None)
            st.session_state.pop("feature_maps_by_category", None)
            st.session_state.pop("feature_maps_cache_key", None)
            st.rerun()

    tab_inspect, tab_features, tab_eval = st.tabs(
        ["🔍 Inspection", "🧠 Feature Analysis", "📊 Evaluation"]
    )

    with tab_inspect:
        _render_inspection_tab(config, model, device, threshold)

    with tab_features:
        _render_feature_analysis_tab(config, model, device, arch)

    with tab_eval:
        _render_evaluation_tab(config, model, device, threshold, ckpt_info)


if __name__ == "__main__":
    main()
