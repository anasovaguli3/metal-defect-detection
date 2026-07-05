"""
Metall nuqsoni aniqlash — Streamlit ilovasi.

Ishga tushirish:
    python scripts/run_app.py
"""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

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
)
from utils import (  # noqa: E402
    CLASS_LABELS,
    count_parameters,
    evaluate_test_set,
    get_checkpoint_info,
    load_model_for_inference,
    run_inference,
)
from visualizations import (  # noqa: E402
    fig_confusion_matrix,
    fig_models_comparison,
    fig_per_class_bars,
    fig_prob_distribution,
    fig_roc_curve,
    fig_training_history,
)
from src.utils import load_json, resolve_path  # noqa: E402


def _show_fig(fig: plt.Figure) -> None:
    st.pyplot(fig, clear_figure=True)


def _render_result(result: dict, title: str | None = None) -> None:
    if title:
        st.markdown(f"### {title}")
    if result["is_defect"]:
        st.error(f"⚠️ **{result['label']}** — {result['confidence']:.1f}% ishonch")
    else:
        st.success(f"✅ **{result['label']}** — {result['confidence']:.1f}% ishonch")
    st.metric(
        "P(defect)",
        f"{result['defect_prob'] * 100:.2f}%",
        delta=f"bashora: {result['pred_class']}",
    )


def _render_batch_summary(rows: list[dict]) -> None:
    defect = sum(1 for r in rows if r["is_defect"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Jami", len(rows))
    c2.metric("Nuqsonsiz", len(rows) - defect)
    c3.metric("Nuqsonli", defect)
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Fayl": r["filename"],
                    "Natija": r["label"],
                    "Ishonch %": round(r["confidence"], 1),
                    "P(defect) %": round(r["defect_prob"] * 100, 1),
                }
                for r in rows
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )


def _render_inspection_tab(model, classes, cfg, device) -> None:
    uploaded_files = st.file_uploader(
        "Detal rasmlarini yuklang",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Bitta yoki bir nechta JPEG/PNG",
    )
    if not uploaded_files:
        st.info("👆 Sinov uchun rasm yuklang yoki `data/raw_images/` dan tanlang.")
        return

    batch: list[dict] = []
    for uploaded in uploaded_files:
        image = Image.open(uploaded).convert("RGB")
        result = run_inference(image, model, classes, cfg, device)
        result["filename"] = uploaded.name
        result["image"] = image
        batch.append(result)

    st.divider()
    st.subheader("Umumiy natija")
    _render_batch_summary(batch)
    st.divider()

    if len(batch) == 1:
        item = batch[0]
        col_l, col_r = st.columns(2)
        with col_l:
            st.image(item["image"], caption=item["filename"], use_container_width=True)
        with col_r:
            _render_result(item)
            for cls in classes:
                p = item["prob_map"][cls]
                name = CLASS_LABELS.get(cls, cls)
                st.progress(min(max(p, 0.0), 1.0), text=f"{name}: {p * 100:.2f}%")
    else:
        for item in batch:
            icon = "⚠️" if item["is_defect"] else "✅"
            with st.expander(f"{icon} {item['filename']} — {item['label']}"):
                col_l, col_r = st.columns(2)
                with col_l:
                    st.image(item["image"], use_container_width=True)
                with col_r:
                    _render_result(item)


def _render_feature_tab(model, classes, cfg, device) -> None:
    st.markdown("Birinchi **conv filtrlari** va **feature map** qatlamlari.")
    layers = feature_layer_names()
    shallow, deep = layers[0], layers[-1]

    st.subheader("1. O'rganilgan filtrlar (birinchi Conv)")
    filters = extract_filter_weights(model)
    _show_fig(fig_filter_grid(filters, title="Conv1 — 32 filtr", max_filters=32))
    st.caption(f"Ko'rsatilgan: 32 / {len(filters)} filtr")

    st.divider()
    st.subheader("2. Feature maps")

    selected_layer = st.selectbox("Feature map qatlami", layers, index=layers.index(deep))
    uploaded = st.file_uploader("Rasm", type=["jpg", "jpeg", "png"], key="feature_upload")
    if not uploaded:
        st.info("Feature map uchun rasm yuklang.")
        return

    image = Image.open(uploaded).convert("RGB")
    activations = capture_feature_maps(model, image, cfg["data"]["image_size"], device)
    col_a, col_b = st.columns(2)
    with col_a:
        _show_fig(fig_input_vs_deep_maps(image, activations, shallow_layer=shallow, deep_layer=deep))
    with col_b:
        _show_fig(fig_feature_map_grid(activations[selected_layer], layer_name=selected_layer))


def _render_saved_training(cfg: dict) -> None:
    hist_path = resolve_path(cfg["paths"]["cnn_history"])
    if not hist_path.exists():
        st.info("O'qitish tarixi topilmadi (`train.py`).")
        return
    df_hist = pd.DataFrame(load_json(hist_path))
    st.dataframe(df_hist.tail(8), use_container_width=True, hide_index=True)
    fig_loss, fig_acc = fig_training_history(df_hist)
    c1, c2 = st.columns(2)
    with c1:
        _show_fig(fig_loss)
    with c2:
        _show_fig(fig_acc)


def _render_comparison_charts(cfg: dict, cnn_metrics: dict | None) -> None:
    baseline_path = resolve_path(cfg["paths"]["baseline_comparison"])
    tuning_path = resolve_path(cfg["paths"]["tuning_comparison"])

    if baseline_path.exists() and cnn_metrics:
        df = pd.read_csv(baseline_path)
        if not df.empty and "model" in df.columns:
            baseline_row = df[df["model"].str.contains("baseline|efficient", case=False, na=False)]
            if baseline_row.empty:
                baseline_row = df.iloc[[-1]]
            b = baseline_row.iloc[0].to_dict()
            b_metrics = {
                "accuracy": float(b.get("accuracy", b.get("val_acc", 0))),
                "precision": float(b.get("precision", 0)),
                "recall": float(b.get("recall", 0)),
                "f1": float(b.get("f1", b.get("f1_weighted", 0))),
            }
            st.markdown("**CNN vs Baseline**")
            _show_fig(fig_models_comparison(cnn_metrics, b_metrics, baseline_label="EfficientNet"))

    if tuning_path.exists():
        st.markdown("**Hyperparameter tuning**")
        st.dataframe(pd.read_csv(tuning_path), use_container_width=True, hide_index=True)


def _render_error_gallery(eval_result: dict) -> None:
    wrong = [r for r in eval_result["rows"] if not r["is_correct"]]
    if not wrong:
        st.success("Barcha test rasmlari to'g'ri tasniflangan.")
        return

    st.subheader("Xato tahlili")
    st.markdown(f"**Noto'g'ri bashoralar:** {len(wrong)}")
    cols = st.columns(min(4, len(wrong)))
    for i, row in enumerate(wrong[:12]):
        with cols[i % len(cols)]:
            img = Image.open(row["path"]).convert("RGB")
            st.image(
                img,
                caption=f"{row['filename']}\n{row['true_class']} → {row['pred_class']}",
                use_container_width=True,
            )


def _render_evaluation_tab(model, classes, cfg, device, ckpt_info: dict) -> None:
    st.markdown("To'liq **test to'plami** bo'yicha baholash, grafiklar va xato tahlili.")

    test_csv = resolve_path(cfg["data"]["processed_dir"]) / "test.csv"
    st.caption(f"Test ma'lumotlari: `{test_csv}`")

    run = st.button("▶ Baholashni ishga tushirish", type="primary")

    cache_key = str(resolve_path(cfg["paths"]["cnn_checkpoint"]))
    if run or st.session_state.get("eval_cache_key") != cache_key:
        if run:
            progress = st.progress(0.0, text="Test to'plami yuklanmoqda…")

            def on_progress(frac: float, name: str) -> None:
                progress.progress(min(frac, 1.0), text=f"Baholash: {name}")

            try:
                result = evaluate_test_set(cfg, model, device, progress_callback=on_progress)
            except FileNotFoundError as exc:
                st.error(str(exc))
                return
            finally:
                progress.empty()

            st.session_state["eval_result"] = result
            st.session_state["eval_cache_key"] = cache_key

    eval_result = st.session_state.get("eval_result")
    if not eval_result:
        st.info("**Baholashni ishga tushirish** tugmasini bosing (~1–2 daqiqa).")
        st.divider()
        st.subheader("Saqlangan o'qitish tarixi")
        _render_saved_training(cfg)
        saved_metrics = resolve_path(cfg["paths"]["test_metrics"])
        if saved_metrics.exists():
            m = load_json(saved_metrics)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy (saqlangan)", f"{m.get('accuracy', 0) * 100:.2f}%")
            c2.metric("Precision", f"{m.get('precision', 0) * 100:.2f}%")
            c3.metric("Recall", f"{m.get('recall', 0) * 100:.2f}%")
            c4.metric("F1", f"{m.get('f1_weighted', 0) * 100:.2f}%")
        cm_path = resolve_path(cfg["paths"]["confusion_matrix"])
        if cm_path.exists():
            st.image(str(cm_path), caption="Saqlangan confusion matrix", use_container_width=True)
        _render_comparison_charts(cfg, None)
        return

    overall = eval_result["overall"]
    metrics = eval_result["metrics"]
    conf = eval_result["confusion"]
    defect_idx = eval_result["defect_idx"]

    st.divider()
    st.subheader("Umumiy natijalar")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Jami", overall["total"])
    c2.metric("To'g'ri", overall["correct"])
    c3.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    c4.metric("F1", f"{metrics['f1']:.3f}")
    c5.metric("Recall", f"{metrics['recall']:.1%}")

    st.divider()
    st.subheader("Sinf bo'yicha")
    summary_rows = []
    for cls in classes:
        cat = eval_result["by_class"].get(cls, {"total": 0, "correct": 0, "errors": 0})
        summary_rows.append(
            {
                "Sinf": CLASS_LABELS.get(cls, cls),
                "Jami": cat["total"],
                "To'g'ri": cat["correct"],
                "Xato": cat["errors"],
                "Aniqlik": f"{cat['correct'] / cat['total']:.1%}" if cat["total"] else "—",
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    csv_buf = StringIO()
    summary_df.to_csv(csv_buf, index=False)
    st.download_button(
        "CSV yuklab olish",
        data=csv_buf.getvalue(),
        file_name="test_baholash_sinf.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Vizualizatsiyalar")
    y_true = eval_result["y_true"]
    y_pred = eval_result["y_pred"]
    scores = eval_result["scores"]
    y_binary = (y_true == defect_idx).astype(int)

    r1a, r1b = st.columns(2)
    with r1a:
        st.markdown("**ROC Curve**")
        _show_fig(fig_roc_curve(y_binary, scores))
    with r1b:
        st.markdown("**Confusion Matrix**")
        _show_fig(fig_confusion_matrix(y_true, y_pred, classes))

    r2a, r2b = st.columns(2)
    with r2a:
        st.markdown("**Ehtimollik taqsimoti**")
        _show_fig(fig_prob_distribution(y_true, scores, classes, defect_idx))
    with r2b:
        st.markdown("**F1 sinf bo'yicha**")
        if eval_result["per_class"]:
            _show_fig(fig_per_class_bars(eval_result["per_class"]))

    st.markdown("**Confusion matrix tafsilotlari**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("TN (normal→normal)", conf["tn"])
    m2.metric("FP (normal→defect)", conf["fp"])
    m3.metric("FN (defect→normal)", conf["fn"])
    m4.metric("TP (defect→defect)", conf["tp"])

    _render_error_gallery(eval_result)

    st.divider()
    st.subheader("O'qitish tarixi")
    _render_saved_training(cfg)

    st.divider()
    st.subheader("Model taqqoslash")
    _render_comparison_charts(cfg, metrics)


@st.cache_resource
def _load_cached():
    return load_model_for_inference()


def main() -> None:
    st.set_page_config(
        page_title="Metall Nuqsoni Aniqlash",
        page_icon="🔍",
        layout="wide",
    )

    st.title("Metall Nuqsoni Aniqlash")
    st.markdown(
        "CNN modeli orqali quyma detal rasmlarini **nuqsonli** yoki **nuqsonsiz** deb tasniflash."
    )

    try:
        model, classes, cfg, device, state, ckpt_path = _load_cached()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    ckpt_info = get_checkpoint_info(ckpt_path)
    if state:
        ckpt_info.setdefault("epoch", state.get("epoch"))
        ckpt_info.setdefault("val_acc", state.get("val_acc"))

    with st.sidebar:
        st.header("Model")
        st.write(f"**Sinflar:** {', '.join(classes)}")
        st.write(f"**Parametrlar:** {count_parameters(model):,}")
        st.write(f"**Qurilma:** {device}")
        if ckpt_info.get("epoch"):
            st.write(f"**Epoch:** {ckpt_info['epoch']}")
        if ckpt_info.get("val_acc"):
            st.write(f"**Val accuracy:** {ckpt_info['val_acc'] * 100:.2f}%")
        st.caption(f"`{ckpt_path.name}`")
        if st.button("Modelni qayta yuklash"):
            st.cache_resource.clear()
            st.session_state.pop("eval_result", None)
            st.session_state.pop("eval_cache_key", None)
            st.rerun()

    tab_inspect, tab_features, tab_eval = st.tabs(
        ["🔍 Tekshiruv", "🧠 Feature Analysis", "📊 Baholash"]
    )

    with tab_inspect:
        _render_inspection_tab(model, classes, cfg, device)

    with tab_features:
        _render_feature_tab(model, classes, cfg, device)

    with tab_eval:
        _render_evaluation_tab(model, classes, cfg, device, ckpt_info)


if __name__ == "__main__":
    main()
