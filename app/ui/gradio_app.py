import time
import random
import gradio as gr
from typing import List
from app.services.ml_service import model_service

NUMERICAL_COLUMNS = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
]

def get_default_value(col):
    if col == "Time":
        return 1000.0
    if col == "Amount":
        return 50.0
    return 0.0

DEFAULT_VALUES = [get_default_value(c) for c in NUMERICAL_COLUMNS]

def build_feature_dict(values: List[float]):
    return dict(zip(NUMERICAL_COLUMNS, values))

def safe_format_probability(p):
    try:
        return f"{float(p):.4f}"
    except Exception:
        return "N/A"

def generate_random_request_inputs(deterministic: bool = False) -> List[float]:
    if deterministic:
        random.seed(42)

    values = []
    values.append(float(random.randint(0, 172800)))  # Time
    for _ in range(28):
        values.append(random.normalvariate(0, 1.5))
    values.append(round(random.uniform(0.01, 5000.00), 2))
    return values

def reset_inputs():
    return DEFAULT_VALUES.copy()

def predict_generator(*args):
    if len(args) != len(NUMERICAL_COLUMNS):
        yield "**Input error**: wrong number of fields."
        return

    yield "⏳ Running model inference…"

    features = build_feature_dict(args)

    try:
        start = time.time()
        result = model_service.predict(features)
        latency = time.time() - start

        if isinstance(result, dict):
            prob_value = result.get("fraud_probability")
            version = result.get("model_version", "unknown")
        else:
            prob_value = result
            version = model_service.model_meta.get("version", "local")

        prob = safe_format_probability(prob_value)

        risk_label = ""
        risk_color = ""
        try:
            p = float(prob_value)
            if p >= 0.7:
                risk_label = "High Risk"
                risk_color = "#dc2626"
            elif p >= 0.4:
                risk_label = "Medium Risk"
                risk_color = "#f59e0b"
            else:
                risk_label = "Low Risk"
                risk_color = "#16a34a"
        except Exception:
            pass

        md = f"""
<div style="padding: 24px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; margin-bottom: 20px;">
    <h2 style="color: white; margin: 0 0 8px 0; font-size: 1.2em;">Fraud Probability</h2>
    <div style="font-size: 3.5em; font-weight: bold; color: white; margin: 12px 0;">{prob}</div>
    <div style="background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px; display: inline-block;">
        <span style="color: white; font-weight: 600; font-size: 1.1em;">{risk_label}</span>
    </div>
</div>

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 16px;">
    <div style="background: #f8fafc; padding: 16px; border-radius: 8px; border-left: 4px solid #3b82f6;">
        <div style="color: #64748b; font-size: 0.85em; margin-bottom: 4px;">Model Version</div>
        <div style="color: #1e293b; font-weight: 600; font-family: monospace;">{version}</div>
    </div>
    <div style="background: #f8fafc; padding: 16px; border-radius: 8px; border-left: 4px solid #8b5cf6;">
        <div style="color: #64748b; font-size: 0.85em; margin-bottom: 4px;">Inference Time</div>
        <div style="color: #1e293b; font-weight: 600;">{latency:.3f}s</div>
    </div>
</div>
"""
        yield md

    except Exception as e:
        yield f"""
<div style="background: #fef2f2; border-left: 4px solid #dc2626; padding: 16px; border-radius: 8px;">
    <div style="color: #dc2626; font-weight: 600; margin-bottom: 4px;">❌ Prediction Error</div>
    <div style="color: #991b1b; font-size: 0.9em;">{e}</div>
</div>
"""

# Enhanced theme with better colors
custom_theme = gr.themes.Soft(
    primary_hue="purple",
    secondary_hue="blue",
    neutral_hue="slate",
).set(
    button_primary_background_fill="linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #5a67d8 0%, #6b3fa0 100%)",
    button_primary_text_color="white",
    block_title_text_weight="600",
    block_label_text_weight="500",
)

with gr.Blocks(
    title="Fraud Detection Service Demo",
    theme=custom_theme,
    css="""
    .gradio-container {
        max-width: 1200px !important;
    }
    .gr-button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    .gr-button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }
    .gr-box {
        border-radius: 12px !important;
        border: 1px solid #e2e8f0 !important;
    }
    .gr-input {
        border-radius: 6px !important;
        border: 1px solid #e2e8f0 !important;
    }
    .gr-input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    .gr-panel {
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    }
    h2, h3 {
        color: #1e293b !important;
    }
    .input-column {
        max-height: 600px;
        overflow-y: auto;
        padding-right: 8px;
    }
    .input-column::-webkit-scrollbar {
        width: 8px;
    }
    .input-column::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 4px;
    }
    .input-column::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }
    .input-column::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
    """
) as demo:
    gr.Markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 2.5em;
            font-weight: 800;
            margin: 0;
        ">🛡️ Fraud Detection Service</h1>
        <p style="color: #64748b; margin-top: 8px; font-size: 1.1em;">
            Real-time transaction fraud prediction using machine learning
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Column(variant="panel"):
                gr.Markdown("### 📊 Transaction Inputs")

                with gr.Row():
                    predict_btn = gr.Button("🚀 Predict", variant="primary", size="lg")
                    random_btn = gr.Button("🎲 Random", size="lg")
                    reset_btn = gr.Button("🔄 Reset", size="lg")

                deterministic_checkbox = gr.Checkbox(
                    label="🎯 Deterministic Random (seed=42)",
                    value=False,
                    info="Use fixed seed for reproducible random values"
                )

                with gr.Column(elem_classes="input-column"):
                    feature_inputs = []
                    for col in NUMERICAL_COLUMNS:
                        comp = gr.Number(
                            label=col,
                            value=get_default_value(col),
                            precision=2
                        )
                        feature_inputs.append(comp)

        with gr.Column(scale=1):
            with gr.Column(variant="panel"):
                gr.Markdown("### 📈 Prediction Output")
                output_md = gr.Markdown("""
                <div style="
                    text-align: center;
                    padding: 40px 20px;
                    color: #64748b;
                    background: #f8fafc;
                    border-radius: 8px;
                    border: 2px dashed #cbd5e1;
                ">
                    <div style="font-size: 3em; margin-bottom: 12px;">🎯</div>
                    <div style="font-size: 1.1em;">Fill the inputs and click <strong>Predict</strong></div>
                </div>
                """)

    random_btn.click(
        fn=generate_random_request_inputs,
        inputs=[deterministic_checkbox],
        outputs=feature_inputs,
        queue=False,
    )

    reset_btn.click(
        fn=reset_inputs,
        inputs=None,
        outputs=feature_inputs,
        queue=False,
    )

    predict_btn.click(
        fn=predict_generator,
        inputs=feature_inputs,
        outputs=output_md,
        queue=True,
    )

gr_app = demo