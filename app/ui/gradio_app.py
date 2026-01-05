import time
import random
import gradio as gr
import pandas as pd
from io import BytesIO
from typing import List
from app.services.ml_service import model_service
from app.db.db import log_request
from app.config.config import logger

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

        try:
            log_request(features, prob_value)
        except Exception as log_err:
            logger.warning(f"Failed to log UI request to DB: {log_err}")

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

def process_batch_csv(file):
    """Process uploaded CSV file and return predictions with DB logging enabled"""
    if file is None:
        return None, """
<div style="background: #fef2f2; border-left: 4px solid #dc2626; padding: 16px; border-radius: 8px;">
    <div style="color: #dc2626; font-weight: 600;">❌ No file uploaded</div>
</div>
"""
    
    try:
        # Read the CSV file
        df = pd.read_csv(file.name)
        
        # Validate required columns
        required_cols = model_service.feature_columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return None, f"""
<div style="background: #fef2f2; border-left: 4px solid #dc2626; padding: 16px; border-radius: 8px;">
    <div style="color: #dc2626; font-weight: 600; margin-bottom: 4px;">❌ Missing Required Columns</div>
    <div style="color: #991b1b; font-size: 0.9em;">Missing: {', '.join(missing_cols)}</div>
</div>
"""
        
        start_time = time.time()
        
        # Make predictions for each row
        predictions = []
        for idx, row in df.iterrows():
            features = row[required_cols].to_dict()
            result = model_service.predict(features)
            
            if isinstance(result, dict):
                prob = result.get("fraud_probability")
            else:
                prob = result
            
            predictions.append(prob)
            
            # Log to database (always enabled)
            try:
                log_request(features, prob)
            except Exception as log_err:
                logger.warning(f"Failed to log batch row {idx} to DB: {log_err}")
        
        # Add predictions to dataframe
        df["fraud_probability"] = predictions
        
        # Calculate statistics
        total_rows = len(df)
        high_risk = sum(1 for p in predictions if p >= 0.7)
        medium_risk = sum(1 for p in predictions if 0.4 <= p < 0.7)
        low_risk = sum(1 for p in predictions if p < 0.4)
        avg_prob = sum(predictions) / len(predictions) if predictions else 0
        latency = time.time() - start_time
        
        # Save to temporary file for download
        output_path = "/tmp/batch_predictions.csv"
        df.to_csv(output_path, index=False)
        
        # Log batch processing event
        logger.info(f"Batch prediction via UI: {total_rows} rows, {latency:.3f}s")
        
        # Create summary markdown
        summary = f"""
<div style="padding: 24px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; margin-bottom: 20px;">
    <h2 style="color: white; margin: 0 0 16px 0; font-size: 1.3em;">📊 Batch Prediction Complete</h2>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px;">
        <div style="background: rgba(255,255,255,0.2); padding: 12px; border-radius: 8px;">
            <div style="color: rgba(255,255,255,0.8); font-size: 0.85em; margin-bottom: 4px;">Total Transactions</div>
            <div style="color: white; font-weight: 700; font-size: 1.8em;">{total_rows}</div>
        </div>
        <div style="background: rgba(255,255,255,0.2); padding: 12px; border-radius: 8px;">
            <div style="color: rgba(255,255,255,0.8); font-size: 0.85em; margin-bottom: 4px;">Average Probability</div>
            <div style="color: white; font-weight: 700; font-size: 1.8em;">{avg_prob:.4f}</div>
        </div>
    </div>
</div>

<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 16px;">
    <div style="background: #fee2e2; padding: 16px; border-radius: 8px; border-left: 4px solid #dc2626;">
        <div style="color: #991b1b; font-size: 0.85em; margin-bottom: 4px;">🔴 High Risk (≥0.7)</div>
        <div style="color: #7f1d1d; font-weight: 700; font-size: 1.5em;">{high_risk}</div>
        <div style="color: #991b1b; font-size: 0.75em; margin-top: 4px;">{(high_risk/total_rows*100):.1f}%</div>
    </div>
    <div style="background: #fef3c7; padding: 16px; border-radius: 8px; border-left: 4px solid #f59e0b;">
        <div style="color: #92400e; font-size: 0.85em; margin-bottom: 4px;">🟡 Medium Risk (0.4-0.7)</div>
        <div style="color: #78350f; font-weight: 700; font-size: 1.5em;">{medium_risk}</div>
        <div style="color: #92400e; font-size: 0.75em; margin-top: 4px;">{(medium_risk/total_rows*100):.1f}%</div>
    </div>
    <div style="background: #dcfce7; padding: 16px; border-radius: 8px; border-left: 4px solid #16a34a;">
        <div style="color: #166534; font-size: 0.85em; margin-bottom: 4px;">🟢 Low Risk (<0.4)</div>
        <div style="color: #14532d; font-weight: 700; font-size: 1.5em;">{low_risk}</div>
        <div style="color: #166534; font-size: 0.75em; margin-top: 4px;">{(low_risk/total_rows*100):.1f}%</div>
    </div>
</div>

<div style="background: #f8fafc; padding: 16px; border-radius: 8px; border-left: 4px solid #8b5cf6;">
    <div style="color: #64748b; font-size: 0.85em; margin-bottom: 4px;">⚡ Processing Time</div>
    <div style="color: #1e293b; font-weight: 600;">{latency:.3f}s ({total_rows/latency:.1f} rows/sec)</div>
</div>

<div style="background: #eff6ff; padding: 16px; border-radius: 8px; margin-top: 16px; border: 1px solid #dbeafe;">
    <div style="color: #1e40af; font-weight: 600; margin-bottom: 8px;">✅ Success</div>
    <div style="color: #1e3a8a; font-size: 0.9em;">Your predictions are ready for download. The CSV includes all original columns plus a new fraud_probability column.</div>
</div>
"""
        
        return output_path, summary
        
    except Exception as e:
        logger.exception(f"Batch processing error in UI: {e}")
        return None, f"""
<div style="background: #fef2f2; border-left: 4px solid #dc2626; padding: 16px; border-radius: 8px;">
    <div style="color: #dc2626; font-weight: 600; margin-bottom: 4px;">❌ Processing Error</div>
    <div style="color: #991b1b; font-size: 0.9em;">{str(e)}</div>
</div>
"""

def generate_csv_template():
    """Generate a sample CSV template with example data"""
    try:
        columns = model_service.feature_columns
        
        sample_data = [
            [0,-1.3598071336738,-0.0727811733098497,2.53634673796914,1.37815522427443,-0.338320769942518,0.462387777762292,0.239598554061257,0.0986979012610507,0.363786969611213,0.0907941719789316,-0.551599533260813,-0.617800855762348,-0.991389847235408,-0.311169353699879,1.46817697209427,-0.470400525259478,0.207971241929242,0.0257905801985591,0.403992960255733,0.251412098239705,-0.018306777944153,0.277837575558899,-0.110473910188767,0.0669280749146731,0.128539358273528,-0.189114843888824,0.133558376740387,-0.0210530534538215,149.62],
            [0,1.19185711131486,0.26615071205963,0.16648011335321,0.448154078460911,0.0600176492822243,-0.0823608088155687,-0.0788029833323113,0.0851016549148104,-0.255425128109186,-0.166974414004614,1.61272666105479,1.06523531137287,0.48909501589608,-0.143772296441519,0.635558093258208,0.463917041022171,-0.114804663102346,-0.183361270123994,-0.145783041325259,-0.0690831352230203,-0.225775248033138,-0.638671952771851,0.101288021253234,-0.339846475529127,0.167170404418143,0.125894532368176,-0.00898309914322813,0.0147241691924927,2.69],
            [1,-1.35835406159823,-1.34016307473609,1.77320934263119,0.379779593034328,-0.503198133318193,1.80049938079263,0.791460956450422,0.247675786588991,-1.51465432260583,0.207642865216696,0.624501459424895,0.066083685268831,0.717292731410831,-0.165945922763554,2.34586494901581,-2.89008319444231,1.10996937869599,-0.121359313195888,-2.26185709530414,0.524979725224404,0.247998153469754,0.771679401917229,0.909412262347719,-0.689280956490685,-0.327641833735251,-0.139096571514147,-0.0553527940384261,-0.0597518405929204,378.66],
            [27219,-25.2663550194138,14.3232538097233,-26.8236729135114,6.34924780743689,-18.664250613469,-4.64740304866878,-17.9712120192706,16.6331030618556,-3.76835097141465,-8.303239351259,4.78325736701241,-6.6992520739678,0.846767864669643,-6.57627643636006,-0.0623303798952992,-5.96165987257541,-12.2184817176797,-4.79184198325342,0.894853521838799,1.65828884445902,1.78070097046593,-1.86131814726914,-1.18816729293127,0.156667050663465,1.76819198236914,-0.219916008250323,1.41185477685432,0.414656383638367,99.99],
            [2,-1.15823309349523,0.877736754848451,1.548717846511,0.403033933955121,-0.407193377311653,0.0959214624684256,0.592940745385545,-0.270532677192282,0.817739308235294,0.753074431976354,-0.822842877946363,0.53819555014995,1.3458515932154,-1.11966983471731,0.175121130008994,-0.451449182813529,-0.237033239362776,-0.0381947870352842,0.803486924960175,0.408542360392758,-0.00943069713232919,0.79827849458971,-0.137458079619063,0.141266983824769,-0.206009587619756,0.502292224181569,0.219422229513348,0.215153147499206,69.99],
            [2,-0.425965884412454,0.960523044882985,1.14110934232219,-0.168252079760302,0.42098688077219,-0.0297275516639742,0.476200948720027,0.260314333074874,-0.56867137571251,-0.371407196834471,1.34126198001957,0.359893837038039,-0.358090652573631,-0.137133700217612,0.517616806555742,0.401725895589603,-0.0581328233640131,0.0686531494425432,-0.0331937877876282,0.0849676720682049,-0.208253514656728,-0.559824796253248,-0.0263976679795373,-0.371426583174346,-0.232793816737034,0.105914779097957,0.253844224739337,0.0810802569229443,3.67],
            [7,-0.644269442348146,1.41796354547385,1.0743803763556,-0.492199018495015,0.948934094764157,0.428118462833089,1.12063135838353,-3.80786423873589,0.615374730667027,1.24937617815176,-0.619467796121913,0.291474353088705,1.75796421396042,-1.32386521970526,0.686132504394383,-0.0761269994382006,-1.2221273453247,-0.358221569869078,0.324504731321494,-0.156741852488285,1.94346533978412,-1.01545470979971,0.057503529867291,-0.649709005559993,-0.415266566234811,-0.0516342969262494,-1.20692108094258,-1.08533918832377,40.8],
            [8528,0.447395553302475,2.48195386638743,-5.66081393141405,4.4559228120932,-2.4437797540431,-2.18504026247234,-4.71614294470093,1.24980325173147,-0.718326066573691,-5.3903302556601,6.45418752494833,-8.48534657377678,0.635281408794591,-7.01990155916612,0.53981407134798,-4.6498642988132,-6.2883575087823,-1.33931232244731,2.26298478762517,0.549612970886705,0.756052550277073,0.140167768675,0.665411100673545,0.131463791724986,-1.90821741154788,0.334807598686864,0.748534284767756,0.175413794422896,1],
            [7,-0.89428608220282,0.286157196276544,-0.113192212729871,-0.271526130088604,2.6695986595986,3.72181806112751,0.370145127676916,0.851084443200905,-0.392047586798604,-0.410430432848439,-0.705116586646536,-0.110452261733098,-0.286253632470583,0.0743553603016731,-0.328783050303565,-0.210077268148783,-0.499767968800267,0.118764861004217,0.57032816746536,0.0527356691149697,-0.0734251001059225,-0.268091632235551,-0.204232669947878,1.0115918018785,0.373204680146282,-0.384157307702294,0.0117473564581996,0.14240432992147,93.2],
            [9,-0.33826175242575,1.11959337641566,1.04436655157316,-0.222187276738296,0.49936080649727,-0.24676110061991,0.651583206489972,0.0695385865186387,-0.736727316364109,-0.366845639206541,1.01761446783262,0.836389570307029,1.00684351373408,-0.443522816876142,0.150219101422635,0.739452777052119,-0.540979921943059,0.47667726004282,0.451772964394125,0.203711454727929,-0.246913936910008,-0.633752642406113,-0.12079408408185,-0.385049925313426,-0.0697330460416923,0.0941988339514961,0.246219304619926,0.0830756493473326,3.68],
        ]
        
        df = pd.DataFrame(sample_data, columns=columns)
        output_path = "/tmp/batch_template.csv"
        df.to_csv(output_path, index=False)
        
        logger.info("CSV template generated via UI")
        return output_path
        
    except Exception as e:
        logger.exception(f"Error generating CSV template: {e}")
        return None

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
        max-width: 1400px !important;
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
    
    with gr.Tabs():
        # Tab 1: Individual Prediction
        with gr.Tab("🎯 Single Prediction"):
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
                            <div style="font-size: 1.1em; color: #475569;">Fill the inputs and click <strong style="color: #475569;">Predict</strong></div>
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
        
        # Tab 2: Batch Prediction
        with gr.Tab("📁 Batch Prediction (CSV)"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Column(variant="panel"):
                        gr.Markdown("### 📤 Upload CSV File")
                        gr.Markdown("""
                        <div style="background: #eff6ff; padding: 16px; border-radius: 8px; margin-bottom: 16px; border: 1px solid #dbeafe;">
                            <div style="color: #1e40af; font-weight: 600; margin-bottom: 8px;">ℹ️ Instructions</div>
                            <ul style="color: #1e40af !important; font-size: 0.9em; margin: 0; padding-left: 20px;">
                                <li style="color: #1e40af !important;">Upload a CSV file with the required 30 columns</li>
                                <li style="color: #1e40af !important;">Each row will be processed and predictions added</li>
                                <li style="color: #1e40af !important;">Download the template below if you need a sample format</li>
                            </ul>
                        </div>
                        """)
                        
                        csv_file = gr.File(
                            label="Upload CSV File",
                            file_types=[".csv"],
                            type="filepath"
                        )
                        
                        with gr.Row():
                            process_btn = gr.Button("🚀 Process Batch", variant="primary", size="lg")
                            template_btn = gr.Button("📥 Download Template", size="lg")
                        
                        template_file = gr.File(label="📥 Template CSV", visible=False)
                
                with gr.Column(scale=1):
                    with gr.Column(variant="panel"):
                        gr.Markdown("### 📊 Batch Results")
                        
                        batch_output_md = gr.Markdown("""
                        <div style="
                            text-align: center;
                            padding: 40px 20px;
                            color: #64748b;
                            background: #f8fafc;
                            border-radius: 8px;
                            border: 2px dashed #cbd5e1;
                        ">
                            <div style="font-size: 3em; margin-bottom: 12px;">📁</div>
                            <div style="font-size: 1.1em; color: #475569;">Upload a CSV file and click <strong style="color: #475569;">Process Batch</strong></div>
                        </div>
                        """)
                    
                    result_file = gr.File(label="📥 Download Results", visible=False)
            
            # Event handlers for batch prediction
            process_btn.click(
                fn=process_batch_csv,
                inputs=[csv_file],
                outputs=[result_file, batch_output_md],
                queue=True,
            ).then(
                fn=lambda: gr.update(visible=True),
                inputs=None,
                outputs=[result_file],
                queue=False,
            )
            
            template_btn.click(
                fn=generate_csv_template,
                inputs=None,
                outputs=[template_file],
                queue=False,
            ).then(
                fn=lambda: gr.update(visible=True),
                inputs=None,
                outputs=[template_file],
                queue=False,
            )

gr_app = demo