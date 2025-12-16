import gradio as gr
import pandas as pd
from engine import recommend

KB_PATH = "data/crops_knowledge_base.csv"

def load_kb():
    return pd.read_csv(KB_PATH)

def infer(ph, ec, oc, rain_mm, tmin, tmax, water, shade_percent, slope_percent, land_size_acre, location):
    df = load_kb()
    inputs = {
        "ph": ph or 6.5,
        "ec": ec or 0.5,
        "oc": oc or 0.75,
        "rain_mm": rain_mm or 900,
        "tmin": tmin or 18,
        "tmax": tmax or 32,
        "water": (water or "medium").lower(),
        "shade_percent": shade_percent or 0,
        "slope_percent": slope_percent or 0,
        "land_acre": land_size_acre or 1.0,
        "location": location or "Karnataka"
    }

    recs = recommend(df, inputs, top_k=8)

    # quick plant count estimate for top 3 (if spacing like "3x3 m")
    plan_rows = []
    for _, r in recs.head(3).iterrows():
        spacing = str(r['spacing'])
        plants = ""
        if 'x' in spacing and 'm' in spacing:
            try:
                s = spacing.lower().replace('m', '').replace(' ', '')
                a, b = s.split('x')
                s1, s2 = float(a), float(b)
                # 1 acre = 4046.86 m2
                n = int((inputs['land_acre']*4046.86) / (s1*s2))
                plants = f" ~{n} plants in {inputs['land_acre']} acre(s)"
            except Exception:
                plants = ""
        plan_rows.append(f"{r['crop']}: {spacing}{plants}".strip())
    plan_text = "\n".join(plan_rows) if plan_rows else ""

    recs['explain'] = recs.apply(
        lambda x: f"Why: {x['reason']}\nIntercrops: {x['intercrops']}\nNotes: {x['notes']}",
        axis=1
    )
    display = recs[['crop', 'score', 'spacing', 'explain']]
    return display, plan_text

with gr.Blocks(title="Agricultural Land Adaptation System") as demo:
    gr.Markdown("# AI-Driven Agricultural Land Adaptation System\nProvide soil & climate to get ranked crop recommendations.")
    with gr.Row():
        with gr.Column():
            ph = gr.Number(label="Soil pH", value=6.5)
            ec = gr.Number(label="EC (dS/m)", value=0.3)
            oc = gr.Number(label="Organic Carbon (%)", value=0.75)
            rain = gr.Number(label="Annual Rainfall (mm)", value=900)
            tmin = gr.Number(label="Min Temp (°C)", value=18)
            tmax = gr.Number(label="Max Temp (°C)", value=32)
            water = gr.Dropdown(choices=["low", "medium", "high"], value="medium", label="Water Availability")
            shade = gr.Slider(0, 100, value=0, label="Shade (%)")
            slope = gr.Slider(0, 30, value=0, label="Slope (%)")
            land = gr.Number(label="Land Size (acre)", value=1.0)
            location = gr.Textbox(label="Location", value="Karnataka")
        with gr.Column():
            btn = gr.Button("Recommend", variant="primary")
            table = gr.Dataframe(headers=["crop", "score", "spacing", "explain"], row_count=8, wrap=True)
            plan = gr.Textbox(label="Planting Plan (Top 3)", lines=4)
    btn.click(infer, [ph, ec, oc, rain, tmin, tmax, water, shade, slope, land, location], [table, plan])

if __name__ == "__main__":
    demo.launch()