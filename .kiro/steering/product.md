# EdgeAgriNet Product Mission

Socio-Economic Agritech Platform

## Project Brief: EdgeAgri-Net

### Overview & Abstract

EdgeAgri-Net is an intelligent, multi-task deep learning platform designed to revolutionize precision agriculture on rural edge-hardware. Traditional agricultural AI models operate in complete isolation—they either classify diseases from leaf photos or forecast yields from weather records, but rarely both. Furthermore, they are typically built as heavy, cloud-dependent architectures that completely break down in rural areas lacking stable internet access.

EdgeAgri-Net solves these issues by introducing a Lightweight Multimodal Cross-Attention Network. It simultaneously ingests high-density spatial crop imagery and sparse, continuous 14-day chronological weather metrics. By fusing these data streams in an intermediate cross-attention layer, the system generates four parallel, real-time outputs: disease diagnoses, yield volume forecasts, optimal annual planting cycle selections, and precision resource/action prescriptions. To bridge the gap between biological insights and a farmer's economic survival, the platform also incorporates a Dynamic Market-Price Elasticity & Simulation Engine to model forward-looking financial choices.

### Active Feature Modules

| Module Name | Input Modalities Required | Output Interface Element | Practical Value for Farmers |
|-------------|---------------------------|--------------------------|----------------------------|
| Multimodal Diagnostics | Leaf Camera Image + 14-Day Temperature, Humidity, Rain Logs | Classification Tag + Sharp Grad-CAM Attention Heatmap | Catches fungal/bacterial infections early before they spread to the rest of the field. |
| Spatiotemporal Yield Tracking | Image Embedding Vector + Cumulative Soil Moisture History | Continuous Numerical Value (e.g., 4.2 Tons / Hectare) | Allows farmers to estimate harvest volumes accurately and arrange shipping logistics in advance. |
| Market Price Simulator | Predicted Yield Volume + Scraped Daily Wholesale Market Prices | Scenario Projection Cards (Bearish / Baseline / Bullish) | Helps farmers decide whether to sell immediately, hold crops in cold storage, or change crops for the next cycle. |
| Prescriptive Agronomist Engine | Fused Cross-Attention Context Matrix Vector | Actionable Text Alert box (Quantified input steps) | Gives clear, immediate instructions on exactly how much fertilizer or water to adjust to save money. |
| Maturity (GDD) Countdown | Historical Heat Accumulation Limits | Visual Timeline Tracker (e.g., 14 Days to Optimal Harvest) | Predicts the exact week of peak crop market maturity to maximize quality and profits. |