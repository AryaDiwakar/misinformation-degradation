# Demo

Install the optional demo dependency and start the Streamlit interface:

```bash
pip install -r requirements-demo.txt
python3 scripts/create_sample_cascade.py
streamlit run demo/app.py
```

The demo expects a metadata-rich checkpoint created by `train.py` and an `.npz`
cascade containing `x` and `edge_index`. `hop_depth`, `parent`, and `n_children`
are optional and are reconstructed when absent.
