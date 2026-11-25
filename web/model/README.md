# web/model/

Aquí va la red FAMACHA-Net exportada a TensorFlow.js. La página la carga sola al abrirla con `python serve.py`.

Se llena con:

```bash
python run_pipeline.py --labels data/<tu_dataset>/labels.csv
# o, si ya entrenaste:
python scripts/export_tfjs.py --run runs/<tu_run> --single --to-web
```

Archivos esperados: `model.json`, `group1-shard*.bin`, `famacha_net_app_config.json`, `model_info.json`.
Mientras esta carpeta esté vacía, la página usa el motor colorimétrico.
