# Metall nuqsoni aniqlash (CNN + baseline)

Ikki klassli tasvir tasnifi: `defect` va `normal`. Loyiha `data.zip` dan boshlanadi.

## Talablar

- Python 3.10+
- NVIDIA GPU (ixtiyoriy, lekin tavsiya etiladi)

## GPU o'rnatish (PyTorch + CUDA)

[R pytorch.org](https://pytorch.org/get-started/locally/) sahifasidan CUDA versiyangizga mos buyruqni oling. Masalan (CUDA 12.1):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Tekshiruv:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

`config/settings.yaml` da `training.device: auto` — GPU bo'lsa CUDA, aks holda CPU.

## O'rnatish (ustoz boshqa kompyuterda ham shu tartibda)

Loyiha **Docker talab qilmaydi**. Ustoz o'z noutbukida faqat Python + `requirements.txt` bilan ishga tushiradi.

```bash
cd "AI assignment"
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Tekshiruv (barcha kutubxonalar va fayllar):

```bash
python scripts/verify_setup.py
```

**Muhim:** `.venv` papkasini loyiha ichida qoldiring — har safar faqat `activate` qiling, qayta `pip install` shart emas. Yangi kompyuterga butun `AI assignment` papkasini (`.venv` siz ham bo'ladi) ko'chirib, yuqoridagi o'rnatishni bir marta bajaring.

**Windows:** `DataLoader` xato bersa `config/settings.yaml` ichida `training.num_workers: 0` qiling.

## Har kuni ishga tushirish (Streamlit)

**Eng oson:** `run_app.bat` faylini ikki marta bosing. Hammasi.

Yoki terminalda **bitta buyruq**:

```powershell
cd "C:\Users\anaso\Desktop\AI assignment"
.\.venv\Scripts\python.exe scripts\run_app.py
```

`pip install` yoki `activate` **har safar shart emas** — faqat birinchi marta o'rnatilgan bo'lsa yetadi.

## Birinchi marta (to'liq pipeline)

```bash
python scripts/unpack_data.py
python scripts/prepare_data.py
python scripts/train.py
python scripts/evaluate.py
python scripts/run_tuning.py
python scripts/train_baseline.py
python -m streamlit run app/app.py
```

Yoki Windowsda ikki marta bosing: **`run_app.bat`**

Brauzerda oching: **http://127.0.0.1:8501** (yoki http://localhost:8501)

### localhost ishlamasa

1. Terminalda xato bormi? Avval model borligini tekshiring: `results/models/cnn_best.pth`
2. `streamlit` emas, shu buyruqni ishlating:
   ```bash
   python -m streamlit run app/app.py
   ```
3. Brauzerni qo'lda oching: `http://127.0.0.1:8501`
4. Port band bo'lsa:
   ```bash
   python -m streamlit run app/app.py --server.port 8502
   ```
   keyin `http://127.0.0.1:8502`
5. `.venv` yoqilganmi: `.venv\Scripts\activate`
6. Streamlit o'rnatilmagan bo'lsa: `pip install streamlit`

Tez sinov uchun `training.epochs` ni 3–5 ga tushirishingiz mumkin.

### Qo'shimcha

| Buyruq | Vazifa |
|--------|--------|
| `python scripts/train.py --resume` | CNN checkpointdan davom |
| `python scripts/run_tuning.py --skip-trained` | Tuning natijalari bor bo'lsa qayta o'qitmaslik |
| `python scripts/predict.py data/raw_images/img_00001.jpeg` | Bitta rasm |

## Loyiha tuzilishi

```
config/settings.yaml
data/          # zip, raw_images, label.csv, processed/
src/           # pipeline, model, train, eval
scripts/       # CLI
app/app.py     # Streamlit
notebooks/report.ipynb
results/       # models, metrics, plots, tuning, baseline, feature_maps
```

## Checkpoint va tarix

- Eng yaxshi CNN: `results/models/cnn_best.pth`
- Epoch tarixi: `results/metrics/training_history.json`
- Resume: `python scripts/train.py --resume` yoki `training.resume: true`

## Ma'lumotlar

- 7284 ta JPEG (300×300), `label.csv` ustunlari: `image`, `choice`
- Ziddiyatli dublikat (`img_01490.jpeg`) `prepare_data` da olib tashlanadi

## Notebook

`notebooks/report.ipynb` — `RUN_TRAINING = False` bilan mavjud natijalarni ko'rsatadi (to'liq o'qitish shart emas).

## cd "AI assignment"
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python scripts/verify_setup.py
python scripts/unpack_data.py
python scripts/prepare_data.py
python scripts/train.py
python scripts/evaluate.py
streamlit run app/app.py