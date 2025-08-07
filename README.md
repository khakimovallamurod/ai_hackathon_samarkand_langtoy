
## O‘rnatish

1. Loyihani klon qiling:

```bash
git clone https://github.com/khakimovallamurod/ai_hackathon_samarkand_langtoy
cd child-safety-monitor
```

2. Virtual muhit yarating va faollashtiring (ixtiyoriy):

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Kerakli paketlarni o‘rnating:

```bash
pip install -r requirements.txt
```

4. YOLO modellarini `models/` papkaga joylashtiring:

   * Asosiy model: `yolo11n.pt`
   * Bolalar modeli: `kid_model.pt`

5. `config.py` faylida quyidagilarni sozlang:

   * Telegram bot tokeni
   * Telegram chat ID

6. ESP32 qurilmangizning WebSocket manzilini `ESP32_WEBSOCKET_URL` ga yozing (masalan, `ws://192.168.4.1/ws`).

---

## Ishga tushirish robot bilan

```bash
python tracking_main.py
```

Yoki test qilish:

```bash
python camera_tracking.py
```

Kamera indeksini sozlash uchun `main()` funksiyasidagi `camera_index` qiymatini o‘zgartiring.

---

## Ishlash tartibi

* Kamera orqali olingan har bir frame YOLO modellar orqali analiz qilinadi.
* Aniqlangan odamlar va bolalar alohida kuzatiladi.
* Xavfli obektlar bilan odamlar orasidagi masofa tekshiriladi.
* Agar xavfli holat aniqlansa:

  * ESP32 ga xavfsizlik statusi va ogohlantirish yuboriladi.
  * Telegram orqali rasmlar va matnli xabarlar keladi.
* Annotatsiyalangan video oqim oynada ko‘rsatiladi.

---

Quyida `text_to_speach.py` ishga tushirish:

---

# 🗣️ Text to Speech (gTTS)

Ushbu loyiha Google Text-to-Speech (gTTS) kutubxonasi orqali matnni ovozga aylantiradi.

### 📌 Foydalanish:

1. Kutubxonani o‘rnating:

```bash
pip install gTTS
```

2. Skriptni ishga tushiring:

```bash
python text_to_speach.py
```

3. Audio `results/output.mp3` faylga saqlanadi.

### ⚙️ Asosiy funksiyalar:

* `text_to_speach_by_lang(text, lang='en', filename='output.mp3')` — Matnni tanlangan tilga ovozga aylantiradi.
* `get_supported_languages()` — Qo‘llab-quvvatlanadigan tillarni ko‘rsatadi.

---

## 🔒 Bolalar xavfsizligi monitoring — Flask ilovasi

### 📌 Asosiy imkoniyatlar:

* 📹 **Real vaqtli kamera oqimi**: `/video_feed` orqali browserda jonli video ko‘rsatadi.
* 🧠 `tracking_main.py` ichidagi `main()` funksiyasi orqali **YOLO asosida aniqlash va kuzatish** ishlaydi.
* 🤖 **ESP32 bilan WebSocket orqali aloqada bo‘ladi** (`robot_command` eventi).
* 🛜 `/send_command/<command>` orqali ESP32 ga buyruq (`forward`, `stop`, `left`, `right`, va h.k.) yuboriladi.
* 📲 **Telegramga alert yuboradi** (config fayldan token va chat\_id olinadi).

---

## ⚙️ Texnik tafsilotlar:

### 📁 Asosiy fayllar:

* `app.py` – Flask ilova kodi (yuqoridagi kod).
* `tracking_main.py` – Ob’ekt aniqlash va ESP32ga signal yuborish funksiyalari.
* `config.py` – Telegram token va chat ID ni saqlovchi fayl.
* `templates/index.html` – Frontend HTML sahifa (jonli video va tugmalar uchun).
* `child_safety_output.mp4` – Saqlanadigan video fayl nomi.

### 🧠 Muhim funksiyalar:

#### `generate_frames()`

`main()` funksiyasidan kadrlarni olib, `Response` orqali video sifatida uzatadi.

#### `@socketio.on('robot_command')`

Frontenddan real vaqtli robot buyruqlarini qabul qiladi.

#### `/send_command/<command>`

ESP32 ga POST so‘rov orqali buyruq yuboradi.

---

## 🚀 Ishga tushirish

```bash
python flask_app.py
```

So'ngra browserda oching: [http://localhost:8000](http://localhost:8000)

---

## 🧩 Foydali

* ESP32 uchun `send_to_esp32()` funktsiyasi aloqani boshqaradi.
* `config.py` quyidagicha bo'lishi mumkin:

```python
def get_token():
    return "YOUR_TELEGRAM_BOT_TOKEN"

def get_chat_id():
    return "YOUR_CHAT_ID"
```

---

# Bolalar Xavfsizligi Monitoring Tizimi

Bu loyiha real vaqt rejimida video oqimidan odamlar va xavfli obektlarni aniqlab, ular orasidagi xavfli masofani tekshiradi hamda ESP32 qurilmaga va Telegram orqali ogohlantirish yuboradi.

---

## Loyihaning asosiy xususiyatlari

* YOLO yordamida odamlar, bolalar (kid) va xavfli obektlarni aniqlash
* Ob'ektlarni individual tarzda kuzatish (tracker)
* Xavfli masofa aniqlanganda ESP32 ga real vaqt ma'lumot yuborish
* Telegram bot orqali xavfli holatlar haqida ogohlantirish
* Video oqimga chizilgan annotatsiyalar bilan ko‘rsatish
* ESP32 bilan WebSocket orqali ikki tomonlama aloqa

---

## Texnik Talablar

* Python 3.8 yoki yuqoriroq
* Quyidagi kutubxonalar:

  * `numpy`
  * `opencv-python`
  * `supervision`
  * `ultralytics`
  * `websocket-client`
  * `python-telegram-bot`
* YOLO modellar (asosiy va bolalar uchun)
* ESP32 qurilma va u bilan bog‘lanish uchun WebSocket manzili
* Telegram bot tokeni va chat ID (ogohlantirish uchun)

---
