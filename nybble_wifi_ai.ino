#include <WiFi.h>
#include <ESPAsyncWebServer.h>
#include <ESP32Servo.h>

Servo FL_Hip, FL_Knee;  // Old chap oyoq
Servo FR_Hip, FR_Knee;  // Old o'ng oyoq
Servo BL_Hip, BL_Knee;  // Orqa chap oyoq
Servo BR_Hip, BR_Knee;  // Orqa o'ng oyoq
Servo neck, head;

#define FL_HIP_PIN 13   // Old chap oyoq - bo'g'im
#define FL_KNEE_PIN 12  // Old chap oyoq - tizza
#define FR_HIP_PIN 14   // Old o'ng oyoq - bo'g'im
#define FR_KNEE_PIN 27  // Old o'ng oyoq - tizza
#define BL_HIP_PIN 26   // Orqa chap oyoq - bo'g'im
#define BL_KNEE_PIN 25  // Orqa chap oyoq - tizza
#define BR_HIP_PIN 33   // Orqa o'ng oyoq - bo'g'im
#define BR_KNEE_PIN 32  // Orqa o'ng oyoq - tizza
#define NECK_PIN 19     // bo'yin boshi
#define HEAD_PIN 18     // bo'yin pasti
#define buzzer 4      

int speed = 20;


const char *ssid = "Robot_Mushuk";
const char *password = "12345678";

AsyncWebServer server(80);
AsyncWebSocket ws("/ws");


void do_stand_up() {
  int frHipStart = FR_Hip.read();
  int frKneeStart = FR_Knee.read();
  int flHipStart = FL_Hip.read();
  int flKneeStart = FL_Knee.read();
  int brHipStart = BR_Hip.read();
  int brKneeStart = BR_Knee.read();
  int blHipStart = BL_Hip.read();
  int blKneeStart = BL_Knee.read();

  for (int i = 0; i <= speed; i++) {  // standart tik turish
    FR_Hip.write(map(i, 0, speed, frHipStart, 80));
    FR_Knee.write(map(i, 0, speed, frKneeStart, 60));
    FL_Hip.write(map(i, 0, speed, flHipStart, 100));
    FL_Knee.write(map(i, 0, speed, flKneeStart, 110));
    BR_Hip.write(map(i, 0, speed, brHipStart, 160));
    BR_Knee.write(map(i, 0, speed, brKneeStart, 110));
    BL_Hip.write(map(i, 0, speed, blHipStart, 10));
    BL_Knee.write(map(i, 0, speed, blKneeStart, 70));
    delay(20);
  }
}

void do_sit_down() {
  for (int i = 0; i <= speed + 20; i++) {  // Orqa oyoqlarni bukish
    BL_Hip.write(map(i, 0, speed, 10, 0));
    BR_Hip.write(map(i, 0, speed, 170, 180));
    BL_Knee.write(map(i, 0, speed, 90, 0));
    BR_Knee.write(map(i, 0, speed, 70, 180));
    delay(20);
  }

  for (int i = 0; i <= speed + 20; i++) {  // Old oyoqlarni bukish
    FL_Hip.write(map(i, 0, speed, 100, 105));
    FR_Hip.write(map(i, 0, speed, 80, 75));
    FL_Knee.write(map(i, 0, speed, 90, 180));
    FR_Knee.write(map(i, 0, speed, 90, 0));
    delay(20);
  }
}

void do_walk(int a) {
  do_stand_up();

  for (int i = 0; i < a; i++) {
    for (int i = 0; i <= speed; i++) {           // old o'ng oyoq ortga
      FR_Hip.write(map(i, 0, speed, 110, 100));  // ort ga 10c
      FR_Knee.write(map(i, 0, speed, 90, 100));  // ort ga 10c
      BL_Knee.write(map(i, 0, speed, 60, 80));   // old ga 20c
      BL_Hip.write(map(i, 0, speed, 0, 10));     // ort ga  30c
      delay(20);
    }

    for (int i = 0; i <= speed; i++) {            //  FL va BR chap oyoq oldinga
      FR_Hip.write(map(i, 0, speed, 100, 70));    // ort ga  30c
      FR_Knee.write(map(i, 0, speed, 100, 100));  // ort ga  0c
      FL_Hip.write(map(i, 0, speed, 100, 60));    // old ga  40c
      FL_Knee.write(map(i, 0, speed, 70, 80));    // old ga  10c
      BR_Hip.write(map(i, 0, speed, 145, 175));   // old ga  25 || 30c
      BR_Knee.write(map(i, 0, speed, 80, 100));   // ort ga  20c
      BL_Hip.write(map(i, 0, speed, 10, 30));     // ort ga  30c
      BL_Knee.write(map(i, 0, speed, 80, 80));    // ort ga  20c
      delay(20);
    }

    for (int i = 0; i <= speed; i++) {           // old chap oyoq ortga
      FL_Hip.write(map(i, 0, speed, 60, 70));    // ort ga 10c
      FL_Knee.write(map(i, 0, speed, 80, 70));   // ort ga 10c
      BR_Knee.write(map(i, 0, speed, 100, 80));  // old ga 20c
      delay(20);
    }

    for (int i = 0; i <= speed; i++) {           // FR va BL o'ng oyoq oldinga
      FR_Hip.write(map(i, 0, speed, 70, 110));   // old ga  40c
      FR_Knee.write(map(i, 0, speed, 100, 90));  // old ga 10c
      FL_Hip.write(map(i, 0, speed, 80, 110));   // ort ga  30c
      FL_Knee.write(map(i, 0, speed, 70, 70));   // ort ga   0c
      BR_Hip.write(map(i, 0, speed, 175, 145));  // ort ga 25 || 30c
      BR_Knee.write(map(i, 0, speed, 80, 80));   // ort ga 20c
      BL_Hip.write(map(i, 0, speed, 30, 0));     // old ga	  30c
      BL_Knee.write(map(i, 0, speed, 80, 60));   // ort ga  20c
      delay(20);
    }

    delay(500);
  }

  for (int i = 0; i <= speed + 20; i++) {  // Orqa va Old oyoqlarni bukish
    BL_Hip.write(map(i, 0, speed, 0, 0));
    BR_Hip.write(map(i, 0, speed, 145, 180));
    FL_Hip.write(map(i, 0, speed, 110, 105));
    FR_Hip.write(map(i, 0, speed, 110, 85));
    delay(20);
  }
  for (int i = 0; i <= speed + 20; i++) {
    BL_Knee.write(map(i, 0, speed, 60, 0));
    BR_Knee.write(map(i, 0, speed, 80, 180));
    FL_Knee.write(map(i, 0, speed, 70, 180));
    FR_Knee.write(map(i, 0, speed, 90, 0));
    delay(20);
  }
}

void neckMove() {
  for (int i = 0; i <= 110; i++) {  // o'ngdan -> chapga
    neck.write(i);
    delay(20);
  }
  delay(20);

  for (int i = 110; i >= 0; i--) {  // chapdan -> o'ngga
    neck.write(i);
    delay(20);
  }
  delay(20);
}

void headMove() {
  for (int i = 0; i <= 90; i++) {  // tepadan -> pastga
    head.write(i);
    delay(20);
  }
  delay(100);

  for (int i = 90; i >= 0; i--) {  // pastdan -> tepaga
    head.write(i);
    delay(20);
  }
  delay(100);
}

void qol_silkish() {
  for (int i = 0; i <= speed + 10; i++) {
    FR_Hip.write(map(i, 0, speed, 90, 160));
    FR_Knee.write(map(i, 0, speed, 90, 65));
    neck.write(map(i, 0, speed, 50, 30));
    head.write(map(i, 0, speed, 20, 50));
    delay(20);
  }

  for (int i = 0; i < 1; i++) {
    for (int i = 0; i <= speed + 10; i++) {
      FR_Hip.write(map(i, 0, speed, 160, 180));
      // FR_Knee.write(map(i, 0, speed, 65, 75)); // ort ga
      delay(20);
    }

    for (int i = 0; i <= speed + 10; i++) {
      head.write(map(i, 0, speed, 50, 30));
      delay(20);
    }
    for (int i = 0; i <= speed + 10; i++) {
      head.write(map(i, 0, speed, 30, 50));
      delay(20);
    }

    for (int i = 0; i <= speed + 10; i++) {
      FR_Hip.write(map(i, 0, speed, 180, 160));
      delay(20);
    }
  }

  for (int i = 0; i <= speed; i++) {
    FR_Hip.write(map(i, 0, speed, 160, 80));
    FR_Knee.write(map(i, 0, speed, 65, 60));
    neck.write(map(i, 0, speed, 30, 50));
    head.write(map(i, 0, speed, 50, 20));
    delay(20);
  }
}

void do_tik_turish() {
  for (int i = 0; i <= speed; i++) {
    FL_Hip.write(map(i, 0, speed, 105, 110));  // old oyoq bilan ko'tarilish
    FL_Knee.write(map(i, 0, speed, 180, 110));
    FR_Hip.write(map(i, 0, speed, 75, 70));
    FR_Knee.write(map(i, 0, speed, 0, 50));

    BL_Hip.write(map(i, 0, speed, 0, 10));  // orqa oyoq bilan o'tirish
    BR_Hip.write(map(i, 0, speed, 180, 140));
    BL_Knee.write(map(i, 0, speed, 0, 0));
    BR_Knee.write(map(i, 0, speed, 180, 170));
    delay(20);
  }
  delay(1000);
  qol_silkish();

  for (int i = 0; i <= speed + 20; i++) {  // Orqa oyoqlarni bukish
    BL_Hip.write(map(i, 0, speed, 10, 0));
    BR_Hip.write(map(i, 0, speed, 170, 180));
    BL_Knee.write(map(i, 0, speed, 90, 0));
    BR_Knee.write(map(i, 0, speed, 70, 180));
    delay(20);
  }
  for (int i = 0; i <= speed + 20; i++) {  // Old oyoqlarni bukish
    FL_Hip.write(map(i, 0, speed, 110, 105));
    FR_Hip.write(map(i, 0, speed, 80, 75));
    FL_Knee.write(map(i, 0, speed, 110, 180));
    FR_Knee.write(map(i, 0, speed, 60, 0));
    delay(20);
  }
}

void do_push_up(int num) {
  for (int i = 0; i <= speed + 20; i++) {  // Orqa oyoqlarni bukish
    BL_Hip.write(map(i, 0, speed, 10, 60));
    BR_Hip.write(map(i, 0, speed, 160, 110));
    BL_Knee.write(map(i, 0, speed, 70, 110));
    BR_Knee.write(map(i, 0, speed, 110, 70));
    delay(20);
  }
  for (int i = 0; i <= speed + 20; i++) {  // Old oyoqlarni ko'tarish
    FR_Hip.write(map(i, 0, speed, 80, 90));
    FL_Hip.write(map(i, 0, speed, 100, 90));
    FR_Knee.write(map(i, 0, speed, 60, 80));
    FL_Knee.write(map(i, 0, speed, 110, 90));
    head.write(map(i, 0, speed, 0, 20));
    delay(20);
  }

  for (int i = 0; i < num; i++) {
    for (int i = 0; i <= speed + 20; i++) {  // Old oyoqni bukish
      FR_Hip.write(map(i, 0, speed, 90, 75));
      FL_Hip.write(map(i, 0, speed, 90, 100));
      FR_Knee.write(map(i, 0, speed, 80, 40));
      FL_Knee.write(map(i, 0, speed, 90, 130));
      head.write(map(i, 0, speed, 20, 0));
      delay(20);
    }
    for (int i = 0; i <= speed + 20; i++) {  // Old oyoqni ko'tarish
      FR_Hip.write(map(i, 0, speed, 75, 90));
      FL_Hip.write(map(i, 0, speed, 100, 90));
      FR_Knee.write(map(i, 0, speed, 40, 80));
      FL_Knee.write(map(i, 0, speed, 130, 90));
      head.write(map(i, 0, speed, 0, 20));
      delay(20);
    }
  }

  for (int i = 0; i <= speed + 20; i++) {  // Orqa oyoqlarni bukish
    BL_Hip.write(map(i, 0, speed, 60, 0));
    BR_Hip.write(map(i, 0, speed, 110, 180));
    BL_Knee.write(map(i, 0, speed, 110, 0));
    BR_Knee.write(map(i, 0, speed, 70, 180));
    delay(20);
  }
  for (int i = 0; i <= speed + 20; i++) {  // Old oyoqlarni bukish
    FL_Hip.write(map(i, 0, speed, 90, 105));
    FR_Hip.write(map(i, 0, speed, 90, 75));
    FL_Knee.write(map(i, 0, speed, 90, 180));
    FR_Knee.write(map(i, 0, speed, 80, 0));
    delay(20);
  }
}

void do_fire_alert() {
  Serial.println("he yoo");
  int start_freq = 400;
  int end_freq = 800;
  int step_delay = 15;
  int freq_step = 3;

  for (int hz = start_freq; hz <= end_freq; hz += freq_step) {
    tone(buzzer, hz);
    delay(step_delay);
  }
  for (int hz = end_freq; hz >= start_freq; hz -= freq_step) {
    tone(buzzer, hz);
    delay(step_delay);
  }
  noTone(buzzer);
}

void onWebSocketEvent(AsyncWebSocket *server, AsyncWebSocketClient *client, AwsEventType type, void *arg, uint8_t *data, size_t len) {
  if (type == WS_EVT_CONNECT) {
    Serial.printf("WebSocket client #%u ulangan!\n", client->id());
  } else if (type == WS_EVT_DISCONNECT) {
    Serial.printf("WebSocket client #%u uzildi\n", client->id());
  } else if (type == WS_EVT_DATA) {
    char command[len + 1];
    memcpy(command, data, len);
    command[len] = '\0';
    String cmdStr = String(command);
    Serial.printf("Buyruq keldi: %s\n", cmdStr.c_str());

    // Kelgan buyruqqa qarab tegishli funksiyani chaqiramiz
    if (cmdStr == "stand_up")        do_stand_up(); 
    else if (cmdStr == "sit_down")   do_sit_down(); 
    else if (cmdStr == "push_up")    do_push_up(1); 
    else if (cmdStr == "tik_turish") do_tik_turish(); 
    else if (cmdStr == "walk")       do_walk(1); 
    else if (cmdStr == "fire_alert") do_fire_alert(); 
  }
}

void setup() {
  Serial.begin(115200);

  FL_Hip.attach(FL_HIP_PIN);
  FL_Knee.attach(FL_KNEE_PIN);
  FR_Hip.attach(FR_HIP_PIN);
  FR_Knee.attach(FR_KNEE_PIN);
  BL_Hip.attach(BL_HIP_PIN);
  BL_Knee.attach(BL_KNEE_PIN);
  BR_Hip.attach(BR_HIP_PIN);
  BR_Knee.attach(BR_KNEE_PIN);

  neck.attach(NECK_PIN);
  head.attach(HEAD_PIN);

  FL_Hip.write(110);
  FR_Hip.write(70);
  delay(50);
  FL_Knee.write(180);
  FR_Knee.write(0);
  delay(50);
  BL_Hip.write(0);
  BR_Hip.write(180);
  delay(50);
  BL_Knee.write(0);
  BR_Knee.write(180);
  delay(50);
  head.write(20);
  delay(50);
  neck.write(40);
  delay(50);

  // Wi-Fi Access Point yaratamiz
  WiFi.softAP(ssid, password);
  IPAddress IP = WiFi.softAPIP();
  Serial.print("AP IP manzili: ");
  Serial.println(IP);

  ws.onEvent(onWebSocketEvent);
  server.addHandler(&ws);

  server.begin();
  Serial.println("HTTP server ishga tushdi");
}

void loop() {
  ws.cleanupClients();
}
