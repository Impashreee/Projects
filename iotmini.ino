#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include "DHT.h"

#define DHTPIN 13
#define DHTTYPE DHT11

DHT dht(DHTPIN, DHTTYPE);
LiquidCrystal_I2C lcd(0x27, 16, 2);

const int mq2 = A0;
const int mq4 = A1;
const int mq5 = A2;
const int mq135 = A3;

float m = -0.6527;
float b = 1.30;
float R0 = 21.91;

void setup() {
  lcd.begin(16, 2);
  lcd.backlight();
  lcd.print("Atmosphere");
  lcd.setCursor(0, 1);
  lcd.print("Gas Analyser");
  delay(3000);
  lcd.clear();

  lcd.print("By");
  lcd.setCursor(0, 1);
  lcd.print("Zeno Modiff");
  delay(3000);
  lcd.clear();

  pinMode(mq4, INPUT);
  pinMode(mq135, INPUT);
  pinMode(mq5, INPUT);
  pinMode(mq2, INPUT);

  Serial.begin(9600);
  dht.begin();
}

void loop() {
  float h = dht.readHumidity();
  float t = dht.readTemperature();
  float f = dht.readTemperature(true);
  
  if (isnan(h) || isnan(t) || isnan(f)) {
    Serial.println(F("Failed to read from DHT sensor!"));
    return;
  }
  
  float hif = dht.computeHeatIndex(f, h);
  float hic = dht.computeHeatIndex(t, h, false);
  
  double mq4_value = analogToPPM(analogRead(mq4));
  double mq135_value = analogToPPM(analogRead(mq135));
  double mq5_value = analogToPPM(analogRead(mq5));
  double mq2_value = analogToPPM(analogRead(mq2));

  Serial.println("************");
  Serial.print("MQ-4 Value: "); Serial.print(mq4_value, DEC); Serial.println(" PPM");
  Serial.print("MQ-135 Value: "); Serial.print(mq135_value, DEC); Serial.println(" PPM");
  Serial.print("MQ-5 Value: "); Serial.print(mq5_value, DEC); Serial.println(" PPM");
  Serial.print("MQ-2 Value: "); Serial.print(mq2_value, DEC); Serial.println(" PPM");
  Serial.print("DHT-Temp: "); Serial.print(t); Serial.println(" °C");
  Serial.print("DHT-Humi: "); Serial.print(h); Serial.println(" %");
  Serial.println("************\n\n");

  displayOnLCD("CNG Value", mq4_value);
  displayOnLCD("AQI Value", mq135_value);
  displayOnLCD("LNG", mq5_value);
  displayOnLCD("SMOKE", mq2_value);
  displayOnLCD("Temp -", t);
  displayOnLCD("Humi -", h);
}

double analogToPPM(int aValue) {
  float sensor_volt;
  float RS_gas;
  float ratio;
  int sensorValue = aValue;

  sensor_volt = sensorValue * (5.0 / 1023.0);
  RS_gas = ((5.0 * 10.0) / sensor_volt) - 10.0;
  ratio = RS_gas / R0;
  double ppm_log = (log10(ratio) - b) / m;
  return ppm_log;
}

void displayOnLCD(const char* label, double value) {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(label);
  lcd.setCursor(0, 1);
  lcd.print(value);
  lcd.setCursor(10, 1);
  lcd.print("PPM");
  delay(3000);
}
