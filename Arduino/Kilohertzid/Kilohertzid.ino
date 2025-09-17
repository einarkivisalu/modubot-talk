void setup() {
  pinMode(3, OUTPUT);  // OC2B (PWM output)
}

void loop() {
  while (1){
  sendBurst(1, 2);  
  }

}

// Saada signaal onTime pikkuseks ajaühikuks ning hoia saattmine kinni offTime ajaühikuks
void sendBurst(unsigned int onTime, unsigned int offTime) {
  enablePWM();                 // Turn on 38kHz signal on pin 3
  //delayMicroseconds(onTime);  // Keep it on
  delay(onTime);
  disablePWM();               // Turn it off
  delay(offTime);
  //delayMicroseconds(offTime); // Wait before next burst
}

void enablePWM() {
  // Set up Timer2 for Fast PWM with OCR2A as TOP
  TCCR2A = _BV(COM2B1) | _BV(WGM20) | _BV(WGM21);
  TCCR2B = _BV(WGM22) | _BV(CS21);  // Prescaler = 8

  OCR2A = 52;   // TOP = 52 → ~37.7kHz
  OCR2B = 26;   // 50% duty cycle
}

void disablePWM() {
  TCCR2A &= ~_BV(COM2B1);  // Disconnect PWM from pin 3
  TCCR2B = 0;              // Stop Timer2
  digitalWrite(3, LOW);    // Ensure pin is off
}
