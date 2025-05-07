
void setup() {
  Serial.begin(9600);
  // Set up motor pins here
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    if (command == 'vasakule') {
      turnLeft();
    }
  }
}

// basic functions forward, back, left, right

void motors_forward() {
    analogWrite(mot1f, mot_speed);
    analogWrite(mot2f, mot_speed);
    digitalWrite(mot1b, LOW);
    digitalWrite(mot2b, LOW);
}


void motors_back() {
    digitalWrite(mot1f, LOW);
    digitalWrite(mot2f, LOW);
    analogWrite(mot1b, mot_speed);
    analogWrite(mot2b, mot_speed);
}


void motors_stop() {
    digitalWrite(mot1f, LOW);
    digitalWrite(mot2f, LOW);
    digitalWrite(mot1b, LOW);
    digitalWrite(mot2b, LOW);
}


void motors_left() {
    analogWrite(mot1f, mot_speed);
    digitalWrite(mot2f, LOW);
    digitalWrite(mot1b, LOW);
    analogWrite(mot2b, LOW);
}


void motors_right() {
    digitalWrite(mot1f, LOW);
    analogWrite(mot2f, mot_speed);
    analogWrite(mot1b, LOW);
    digitalWrite(mot2b, LOW);
}

}
//push for opening straight from arduino ide