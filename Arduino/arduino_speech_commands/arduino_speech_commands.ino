
// Mootorid
const int mot1f = 6, mot1b = 5; //parem
const int mot2f = 3, mot2b = 2; //vasak
int mot_speed = 100;
const int errorLED = 13;


String input = "";

void setup() {
  Serial.begin(9600);

  // Set motor pins as outputs
  pinMode(mot1f, OUTPUT);
  pinMode(mot1b, OUTPUT);
  pinMode(mot2f, OUTPUT);
  pinMode(mot2b, OUTPUT);

  // Optional: Blink LED to indicate startup
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);
  delay(200);
  digitalWrite(LED_BUILTIN, LOW);
}

void loop() {
  // Read input string from serial until newline
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      handleCommand(input);
      input = "";
    } else {
      input += c;
    }
  }
}

void handleCommand(String command) {
  Serial.print("Received command: ");
  Serial.println(command);

  if (command.equals("vasakule")) {
    motors_left();
  } else if (command.equals("paremale")) {
    motors_right();
  } else if (command.equals("otse")) {
    motors_forward();
  } else if (command.equals("tagasi")) {
    motors_back();
  } else if (command.equals("stop")) {
    motors_stop();
  } else {
    Serial.println("Unknown command.");
  }
}

// === Motor movement functions ===

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
  analogWrite(mot2b, mot_speed);
}

void motors_right() {
  digitalWrite(mot1f, LOW);
  analogWrite(mot2f, mot_speed);
  analogWrite(mot1b, mot_speed);
  digitalWrite(mot2b, LOW);
}