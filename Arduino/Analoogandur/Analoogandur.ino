// Motor control pins
const int IN1 = 5;   // Left motor forward
const int IN2 = 6;   // Left motor backward
const int IN3 = 9;   // Right motor forward
const int IN4 = 10;  // Right motor backward

// IR sensor pins
const int IR_SENSOR_LEFT = A0;  // Left IR sensor pin
const int IR_SENSOR_RIGHT = A1; // Right IR sensor pin

// Motor speed (PWM value)
int motorSpeed = 10; // PWM speed (0–255)

// Threshold for comparison (sensitivity for matching values)
int threshold = 5;  // Mõõtsime tulemusi ning 5ga võiks juba otse minna

void setup() {
  // Set motor pins as output
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  Serial.begin(9600);
}

void loop() {
  int leftSensorValue = analogRead(IR_SENSOR_LEFT);
  int rightSensorValue = analogRead(IR_SENSOR_RIGHT);

  Serial.print("Left Sensor: ");//Kui loeme serial printi ss võib jätta, aga muidu pole esmatähtis
  Serial.print(leftSensorValue);
  Serial.print("  Right Sensor: ");
  Serial.println(rightSensorValue);

  int diff = abs(leftSensorValue - rightSensorValue);//

  if (diff > threshold) {
    if (leftSensorValue > rightSensorValue) {
      turnLeft();
      Serial.println("Turning Left.");
      delay(1000);
    } else {
      turnRight();
      Serial.println("Turning Right.");
      delay(1000);
    }
  } else {
    moveForward();
    Serial.println("Moving Forward.");
    delay(1000);
  }

  delay(100);
}

void moveForward() {
  analogWrite(IN1, motorSpeed);     // Left motor forward
  digitalWrite(IN2, LOW);
  analogWrite(IN3, motorSpeed);     // Right motor forward
  digitalWrite(IN4, LOW);
}


// Reversed moveBackward
void moveBackward() {
  analogWrite(IN1, motorSpeed);
  digitalWrite(IN2, LOW);
  analogWrite(IN3, motorSpeed);
  digitalWrite(IN4, LOW);
}

// Reversed turnLeft
void turnLeft() {
  digitalWrite(IN1, LOW);
  analogWrite(IN2, motorSpeed);     // Left wheel backward
  analogWrite(IN3, motorSpeed);     // Right wheel forward
  digitalWrite(IN4, LOW);
}



// Reversed turnRight
void turnRight() {
  analogWrite(IN1, motorSpeed);     // Left wheel forward
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  analogWrite(IN4, motorSpeed);     // Right wheel backward
}

