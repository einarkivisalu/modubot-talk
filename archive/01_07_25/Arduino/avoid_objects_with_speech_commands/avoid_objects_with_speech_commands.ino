// --- Motors ---
const int mot1f = 6, mot1b = 5; // Right motor
const int mot2f = 3, mot2b = 2; // Left motor
int mot_speed = 40;

// --- Ultrasonic Sensors ---
const int trigPins[] = {49, 51, 53};  // Left, Center, Right
const int echoPins[] = {43, 45, 47};
const int numSensors = 3;
const int distStop = 50;
const int distSlow = 100;
const int maxRange = 800;
const int minRange = 0;
const int Vcc = 9;

// --- Global Variables ---
String input = "";
String currentCommand = "";  // store the last manual command
bool isMoving = false;      // if robot should move

void setup() {
  Serial.begin(9600);

  // Motor pins
  pinMode(mot1f, OUTPUT);
  pinMode(mot1b, OUTPUT);
  pinMode(mot2f, OUTPUT);
  pinMode(mot2b, OUTPUT);

  // Sensors
  pinMode(Vcc, OUTPUT);
  digitalWrite(Vcc, HIGH);

  for (int i = 0; i < numSensors; i++) {
    pinMode(trigPins[i], OUTPUT);
    pinMode(echoPins[i], INPUT);
  }

  motors_stop();
}

void loop() {
  // Read serial input for speech commands
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      handleCommand(input);
      input = "";
    } else {
      input += c;
    }
  }

  // If a command was set, execute it once with obstacle check
  if (isMoving && currentCommand != "") {
    int left = getDistance(trigPins[0], echoPins[0]);
    int center = getDistance(trigPins[1], echoPins[1]);
    int right = getDistance(trigPins[2], echoPins[2]);

    if (center <= distStop || left <= distStop || right <= distStop) {
      // Obstacle detected: avoid it once
      avoidObstacle(left, center, right);
      motors_stop();
    } else {
      // No obstacle, execute the command once
      executeCommand(currentCommand);
      delay(500);  // Move a bit (adjust as needed)
      motors_stop();
    }

    // Reset flags so it doesn’t repeat endlessly
    isMoving = false;
    currentCommand = "";
  }
}

// Handle speech commands, start/stop movement
void handleCommand(String command) {
  command.trim();
  Serial.print("Received command: ");
  Serial.println(command);

  if (command.equalsIgnoreCase("stop")) {
    motors_stop();
    isMoving = false;
    currentCommand = "";
  } else if (
      command.equalsIgnoreCase("otse") ||
      command.equalsIgnoreCase("vasakule") ||
      command.equalsIgnoreCase("paremale") ||
      command.equalsIgnoreCase("tagasi")) {
    currentCommand = command;
    isMoving = true;
    executeCommand(command);
  } else {
    Serial.println("Tundmatu käsklus.");
  }
}

// Executes motor functions based on command string
void executeCommand(String command) {
  if (command.equalsIgnoreCase("otse")) {
    motors_forward();
  } else if (command.equalsIgnoreCase("vasakule")) {
    motors_left();
  } else if (command.equalsIgnoreCase("paremale")) {
    motors_right();
  } else if (command.equalsIgnoreCase("tagasi")) {
    motors_back();
  }
}

// Obstacle avoidance routine (simple)
void avoidObstacle(int left, int center, int right) {
  motors_stop();
  delay(200);
  motors_back();
  delay(600);

  if (left > right) {
    motors_left();
  } else {
    motors_right();
  }
  delay(500);
  motors_stop();
  delay(200);
}

int getDistance(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  long duration = pulseIn(echoPin, HIGH, 20000); // 20ms timeout
  int distance = duration / 58;

  if (distance <= minRange || distance >= maxRange) {
    return maxRange;
  }
  return distance;
}

// Motor movement functions
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

// Slow down motors based on proximity
void motors_slows(int distance) {
  const int minSpeed = 40;
  int speed = map(distance, distStop, distSlow, minSpeed, mot_speed);
  speed = constrain(speed, minSpeed, mot_speed);

  analogWrite(mot1f, speed);
  analogWrite(mot2f, speed);
  digitalWrite(mot1b, LOW);
  digitalWrite(mot2b, LOW);
}
