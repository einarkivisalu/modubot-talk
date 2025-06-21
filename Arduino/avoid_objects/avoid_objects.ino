// Programm töötab Arduino Mega 2560 peal

// Ultraheliandurid
const int trigPins[] = {49, 51, 53};
const int echoPins[] = {43, 45, 47};
const int numSensors = 3;
const int distStop = 50;
const int distSlow = 100;
const int maxRange = 800;
const int minRange = 0;
const int Vcc = 9;


// Mootorid
const int mot1f = 6, mot1b = 5; //parem
const int mot2f = 3, mot2b = 2; //vasak
int mot_speed = 100;
const int errorLED = 13;


// input output
void setup() {
    pinMode(Vcc, OUTPUT);
    digitalWrite(Vcc, HIGH);
    for (int i = 0; i < numSensors; i++) {
        pinMode(trigPins[i], OUTPUT);
        pinMode(echoPins[i], INPUT);
    }
    pinMode(mot1f, OUTPUT);
    pinMode(mot1b, OUTPUT);
    pinMode(mot2f, OUTPUT);
    pinMode(mot2b, OUTPUT);
}


// --------PEATSÜKKEL-------- (objektide vältimine)
void loop() {
  autopilot();
}

//AUTOPILOOT
void autopilot() {
    int distances[numSensors];
    for (int i = 0; i < numSensors; i++) {
        distances[i] = getDistance(trigPins[i], echoPins[i]);
    }
    int left = distances[0];
    int center = distances[1];
    int right = distances[2];


    if (center <= distStop || left <= distStop || right <= distStop) {
        motors_stop();        // Peatu
        motors_back();        // Tagurda
        delay(800);           // Väike paus


        if (left > right) {
            motors_left();   // Kui vasakul rohkem ruumi, pööra sinna
            delay(500);
        } else {
            motors_right();  // Muidu paremale
            delay(500);
        }


        motors_stop()
        delay(200);           // Paus enne edasiminekut


    } else if (center < distSlow || left < distSlow || right < distSlow) {
        int nearest = min(left, min(center, right));
        motors_slows(nearest);     // Aeglusta vastavalt lähima takistuse kaugusele
        delay(100);                // Väike viide
    } else {
        motors_forward();          // Liigu edasi
    }
}



// ----ANDURID----
int getDistance(int trigPin, int echoPin) {
    digitalWrite(trigPin, LOW);
    delayMicroseconds(2);
    digitalWrite(trigPin, HIGH);
    delayMicroseconds(10);
    digitalWrite(trigPin, LOW);
    long duration = pulseIn(echoPin, HIGH);
    delay(50);
    int distance = duration / 58;
    return (distance >= maxRange || distance <= minRange) ? maxRange : distance;
}


// ----MOOTORIFUNKTSIOONID----
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


void motors_forward_left() {
    analogWrite(mot1f, mot_speed);
    analogWrite(mot2f, mot_speed * 0.5);
    digitalWrite(mot1b, LOW);
    digitalWrite(mot2b, LOW);


}


void motors_forward_right() {
    analogWrite(mot1f, mot_speed * 0.5);
    analogWrite(mot2f, mot_speed);
    digitalWrite(mot1b, LOW);
    digitalWrite(mot2b, LOW);
}


void motors_back_left() {
    analogWrite(mot1b, mot_speed * 0.5);
    analogWrite(mot2b, mot_speed);
    digitalWrite(mot1f, LOW);
    digitalWrite(mot2f, LOW);
}


void motors_back_right() {
    analogWrite(mot1b, mot_speed);
    analogWrite(mot2b, mot_speed * 0.5);
    digitalWrite(mot1f, LOW);
    digitalWrite(mot2f, LOW);
}


void motors_slows(int distance) {
    const int stopDistance = distStop;
    const int slowDownDistance = distSlow;
    const int minSpeed = 40;
    const int maxSpeed = mot_speed;




    int speed;
    if (distance <= stopDistance + 5) {
        speed = 40;
    } else if (distance <= slowDownDistance) {
        speed = map(distance, stopDistance, slowDownDistance, minSpeed, maxSpeed);
    } else {
        speed = maxSpeed;
    }

    analogWrite(mot1f, speed);
    analogWrite(mot2f, speed);
    digitalWrite(mot1b, LOW);
    digitalWrite(mot2b, LOW);
}
