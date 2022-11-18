void setup() {
// put your setup code here, to run once:
pinMode(9,OUTPUT);
pinMode(10, OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
analogWrite(9, 300);
delay(500);
analogWrite(10, 300);
delay(500);
}
