from gtts import gTTS
import os
text = """Neljapäeva õhtul rääkisid „Ringvaates“ homme ETV ekraanil alustava krimidraama „Von Fock“ peaosatäitja Priit 
Pius ja operaator Mart Ratassepp viraalseks läinud stseenist, kus Piusi tegelane peab veel all tõllast põgenema. Stseeni
 oli Piusil äge filmida, kuid vajas hingamisregulaatoriga ettevalmistumist.

Sarja tarbeks võeti linti üks veealune stseen, kus peaosalist Paul von Focki kehastav näitleja Priit Pius pidi vette
vajuvast tõllast põgenema. Ta võitleb õhupuudusega ning tõllast põgenemine kaskadöörilikul viisil on omaette katsumus.
Stseenist postitas režissöör Arun Tamm katkendi sotsiaalmeediasse ning tänaseks on klippi vaadatud Instagramis üle 
29 miljoni korra."""

speech = gTTS(text=text, lang='et', slow=False)
speech.save("voice.mp3")
os.system("voice.mp3")
