# modubot-talk
## Taltech IT Kolledži robootikaringi näpuharjutused kõneleva roboti tegemiseks
=======
Eesmärk on tekitada moodul, mis suudab teisendada eestikeelse kõne tekstiks ja vastupidi

### record_and_transcribe
Praegune kood kõnesünteesiks. Text-to-speech ja speech-to-text. 
- Vajalik on Huggingface Whisper AI.
- Vajab internetti transkriptsiooniks.

### Roboti juhtimine häälkäsklustega
raspberry_commands_threading.py ja (avoid_objects_with_speech_commands.ino või AOWSC_continuous) 

avoid_objects_with_speech_commands.ino == anna robotile käsklus, ta täidab seda 1x, samal ajal vältides objekte.
AOWSC_continuous.ino == robot täidab eelmist käsku ja väldib objekte kuni järgmise käsuni. (vaja veel testida-mõelda parim algoritm/liikumine)

### Credits
This project uses the FLEURS dataset by Google Research, licensed under CC BY 4.0.
Available at: https://huggingface.co/datasets/google/fleurs
