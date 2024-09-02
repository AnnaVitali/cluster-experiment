# cluster-experiment

All'interno della directory  `forward_top_bottom_lamp_simulation` vi è il file train.py che deve essere eseguito per iniziare l'addestramento della rete neurale.
L'addestramento è costituito da 150 epoche e una volta fatto partire all'interno del file `forward_top_bottom_lamp_simulation/outputs/launch.log` possono essere visualizzati i dettagli della loss.
Quando l'addestramento è completato all'interno della directory `forward_top_bottom_lamp_simulation/checkpoints` sarà possibile trovare i file contenenti i dati del modello salvato per le diverse epoche effettuate, considerando che il salvataggio avviene ogni 50 epoche.

## Note importanti
Per poter lanciare il training è necessario **disporre di una GPU nvidia**, come richiesto dalla libreria [Nvidia Modulus](https://docs.nvidia.com/deeplearning/modulus/getting-started/index.html).
