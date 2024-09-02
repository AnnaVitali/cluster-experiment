# cluster-experiment

All'interno della directory  `forward_top_bottom_lamp_simulation` vi è il file train.py che deve essere eseguito per iniziare l'addestramento della rete neurale.
L'addestramento è costituito da 150 epoche e una volta fatto partire all'interno del file `forward_top_bottom_lamp_simulation/outputs/launch.log` possono essere visualizzati i dettagli della loss.
Quando l'addestramento è completato all'interno della directory `forward_top_bottom_lamp_simulation/checkpoints` sarà possibile trovare i file contenenti i dati del modello salvato per le diverse epoche effettuate, considerando che il salvataggio avviene ogni 50 epoche.

# Istruzioni deployment
Fra i file forniti nel repository è possibile trovare un `Dockerfile` e un `docker-compose.yaml`. Questi file consentono rispettivamente di creare l'immagine del container e lanciare la sua esecuzione sul cluster almai. Prima di lanciare il file docker-compose è necessario effettuare la build dell'immagine tramite ad esempio il seguente comando:

```bash
docker build --build-arg USER_NAME=anna.vitali7 --build-arg APP_VERSION=1.0 -t thermoforming_simulazion:1.0 .
```
Questa operazione può richiedere abbastanza tempo, in quanto l'immagine fornita da nvidia (che rappresenta il punto di partenza per costruire quest'immagine) è abbastanza pesante.

All'interno del `docker-compose.yaml`, invece, una volta ottenuta l'immagine occorre impostare, oltre che il nome dell'immagine, con quello assegnato al momento della build, anche i valori delle variabili d'ambiente.

## Note importanti
Per poter lanciare il training è necessario **disporre di una GPU nvidia**, come richiesto dalla libreria [Nvidia Modulus](https://docs.nvidia.com/deeplearning/modulus/getting-started/index.html).
