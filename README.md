# cluster-experiment

All'interno della directory  `forward_top_bottom_lamp_simulation` è presente il file train.py il quale consente di iniziare l'addestramento della rete neurale. Tale addestramento è costituito da 150 epoche, i dettagli dell'andamento sono contenuti all'interno del file `forward_top_bottom_lamp_simulation/outputs/launch.log`, in cui è possibile visualizzare i valori delle loss di train e di validation, nonché il numero delle epoche raggiunto.
Quando l'addestramento è completato all'interno della directory `forward_top_bottom_lamp_simulation/checkpoints` sarà possibile trovare i file contenenti i dati del modello salvato per le diverse epoche, considerando che il salvataggio avviene ogni 50 epoche.

# Istruzioni deployment
Fra i file forniti nel repository è possibile trovare un `Dockerfile` e un `docker-compose.yaml`; questi file consentono rispettivamente di creare l'immagine del container e lanciare la sua esecuzione all'interno del cluster alma-ai. 

Prima di lanciare il file docker-compose è necessario effettuare la build dell'immagine tramite il seguente comando:

```bash
docker build --build-arg USER_NAME="your user name" --build-arg APP_VERSION=1.0 -t "your image name":1.0 .
```
Questa operazione può richiedere abbastanza tempo, in quanto l'immagine fornita da nvidia (che rappresenta il punto di partenza per costruire quest'immagine) è abbastanza pesante.

All'interno del `docker-compose.yaml`, invece, una volta ottenuta l'immagine occorre impostare, oltre che il nome dell'immagine, anche i valori delle variabili d'ambiente.

## Note importanti
Per poter lanciare il training è necessario **disporre di una GPU nvidia**, come richiesto dalla libreria [Nvidia Modulus](https://docs.nvidia.com/deeplearning/modulus/getting-started/index.html).
