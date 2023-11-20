from confluent_kafka import Producer, Consumer, KafkaError
from flask import Flask, jsonify
from flask import request
import json
from flask_cors import CORS
import pandas as pd
import joblib  # Si vous utilisez un modèle sklearn
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)
CORS(app)

# Configuration du producteur Kafka
producer_conf = {'bootstrap.servers': 'localhost:9092'}

# Configuration du consommateur Kafka
consumer_conf = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'mygroup',
    'auto.offset.reset': 'earliest'
}
# Remplacez 'best_model.pkl' par le nom réel de votre modèle
best_model = joblib.load('meilleur_modele_svm.pkl')


@app.route('/data', methods=['POST'])
def process_data():
    # Supposons que les données sont envoyées en tant que JSON dans la requête POST
    data = request.json
    result = producer_consumer(data)
    return result


def producer_consumer(data):
    def consume_messages(consumer):
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print("Erreur lors de la consommation : {}".format(msg.error()))
                continue

            message_data = json.loads(msg.value().decode('utf-8'))
            # Afficher les données reçues
            print(f"Message reçu: {message_data}")
            data_df = pd.DataFrame([message_data])
            print(data_df)
            # Supprimer les colonnes non pertinentes ou uniques pour chaque entrée
            columns_to_drop = ['name', 'onboard_date',
                               'location', 'company']
            datax = data_df.drop(columns=columns_to_drop)
            print(datax)
            sc_x = StandardScaler()
            datac = sc_x.fit_transform(datax)
            print(datac)
            # Effectuer des prédictions en temps réel
            # Utiliser le modèle pour prédire
            prediction = best_model.predict(pd.DataFrame(datac))
            print(prediction)
            # Publier les prédictions sur un autre topic Kafka
            # producer.produce(producer_topic, json.dumps({'prediction': prediction.tolist()}))
            return jsonify({'prediction': prediction.tolist()}), 200

    producer = Producer(producer_conf)
    consumer = Consumer(consumer_conf)
    # Remplacez 'topic_name' par votre nom de topic Kafka
    consumer.subscribe(['topic'])

    producer.produce('topic', json.dumps(data).encode('utf-8'))
    producer.flush()

    result = consume_messages(consumer)

    consumer.close()
    producer.flush()

    return result


if __name__ == '__main__':
    app.run(debug=True)
