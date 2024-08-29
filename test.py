from flask import Flask, request, jsonify
import redis
import json
import os
import time

app = Flask(__name__)

# Set up Redis connection
try:
    redis_client = redis.Redis(
        host='redis-14670.c309.us-east-2-1.ec2.redns.redis-cloud.com',
        port=14670,
        password='LMtnadv5hvh8EYu43Ceep4YkbcU6o7Dz',
        decode_responses=True
    )
    # Check if Redis server is reachable
    redis_client.ping()
    print("Successfully connected to Redis!")
except redis.RedisError as e:
    print(f"Error connecting to Redis: {e}")
    redis_client = None  # Set to None or handle the error accordingly

@app.route('/webhook', methods=['POST'])
def webhook():
    if redis_client is None:
        return jsonify({'error': 'Redis connection not available'}), 500

    try:
        # Get GitHub event and payload
        event = request.headers.get('X-GitHub-Event')
        payload = request.json

        # Store the data in Redis
        key = f'github-webhook:{int(time.time())}'
        redis_client.set(key, json.dumps({'event': event, 'payload': payload}))

        return jsonify({'message': 'Webhook received'}), 200
    except Exception as e:
        print(f'Error handling webhook: {e}')
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
