// server.js
const express = require('express');
const cors = require('cors');
const { Worker } = require('worker_threads');
require('dotenv').config();

process.on('unhandledRejection', (reason, promise) => {
    console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

process.on('uncaughtException', (err) => {
    console.error('Uncaught Exception:', err);
});

// Set up Express app
const app = express();
app.use(cors());
app.use(express.json({ limit: "10mb" }));

// Worker thread function
function runWorker(payload) {
    return new Promise((resolve, reject) => {
        const worker = new Worker('./worker.js', { workerData: payload });
        worker.on('message', resolve);
        worker.on('error', reject);
        worker.on('exit', (code) => {
            if (code !== 0) reject(new Error(`Worker stopped with exit code ${code}`));
        });
    });
}

// Prediction endpoint
app.post('/predict', async (req, res) => {
    const { base64 } = req.body;

    if (!base64) {
        return res.status(400).json({ error: 'No image data provided' });
    }

    try {
        const result = await runWorker({ base64 });

        if(result.error) {
            return res.status(500).json({ error: 'Error during prediction attemp' });
        }

        res.json(result);
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: 'Error during prediction' });
    }
});

const HOST = '0.0.0.0';
const PORT = process.env.PORT || 3000;
app.listen(PORT, HOST, () => console.log(`Server running on ${HOST}:${PORT}`));
