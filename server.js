// server.js
const express = require('express');
const cors = require('cors');
const { Worker } = require('worker_threads');
const {supabase} = require("./supabase");
const tf = require("@tensorflow/tfjs-node");
require('dotenv').config();

console.log(`Running Node.js version: ${process.version}`);


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

let datas = {model: undefined, labels: []};
let modelLoaded = false;

const WORKER_COUNT = 2; // Number of workers to preload
const workers = [];
const workerQueue = [];

const getSignedUrls = async () => {
    try {
        const { data: modelJsonUrl, error: modelJsonError } = supabase
            .storage
            .from('models')
            .getPublicUrl('model2/model.json');

        const { data: weightsUrl, error: weightsError } = supabase
            .storage
            .from('models')
            .getPublicUrl('model2/weights.bin');

        const { data: metaData, error: metaDataError } = supabase
            .storage
            .from('models')
            .getPublicUrl('model2/metadata.json');

        if (modelJsonError || weightsError || metaDataError) {
            console.error('Error generating signed URLs:', modelJsonError || weightsError || metaDataError);
            return null;
        }

        return {
            modelJsonUrl: modelJsonUrl.publicUrl,
            weightsUrl: weightsUrl.publicUrl,
            metaDataUrl: metaData.publicUrl
        };
    } catch (error) {
        console.error('Error during URL fetch:', error);
        return null;
    }
};

let signUrls = null;

const createWorker = () => {
    return new Promise((resolve, reject) => {
        const worker = new Worker('./worker.js', {
            workerData: {
                modelJsonUrl: signUrls.modelJsonUrl,
                metaDataUrl: signUrls.metaDataUrl,
            },
        });

        worker.on('message', (message) => {
            if (message === 'ready') {
                console.log('Worker ready.');
                workers.push(worker);
                resolve();
            }
        });

        worker.on('error', (err) => {
            console.error('Worker error:', err);
            reject(err);
        });

        worker.on('exit', (code) => {
            if (code !== 0) console.error(`Worker exited with code ${code}`);
        });

        workerQueue.push(worker); // Add worker to the queue
    });
}

const getAvailableWorker = () => {
    if (workerQueue.length === 0) {
        throw new Error('No workers available');
    }
    return workerQueue.shift();
}

const returnWorker = (worker) => {
    workerQueue.push(worker);
}


// Prediction endpoint
app.post('/predict', async (req, res) => {

    const { base64 } = req.body;

    if (!base64) {
        console.log("Image not provided")
        return res.status(400).json({ error: 'No image data provided' });
    }

    try {
        const worker = getAvailableWorker();

        const result = await new Promise((resolve, reject) => {
            worker.once('message', resolve);
            worker.once('error', reject);
            worker.postMessage({ base64 });
        });

        returnWorker(worker); // Return the worker to the queue
        res.json(result);
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: 'Error during prediction' });
    }
});

const HOST = '0.0.0.0';
const PORT = process.env.PORT || 3000;

(async () => {
    console.log('Preloading urls...');
    signUrls = await getSignedUrls();
    console.log('Urls loaded');
    console.log('Preloading workers...');
    await Promise.all(Array.from({ length: WORKER_COUNT }, createWorker));
    console.log(`All ${WORKER_COUNT} workers ready.`);
    app.listen(PORT, HOST, () => console.log(`Server running on ${HOST}:${PORT}`));
})();
