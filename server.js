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

const fetchLabels = async (url) => {
    try {
        const response = await fetch(url);

        if(!response.ok) {
            console.log("Error fetching labels from url")
            return null;
        }
        const metadata = await response.json();
        return metadata.labels;
    }catch (error) {
        console.log("Error fetching labels");
        return null;
    }
}

// Load model and labels once
(async () => {
    try {
        const signUrls = await getSignedUrls();
        if(signUrls) {
            const model = await tf.loadLayersModel(signUrls.modelJsonUrl);
            console.log("v2: Model has been loaded!");
            const labels = await fetchLabels(signUrls.metaDataUrl);
            console.log("v2: Labels has been loaded");
            datas = {model, labels};
            modelLoaded = true;
        }
    } catch (err) {
        console.error('Error loading models:', err);
    }
})();

// Worker thread function
function runWorker(payload) {
    return new Promise((resolve, reject) => {
        const { model, labels } = datas;
        const worker = new Worker('./worker.js', { workerData: { model, labels } });
        // This will log if the worker is instantiated correctly
        console.log("Worker created, sending data:");
        worker.postMessage(payload);

        worker.on('message', resolve);
        worker.on('error', reject);
        worker.on('exit', (code) => {
            if (code !== 0) reject(new Error(`Worker stopped with exit code ${code}`));
        });
    });
}

// Prediction endpoint
app.post('/predict', async (req, res) => {

    const { model, labels } = datas;
    if(!modelLoaded || !model || !labels) {
        console.log("Model not loaded!");
        return res.status(400).json({ error: 'Model not loaded' });
    }

    const { base64 } = req.body;

    if (!base64) {
        console.log("Image not provided")
        return res.status(400).json({ error: 'No image data provided' });
    }

    try {
        console.log("Predicting...")
        const result = await runWorker({ base64 });
        console.log("Result received!")
        if(result.error) {
            console.log("Error during prediction")
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
