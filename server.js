const express = require('express');
const tf = require('@tensorflow/tfjs-node');
const cors = require('cors');
const { supabase } = require('./supabase');
const fetch = require('node-fetch');
require('dotenv').config();

// Handle exceptions and rejections
process.on('unhandledRejection', (reason, promise) => {
    console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});
process.on('uncaughtException', (err) => {
    console.error('Uncaught Exception:', err);
});

// Set up Express app
const app = express();
app.use(cors());
app.use(express.json({ limit: '10mb' }));

let model;
let labels = [];
let modelLoaded = false;

// Function to fetch dataset images from Supabase
const getImagesFromSupabase = async () => {
    try {
        const { data: files, error } = await supabase
            .storage
            .from('models')
            .list('datasets/', { prefix: 'datasets/' });

        if (error) {
            console.error('Error fetching files:', error);
            return null;
        }

        // Filter and organize images by class
        const imageClasses = {};
        for (const file of files) {
            const className = file.name.split('/')[1]; // Extract class name
            const imageUrl = supabase.storage.from('models').getPublicUrl(file.name).publicUrl;

            if (!imageClasses[className]) {
                imageClasses[className] = [];
            }
            imageClasses[className].push(imageUrl);
        }

        return imageClasses;
    } catch (error) {
        console.error('Error fetching images:', error);
        return null;
    }
};

// Function to fetch the MobileNet base model
const loadBaseModel = async () => {
    try {
        model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v2_1.0_224/model.json');
        console.log('MobileNet base model loaded.');
    } catch (error) {
        console.error('Error loading the model:', error);
    }
};

// Function to prepare image tensor
const prepareImage = (imageUrl) => {
    return new Promise((resolve, reject) => {
        fetch(imageUrl)
            .then(response => response.buffer())
            .then(imageBuffer => {
                const tensor = tf.node.decodeImage(imageBuffer)
                    .resizeBilinear([224, 224])
                    .div(tf.scalar(255))
                    .expandDims(0);
                resolve(tensor);
            })
            .catch(reject);
    });
};

// Train model using dataset from Supabase
const trainModel = async () => {
    try {
        const imageClasses = await getImagesFromSupabase();
        if (!imageClasses) return;

        const classNames = Object.keys(imageClasses);
        labels = classNames;

        // Prepare images and labels for training
        const imageTensors = [];
        const classLabels = [];
        for (const [className, imageUrls] of Object.entries(imageClasses)) {
            for (const imageUrl of imageUrls) {
                const tensor = await prepareImage(imageUrl);
                imageTensors.push(tensor);
                classLabels.push(classNames.indexOf(className)); // Use index as class label
            }
        }

        const xs = tf.concat(imageTensors, 0);
        const ys = tf.tensor(classLabels, [classLabels.length], 'int32');

        // Fine-tuning MobileNet: remove the final layer and add custom layers
        const baseModel = model;
        const x = baseModel.layers[baseModel.layers.length - 2].output; // Remove the last layer
        const x2 = tf.layers.dense({ units: 128, activation: 'relu' }).apply(x);
        const output = tf.layers.dense({ units: labels.length, activation: 'softmax' }).apply(x2);
        const newModel = tf.model({ inputs: baseModel.inputs, outputs: output });

        newModel.compile({ optimizer: 'adam', loss: 'sparseCategoricalCrossentropy', metrics: ['accuracy'] });

        // Train the model
        await newModel.fit(xs, ys, { epochs: 5, batchSize: 32 });
        console.log('Training complete.');

        // Save the fine-tuned model
        await newModel.save('file://./fine_tuned_model');
        console.log('Fine-tuned model saved.');
        model = newModel;
        modelLoaded = true;
    } catch (error) {
        console.error('Error during training:', error);
    }
};

// Load model and start training
(async () => {
    await loadBaseModel();
    await trainModel();
})();

// Prediction route
app.post('/predict', async (req, res) => {
    try {
        if (!modelLoaded) {
            return res.status(500).json({ error: 'Model not loaded' });
        }

        const { base64 } = req.body;

        if (!base64) {
            return res.status(400).json({ error: 'No image data provided' });
        }

        const prediction = tf.tidy(() => {
            const imageBuffer = Buffer.from(base64, 'base64');
            const tensor = tf.node.decodeImage(imageBuffer, 3)
                .resizeBilinear([224, 224])
                .div(tf.scalar(255))
                .expandDims(0);

            const prediction = model.predict(tensor);
            const predictionData = prediction.arraySync(); // Get prediction array

            const maxIndex = predictionData[0].indexOf(Math.max(...predictionData[0]));
            const confidence = Math.max(...predictionData[0]);

            return {
                label: labels[maxIndex],
                confidence
            };
        });

        res.json(prediction);
    } catch (error) {
        console.error('Error during prediction:', error);
        res.status(500).json({ error: 'Error during prediction' });
    }
});

const HOST = '0.0.0.0';
const PORT = process.env.PORT || 3000;
app.listen(PORT, HOST, () => console.log(`Server running on ${HOST}:${PORT}`));
