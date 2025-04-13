package com.karolbystrek.neuralnetwork;

import java.util.Arrays;
import java.util.Collections;

public class NeuralNetwork {

    private final Layer[] layers;

    public NeuralNetwork(int[] layerSizes) {
        if (layerSizes == null || layerSizes.length < 2) {
            throw new IllegalArgumentException("The neural network must have at least two layers.");
        }
        layers = new Layer[layerSizes.length - 1];

        for (int i = 0; i < layers.length - 1; i++) {
            layers[i] = new HiddenLayer(layerSizes[i], layerSizes[i + 1]);
        }
        layers[layers.length - 1] = new OutputLayer(layerSizes[layerSizes.length - 2], layerSizes[layerSizes.length - 1]);
    }

    public float[] predict(DataPoint dataPoint) {
        return forward(dataPoint.getInput());
    }

    public void fit(DataPoint[] trainingData, int epochs, int batchSize, float learningRate) {
        System.out.println("Beginning training...");

        for (int epoch = 0; epoch < epochs; epoch++) {
            long startTime = System.nanoTime();
            System.out.print("Epoch " + (epoch + 1) + "/" + epochs + ": ");

            Collections.shuffle(Arrays.asList(trainingData));
            float averageCost = train(trainingData, batchSize, learningRate);

            System.out.print("Average cost: " + averageCost + " | ");
            System.out.println("Total time: " + (System.nanoTime() - startTime) / 1.0e9 + " ms");
        }
    }

    private float train(DataPoint[] trainingData, int batchSize,float learningRate) {
        float totalCost = 0.0f;
        int batchIndex = 0;

        for (DataPoint dataPoint : trainingData) {
            float[] output = forward(dataPoint.getInput());
            float[] expectedOutput = dataPoint.getExpectedOutput();

            totalCost += calculateCost(output, expectedOutput);
            float[] outputGradient = calculateOutputGradient(output, expectedOutput);
            backward(outputGradient);

            batchIndex++;
            if (batchIndex >= batchSize) {
                updateParameters(learningRate);
                batchIndex = 0;
            }

        }
        if (batchIndex > 0) {
            updateParameters(learningRate);
        }
        return totalCost / trainingData.length;
    }

    private float calculateCost(float[] output, float[] expectedOutput) {
        float cost = 0.0f;
        float EPSILON = 1e-13f;

        for (int i = 0; i < output.length; i++) {
            cost -= (float) (expectedOutput[i] * Math.log(output[i] + EPSILON));
        }
        return cost;
    }

    private float[] forward(float[] input) {
        float[] output = input;
        for (Layer layer : layers) {
            output = layer.forward(output);
        }
        return output;
    }

    private void backward(float[] outputGradient) {
        for (int i = layers.length - 1; i >= 0; i--) {
            outputGradient = layers[i].backward(outputGradient);
        }
    }

    private void updateParameters(float learningRate) {
        for (Layer layer : layers) {
            layer.updateParameters(learningRate);
        }
    }

    private float[] calculateOutputGradient(float[] output, float[] expectedOutput) {
        float[] outputGradient = new float[output.length];
        for (int i = 0; i < output.length; i++) {
            outputGradient[i] = output[i] - expectedOutput[i];
        }
        return outputGradient;
    }
}
