package com.karolbystrek.neuralnetwork;

import java.io.IOException;

public class App {

    private static final float LEARNING_RATE = 0.01f;
    private static final int EPOCHS = 100;
    private static final int BATCH_SIZE = 32;

    public static void main(String[] args) {
        DataPoint[] trainingData;
        try {
            trainingData = MnistDataReader.readData("data/MNIST/train-images.idx3-ubyte", "data/MNIST/train-labels.idx1-ubyte");
        } catch (IOException e) {
            System.err.println("Error reading training data: " + e.getMessage());
            return;
        }
        NeuralNetwork model = new NeuralNetwork(new int[]{784, 128, 64, 10});
        model.fit(trainingData, EPOCHS, BATCH_SIZE, LEARNING_RATE);

        int correctPredictions = 0;
        int totalPredictions = 0;
        for (DataPoint dataPoint : trainingData) {
            float[] output = model.predict(dataPoint);
            float[] expectedOutput = dataPoint.getExpectedOutput();
            int predictedLabel = getPredictedLabel(output);
            int expectedLabel = getExpectedLabel(expectedOutput);

            System.out.println("Network output: " + predictedLabel + ", Expected output: " + expectedLabel);

            totalPredictions++;
            correctPredictions += (predictedLabel == expectedLabel) ? 1 : 0;
        }
        System.out.println("Accuracy: " + correctPredictions / totalPredictions * 100.0f + "%");
    }

    private static int getPredictedLabel(float[] output) {
        int predictedLabel = 0;
        float maxProbability = output[0];
        for (int i = 1; i < output.length; i++) {
            if (output[i] > maxProbability) {
                maxProbability = output[i];
                predictedLabel = i;
            }
        }
        return predictedLabel;
    }

    private static int getExpectedLabel(float[] expectedOutput) {
        int expectedLabel = 0;
        for (int i = 1; i < expectedOutput.length; i++) {
            if (expectedOutput[i] > expectedOutput[expectedLabel]) {
                expectedLabel = i;
            }
        }
        return expectedLabel;
    }
}
